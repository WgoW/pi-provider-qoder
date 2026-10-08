import { rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type {
  Api,
  AssistantMessage,
  AssistantMessageEvent,
  AssistantMessageEventStream,
  Context,
  Model,
  ToolCall,
} from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { clearQoderModelsMemCache } from "../catalog.js";
import type { QoderUsage } from "../credits.js";
import { qoderDecodeBody } from "../protocol/encoding.js";
import { streamQoder } from "../protocol/stream.js";
import { loadLiveFixture } from "./live-fixture.js";

// Pin the identity so the mocked fetch below only ever serves the chat request.
// Without a resolved identity, streamQoder fetches /userinfo first and consumes
// the mock response, leaving the chat read to fail on a locked stream.
vi.mock("../auth/oauth.js", () => ({
  resolveQoderIdentity: vi.fn().mockResolvedValue({
    access: "fake",
    userID: "test-user",
    email: "test@example.com",
    name: "Test User",
    machineID: "test-machine",
    refresh: "",
    expires: 0,
  }),
}));

/**
 * Build a single SSE `data:` line carrying a Qoder envelope:
 *   { headers, body: <JSON string>, statusCodeValue, statusCode }
 * The server wraps the OpenAI-style chunk inside `body` as a JSON string.
 */
function sseEnvelope(body: object, statusCodeValue = 200, statusCode = "OK"): string {
  return (
    "data:" +
    JSON.stringify({
      headers: { "Content-Type": ["application/json"] },
      body: JSON.stringify(body),
      statusCodeValue,
      statusCode,
    }) +
    "\n\n"
  );
}

const DONE_SSE =
  "data:" +
  JSON.stringify({
    headers: { "Content-Type": ["application/json"] },
    body: "[DONE]",
    statusCodeValue: 200,
    statusCode: "OK",
  }) +
  "\n\n";

function chunk(delta: object, extra: object = {}): object {
  return {
    choices: [{ delta, index: 0 }],
    created: 1,
    id: "test-id",
    model: "auto",
    object: "chat.completion.chunk",
    ...extra,
  };
}

function finishChunk(finish_reason: string, extra: object = {}): object {
  return {
    choices: [{ finish_reason, index: 0 }],
    created: 1,
    id: "test-id",
    model: "auto",
    object: "chat.completion.chunk",
    usage: { completion_tokens: 1, prompt_tokens: 1, total_tokens: 2 },
    ...extra,
  };
}

const SUCCESS_SSE = loadLiveFixture("global").interactions.chat.response.body as string;

const BLOCKED_SSE = sseEnvelope(
  { code: "provider_error", message: "Session blocked", request_id: "r", type: "provider_error" },
  406,
  "Not Acceptable",
);

const QUEUED_SSE = sseEnvelope(
  {
    code: "403",
    message: JSON.stringify({
      code: "10605",
      message: JSON.stringify({
        isQueued: false,
        modelKey: "qfmodel",
        queueCount: 0,
        queueType: "ultrafast",
        retryAfterSeconds: 0,
        serviceAvailable: true,
        waitTime: 0,
      }),
    }),
  },
  403,
  "Forbidden",
);

function mockFetch(body: string): typeof fetch {
  const response = new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
  return vi.fn(async () => response) as unknown as typeof fetch;
}

function makeModel(provider = "qoder", id = "Lite"): Model<Api> {
  return { id, api: "qoder-api" as Api, provider } as Model<Api>;
}

/**
 * Write an account-scoped catalogue for the mocked identity ("test-user") so the
 * entitlement pre-flight has something to check. `servedModelIds` is the raw
 * answer of the account's last `/model/list`.
 */
function accountCatalogPath(): string {
  return join(process.env.HOME || process.env.USERPROFILE || "", ".pi", "agent", "qoder-models-cache.json");
}

/**
 * Write an account-scoped catalogue for the mocked identity ("test-user") so the
 * entitlement pre-flight has something to check. `servedModelIds` is the raw
 * answer of the account's last `/model/list`.
 */
function writeAccountCatalog(servedModelIds: string[], modelIds: string[] = servedModelIds): void {
  const configs = Object.fromEntries(
    modelIds.map((id) => [id, { key: id.toLowerCase(), enable: true, display_name: id }]),
  );
  const slot = (ids: string[]) => ({
    updatedAt: Date.now(),
    identity: { userID: "test-user" },
    servedModelIds: ids,
    models: ids.map((id) => ({ id, name: id })),
    configs: Object.fromEntries(ids.map((id) => [id, { key: id.toLowerCase(), enable: true, display_name: id }])),
  });
  writeFileSync(
    accountCatalogPath(),
    JSON.stringify({
      version: 2,
      updatedAt: Date.now(),
      models: modelIds.map((id) => ({ id, name: id })),
      configs,
      accounts: {
        "test-user": slot(servedModelIds),
        // A second account in the region that does list the model: the
        // entitlement veto only fires when the model exists elsewhere.
        "other-user": slot(["Kimi-K3"]),
      },
    }),
    "utf8",
  );
  clearQoderModelsMemCache();
}

function clearAccountCatalog(): void {
  rmSync(accountCatalogPath(), { force: true });
  clearQoderModelsMemCache();
}

function makeContext(): Context {
  return {
    systemPrompt: "test",
    messages: [{ role: "user", content: "hi" }],
    tools: [],
  } as unknown as Context;
}

async function consume(stream: AssistantMessageEventStream): Promise<AssistantMessageEvent[]> {
  const events: AssistantMessageEvent[] = [];
  for await (const ev of stream) {
    events.push(ev);
    if (ev.type === "done" || ev.type === "error") break;
  }
  return events;
}

describe("streamQoder", () => {
  const originalFetch = globalThis.fetch;
  const originalCnPat = process.env.QODERCN_PERSONAL_ACCESS_TOKEN;
  afterEach(() => {
    globalThis.fetch = originalFetch;
    if (originalCnPat === undefined) delete process.env.QODERCN_PERSONAL_ACCESS_TOKEN;
    else process.env.QODERCN_PERSONAL_ACCESS_TOKEN = originalCnPat;
    vi.restoreAllMocks();
  });

  it("replays a recorded-format SSE fixture into text + stop", async () => {
    globalThis.fetch = mockFetch(SUCCESS_SSE);
    const stream = streamQoder(makeModel(), makeContext(), { apiKey: "fake" });
    const events = await consume(stream);

    const done = events.find((e) => e.type === "done");
    expect(done, "expected a done event").toBeDefined();
    const msg = (done as { message: AssistantMessage }).message;
    expect(msg.stopReason).toBe("stop");
    const text = msg.content.find((c) => c.type === "text");
    expect(text && "text" in text ? text.text : "").toBe("OK");
  });

  it("sends the internal upstream key for a friendly model id", async () => {
    globalThis.fetch = mockFetch(SUCCESS_SSE);
    await consume(streamQoder(makeModel("qoder", "Lite"), makeContext(), { apiKey: "fake" }));

    const init = vi.mocked(globalThis.fetch).mock.calls[0][1];
    expect(init?.headers).toEqual(expect.objectContaining({ "X-Model-Key": "lite" }));

    const custom = "_doRTgHZBKcGVjlvpC,@aFSx#DPuNJme&i*MzLOEn)sUrthbf%Y^w.(kIQyXqWA!";
    const standard = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const encoded = Buffer.from(init?.body as Uint8Array).toString("utf8");
    const rearranged = [...encoded]
      .map((character) => (character === "$" ? "=" : standard[custom.indexOf(character)] || character))
      .join("");
    const third = Math.floor(rearranged.length / 3);
    const base64 =
      rearranged.slice(rearranged.length - third) +
      rearranged.slice(third, rearranged.length - third) +
      rearranged.slice(0, third);
    const body = JSON.parse(Buffer.from(base64, "base64").toString("utf8")) as {
      chat_context: { extra: { modelConfig: { key: string } } };
      model_config: { key: string };
    };
    expect(body.chat_context.extra.modelConfig.key).toBe("lite");
    expect(body.model_config.key).toBe("lite");
  });

  it("binds chat hosts to provider ids even when only a CN PAT is set", async () => {
    process.env.QODERCN_PERSONAL_ACCESS_TOKEN = "pt-cn-only";

    globalThis.fetch = mockFetch(SUCCESS_SSE);
    await consume(streamQoder(makeModel("qoder"), makeContext(), { apiKey: "fake" }));
    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.stringMatching(/^https:\/\/api3\.qoder\.sh\//),
      expect.any(Object),
    );

    globalThis.fetch = mockFetch(SUCCESS_SSE);
    await consume(streamQoder(makeModel("qoder-cn", "Qwen3.7-Plus"), makeContext(), { apiKey: "fake" }));
    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.stringMatching(/^https:\/\/gateway\.qoder\.com\.cn\//),
      expect.any(Object),
    );
  });

  it("fails fast when the account's own catalogue cannot serve the model", async () => {
    writeAccountCatalog(["Lite"]);
    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const events = await consume(streamQoder(makeModel("qoder", "Kimi-K3"), makeContext(), { apiKey: "fake" }));

    const err = events.find((e) => e.type === "error") as { error: AssistantMessage } | undefined;
    expect(err, "expected an entitlement error event").toBeDefined();
    expect(err?.error.errorMessage).toMatch(/entitlement/i);
    expect(err?.error.errorMessage).toMatch(/status 403/);
    // Qoder only answers this after ~3 minutes, so it must never be attempted.
    expect(fetchMock).not.toHaveBeenCalled();

    clearAccountCatalog();
  });

  it("still sends a model the account's own catalogue lists", async () => {
    writeAccountCatalog(["Lite"]);
    globalThis.fetch = mockFetch(SUCCESS_SSE);

    await consume(streamQoder(makeModel("qoder", "Lite"), makeContext(), { apiKey: "fake" }));

    expect(globalThis.fetch).toHaveBeenCalled();
    clearAccountCatalog();
  });

  it("surfaces an upstream 406 'Session blocked' as an error event, not a silent stop", async () => {
    globalThis.fetch = mockFetch(BLOCKED_SSE);
    const stream = streamQoder(makeModel(), makeContext(), { apiKey: "fake" });
    const events = await consume(stream);

    const err = events.find((e) => e.type === "error");
    expect(err, "expected an error event").toBeDefined();
    const msg = (err as { error: AssistantMessage }).error;
    expect(msg.stopReason).toBe("error");
    expect(msg.errorMessage).toMatch(/Session blocked/);
    expect(msg.errorMessage).toMatch(/406/);
    // Must NOT emit a silent done/stop.
    expect(events.find((e) => e.type === "done")).toBeUndefined();
  });

  it("surfaces premature disconnect as an error when stream ends before [DONE]", async () => {
    // If the server terminates the connection or times out without sending [DONE],
    // it must surface as an error event, not be swallowed as a successful stop.
    const truncatedSse = sseEnvelope(chunk({ content: "thinking...", role: "assistant" }));
    globalThis.fetch = mockFetch(truncatedSse);
    const stream = streamQoder(makeModel(), makeContext(), { apiKey: "fake" });
    const events = await consume(stream);

    const err = events.find((e) => e.type === "error");
    expect(err, "expected an error event").toBeDefined();
    const msg = (err as { error: AssistantMessage }).error;
    expect(msg.stopReason).toBe("error");
    expect(msg.errorMessage).toMatch(/disconnected prematurely/i);
    expect(events.find((e) => e.type === "done")).toBeUndefined();
  });

  it("preserves finish_reason=length instead of overwriting to stop", async () => {
    const sse =
      sseEnvelope(chunk({ content: "partial", role: "assistant" })) + sseEnvelope(finishChunk("length")) + DONE_SSE;
    globalThis.fetch = mockFetch(sse);
    const stream = streamQoder(makeModel(), makeContext(), { apiKey: "fake" });
    const events = await consume(stream);

    const done = events.find((e) => e.type === "done");
    const msg = (done as { message: AssistantMessage }).message;
    expect(msg.stopReason).toBe("length");
  });

  it("captures usage, responseId and responseModel from the finish chunk", async () => {
    const sse =
      sseEnvelope(chunk({ content: "OK", role: "assistant" })) +
      sseEnvelope(
        finishChunk("stop", {
          id: "chatcmpl-abc123",
          model: "qmodel_latest",
          usage: {
            prompt_tokens: 42,
            completion_tokens: 7,
            total_tokens: 49,
            completion_tokens_details: { reasoning_tokens: 3 },
            // prompt_tokens (42) INCLUDES cached_tokens (5) per OpenAI
            // semantics; pi-core expects `input` to exclude them
            // (promptTokens = input + cacheRead + cacheWrite), so input =
            // 42 - 5 - 10 = 27. cacheable_tokens is a capacity metric, not a
            // write count, and must not be mapped to cacheWrite.
            prompt_tokens_details: { cacheable_tokens: 99, cache_write_tokens: 10, cached_tokens: 5 },
            credits: 0.000613272,
            original_credits: 0.001226544,
            billable: true,
          },
        }),
      ) +
      DONE_SSE;
    globalThis.fetch = mockFetch(sse);
    const stream = streamQoder(makeModel(), makeContext(), { apiKey: "fake" });
    const events = await consume(stream);

    const done = events.find((e) => e.type === "done");
    const msg = (done as { message: AssistantMessage }).message;
    expect(msg.responseId).toBe("chatcmpl-abc123");
    expect(msg.responseModel).toBe("qmodel_latest");
    expect(msg.usage.input).toBe(27);
    expect(msg.usage.output).toBe(7);
    expect(msg.usage.totalTokens).toBe(49);
    expect(msg.usage.cacheRead).toBe(5);
    expect(msg.usage.cacheWrite).toBe(10);
    const creditUsage = msg.usage as QoderUsage;
    expect(creditUsage.credits).toBe(0.000613272);
    expect(creditUsage.original_credits).toBe(0.001226544);
    expect(creditUsage.billable).toBe(true);
    expect(creditUsage.charged_credits).toBe(0.000613272);
    expect(msg.usage.cost.total).toBe(0);
  });

  it("emits a done event with reason=length when finish_reason is length", async () => {
    const sse =
      sseEnvelope(chunk({ content: "partial", role: "assistant" })) + sseEnvelope(finishChunk("length")) + DONE_SSE;
    globalThis.fetch = mockFetch(sse);
    const stream = streamQoder(makeModel(), makeContext(), { apiKey: "fake" });
    const events = await consume(stream);

    const done = events.find((e) => e.type === "done");
    expect(done, "expected a done event").toBeDefined();
    expect((done as { reason: string }).reason).toBe("length");
  });

  it("reports a tool_use stop reason when the stream emits tool calls", async () => {
    const sse =
      sseEnvelope(
        chunk({
          tool_calls: [
            {
              index: 0,
              id: "call_1",
              function: { name: "bash", arguments: '{"command":"ls"}' },
            },
          ],
        }),
      ) +
      sseEnvelope(finishChunk("tool_calls")) +
      DONE_SSE;
    globalThis.fetch = mockFetch(sse);
    const stream = streamQoder(makeModel(), makeContext(), { apiKey: "fake" });
    const events = await consume(stream);

    const done = events.find((e) => e.type === "done");
    const msg = (done as { message: AssistantMessage }).message;
    expect(msg.stopReason).toBe("toolUse");
    const toolCall = msg.content.find((c) => c.type === "toolCall");
    expect(toolCall).toBeDefined();
  });

  it("assembles reasoning chunks before the final answer", async () => {
    const sse =
      sseEnvelope(chunk({ reasoning_content: "check " })) +
      sseEnvelope(chunk({ reasoning_content: "twice" })) +
      sseEnvelope(chunk({ content: "done" })) +
      sseEnvelope(finishChunk("stop")) +
      DONE_SSE;
    globalThis.fetch = mockFetch(sse);

    const events = await consume(streamQoder(makeModel(), makeContext(), { apiKey: "fake", reasoning: "high" }));
    const done = events.find((event) => event.type === "done") as { message: AssistantMessage };

    expect(done.message.content).toEqual([
      { type: "thinking", thinking: "check twice" },
      { type: "text", text: "done" },
    ]);
    expect(events.map((event) => event.type)).toContain("thinking_delta");
  });

  it("reports a degenerate turn instead of a silent stop", async () => {
    // The gateway failed to parse the model's tool call and leaked its DSML
    // markup into reasoning_content: the turn carries no text and no tool call,
    // and finish_reason is still "stop". Reading that as a finished task ended
    // the agent turn with nothing to show and nothing to retry.
    const sse =
      sseEnvelope(chunk({ reasoning_content: "Let me look.</invoke>\n</invoke>\n</invoke>" })) +
      sseEnvelope(finishChunk("stop")) +
      DONE_SSE;
    globalThis.fetch = mockFetch(sse);

    const events = await consume(streamQoder(makeModel(), makeContext(), { apiKey: "fake", reasoning: "high" }));

    const error = events.find((event) => event.type === "error") as { error: AssistantMessage };
    expect(error, "expected an error event").toBeDefined();
    expect(error.error.errorMessage).toMatch(/degenerate model output/);
    expect(events.find((event) => event.type === "done")).toBeUndefined();
  });

  it("keeps a normal turn and strips the residue from its thinking", async () => {
    const sse =
      sseEnvelope(chunk({ reasoning_content: "thinking</invoke>\n</invoke>" })) +
      sseEnvelope(chunk({ content: "the answer" })) +
      sseEnvelope(finishChunk("stop")) +
      DONE_SSE;
    globalThis.fetch = mockFetch(sse);

    const events = await consume(streamQoder(makeModel(), makeContext(), { apiKey: "fake", reasoning: "high" }));

    const done = events.find((event) => event.type === "done") as { message: AssistantMessage };
    expect(done, "expected a done event").toBeDefined();
    const thinking = done.message.content.find((block) => block.type === "thinking") as { thinking: string };
    expect(thinking.thinking).toBe("thinking\n");
    expect(done.message.content.find((block) => block.type === "text")).toMatchObject({ text: "the answer" });
  });

  it("assembles parallel tool calls by their stream indexes", async () => {
    const sse =
      sseEnvelope(
        chunk({
          tool_calls: [
            { index: 0, id: "call_a", function: { name: "read", arguments: '{"path":' } },
            { index: 1, id: "call_b", function: { name: "search", arguments: '{"query":' } },
          ],
        }),
      ) +
      sseEnvelope(
        chunk({
          tool_calls: [
            { index: 0, function: { arguments: '"/a"}' } },
            { index: 1, function: { arguments: '"needle"}' } },
          ],
        }),
      ) +
      sseEnvelope(finishChunk("tool_calls")) +
      DONE_SSE;
    globalThis.fetch = mockFetch(sse);

    const events = await consume(streamQoder(makeModel(), makeContext(), { apiKey: "fake" }));
    const done = events.find((event) => event.type === "done") as { message: AssistantMessage };
    const calls = done.message.content.filter((block): block is ToolCall => block.type === "toolCall");

    expect(calls).toEqual([
      { type: "toolCall", id: "call_a", name: "read", arguments: { path: "/a" } },
      { type: "toolCall", id: "call_b", name: "search", arguments: { query: "needle" } },
    ]);
  });

  it("preserves text emitted before and after a tool call", async () => {
    const sse =
      sseEnvelope(chunk({ content: "before" })) +
      sseEnvelope(chunk({ tool_calls: [{ index: 0, id: "call_1", function: { name: "lookup", arguments: "{}" } }] })) +
      sseEnvelope(chunk({ content: " after" })) +
      sseEnvelope(finishChunk("tool_calls")) +
      DONE_SSE;
    globalThis.fetch = mockFetch(sse);

    const events = await consume(streamQoder(makeModel(), makeContext(), { apiKey: "fake" }));
    const done = events.find((event) => event.type === "done") as { message: AssistantMessage };

    expect(done.message.content).toEqual([
      { type: "text", text: "before after" },
      { type: "toolCall", id: "call_1", name: "lookup", arguments: {} },
    ]);
  });

  it("emits a tool call that arrives with no arguments", async () => {
    // A no-argument tool, or a model that sends id+name and stops. The block
    // used to be created only inside `if (tc.function?.arguments)`, so this
    // produced a toolCallsState entry and NO content block — and the finalizer
    // then set stopReason "toolUse" on a message with no tool call in it. pi's
    // agent loop had nothing to execute and the turn ended silently, mid-task.
    const sse =
      sseEnvelope(
        chunk({
          tool_calls: [{ index: 0, id: "call_1", function: { name: "advisor", arguments: "" } }],
        }),
      ) +
      sseEnvelope(finishChunk("tool_calls")) +
      DONE_SSE;
    globalThis.fetch = mockFetch(sse);
    const stream = streamQoder(makeModel(), makeContext(), { apiKey: "fake" });
    const events = await consume(stream);

    const done = events.find((e) => e.type === "done");
    const msg = (done as { message: AssistantMessage }).message;
    const toolCall = msg.content.find((c) => c.type === "toolCall") as ToolCall | undefined;
    expect(toolCall, "a named tool call must reach the message even with no arguments").toBeDefined();
    expect(toolCall?.name).toBe("advisor");
    expect(toolCall?.id).toBe("call_1");
    expect(toolCall?.arguments).toEqual({});
    expect(msg.stopReason).toBe("toolUse");
  });

  it("picks up an id and name that arrive after the block is open", async () => {
    // Streamed the other way round: arguments first, identity later.
    const sse =
      sseEnvelope(chunk({ tool_calls: [{ index: 0, function: { name: "bash", arguments: '{"comm' } }] })) +
      sseEnvelope(chunk({ tool_calls: [{ index: 0, id: "call_9", function: { arguments: 'and":"ls"}' } }] })) +
      sseEnvelope(finishChunk("tool_calls")) +
      DONE_SSE;
    globalThis.fetch = mockFetch(sse);
    const stream = streamQoder(makeModel(), makeContext(), { apiKey: "fake" });
    const events = await consume(stream);

    const done = events.find((e) => e.type === "done");
    const msg = (done as { message: AssistantMessage }).message;
    const toolCall = msg.content.find((c) => c.type === "toolCall") as ToolCall | undefined;
    expect(toolCall?.id).toBe("call_9");
    expect(toolCall?.name).toBe("bash");
    expect(toolCall?.arguments).toEqual({ command: "ls" });
  });

  it("does not claim toolUse when no tool call reached the message", async () => {
    // A malformed stream: a tool_calls delta with neither id nor name. Better a
    // clean "stop" than a message that says toolUse and carries nothing, which
    // the agent loop cannot act on and cannot report.
    const sse =
      sseEnvelope(chunk({ content: "thinking about it", role: "assistant" })) +
      sseEnvelope(chunk({ tool_calls: [{ index: 0, function: {} }] })) +
      sseEnvelope(finishChunk("stop")) +
      DONE_SSE;
    globalThis.fetch = mockFetch(sse);
    const stream = streamQoder(makeModel(), makeContext(), { apiKey: "fake" });
    const events = await consume(stream);

    const done = events.find((e) => e.type === "done");
    const msg = (done as { message: AssistantMessage }).message;
    expect(msg.content.find((c) => c.type === "toolCall")).toBeUndefined();
    expect(msg.stopReason).toBe("stop");
  });
  it("retries a transient 10605 queue response", async () => {
    const responses = [QUEUED_SSE, SUCCESS_SSE].map(
      (body) => new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } }),
    );
    const chatCalls: string[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/userinfo")) {
        return new Response(JSON.stringify({}), { status: 200, headers: { "content-type": "application/json" } });
      }
      chatCalls.push(url);
      return responses.shift() as Response;
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const stream = streamQoder(makeModel(), makeContext(), { apiKey: "fake" });
    const events = await consume(stream);

    expect(chatCalls).toHaveLength(2);
    expect(events.some((event) => event.type === "error")).toBe(false);
    expect(events.some((event) => event.type === "done")).toBe(true);
  });
  it("finishes when the gateway sends [DONE] but keeps the body open", async () => {
    // Qoder's gateway does not always close the HTTP body after the sentinel.
    // The read loop used to keep awaiting reader.read() until the socket went
    // away, so a fully streamed reply never produced a done event and the
    // agent appeared to hang with no error.
    const sse = sseEnvelope(chunk({ content: "OK", role: "assistant" })) + sseEnvelope(finishChunk("stop")) + DONE_SSE;
    let cancelled = false;
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(sse));
        // Deliberately never call controller.close().
      },
      cancel() {
        cancelled = true;
      },
    });
    globalThis.fetch = vi.fn(
      async () => new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } }),
    ) as unknown as typeof fetch;

    const stream = streamQoder(makeModel(), makeContext(), { apiKey: "fake" });
    const events = await consume(stream);

    const done = events.find((e) => e.type === "done");
    expect(done, "expected a done event even though the body stayed open").toBeDefined();
    const msg = (done as { message: AssistantMessage }).message;
    expect(msg.stopReason).toBe("stop");
    const text = msg.content.find((c) => c.type === "text");
    expect(text && "text" in text ? text.text : "").toBe("OK");
    // The reader is released rather than left holding the connection.
    expect(cancelled).toBe(true);
  });

  it("finishes on a bare 'data: [DONE]' line with the body left open", async () => {
    // Same sentinel, unwrapped.
    const sse = `${sseEnvelope(chunk({ content: "hi", role: "assistant" }))}data: [DONE]\n\n`;
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(sse));
      },
    });
    globalThis.fetch = vi.fn(
      async () => new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } }),
    ) as unknown as typeof fetch;

    const stream = streamQoder(makeModel(), makeContext(), { apiKey: "fake" });
    const events = await consume(stream);

    const done = events.find((e) => e.type === "done");
    expect(done, "expected a done event for the bare sentinel").toBeDefined();
    const msg = (done as { message: AssistantMessage }).message;
    const text = msg.content.find((c) => c.type === "text");
    expect(text && "text" in text ? text.text : "").toBe("hi");
  });
  it("reports aborted when the request is cancelled before streaming starts", async () => {
    const controller = new AbortController();
    globalThis.fetch = vi.fn(
      (_url: URL | RequestInfo, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          if (init?.signal?.aborted) {
            reject(new DOMException("Aborted", "AbortError"));
            return;
          }
          init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), {
            once: true,
          });
        }),
    ) as unknown as typeof fetch;

    const eventsPromise = consume(
      streamQoder(makeModel(), makeContext(), { apiKey: "fake", signal: controller.signal }),
    );
    controller.abort();
    const events = await eventsPromise;

    const error = events.find((event) => event.type === "error") as { error: AssistantMessage };
    expect(error.error.stopReason).toBe("aborted");
    expect(events.find((event) => event.type === "done")).toBeUndefined();
  });

  it("clamps and falls back unsupported reasoning effort to a valid upstream effort", async () => {
    globalThis.fetch = mockFetch(SUCCESS_SSE);
    // Request GLM-5.3-Flash with medium (upstream only supports high / max)
    const stream = streamQoder(makeModel("qoder", "GLM-5.3-Flash"), makeContext(), {
      apiKey: "fake",
      reasoning: "medium",
    });
    await consume(stream);

    const init = vi.mocked(globalThis.fetch).mock.calls[0][1];
    const decoded = qoderDecodeBody(Buffer.from(init?.body as Uint8Array)).toString("utf8");
    const parsed = JSON.parse(decoded) as {
      parameters: { enable_thinking: boolean; reasoning_effort?: string };
    };

    expect(parsed.parameters.enable_thinking).toBe(true);
    // Must NOT send "medium"; must fallback to a valid effort like "high" or "max"
    expect(parsed.parameters.reasoning_effort).not.toBe("medium");
    expect(["high", "max"]).toContain(parsed.parameters.reasoning_effort);
  });

  it("keeps thinking enabled with valid effort for always-thinking models when off is requested", async () => {
    globalThis.fetch = mockFetch(SUCCESS_SSE);
    const stream = streamQoder(makeModel("qoder", "GLM-5.3-Flash"), makeContext(), {
      apiKey: "fake",
      reasoning: "off" as unknown as "high",
    });
    await consume(stream);

    const init = vi.mocked(globalThis.fetch).mock.calls[0][1];
    const decoded = qoderDecodeBody(Buffer.from(init?.body as Uint8Array)).toString("utf8");
    const parsed = JSON.parse(decoded) as {
      parameters: { enable_thinking: boolean; reasoning_effort?: string };
    };

    // GLM-5.3-Flash is always-on; cannot disable thinking without causing error 1210
    expect(parsed.parameters.enable_thinking).toBe(true);
    expect(["high", "max"]).toContain(parsed.parameters.reasoning_effort);
  });

  it("parses XML <tool_call> tags from text stream as real tool calls", async () => {
    const sse =
      sseEnvelope(
        chunk({
          content:
            "Let me read the file.\n\n<tool_call>\n<function=read>\n<parameter=file_path>/test/file.md</parameter>\n</function>\n</tool_call>",
          role: "assistant",
        }),
      ) +
      sseEnvelope(finishChunk("stop")) +
      DONE_SSE;

    globalThis.fetch = mockFetch(sse);
    const stream = streamQoder(makeModel(), makeContext(), { apiKey: "fake" });
    const events = await consume(stream);

    const done = events.find((e) => e.type === "done");
    const msg = (done as { message: AssistantMessage }).message;

    expect(msg.stopReason).toBe("toolUse");
    const toolCall = msg.content.find((c) => c.type === "toolCall") as ToolCall | undefined;
    expect(toolCall).toBeDefined();
    expect(toolCall?.name).toBe("read");
    expect(toolCall?.arguments.path).toBe("/test/file.md");

    const text = msg.content.find((c) => c.type === "text") as { text: string } | undefined;
    expect(text?.text).toBe("Let me read the file.");
  });
});
