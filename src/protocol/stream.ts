import crypto from "node:crypto";
import * as PiAi from "@earendil-works/pi-ai";
import {
  type Api,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  clampThinkingLevel,
  type Model,
  type SimpleStreamOptions,
  type TextContent,
  type ThinkingContent,
  type ToolCall,
} from "@earendil-works/pi-ai";
import { resolveQoderIdentity } from "../auth/oauth.js";
import { buildThinkingLevelMap, checkAccountEntitlement, MAX_OUTPUT_TOKENS, resolveModelConfig } from "../catalog.js";
import { buildAuthHeaders, getMachineId } from "../cosy.js";
import { applyQoderCredits } from "../credits.js";
import { getQoderChatURL, getQoderRegionConfig } from "../region.js";
import { qoderEncodeBody } from "./encoding.js";
import { isDegenerateDsmlTurn, stripDsmlResidue, stripThinkingTags, ThinkingTagParser } from "./thinking.js";
import { parseToolCallsFromText } from "./tool-parser.js";
import { extractTools, transformMessagesForQoder, transformTools } from "./transform.js";

interface ToolCallState {
  arguments: string;
  id: string;
  name: string;
  emittedStart?: boolean;
  emittedEnd?: boolean;
  contentIndex: number;
}

function stableHash(prefix: string, ...inputs: string[]): string {
  const hash = crypto.createHash("sha256");
  hash.update(prefix);
  for (const input of inputs) {
    hash.update("\0");
    hash.update(input);
  }
  return hash.digest("hex").slice(0, 16);
}

function stableChatRecordID(
  model: string,
  messages: Array<{ role?: string; content?: unknown }>,
  tools: unknown,
  maxTokens: number,
): string {
  const hash = crypto.createHash("sha256");
  hash.update("qoder-record");
  hash.update("\0");
  hash.update(model);
  for (const msg of messages) {
    if (msg?.role) {
      hash.update("\0");
      hash.update(msg.role);
    }
    if (msg?.content) {
      hash.update("\0");
      hash.update(typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content));
    }
  }
  if (tools) {
    hash.update("\0");
    hash.update(JSON.stringify(tools));
  }
  hash.update("\0");
  hash.update(`mt=${maxTokens}`);
  return hash.digest("hex").slice(0, 16);
}

const MAX_QUEUE_RETRIES = 4;
const MAX_QUEUE_RETRY_DELAY_SECONDS = 60;

type QueueRetryInfo = {
  retryAfterSeconds: number;
};

/** Find Qoder's nested 10605 queue response in an SSE envelope or JSON body. */
function findQueueRetryInfo(value: unknown, depth = 0): QueueRetryInfo | null {
  if (depth > 5) return null;

  if (typeof value === "string") {
    try {
      return findQueueRetryInfo(JSON.parse(value), depth + 1);
    } catch {
      return null;
    }
  }

  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  if (String(record.code) === "10605") {
    const retryAfterSeconds =
      typeof record.retryAfterSeconds === "number" && Number.isFinite(record.retryAfterSeconds)
        ? Math.max(0, record.retryAfterSeconds)
        : 3;
    return { retryAfterSeconds };
  }

  for (const key of ["body", "message"]) {
    const nested = findQueueRetryInfo(record[key], depth + 1);
    if (nested) return nested;
  }
  return null;
}

type QueueInspection = {
  response: Response;
  queue: QueueRetryInfo | null;
};

function responseWithBody(response: Response, body: BodyInit | null): Response {
  return new Response(body, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
}

/** Inspect the first response event while preserving it for the real parser. */
async function inspectResponseForQueue(response: Response): Promise<QueueInspection> {
  if (!response.body) return { response, queue: null };

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  const decoder = new TextDecoder();
  let buffer = "";
  let sourceDone = false;
  let firstDataLineSeen = false;
  let queue: QueueRetryInfo | null = null;

  if (!response.ok) {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }
    const text = decoder.decode(Buffer.concat(chunks));
    return { response: responseWithBody(response, text), queue: findQueueRetryInfo(text) };
  }

  while (!firstDataLineSeen && buffer.length < 64 * 1024) {
    const { done, value } = await reader.read();
    if (done) {
      sourceDone = true;
      break;
    }
    chunks.push(value);
    buffer += decoder.decode(value, { stream: true });

    while (true) {
      const lineEnd = buffer.indexOf("\n");
      if (lineEnd === -1) break;
      const line = buffer.substring(0, lineEnd).trim();
      buffer = buffer.substring(lineEnd + 1);
      if (!line.startsWith("data:")) continue;

      firstDataLineSeen = true;
      const dataStr = line.substring(5).trim();
      if (dataStr !== "[DONE]") {
        try {
          queue = findQueueRetryInfo(JSON.parse(dataStr));
        } catch {}
      }
      break;
    }
  }

  if (queue) {
    await reader.cancel().catch(() => {});
    return { response, queue };
  }

  const replayBody = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      if (sourceDone) controller.close();
    },
    async pull(controller) {
      if (sourceDone) return;
      try {
        const { done, value } = await reader.read();
        if (done) {
          sourceDone = true;
          controller.close();
        } else {
          controller.enqueue(value);
        }
      } catch (error) {
        controller.error(error);
      }
    },
    cancel(reason) {
      return reader.cancel(reason);
    },
  });
  return { response: responseWithBody(response, replayBody), queue: null };
}

function waitForQueueRetry(seconds: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.reject(new DOMException("Operation aborted", "AbortError"));
  const delay = Math.min(Math.max(seconds, 0), MAX_QUEUE_RETRY_DELAY_SECONDS) * 1000;
  if (delay === 0) return Promise.resolve();

  return new Promise((resolve, reject) => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const cleanup = () => {
      if (timer) clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    };
    const onAbort = () => {
      cleanup();
      reject(new DOMException("Operation aborted", "AbortError"));
    };
    timer = setTimeout(() => {
      cleanup();
      resolve();
    }, delay);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

async function fetchWithQueueRetry(url: string, init: RequestInit, signal?: AbortSignal): Promise<Response> {
  for (let attempt = 0; ; attempt++) {
    const inspected = await inspectResponseForQueue(await fetch(url, init));
    if (!inspected.queue) return inspected.response;
    if (attempt >= MAX_QUEUE_RETRIES) {
      throw new Error(`Qoder queue remained busy (10605) after ${MAX_QUEUE_RETRIES} retries`);
    }

    if (process.env.QODER_DEBUG) {
      console.error(
        `[pi-provider-qoder] Qoder queue busy (10605), retrying in ${inspected.queue.retryAfterSeconds}s (${attempt + 1}/${MAX_QUEUE_RETRIES})`,
      );
    }
    await waitForQueueRetry(inspected.queue.retryAfterSeconds, signal);
  }
}
function contentToText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part === "object" && "text" in part) return (part as { text: string }).text;
        return "";
      })
      .join("\n");
  }
  return "";
}

export function streamQoder(
  model: Model<Api>,
  context: Context,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  const StreamCtor = (PiAi as unknown as { AssistantMessageEventStream: new () => AssistantMessageEventStream })
    .AssistantMessageEventStream;
  const stream = new StreamCtor();

  const output: AssistantMessage = {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "stop",
    timestamp: Date.now(),
  };

  (async () => {
    try {
      const providerMode = model.provider.startsWith("qoder-cn") ? "cn" : "global";
      const region = getQoderRegionConfig(providerMode);
      const accessToken = options?.apiKey;
      if (!accessToken) {
        throw new Error(
          providerMode === "cn"
            ? "Qoder CN credentials not set. Run /login qoder-cn or set QODERCN_PERSONAL_ACCESS_TOKEN."
            : "Qoder credentials not set. Run /login qoder or set QODER_PERSONAL_ACCESS_TOKEN.",
        );
      }

      // Resolve the real Qoder identity from the job token. OMP keeps login
      // credentials in its own agent.db, not in ~/.pi/agent/auth.json, so a
      // cache miss would otherwise send uid "qoder-user" and Qoder CN rejects
      // it with "Login expired" (105).
      const ident = await resolveQoderIdentity(accessToken, model.provider, providerMode);
      const userID = ident.userID || "qoder-user";
      const name = ident.name || region.userNameFallback;
      const email = ident.email || region.userEmailFallback;
      const machineID = ident.machineID || getMachineId();

      // Both providers expose the upstream display_name (whitespace stripped)
      // as the pi id. Read the original key from cached/static config so the
      // gateway still receives identifiers such as `lite` or `qmodel`.
      const modelConfig = resolveModelConfig(model.id, providerMode);
      if (!modelConfig?.key) {
        throw new Error(`Unknown Qoder model id: ${model.id}`);
      }
      const qoderModel = modelConfig.key;

      // Pre-flight entitlement: Qoder only rejects a model the account cannot
      // serve after holding the SSE connection open for ~3 minutes (403 code
      // 112), so a misrouted pooled request costs minutes. The per-account
      // catalogue is already cached, so fail locally in milliseconds instead;
      // accounts with an unknown or stale catalogue keep failing open.
      const entitlement = checkAccountEntitlement(providerMode, userID, model.id);
      if (entitlement.checked && !entitlement.served) {
        // "upstream status 403" keeps the wording classifiable by hosts that
        // route pooled accounts on HTTP status (multiprovider's default
        // failure classification), so the request fails over instead of dying.
        throw new Error(
          `Qoder entitlement: account ${email} cannot serve model "${model.id}" ` +
            "(upstream status 403 code 112 entitlement). Route it to an account whose plan includes it.",
        );
      }

      const isReasoning = !!modelConfig.is_reasoning;

      const normalizedMessages = transformMessagesForQoder(context.messages);
      // OMP may supply the system prompt as a single-element content array;
      // Qoder MessagesInputDto#content is a String and rejects an array with
      // "Execution failed: set property ... MessagesInputDto#content". Normalize.
      const systemText = contentToText(context.systemPrompt || "");

      const finalMessages: ReturnType<typeof transformMessagesForQoder> = [];
      let combinedSystemPrompt = systemText;

      for (const msg of normalizedMessages) {
        if (msg.role === "system") {
          const sysContent = typeof msg.content === "string" ? msg.content : "";
          if (sysContent) {
            combinedSystemPrompt = combinedSystemPrompt ? `${combinedSystemPrompt}\n\n${sysContent}` : sysContent;
          }
        } else {
          finalMessages.push(msg);
        }
      }

      if (combinedSystemPrompt) {
        finalMessages.unshift({ role: "system", content: combinedSystemPrompt });
      }

      let lastUserText = "";
      for (let i = finalMessages.length - 1; i >= 0; i--) {
        if (finalMessages[i].role === "user") {
          const content = finalMessages[i].content;
          const text =
            typeof content === "string"
              ? content
              : Array.isArray(content)
                ? content.map((c) => ("text" in c ? c.text : "")).join("")
                : "";
          // Skip synthetic image carrier messages when resolving original user prompt
          if (text.startsWith("[") && text.includes("image") && text.includes("returned by")) {
            continue;
          }
          lastUserText = text;
          break;
        }
      }

      // Use a stable session id when pi provides one (per agent session) so
      // the Qoder server can maintain prompt cache affinity across consecutive
      // requests. Fall back to a random id only when no sessionId is available.
      const stablePart = stableHash("qoder-session", userID, qoderModel);
      const sessionID = options?.sessionId
        ? `${stablePart}-${options.sessionId}`
        : `${stablePart}-${crypto.randomUUID()}`;

      // Qoder's catalog exposes no per-model output cap, so we use the
      // documented upstream ceiling (MAX_OUTPUT_TOKENS = 131072, see models.ts)
      // and let pi cap it lower when the caller sets options.maxTokens (e.g.
      // compaction at 40K). This avoids truncating reasoning chains / long
      // generations that the 32K default would cut off.
      let maxTokens = MAX_OUTPUT_TOKENS;
      if (options?.maxTokens && options.maxTokens < maxTokens) {
        maxTokens = options.maxTokens;
      }

      const effectiveTools = extractTools(context);
      const toolsRaw = effectiveTools.length > 0 ? transformTools(effectiveTools) : undefined;
      const recordID = stableChatRecordID(qoderModel, finalMessages, toolsRaw, maxTokens);

      // Map pi's thinking level (options.reasoning) to Qoder's request fields.
      // Confirmed from @qoder-ai/qodercli: the chat body carries `reasoning_effort`
      // ("none"|"low"|"medium"|"high"|"xhigh"|"max") and `enable_thinking` (bool)
      // inside `parameters`, alongside `max_tokens`.
      //
      // This mirrors the pattern the pi-ai OpenAI provider uses: clamp the
      // requested level to what the model advertises via thinkingLevelMap, then
      // map to the upstream effort name. clampThinkingLevel returns "off" when
      // the level is unsupported or the user disabled thinking.
      const requestedLevel = options?.reasoning;
      const effectiveModel: Model<Api> =
        model.thinkingLevelMap || !modelConfig
          ? model
          : { ...model, thinkingLevelMap: buildThinkingLevelMap(modelConfig) };

      const clamped = requestedLevel ? clampThinkingLevel(effectiveModel, requestedLevel) : undefined;
      const reasoningLevel = clamped === "off" ? undefined : clamped;
      const parameters: Record<string, unknown> = { max_tokens: maxTokens };

      const tc = modelConfig?.thinking_config;
      const efforts = tc?.enabled?.efforts;
      const isAlwaysThinking = isReasoning && !!tc?.enabled && !tc.disabled;

      if (reasoningLevel) {
        parameters.enable_thinking = true;
        // Effort-based models advertise concrete effort names in the map
        // (low/medium/xhigh/max). Toggle-only models map every level to
        // "enabled"/"disabled" and accept no effort value — only the on/off
        // switch matters, so we send enable_thinking alone.
        const mapped = effectiveModel.thinkingLevelMap?.[reasoningLevel];
        let effort = mapped && mapped !== "enabled" && mapped !== "disabled" ? mapped : reasoningLevel;
        // Only send reasoning_effort when the upstream model actually exposes
        // effort levels (thinking_config.enabled.efforts).
        if (efforts && typeof efforts === "object") {
          const supportedEfforts = Object.keys(efforts);
          if (!supportedEfforts.includes(effort as string)) {
            // Pick default effort if marked, or fallback to supported effort
            const defaultEffort =
              Object.entries(efforts).find(([_, v]) => (v as { is_default?: boolean })?.is_default)?.[0] ||
              supportedEfforts[supportedEfforts.length - 1];
            effort = defaultEffort;
          }
          parameters.reasoning_effort = effort;
        }
      } else if (isAlwaysThinking) {
        // Upstream model enforces always-on thinking (e.g. GLM-5.3-Flash, error 1210).
        // If the user requested "off" or level was clamped to off, keep thinking enabled
        // using the model's default effort rather than triggering an upstream 400.
        parameters.enable_thinking = true;
        if (efforts && typeof efforts === "object") {
          const defaultEffort =
            Object.entries(efforts).find(([_, v]) => (v as { is_default?: boolean })?.is_default)?.[0] ||
            Object.keys(efforts)[0];
          if (defaultEffort) {
            parameters.reasoning_effort = defaultEffort;
          }
        }
      } else {
        // No reasoning level selected (or clamped to off): explicitly disable
        // thinking so the model does not reason by default.
        parameters.enable_thinking = false;
      }

      const reqBody: Record<string, unknown> = {
        request_id: crypto.randomUUID(),
        request_set_id: recordID,
        chat_record_id: recordID,
        session_id: sessionID,
        stream: true,
        chat_task: "FREE_INPUT",
        is_reply: true,
        is_retry: false,
        source: 1,
        version: "3",
        session_type: "qodercli",
        agent_id: "agent_common",
        task_id: "common",
        code_language: "",
        chat_prompt: "",
        image_urls: null,
        aliyun_user_type: "",
        // Qoder's server ignores the top-level `system` field (verified: the
        // model never sees it). Inject the system prompt as a leading
        // role:system message instead, which the server does honor.
        system: "",
        messages: finalMessages,
        tools: toolsRaw || [],
        parameters,
        chat_context: {
          chatPrompt: "",
          imageUrls: null,
          extra: {
            context: [],
            modelConfig: {
              key: qoderModel,
              is_reasoning: isReasoning,
            },
            originalContent: lastUserText,
          },
          features: [],
          text: lastUserText,
        },
        model_config: modelConfig,
        business: {
          product: "cli",
          version: "1.0.0",
          type: "agent",
          stage: "start",
          id: crypto.randomUUID(),
          name: lastUserText.substring(0, 30),
          begin_at: Date.now(),
        },
      };

      const bodyBytes = Buffer.from(JSON.stringify(reqBody));
      const encodedBytes = qoderEncodeBody(bodyBytes);

      const chatURL = getQoderChatURL(providerMode);

      const headers = buildAuthHeaders(encodedBytes, chatURL, {
        userID,
        authToken: accessToken,
        name,
        email,
        machineID,
      });

      const modelSource = modelConfig.source || "system";

      const response = await fetchWithQueueRetry(
        chatURL,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
            "Cache-Control": "no-cache",
            "Accept-Encoding": "identity",
            Connection: "keep-alive",
            "X-Model-Key": qoderModel,
            "X-Model-Source": modelSource,
            ...headers,
          },
          body: encodedBytes as unknown as BodyInit,
          signal: options?.signal,
        },
        options?.signal,
      );

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Qoder API request failed: ${response.status} ${response.statusText}. Response: ${errText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body");
      const decoder = new TextDecoder();
      let buffer = "";
      let bufferStart = 0;

      let contentBlockIndex = -1;
      let thinkingBlockIndex = -1;
      // Tail of the reasoning channel before DSML residue is stripped. Only the
      // tail matters (the degenerate-turn check looks for trailing markup), so
      // this stays bounded instead of growing with the reply.
      let rawReasoningTail = "";
      const toolCallsState: ToolCallState[] = [];

      const thinkingEnabled = (options?.reasoning as unknown) !== false && (options?.reasoning as unknown) !== "off";
      const thinkingParser = thinkingEnabled ? new ThinkingTagParser(output, stream) : null;

      stream.push({ type: "start", partial: output });

      // `data: [DONE]` is the end of the response. Break the read loop too, not
      // just the line loop: Qoder's gateway keeps the HTTP body open after the
      // sentinel, so waiting for `done` from reader.read() hung until the
      // server or the OS eventually closed the socket. The full reply had
      // already been streamed by then, so the agent looked stuck with no error.
      let sawDone = false;

      while (!sawDone) {
        const { done, value } = await reader.read();
        if (done) break;

        // Drop consumed prefix before appending so we do not keep growing a
        // dead head of the string across chunks.
        if (bufferStart > 0) {
          buffer = buffer.substring(bufferStart);
          bufferStart = 0;
        }
        buffer += decoder.decode(value, { stream: true });

        while (true) {
          const lineEnd = buffer.indexOf("\n", bufferStart);
          if (lineEnd === -1) break;

          const line = buffer.substring(bufferStart, lineEnd).trim();
          bufferStart = lineEnd + 1;

          if (!line.startsWith("data:")) continue;

          const dataStr = line.substring(5).trim();
          if (dataStr === "[DONE]") {
            sawDone = true;
            break;
          }

          try {
            const envelope = JSON.parse(dataStr);
            if (envelope.statusCodeValue && envelope.statusCodeValue !== 200) {
              throw new Error(`Upstream status ${envelope.statusCodeValue}: ${envelope.body}`);
            }

            let inner: Record<string, unknown> | null = null;
            if (typeof envelope.body === "string") {
              if (envelope.body === "[DONE]") {
                sawDone = true;
                break;
              }
              if (envelope.body) {
                inner = JSON.parse(envelope.body);
              }
            } else if (envelope.choices || envelope.id) {
              inner = envelope;
            }
            if (!inner) continue;
            if (inner.id) output.responseId = inner.id as string;
            // Keep Pi's selected model identity in `output.model`. Qoder may
            // report a different concrete backend route here; preserve that
            // separately for diagnostics without changing assistant identity.
            if (inner.model) output.responseModel = inner.model as string;
            if (inner.usage) {
              applyQoderCredits(output.usage, inner.usage as Record<string, unknown>, model.provider);
              const u = inner.usage as {
                prompt_tokens?: number;
                completion_tokens?: number;
                total_tokens?: number;
                completion_tokens_details?: { reasoning_tokens?: number };
                prompt_tokens_details?: {
                  cacheable_tokens?: number;
                  cached_tokens?: number;
                  cache_write_tokens?: number;
                };
              };
              // pi-core computes `promptTokens = input + cacheRead + cacheWrite`
              // (Anthropic convention: `input` EXCLUDES cached/written tokens).
              // Qoder follows OpenAI semantics where `prompt_tokens` INCLUDES
              // `cached_tokens`, so subtract cacheRead (and cache_write_tokens
              // when reported) to match the contract pi-ai's own OpenAI
              // provider uses. `cacheable_tokens` is a capacity metric, not a
              // write count (it is 0 even on first-turn writes), so it is NOT
              // mapped to cacheWrite.
              const promptTokens = u.prompt_tokens ?? 0;
              const cacheReadTokens = u.prompt_tokens_details?.cached_tokens ?? 0;
              const cacheWriteTokens = u.prompt_tokens_details?.cache_write_tokens ?? 0;
              output.usage.input = Math.max(0, promptTokens - cacheReadTokens - cacheWriteTokens);
              output.usage.output = u.completion_tokens ?? 0;
              output.usage.totalTokens = u.total_tokens ?? 0;
              output.usage.cacheRead = cacheReadTokens;
              output.usage.cacheWrite = cacheWriteTokens;
            }
            const choices = inner.choices as
              | Array<{
                  delta?: {
                    reasoning_content?: string;
                    content?: string;
                    tool_calls?: Array<{
                      index?: number;
                      id?: string;
                      function?: { name?: string; arguments?: string };
                    }>;
                  };
                  finish_reason?: string;
                }>
              | undefined;
            if (Array.isArray(choices) && choices.length > 0) {
              const choice = choices[0];
              const delta = choice.delta;

              if (delta) {
                // 1. Process reasoning/thinking content (API reasoning)
                if (delta.reasoning_content) {
                  // Qoder's backend sometimes routes a literal `<thinking>`
                  // opener into reasoning_content (with the matching
                  // `</thinking>` closer landing in the content stream), and
                  // sometimes leaks unparsed DSML tool-call markup there. Strip
                  // both so the thinking block stays clean, matching the SDK's
                  // ContentBlock model.
                  rawReasoningTail = (rawReasoningTail + delta.reasoning_content).slice(-512);
                  const reasoningChunk = stripDsmlResidue(stripThinkingTags(delta.reasoning_content));
                  if (reasoningChunk) {
                    if (thinkingBlockIndex === -1) {
                      thinkingBlockIndex = output.content.length;
                      output.content.push({ type: "thinking", thinking: "" });
                      stream.push({ type: "thinking_start", contentIndex: thinkingBlockIndex, partial: output });
                    }
                    const block = output.content[thinkingBlockIndex] as ThinkingContent;
                    block.thinking += reasoningChunk;
                    stream.push({
                      type: "thinking_delta",
                      contentIndex: thinkingBlockIndex,
                      delta: reasoningChunk,
                      partial: output,
                    });
                  }
                }

                // 2. Process text content
                if (delta.content) {
                  // End API thinking block if active
                  if (thinkingBlockIndex !== -1) {
                    const block = output.content[thinkingBlockIndex] as ThinkingContent;
                    stream.push({
                      type: "thinking_end",
                      contentIndex: thinkingBlockIndex,
                      content: block.thinking,
                      partial: output,
                    });
                    thinkingBlockIndex = -1;
                  }

                  if (thinkingParser) {
                    thinkingParser.processChunk(delta.content);
                  } else {
                    if (contentBlockIndex === -1) {
                      contentBlockIndex = output.content.length;
                      output.content.push({ type: "text", text: "" });
                      stream.push({ type: "text_start", contentIndex: contentBlockIndex, partial: output });
                    }
                    const block = output.content[contentBlockIndex] as TextContent;
                    block.text += delta.content;
                    stream.push({
                      type: "text_delta",
                      contentIndex: contentBlockIndex,
                      delta: delta.content,
                      partial: output,
                    });
                  }
                }

                // 3. Process tool calls
                if (delta.tool_calls && Array.isArray(delta.tool_calls)) {
                  if (thinkingBlockIndex !== -1) {
                    const block = output.content[thinkingBlockIndex] as ThinkingContent;
                    stream.push({
                      type: "thinking_end",
                      contentIndex: thinkingBlockIndex,
                      content: block.thinking,
                      partial: output,
                    });
                    thinkingBlockIndex = -1;
                  }
                  if (thinkingParser) {
                    thinkingParser.finalize();
                  }

                  for (const tc of delta.tool_calls) {
                    const idx = tc.index ?? 0;
                    if (!toolCallsState[idx]) {
                      toolCallsState[idx] = {
                        arguments: "",
                        id: tc.id || "",
                        name: "",
                        contentIndex: 0,
                      };
                    }
                    const state = toolCallsState[idx];
                    if (tc.id) state.id = tc.id;
                    if (tc.function?.name) state.name = tc.function.name;

                    // Open the block as soon as the call is IDENTIFIABLE, not
                    // when its first argument byte arrives. A call whose
                    // arguments are absent or an empty string — a no-argument
                    // tool, or a model that sends id+name and then stops — used
                    // to create a toolCallsState entry and no content block, so
                    // the finalizer below saw a non-empty state array, set
                    // stopReason "toolUse", and handed back a message with no
                    // tool call in it. The agent loop then had nothing to run
                    // and the turn simply ended, mid-task and without an error.
                    if (state.emittedStart === undefined && (state.id || state.name)) {
                      state.emittedStart = true;
                      if (!state.id) state.id = `call_${crypto.randomUUID().slice(0, 8)}`;
                      state.contentIndex = output.content.length;
                      output.content.push({
                        type: "toolCall",
                        id: state.id,
                        name: state.name,
                        arguments: {},
                      } satisfies ToolCall);
                      stream.push({ type: "toolcall_start", contentIndex: state.contentIndex, partial: output });
                    }

                    // id and name can arrive after the block is open; keep it
                    // in step, since the finalizer only rewrites `arguments`.
                    if (state.emittedStart) {
                      const block = output.content[state.contentIndex] as ToolCall;
                      block.id = state.id;
                      block.name = state.name;
                    }

                    if (tc.function?.arguments) {
                      const argDelta = tc.function.arguments;
                      state.arguments += argDelta;
                      stream.push({
                        type: "toolcall_delta",
                        contentIndex: state.contentIndex,
                        delta: argDelta,
                        partial: output,
                      });
                    }
                  }
                }
              }

              if (choice.finish_reason) {
                // Preserve the real upstream finish_reason (e.g. "length",
                // "content_filter") instead of forcing "stop" later.
                output.stopReason = choice.finish_reason as AssistantMessage["stopReason"];
              }
            }
          } catch (e) {
            // A single malformed SSE line shouldn't kill the stream — skip it.
            // But a genuine upstream error (thrown below) must propagate to the
            // outer catch and surface as stopReason="error", not be swallowed.
            if (e instanceof SyntaxError) {
              if (process.env.QODER_DEBUG) {
                console.error("[pi-provider-qoder] skipping malformed SSE line:", dataStr.slice(0, 200));
              }
              continue;
            }
            throw e;
          }
        }
      }

      // Stop reading and let the connection go once the reply is complete.
      // Without this the body stays open until the server times it out.
      await reader.cancel().catch(() => {});

      if (!sawDone) {
        throw new Error(
          `Qoder stream disconnected prematurely (connection closed before [DONE] after ${output.usage.output || 0} output tokens)`,
        );
      }

      if (thinkingParser) {
        thinkingParser.finalize();
      }

      if (thinkingBlockIndex !== -1) {
        const block = output.content[thinkingBlockIndex] as ThinkingContent;
        stream.push({
          type: "thinking_end",
          contentIndex: thinkingBlockIndex,
          content: block.thinking,
          partial: output,
        });
      }

      for (const state of toolCallsState) {
        if (state?.emittedStart && !state.emittedEnd) {
          state.emittedEnd = true;
          let args = {};
          try {
            args = JSON.parse(state.arguments || "{}");
          } catch {}
          const block = output.content[state.contentIndex] as ToolCall;
          block.arguments = args;
          stream.push({
            type: "toolcall_end",
            contentIndex: state.contentIndex,
            toolCall: {
              type: "toolCall",
              id: state.id,
              name: state.name,
              arguments: args,
            },
            partial: output,
          });
        }
      }

      // Fallback: If no native tool_calls were emitted, check if the model output
      // XML/text <tool_call> tags in text blocks (e.g. Qwen / DeepSeek XML format).
      if (!toolCallsState.some((state) => state?.emittedStart)) {
        for (let i = 0; i < output.content.length; i++) {
          const block = output.content[i];
          if (block.type === "text" && block.text.includes("<tool_call>")) {
            const { cleanText, toolCalls } = parseToolCallsFromText(block.text);
            if (toolCalls.length > 0) {
              block.text = cleanText;
              for (const tc of toolCalls) {
                const callId = `call_${crypto.randomUUID().slice(0, 8)}`;
                const toolCallBlock: ToolCall = {
                  type: "toolCall",
                  id: callId,
                  name: tc.name,
                  arguments: tc.arguments,
                };
                const contentIndex = output.content.length;
                output.content.push(toolCallBlock);
                stream.push({
                  type: "toolcall_start",
                  contentIndex,
                  partial: output,
                });
                stream.push({
                  type: "toolcall_end",
                  contentIndex,
                  toolCall: toolCallBlock,
                  partial: output,
                });
                toolCallsState.push({
                  id: callId,
                  name: tc.name,
                  arguments: JSON.stringify(tc.arguments),
                  emittedStart: true,
                  emittedEnd: true,
                  contentIndex,
                });
              }
            }
          }
        }
      }

      // Guarded on blocks that actually reached the message, not on the state
      // array being non-empty. Claiming "toolUse" for a message carrying no
      // tool call is what turned a malformed stream into a silent dead end.
      // Never overwrite a terminal finish_reason ("length", "content_filter")
      // because truncated tool calls are incomplete.
      if (toolCallsState.some((state) => state?.emittedStart)) {
        if (output.stopReason !== "length") {
          output.stopReason = "toolUse";
        }
      }
      // A turn the gateway stripped the tool call out of has no text and no tool
      // call, so `stop` reads as a finished task: the agent loop ends the turn
      // and the work dies with no error to retry. Report it instead.
      if (isDegenerateDsmlTurn(output, rawReasoningTail)) {
        output.stopReason = "error";
        // "server error" is deliberate: pi-ai classifies an assistant error by
        // matching its message (RETRYABLE_PROVIDER_ERROR_PATTERN), and that
        // phrase is what makes the turn retryable instead of a dead end.
        output.errorMessage =
          "Qoder server error: degenerate model output (unparseable DSML tool-call markup, no tool call executed)";
        stream.push({ type: "error", reason: "error", error: output });
        stream.end();
        return;
      }
      // Otherwise keep whatever finish_reason set upstream (defaults to "stop").
      // Never overwrite a meaningful finish_reason ("length", "content_filter",
      // ...) with "stop".
      stream.push({
        type: "done",
        reason: output.stopReason as Extract<AssistantMessage["stopReason"], "stop" | "length" | "toolUse">,
        message: output,
      });
      stream.end();
    } catch (e: unknown) {
      output.stopReason = options?.signal?.aborted ? "aborted" : "error";
      output.errorMessage = e instanceof Error ? e.message : String(e);
      stream.push({ type: "error", reason: output.stopReason, error: output });
      try {
        stream.end();
      } catch {}
    }
  })();

  return stream;
}
