import crypto from "node:crypto";
import * as PiAi from "@earendil-works/pi-ai";
import {
  type Api,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  clampThinkingLevel,
  type Message,
  type Model,
  type SimpleStreamOptions,
  type TextContent,
  type ThinkingContent,
  type Tool,
  type ToolCall,
  type TranscriptContext,
} from "@earendil-works/pi-ai";
import { resolveQoderIdentity } from "../auth/oauth.js";
import {
  buildThinkingLevelMap,
  checkAccountEntitlement,
  isAlwaysThinkingModel,
  MAX_OUTPUT_TOKENS,
  resolveModelConfig,
} from "../catalog.js";
import { buildAuthHeaders, getMachineId } from "../cosy.js";
import { applyQoderCredits } from "../credits.js";
import { getQoderChatURL, getQoderRegionConfig } from "../region.js";
import { qoderEncodeBody } from "./encoding.js";
import { isDegenerateDsmlTurn, stripDsmlResidue, stripThinkingTags, ThinkingTagParser } from "./thinking.js";
import { parseToolCallsFromText } from "./tool-parser.js";
import { extractTools, transformMessagesForQoder, transformTools } from "./transform.js";

/**
 * Qoder's scheduler reports a busy model as error code 10605. The official
 * client may return either a ready-to-retry snapshot or an explicitly queued
 * snapshot (`isQueued:true`, `serviceAvailable:false`). Both are transient
 * and must honor `retryAfterSeconds`; otherwise a real queue rejection kills
 * the whole turn even though the request could have succeeded moments later.
 */
const MODEL_BUSY_CODE = "10605";
const DEFAULT_MODEL_BUSY_RETRIES = 10;
const MAX_MODEL_BUSY_RETRIES = 10;
const MODEL_BUSY_MAX_WAIT_MS = 540_000;

/** A recovery-eligible "model busy" rejection from Qoder's queue scheduler. */
class QoderModelBusyError extends Error {
  constructor(
    message: string,
    readonly retryAfterMs: number | undefined,
    readonly sawModelOutput = false,
  ) {
    super(message);
    this.name = "QoderModelBusyError";
  }
}

interface ModelBusyQueueSnapshot {
  retryAfterMs?: number;
}

/** Parse `unknown | JSON string` layers until an object is found (bounded). */
function parseJsonObject(value: unknown, depth = 0): Record<string, unknown> | undefined {
  if (depth > 4) return undefined;
  if (typeof value === "string") {
    try {
      return parseJsonObject(JSON.parse(value), depth + 1);
    } catch {
      return undefined;
    }
  }
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

/**
 * Recognize Qoder's model-busy envelope and extract the retry hint.
 *
 * Envelope body shape (three nested JSON layers):
 *   {code:"403",message:"{\"code\":\"10605\",\"message\":\"{\\\"isQueued\\\":false,...}\",\"subCode\":\"RESOURCE_LIMIT\"}"}
 *
 * Both documented transient combinations are eligible:
 * `isQueued:false` + `serviceAvailable:true` (retry when the slot opens) and
 * `isQueued:true` + `serviceAvailable:false` (already waiting in the queue).
 * Missing flags, `isQueued:false` + `serviceAvailable:false`, or any other
 * code keep the failure behavior. `subCode` is diagnostic only.
 */
export function parseModelBusyRejection(body: unknown): ModelBusyQueueSnapshot | undefined {
  const outer = parseJsonObject(body);
  const busy = parseJsonObject(outer?.message);
  if (busy?.code !== MODEL_BUSY_CODE) return undefined;
  const queue = parseJsonObject(busy.message);
  if (!queue) return undefined;
  const readyToRetry = queue.isQueued === false && queue.serviceAvailable === true;
  const queued = queue.isQueued === true && queue.serviceAvailable === false;
  if (!readyToRetry && !queued) return undefined;
  const seconds = queue.retryAfterSeconds;
  const retryAfterMs =
    typeof seconds === "number" && Number.isFinite(seconds) && seconds > 0 ? Math.round(seconds * 1000) : 0;
  return { retryAfterMs };
}

/** Sleep that rejects as soon as the caller aborts. */
function sleepWithAbort(ms: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.reject(new Error("Aborted"));
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error("Aborted"));
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(new Error("Aborted"));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

interface ToolCallCandidates {
  states: Set<ToolCallState>;
  complete: Set<ToolCallState>;
  unique: ToolCallState | undefined;
}

interface ToolCallState {
  argumentChunks: string[];
  argumentLength: number;
  objectRoot?: boolean;
  depth: number;
  inString: boolean;
  escaped: boolean;
  complete: boolean;
  parsedArguments?: ToolCall["arguments"];
  id: string;
  name: string;
  block?: ToolCall;
  contentIndex: number;
  candidates: Set<ToolCallCandidates>;
  unindexed: boolean;
}

interface ToolCallAggregation {
  states: ToolCallState[];
  byId: Map<string, ToolCallState>;
  byIndex: Map<number, ToolCallCandidates>;
  unscopedCandidates: Set<ToolCallState>;
  statesByBlock: Map<ToolCall, ToolCallState>;
  lastToolState?: ToolCallState;
}

function refreshToolIndexes(output: AssistantMessage, toolCalls: ToolCallAggregation): void {
  const lastToolState = toolCalls.lastToolState;
  if (!lastToolState || output.content[lastToolState.contentIndex] === lastToolState.block) return;
  // ThinkingTagParser.emitThinking() can splice only when creating its first
  // thinking block. All other layout changes append. Refresh in one scan.
  for (let index = 0; index < output.content.length; index++) {
    const block = output.content[index];
    if (block.type !== "toolCall") continue;
    const state = toolCalls.statesByBlock.get(block);
    if (state) state.contentIndex = index;
  }
}

function retireCompleteCandidates(toolCalls: ToolCallAggregation, candidates: ToolCallCandidates): void {
  for (const state of candidates.complete) {
    // Structural closure alone is not sufficient evidence of a boundary.
    parseToolArguments(state);
    candidates.states.delete(state);
    state.candidates.delete(candidates);
    if (state.candidates.size === 0 && !state.unindexed) toolCalls.unscopedCandidates.delete(state);
  }
  candidates.complete.clear();
  candidates.unique = candidates.states.size === 1 ? candidates.states.values().next().value : undefined;
}

const TOOL_ARGUMENT_STRUCTURE = /["\\{}[\]]/g;
const NON_JSON_WHITESPACE = /[^ \t\r\n]/;

function invalidToolArguments(state: ToolCallState): Error {
  return new Error(`Invalid or incomplete tool-call JSON (characters=${state.argumentLength})`);
}

// Only locate a possible root-object boundary here. JSON.parse remains the
// authority on grammar; scanning each new fragment avoids parsing prefixes.
function appendToolArguments(state: ToolCallState, fragment: string): void {
  state.argumentChunks.push(fragment);
  state.argumentLength += fragment.length;
  if (state.complete) {
    if (NON_JSON_WHITESPACE.test(fragment)) throw invalidToolArguments(state);
    return;
  }
  let start = 0;
  if (state.objectRoot === undefined) {
    start = fragment.search(NON_JSON_WHITESPACE);
    if (start === -1) return;
    state.objectRoot = fragment[start] === "{";
  }
  if (!state.objectRoot) return;
  if (state.escaped) {
    start = 1;
    state.escaped = false;
  }
  TOOL_ARGUMENT_STRUCTURE.lastIndex = start;
  while (true) {
    const match = TOOL_ARGUMENT_STRUCTURE.exec(fragment);
    if (!match) break;
    const character = match[0];
    if (state.inString) {
      if (character === "\\") {
        if (TOOL_ARGUMENT_STRUCTURE.lastIndex === fragment.length) state.escaped = true;
        else TOOL_ARGUMENT_STRUCTURE.lastIndex++;
      } else if (character === '"') {
        state.inString = false;
      }
    } else if (character === '"') {
      state.inString = true;
    } else if (character === "{" || character === "[") {
      state.depth++;
    } else if (character === "}" || character === "]") {
      state.depth--;
      if (state.depth === 0) {
        state.complete = true;
        if (NON_JSON_WHITESPACE.test(fragment.slice(TOOL_ARGUMENT_STRUCTURE.lastIndex))) {
          throw invalidToolArguments(state);
        }
        return;
      }
    }
  }
}

function parseToolArguments(state: ToolCallState): ToolCall["arguments"] {
  if (state.parsedArguments) return state.parsedArguments;
  let args: unknown = {};
  if (state.argumentLength > 0) {
    try {
      args = JSON.parse(state.argumentChunks.join(""));
    } catch {
      // Do not let the malformed-SSE SyntaxError handler swallow this failure,
      // and never include the parser's message (which can expose arguments).
      throw invalidToolArguments(state);
    }
  }
  if (args === null || typeof args !== "object" || Array.isArray(args)) {
    throw new Error(`Non-object tool-call arguments (characters=${state.argumentLength})`);
  }
  state.parsedArguments = args as ToolCall["arguments"];
  return state.parsedArguments;
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

// fork: XML `<tool_call>` recovery is this fork's addition; upstream has no
// text-markup fallback. Keep it intact when re-syncing stream.ts.
/**
 * Recover tool calls a model emitted as `<tool_call>` XML inside text blocks.
 *
 * Some Qoder upstreams (Qwen / DeepSeek XML tool mode) answer with the call as
 * markup instead of native `tool_calls` deltas. The markup is stripped from the
 * text and turned into real tool-call blocks, emitting the same start/end events
 * a native call produces so the host runs it normally.
 *
 * Mutates `output.content` and returns the number of recovered calls.
 */
function recoverXmlToolCalls(output: AssistantMessage, stream: AssistantMessageEventStream): number {
  let recovered = 0;
  for (let i = 0; i < output.content.length; i++) {
    const block = output.content[i];
    if (block.type !== "text" || !block.text.includes("<tool_call>")) continue;

    const parsed = parseToolCallsFromText(block.text);
    if (parsed.toolCalls.length === 0) continue;
    block.text = parsed.cleanText;

    for (const call of parsed.toolCalls) {
      const toolCallBlock: ToolCall = {
        type: "toolCall",
        id: `call_${crypto.randomUUID().slice(0, 8)}`,
        name: call.name,
        arguments: call.arguments as ToolCall["arguments"],
      };
      const contentIndex = output.content.length;
      output.content.push(toolCallBlock);
      stream.push({ type: "toolcall_start", contentIndex, partial: output });
      stream.push({ type: "toolcall_end", contentIndex, toolCall: toolCallBlock, partial: output });
      recovered++;
    }
  }
  return recovered;
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

/**
 * The pi-ai transcript helpers we feature-detect at runtime. Derived from the
 * pi-ai namespace so the signatures stay in step; every member is optional
 * because pi-ai <=0.85 does not export them at all.
 */
type TranscriptHelpers = Partial<
  Pick<
    typeof PiAi,
    | "collapseSystemMessages"
    | "getCurrentTools"
    | "getInitialSystemMessage"
    | "getSystemMessageText"
    | "withoutInitialSystemMessage"
  >
>;

/**
 * Resolve the messages, system-prompt text, and tools for the Qoder request
 * from EITHER pi-ai provider contract:
 *
 *  - pi-ai >=0.86 hands providers a `TranscriptContext` (`{ messages }` only):
 *    `normalizeContext()` folds the system prompt and tool declarations into the
 *    transcript's leading system message, so the legacy top-level
 *    `context.systemPrompt` / `context.tools` are undefined. Resolve them via the
 *    transcript helpers, and drop the consolidated leading system message from
 *    the conversation (reqBody re-injects `systemText` as its own role:system
 *    message, because Qoder ignores the top-level `system` field).
 *  - pi-ai <=0.85 hands providers a flat `Context` with top-level
 *    `systemPrompt` / `tools` and no transcript helpers.
 *  - Some >=0.86 hosts hand providers a flat `Context` anyway while still
 *    exporting the transcript toolkit to legacy extensions (omp 18.12.x does:
 *    its compat root exports the helpers, but `streamQoder` receives
 *    `systemPrompt` + `tools` unfolded). The leading system message is what
 *    marks a folded transcript, so its absence means the flat fields still
 *    carry the prompt and tools.
 *
 * The helpers are passed in (as the pi-ai namespace) and feature-detected, so a
 * single bundle runs on all three: on 0.85 the namespace simply lacks them and
 * we use the flat fields. Using a namespace import for this is deliberate — a
 * named import of a 0.86-only export would throw at module-load time on 0.85.
 */
export function resolveRequestContext(
  piAi: TranscriptHelpers,
  context: Context,
): { messages: Message[]; systemText: string; tools: Tool[] } {
  const {
    collapseSystemMessages,
    getCurrentTools,
    getInitialSystemMessage,
    getSystemMessageText,
    withoutInitialSystemMessage,
  } = piAi;

  // Require the FULL transcript toolkit — including getSystemMessageText — before
  // taking the transcript path. If any helper is missing we cannot faithfully
  // reconstruct the prompt + tools from the transcript, so fall back to the flat
  // Context fields rather than silently dropping the system prompt (which is what
  // happens if we resolve tools from the transcript but lack getSystemMessageText:
  // context.systemPrompt is undefined under the >=0.86 contract).
  const hasTranscript =
    typeof collapseSystemMessages === "function" &&
    typeof getCurrentTools === "function" &&
    typeof getInitialSystemMessage === "function" &&
    typeof getSystemMessageText === "function" &&
    typeof withoutInitialSystemMessage === "function";

  if (!hasTranscript) {
    // pi-ai <=0.85: flat Context carries the prompt and tools directly.
    return {
      messages: context.messages,
      systemText: contentToText(context.systemPrompt || ""),
      tools: context.tools ?? [],
    };
  }

  const collapsed = collapseSystemMessages(context as unknown as TranscriptContext);
  const transcriptMessages = collapsed.messages;
  const initialSystem = getInitialSystemMessage(transcriptMessages);

  // The leading system message is the only proof that this host folded
  // `systemPrompt` and `tools` into the transcript. Without it the helpers
  // replay nothing — `getCurrentTools` reads `toolsAdded` off system messages —
  // so a host that exports the toolkit but still passes a flat `Context` would
  // otherwise send `tools: []` and lose the tool-call protocol entirely.
  if (!initialSystem) {
    return {
      messages: context.messages,
      systemText: contentToText(context.systemPrompt || ""),
      tools: context.tools ?? [],
    };
  }

  const tools = getCurrentTools(transcriptMessages);
  const systemText = contentToText(getSystemMessageText?.(initialSystem) ?? context.systemPrompt ?? "");
  const messages = withoutInitialSystemMessage(transcriptMessages);
  return { messages, systemText, tools };
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
      // fork: multi-account providers are `qoder-2` / `qoder-cn-2` / ...; upstream
      // compares against the exact id only.
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
      // fork: per-account entitlement pre-flight. Upstream has no per-account
      // catalogue, so this whole block is fork-only.
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

      // Resolve messages / system prompt / tools from whichever pi-ai contract
      // is running (>=0.86 TranscriptContext, or <=0.85 flat Context). The
      // transcript helpers are feature-detected inside, so one bundle runs on
      // both. See resolveRequestContext.
      const resolved = resolveRequestContext(PiAi, context);
      const normalizedMessages = transformMessagesForQoder(resolved.messages);
      // OMP may supply the system prompt as a single-element content array;
      // Qoder MessagesInputDto#content is a String and rejects an array with
      // "Execution failed: set property ... MessagesInputDto#content". Normalize.
      const systemText = resolved.systemText;

      // Qoder ignores the top-level `system` field, so the prompt has to ride
      // as a leading role:system message. Fold every system message the
      // transcript still carries (pi splits some into `sections`) into that one
      // leading message instead of scattering them through the conversation.
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

      // resolved.tools already prefers the transcript (pi-ai >=0.86) and falls
      // back to the legacy top-level context.tools (<=0.85). Without a bound
      // tool list the request ships `tools: []`, the model gets no function
      // schema, and it free-forms ChatML tool calls as plain text that never
      // execute — the regression this fixes. extractTools additionally walks
      // `toolsAdded`/`toolsRemoved` on host system messages, which covers hosts
      // that pass a flat Context whose tools only live in the transcript.
      // fork: the extractTools fallback (and combinedSystemPrompt folding above)
      // is fork-only; upstream resolves tools from resolveRequestContext alone.
      const effectiveTools = resolved.tools.length > 0 ? resolved.tools : extractTools(context);
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

      // Normalize every host's "thinking off" signal into one flag. pi and OMP
      // both resolve level "off" to `reasoning: undefined`; OMP's fork also sets
      // `disableReasoning` / `forceReasoningOff`; `false` / `"off"` cover older
      // hosts. When off, skip clamping so the request never snaps back up to a
      // concrete level.
      const extOptions = options as
        | (SimpleStreamOptions & { disableReasoning?: boolean; forceReasoningOff?: boolean })
        | undefined;
      const explicitlyOff =
        extOptions?.disableReasoning === true ||
        extOptions?.forceReasoningOff === true ||
        (requestedLevel as unknown) === false ||
        (requestedLevel as unknown) === "off";

      // `undefined` here (explicit off, omitted level, or clamped to off) means
      // "disable thinking"; the else branch below turns it into the wire signals.
      //
      // The clamp runs against a model whose thinkingLevelMap is rebuilt from the
      // cached catalog when the host did not supply one, so the requested level
      // is only clamped to levels this model actually advertises.
      const effectiveModel: Model<Api> =
        model.thinkingLevelMap || !modelConfig
          ? model
          : { ...model, thinkingLevelMap: buildThinkingLevelMap(modelConfig) };

      const clamped = !explicitlyOff && requestedLevel ? clampThinkingLevel(effectiveModel, requestedLevel) : undefined;
      const reasoningLevel = clamped === "off" ? undefined : clamped;
      const parameters: Record<string, unknown> = { max_tokens: maxTokens };

      const tc = modelConfig?.thinking_config;
      const efforts = tc?.enabled?.efforts;
      // Upstream model enforces always-on thinking (e.g. GLM-5.3-Flash, error
      // 1210): it must never receive enable_thinking:false, so "off" has to be
      // satisfied with the model's default effort instead.
      //
      // The catalogue shape cannot tell an always-on model from a merely
      // effort-limited one (both lack a `disabled` branch), so this comes from
      // the curated seed table. The `off: null` pin is the only recorded signal.
      // fork: the "mandatory reasoning" decision comes from the curated seed
      // table (`off: null`), not from the catalogue shape — upstream infers it
      // from `thinking_config.disabled` and would misread every model whose
      // config this fork synthesizes from `is_reasoning`.
      const isAlwaysThinking = isAlwaysThinkingModel(model.id, providerMode);

      if (reasoningLevel) {
        parameters.enable_thinking = true;
        // Effort-based models advertise concrete effort names in the map
        // (low/medium/xhigh/max). Toggle-only models map every level to
        // "enabled"/"disabled" and accept no effort value — only the on/off
        // switch matters, so we send enable_thinking alone.
        const mapped = effectiveModel.thinkingLevelMap?.[reasoningLevel];
        let effort = mapped && mapped !== "enabled" && mapped !== "disabled" ? mapped : reasoningLevel;
        // Only send reasoning_effort when the upstream model actually exposes
        // effort levels (thinking_config.enabled.efforts). The clamped level can
        // still be one the catalog does not list (a host-side alias), so fall
        // back to the catalog default or its last supported effort.
        if (efforts && typeof efforts === "object") {
          const supportedEfforts = Object.keys(efforts);
          if (!supportedEfforts.includes(effort as string)) {
            const defaultEffort =
              Object.entries(efforts).find(([_, v]) => (v as { is_default?: boolean })?.is_default)?.[0] ||
              supportedEfforts[supportedEfforts.length - 1];
            effort = defaultEffort;
          }
          parameters.reasoning_effort = effort;
        }
      } else if (isAlwaysThinking) {
        // No level selected (or the host asked for off): keep thinking enabled
        // with the model's default effort rather than triggering an upstream 400.
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
        // Explicitly disable thinking. `enable_thinking: false` alone is not
        // enough: qodercli also sends `reasoning_effort: "none"` (and flips
        // `model_config.is_reasoning` below), otherwise the gateway still
        // routes the reasoning model as reasoning-enabled.
        //
        // Exception: when the catalog positively shows the model has no
        // disabled branch (effort-based thinking_config without `disabled`,
        // or thinkingLevelMap pinning `off` to null), some upstreams reject
        // `reasoning_effort: "none"` outright (Sonus: 400 "Unsupported
        // value: 'none'"). For those, omit the effort signal — the request
        // stays valid and `enable_thinking: false` is the strongest disable
        // the upstream accepts. With no catalog signal either way (static
        // fallback / tests), keep sending "none".
        parameters.enable_thinking = false;
        const lacksDisabledBranch = (efforts && !tc?.disabled) || effectiveModel.thinkingLevelMap?.off === null;
        if (!lacksDisabledBranch) {
          parameters.reasoning_effort = "none";
        }
      }

      // An always-on-thinking model keeps `is_reasoning: true` even when the host
      // asked for "off" — the request still enables thinking (see the
      // isAlwaysThinking branch above), so flipping the flag here would describe
      // the request incorrectly. Only a genuine disable clears it.
      const thinkingActive = !!reasoningLevel || isAlwaysThinking;

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
        session_type: providerMode === "cn" ? "qoderclicn" : "qodercli",
        agent_id: "agent_common",
        task_id: "common",
        code_language: "",
        chat_prompt: "",
        image_urls: null,
        aliyun_user_type: "",
        // Qoder's server ignores the top-level `system` field (verified: the
        // model never sees it). The prompt rides as the leading role:system
        // message folded above (systemText + any system messages the transcript
        // still carried), so use finalMessages rather than re-prepending
        // systemText to a list that may already contain system messages.
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
              is_reasoning: thinkingActive ? isReasoning : false,
            },
            originalContent: lastUserText,
          },
          features: [],
          text: lastUserText,
        },
        model_config: thinkingActive ? modelConfig : { ...modelConfig, is_reasoning: false },
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

      const chatURL = getQoderChatURL(providerMode);
      const modelSource = modelConfig.source || "system";

      // A caller-provided retry budget (including an explicit 0) wins; otherwise
      // retry up to the bounded queue budget. Non-finite values
      // fall back to the default so the caps stay finite.
      const requestedRetries = options?.maxRetries;
      const modelBusyRetries = Math.min(
        Math.max(
          0,
          typeof requestedRetries === "number" && Number.isFinite(requestedRetries)
            ? Math.floor(requestedRetries)
            : DEFAULT_MODEL_BUSY_RETRIES,
        ),
        MAX_MODEL_BUSY_RETRIES,
      );
      // pi's contract: `maxRetryDelayMs: 0` disables the caller cap. The
      // provider still enforces a bounded ceiling so a bad server hint cannot
      // block a session indefinitely.
      const requestedMaxWaitMs = options?.maxRetryDelayMs;
      const modelBusyMaxWaitMs =
        typeof requestedMaxWaitMs === "number" && Number.isFinite(requestedMaxWaitMs) && requestedMaxWaitMs > 0
          ? Math.min(MODEL_BUSY_MAX_WAIT_MS, requestedMaxWaitMs)
          : MODEL_BUSY_MAX_WAIT_MS;

      let encodedBytes = qoderEncodeBody(Buffer.from(JSON.stringify(reqBody)));
      let headers = buildAuthHeaders(encodedBytes, chatURL, {
        userID,
        authToken: accessToken,
        name,
        email,
        machineID,
      });

      const thinkingEnabled = (options?.reasoning as unknown) !== false && (options?.reasoning as unknown) !== "off";

      // Exactly one `start` per logical stream: recovery retries reuse it.
      stream.push({ type: "start", partial: output });

      const runAttempt = async (): Promise<void> => {
        // Reset every mutable output field BEFORE the request: a failed attempt
        // may otherwise leak metadata, usage (including Qoder credits) or a
        // finish_reason into the attempt that succeeds, or into the terminal
        // error when the next attempt fails at fetch/HTTP level.
        options?.signal?.throwIfAborted();
        output.stopReason = "stop";
        output.errorMessage = undefined;
        output.responseId = undefined;
        output.responseModel = undefined;
        output.usage = {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        };

        const response = await fetch(chatURL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
            "Cache-Control": "no-cache",
            "Accept-Encoding": "identity",
            // Long SSE sessions get dropped by intermediate proxies when they
            // reuse a connection the server is about to close; asking for
            // keep-alive keeps a single long-lived stream instead.
            // fork: keep-alive for long SSE sessions; upstream omits it.
            Connection: "keep-alive",
            "X-Model-Key": qoderModel,
            "X-Model-Source": modelSource,
            ...headers,
          },
          body: encodedBytes as unknown as BodyInit,
          signal: options?.signal,
        });

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
        let toolCalls: ToolCallAggregation | undefined;

        const thinkingParser = thinkingEnabled ? new ThinkingTagParser(output, stream) : null;

        let sawModelOutput = false;

        // `data: [DONE]` is the end of the response. Break the read loop too, not
        // just the line loop: Qoder's gateway keeps the HTTP body open after the
        // sentinel, so waiting for `done` from reader.read() hung until the
        // server or the OS eventually closed the socket. The full reply had
        // already been streamed by then, so the agent looked stuck with no error.
        let sawDone = false;

        try {
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
                  const busy = parseModelBusyRejection(envelope.body);
                  if (busy) {
                    throw new QoderModelBusyError(
                      `Upstream status ${envelope.statusCodeValue}: ${envelope.body}`,
                      busy.retryAfterMs,
                      sawModelOutput,
                    );
                  }
                  throw new Error(`Upstream status ${envelope.statusCodeValue}: ${envelope.body}`);
                }

                const innerStr = envelope.body;
                // The gateway sends the sentinel wrapped in an envelope
                // (`body: "[DONE]"`) as well as bare, and both mean the reply is
                // over, so both have to end the read loop.
                if (innerStr === "[DONE]") {
                  sawDone = true;
                  break;
                }
                if (!innerStr) continue;

                const inner = JSON.parse(innerStr);
                if (inner.id) output.responseId = inner.id as string;
                if (inner.model) output.responseModel = inner.model as string;
                if (inner.usage) {
                  // fork: pass the raw provider id — multi-account providers are
                  // `qoder-2` / `qoder-cn-2`, and applyQoderCredits normalizes
                  // them itself. Collapsing to a two-value enum here would tag
                  // every secondary account as the primary one. Upstream passes
                  // the collapsed enum instead.
                  applyQoderCredits(output.usage, inner.usage, model.provider);
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
                if (inner.choices && inner.choices.length > 0) {
                  const choice = inner.choices[0];
                  const delta = choice.delta;

                  if (delta) {
                    // Any raw generation field ends recovery for this stream: a
                    // retry could duplicate text, thinking, or tool arguments
                    // (including fragments still buffered in the tag parser).
                    if (
                      delta.reasoning_content ||
                      delta.content ||
                      (Array.isArray(delta.tool_calls) && delta.tool_calls.length > 0)
                    ) {
                      sawModelOutput = true;
                    }
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
                        if (toolCalls?.lastToolState) refreshToolIndexes(output, toolCalls);
                      } else {
                        // A tool call may have been appended after the current
                        // text block, so appending to it again would place this
                        // text BEFORE the call in `content` order (the block is
                        // still a text block, just no longer the trailing one).
                        // Continue the cached block only while it is still last.
                        const trailing = output.content[contentBlockIndex];
                        const continuesTrailingBlock =
                          contentBlockIndex !== -1 &&
                          contentBlockIndex === output.content.length - 1 &&
                          trailing?.type === "text";
                        if (!continuesTrailingBlock) {
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
                      for (const tc of delta.tool_calls) {
                        const id = tc.id || "";
                        const name = tc.function?.name || "";
                        const argDelta = tc.function?.arguments ?? "";
                        if (typeof id !== "string" || typeof name !== "string") {
                          throw new Error("Tool-call identity conflict: invalid metadata");
                        }
                        if (typeof argDelta !== "string") throw new Error("Invalid tool-call argument fragment");
                        if (!id && !name && !argDelta) continue;

                        toolCalls ??= {
                          states: [],
                          byId: new Map(),
                          byIndex: new Map(),
                          unscopedCandidates: new Set(),
                          statesByBlock: new Map(),
                        };
                        const index = typeof tc.index === "number" ? tc.index : undefined;
                        let candidates: ToolCallCandidates | undefined;
                        if (index !== undefined) {
                          candidates = toolCalls.byIndex.get(index);
                          if (!candidates) {
                            candidates = { states: new Set(), complete: new Set(), unique: undefined };
                            toolCalls.byIndex.set(index, candidates);
                          }
                        }
                        let state = id ? toolCalls.byId.get(id) : undefined;
                        let associated = !!state && !!candidates && candidates.states.has(state);
                        if (id && candidates && !associated && candidates.complete.size > 0) {
                          retireCompleteCandidates(toolCalls, candidates);
                        }
                        const available = candidates?.states ?? toolCalls.unscopedCandidates;
                        if (!state) {
                          const only =
                            available.size === 1
                              ? candidates
                                ? candidates.unique
                                : available.values().next().value
                              : undefined;
                          if (!id && available.size > 1) {
                            throw new Error(`Ambiguous tool-call continuation (candidates=${available.size})`);
                          }
                          // An explicit new ID never overwrites another identity.
                          // Only a unique, still-unidentified provisional can bind.
                          if (only && (!id || !only.id)) {
                            state = only;
                            associated = candidates !== undefined;
                          } else {
                            state = {
                              argumentChunks: [],
                              argumentLength: 0,
                              depth: 0,
                              inString: false,
                              escaped: false,
                              complete: false,
                              id: "",
                              name: "",
                              contentIndex: -1,
                              candidates: new Set(),
                              unindexed: index === undefined,
                            };
                            toolCalls.states.push(state);
                          }
                        }
                        if (name && state.name && name !== state.name) {
                          throw new Error("Tool-call identity conflict: inconsistent name");
                        }
                        const newlyIdentified = !!id && !state.id;
                        const newlyNamed = !!name && !state.name;
                        if (newlyIdentified) {
                          state.id = id;
                          toolCalls.byId.set(id, state);
                        }
                        if (newlyNamed) state.name = name;
                        if (!associated) {
                          // Indexed continuations already have all three links.
                          // Unindexed explicit IDs can reactivate retired states.
                          toolCalls.unscopedCandidates.add(state);
                          if (candidates) {
                            state.unindexed = false;
                            state.candidates.add(candidates);
                            candidates.states.add(state);
                            candidates.unique = candidates.states.size === 1 ? state : undefined;
                            if (state.complete && state.id) candidates.complete.add(state);
                          }
                        }

                        const wasComplete = state.complete;
                        if (argDelta) appendToolArguments(state, argDelta);
                        if (state.complete && state.id && (!wasComplete || newlyIdentified)) {
                          // Each association is registered at most once on closure,
                          // not revisited on every fragment or repeated ID.
                          for (const scope of state.candidates) scope.complete.add(state);
                        }

                        if (!state.block && (state.id || state.name)) {
                          state.block = { type: "toolCall", id: state.id, name: state.name, arguments: {} };
                          state.contentIndex = output.content.length;
                          output.content.push(state.block);
                          toolCalls.statesByBlock.set(state.block, state);
                          toolCalls.lastToolState = state;
                          stream.push({ type: "toolcall_start", contentIndex: state.contentIndex, partial: output });
                          // Argument-only provisional fragments have no content
                          // position until this block exists. Replay them in order.
                          for (const fragment of state.argumentChunks) {
                            stream.push({
                              type: "toolcall_delta",
                              contentIndex: state.contentIndex,
                              delta: fragment,
                              partial: output,
                            });
                          }
                        } else if (state.block) {
                          if (newlyIdentified) state.block.id = state.id;
                          if (newlyNamed) state.block.name = state.name;
                          if (argDelta) {
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
        } finally {
          // Release on DONE, abort, and identity/argument failures alike.
          try {
            await reader.cancel().catch(() => {});
          } finally {
            reader.releaseLock();
          }
        }

        // The gateway can close the body without ever sending the [DONE]
        // sentinel. Treating that as a normal end reports a truncated reply as a
        // finished turn, so the agent loop sees `stop` and never retries. Fail
        // loudly instead — pi-ai classifies this as a retryable provider error.
        // fork: surface a truncated stream as a retryable error instead of a
        // silent stop; upstream has no such check.
        if (!sawDone) {
          throw new Error(
            `Qoder stream disconnected prematurely (connection closed before [DONE] after ${output.usage.output || 0} output tokens)`,
          );
        }

        if (thinkingParser) {
          thinkingParser.finalize();
          if (toolCalls?.lastToolState) refreshToolIndexes(output, toolCalls);
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

        if (toolCalls) {
          // Validate the whole batch before publishing any successful end event.
          for (const state of toolCalls.states) {
            if (!state.id || !state.name) {
              throw new Error(`Unresolved tool-call identity (characters=${state.argumentLength})`);
            }
            parseToolArguments(state);
          }
          // Map insertion order is the order calls first became identifiable,
          // independent of sparse/reused upstream indexes and provisional buffers.
          for (const [block, state] of toolCalls.statesByBlock) {
            block.arguments = parseToolArguments(state);
            stream.push({
              type: "toolcall_end",
              contentIndex: state.contentIndex,
              toolCall: block,
              partial: output,
            });
          }

          if (toolCalls.statesByBlock.size > 0 && output.stopReason !== "length") {
            output.stopReason = "toolUse";
          }
        }

        // Fallback: some upstreams (Qwen / DeepSeek XML tool mode) emit the call
        // as plain `<tool_call>` markup inside a text block instead of native
        // tool_calls deltas. Without this the markup is either shown to the user
        // verbatim or classed as a degenerate turn, and the tool never runs.
        // Only applies when no native call parsed — a real tool_calls stream is
        // authoritative.
        if (!toolCalls || toolCalls.statesByBlock.size === 0) {
          const recovered = recoverXmlToolCalls(output, stream);
          if (recovered > 0 && output.stopReason !== "length") {
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
      };

      for (let attempt = 0; ; attempt++) {
        try {
          await runAttempt();
          return;
        } catch (e) {
          if (!(e instanceof QoderModelBusyError)) throw e;
          if (e.sawModelOutput || output.content.length > 0) {
            // Keep this wording out of pi-ai's retryable pattern: retrying a
            // turn that already emitted content would duplicate it.
            throw new Error("Qoder model rejected the request after partial output (10605 RESOURCE_LIMIT)");
          }
          const requestedWaitMs = e.retryAfterMs ?? 0;
          if (attempt >= modelBusyRetries) {
            throw new QoderModelBusyError(
              `Qoder server error: model is busy (10605 RESOURCE_LIMIT, retryAfterSeconds=${Math.round(requestedWaitMs / 1000)}, attempts=${attempt + 1})`,
              requestedWaitMs,
            );
          }
          if (requestedWaitMs > modelBusyMaxWaitMs) {
            throw new QoderModelBusyError(
              `Qoder server error: model is busy (10605 RESOURCE_LIMIT, requested delay ${requestedWaitMs}ms exceeds local cap ${modelBusyMaxWaitMs}ms)`,
              requestedWaitMs,
            );
          }
          await sleepWithAbort(requestedWaitMs, options?.signal);
          // Covers an abort that lands between the wait and the next attempt.
          options?.signal?.throwIfAborted();
          // qodercli resets request_id and chat_record_id to ONE new value per
          // recovery attempt and keeps request_set_id, so the server can
          // correlate the retry with the same request set.
          const nextId = crypto.randomUUID();
          reqBody.request_id = nextId;
          reqBody.chat_record_id = nextId;
          encodedBytes = qoderEncodeBody(Buffer.from(JSON.stringify(reqBody)));
          headers = buildAuthHeaders(encodedBytes, chatURL, {
            userID,
            authToken: accessToken,
            name,
            email,
            machineID,
          });
        }
      }
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
