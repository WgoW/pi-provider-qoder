import { createHash, randomUUID } from "node:crypto";
import { mkdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import type { ThinkingLevel, ThinkingLevelMap } from "@earendil-works/pi-ai";
import { buildAuthHeaders } from "./cosy.js";
import { refreshGatewayCosyVersion } from "./cosy-version.js";
import { getQoderBaseUrl, getQoderModelListURL, getQoderRegionConfig, type QoderMode } from "./region.js";

export const ZERO_COST = Object.freeze({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });

/**
 * Maximum output tokens sent per request. Aliyun Model Studio (the upstream
 * behind Qoder's CN catalog) documents Max Output Length = 131072 for every
 * model we expose (qwen3.8-max/flash, qwen3.7-max/plus/flash), in both normal
 * and thinking modes (thinking chain alone goes up to 262144). The Qoder
 * /model/list catalog does not return a per-model output cap, so this single
 * constant is the source of truth for both static models and request sending.
 * qodercli ships a conservative 32e3 default and caps its UI at 65536; we use
 * the documented upstream ceiling so reasoning chains and long generations
 * are not truncated.
 */
export const MAX_OUTPUT_TOKENS = 131072;

/**
 * Fallback context window when the catalog omits `context_config`.
 *
 * Qoder's `/model/list` often ships `max_input_tokens` as a stale 180K floor
 * even for models that accept 1M-token prompts (verified against global `lite`
 * through 1,000K tokens). When `context_config` is present we use its largest
 * `token_count` instead, so models that truly advertise 200K/256K stay there.
 */
export const DEFAULT_CONTEXT_WINDOW = 1000000;

/** Shape of a single entry returned by the Qoder /model/list endpoint. */
export interface QoderModelEntry {
  key?: string;
  enable?: boolean;
  display_name?: string;
  max_input_tokens?: number;
  price_factor?: number;
  context_config?: Record<string, { token_count?: number; is_default?: boolean }>;
  is_vl?: boolean;
  is_reasoning?: boolean;
  thinking_config?: {
    disabled?: unknown;
    enabled?: { efforts?: Record<string, { is_default?: boolean }>; is_default?: boolean };
  };
  source?: string;
  [key: string]: unknown;
}

/**
 * omp fork's `ThinkingConfig` shape. The fork reads `model.thinking` (not the
 * `thinkingLevelMap` that earendil pi-ai exposes), and when it is absent omp
 * fills a generic default ladder (`minimal`..`high`) instead of the model's
 * real efforts - so `DeepSeek-V4-Pro` would show `low`/`medium` it does not
 * support. Declared inline because this plugin compiles against
 * @earendil-works/pi-ai, which has no such field.
 */
export interface QoderThinkingConfig {
  mode: "effort";
  efforts: readonly ThinkingLevel[];
  defaultLevel?: ThinkingLevel;
  requiresEffort?: boolean;
}

export interface QoderModelDef {
  id: string;
  upstreamKey?: string;
  name: string;
  priceFactor?: number;
  api: "qoder-api";
  provider: "qoder" | "qoder-cn";
  baseUrl: string;
  reasoning: boolean;
  supportsEffort: boolean;
  thinkingLevelMap?: ThinkingLevelMap;
  /** omp-only effort metadata; earendil pi-ai ignores it. */
  thinking?: QoderThinkingConfig;
  input: ("text" | "image")[];
  cost: typeof ZERO_COST;
  contextWindow: number;
  maxTokens: number;
  description?: string;
}

// fork: model price factor display (upstream has no price_factor handling).
export function getPriceFactor(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined;
}

export function addPriceFactorToName(name: string, priceFactor: number | undefined): string {
  if (priceFactor === undefined) return name;
  const suffix = `(${priceFactor}x)`;
  return name.endsWith(suffix) ? name : `${name} ${suffix}`;
}

export function withPriceFactor(model: QoderModelDef, priceFactor: number | undefined): QoderModelDef {
  return {
    ...model,
    name: addPriceFactorToName(model.name, priceFactor),
    ...(priceFactor === undefined ? {} : { priceFactor }),
  };
}

function getHomeDir(): string {
  // Prefer process.env.HOME so vitest setup can isolate caches. Node 26+ caches
  // os.homedir() from process start, ignoring later HOME changes.
  return process.env.HOME || process.env.USERPROFILE || homedir();
}

function getQoderCachePath(mode: QoderMode): string {
  return join(getHomeDir(), ".pi", "agent", getQoderRegionConfig(mode).modelCacheFile);
}

/**
 * On-disk catalogue format.
 *
 * v1 kept a single catalogue per region. That is wrong for Qoder: `/model/list`
 * is account-scoped, so a quota-exhausted (or free-plan) account answers with
 * its `is_free` models only. One shared slot let that degraded answer erase a
 * funded account's full catalogue, which collapsed the picker to Qwen-only.
 *
 * v2 keeps one slot per account and mirrors the merged catalogue at the top
 * level, so both the union view and v1-era readers stay valid.
 */
const CACHE_VERSION = 2;

/** Slot key for a pre-v2 snapshot whose owning account is unknown. */
// fork: per-account catalogue (upstream keeps a single catalogue per region).
// Everything from here through checkAccountEntitlement is fork-only: the account
// key, the per-account slots inside the cache document, and the entitlement
// verdict used to fail a misrouted pooled request before it costs minutes.
export const UNKNOWN_ACCOUNT_KEY = "__unknown__";

export interface QoderAccountCatalog {
  updatedAt: number;
  identity: { userID?: string; email?: string; name?: string };
  /**
   * True when the account looks quota-degraded (free-only catalogue that is
   * smaller than the last known-good one). Recovery from the account itself
   * clears the flag; a stored list kept by the downgrade guard stays flagged.
   */
  degraded?: boolean;
  models: QoderModelDef[];
  configs?: Record<string, QoderModelEntry>;
  /**
   * Model ids from the most recent successful `/model/list` response for this
   * account, kept separately from `models`.
   *
   * `models` may have been preserved from an earlier, richer response by the
   * quota-downgrade guard, so it must never be used to decide whether this
   * account is allowed to call a model — `servedModelIds` is the raw answer.
   */
  servedModelIds?: string[];
}

interface ParsedModelCache {
  version?: number;
  // v1 top-level view: mirror of the merged catalogue for older readers.
  updatedAt?: number;
  models?: QoderModelDef[];
  configs?: Record<string, QoderModelEntry>;
  /** v2 account slots — source of truth when present. */
  accounts?: Record<string, QoderAccountCatalog>;
}

interface CatalogSlot {
  key: string;
  data: QoderAccountCatalog;
}

/** In-memory cache keyed by absolute cache path (HOME-safe across tests). */
const modelCacheMem = new Map<string, { sig: string; data: ParsedModelCache }>();

/** Clear process-memory model caches (also used by tests that mutate cache files). */
export function clearQoderModelsMemCache(): void {
  modelCacheMem.clear();
}

/**
 * Identify an account slot. Preferred key is the stable Qoder userID, with an
 * email digest as fallback, so a rotated access token never creates a new slot.
 */
export function qoderAccountKey(identity: { userID?: string; email?: string }): string {
  const userID = identity.userID?.trim();
  if (userID) return userID;
  const email = identity.email?.trim().toLowerCase();
  if (email) return `email:${createHash("sha1").update(email).digest("hex").slice(0, 12)}`;
  return UNKNOWN_ACCOUNT_KEY;
}

/** `mtime:size` signature so a write from another pane/process invalidates the memo. */
function cacheSignature(cachePath: string): string {
  try {
    const stat = statSync(cachePath);
    return `${stat.mtimeMs}:${stat.size}`;
  } catch {
    return "missing";
  }
}

/** Signature of a region's catalogue file; used to detect cross-process updates. */
export function qoderCatalogCacheSignature(mode: QoderMode): string {
  return cacheSignature(getQoderCachePath(mode));
}

function readParsedModelCache(mode: QoderMode): ParsedModelCache | null {
  const cachePath = getQoderCachePath(mode);
  const sig = cacheSignature(cachePath);
  const mem = modelCacheMem.get(cachePath);
  if (mem && mem.sig === sig) {
    return mem.data;
  }
  if (sig === "missing") {
    // File removed externally: keep serving the memoized catalogue (hot path),
    // and fall back to static models only when nothing was ever loaded.
    return mem?.data ?? null;
  }
  try {
    const raw = readFileSync(cachePath, "utf8");
    if (!raw.trim()) return mem?.data ?? null;
    const data = JSON.parse(raw) as ParsedModelCache;
    if (!data || typeof data !== "object") return mem?.data ?? null;
    const hasAccounts = !!data.accounts && typeof data.accounts === "object" && Object.keys(data.accounts).length > 0;
    const hasLegacy = Array.isArray(data.models) && data.models.length > 0;
    if (!hasAccounts && !hasLegacy) return null;
    modelCacheMem.set(cachePath, { sig, data });
    return data;
  } catch {
    return mem?.data ?? null;
  }
}

function writeParsedModelCache(mode: QoderMode, data: ParsedModelCache): void {
  const cachePath = getQoderCachePath(mode);
  mkdirSync(dirname(cachePath), { recursive: true });
  const temporaryPath = `${cachePath}.${process.pid}.${randomUUID()}.tmp`;
  try {
    writeFileSync(temporaryPath, JSON.stringify(data, null, 2), "utf-8");
    renameSync(temporaryPath, cachePath);
  } finally {
    rmSync(temporaryPath, { force: true });
  }
  modelCacheMem.set(cachePath, { sig: cacheSignature(cachePath), data });
}

/**
 * Derive the only public model id from Qoder's display name.
 * The upstream key remains available solely inside the matching config entry.
 */
export function toQoderModelId(displayName?: string): string {
  return (displayName || "QoderModel").replace(/\s+/g, "");
}

export const staticModels: QoderModelDef[] = [
  {
    id: "Auto",
    upstreamKey: "auto",
    name: "Auto",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: true,
    supportsEffort: false,
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
  {
    id: "Ultimate",
    upstreamKey: "ultimate",
    name: "Ultimate",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: true,
    supportsEffort: true,
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
  {
    id: "Performance",
    upstreamKey: "performance",
    name: "Performance",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: true,
    supportsEffort: true,
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
  {
    id: "Efficient",
    upstreamKey: "efficient",
    name: "Efficient",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: false,
    supportsEffort: false,
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
  {
    id: "Lite",
    upstreamKey: "lite",
    name: "Lite",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: false,
    supportsEffort: false,
    input: ["text"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
  {
    id: "Qwen3.7Plus",
    upstreamKey: "qmodel",
    name: "Qwen3.7 Plus",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: false,
    supportsEffort: false,
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
  {
    id: "Cantus",
    upstreamKey: "cmodel",
    name: "Cantus",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: true,
    supportsEffort: true,
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
  {
    id: "Qwen3.8-Max",
    upstreamKey: "qmodel_preview",
    name: "Qwen3.8-Max",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: true,
    supportsEffort: true,
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
  {
    id: "Qwen3.8-Flash",
    upstreamKey: "qfmodel",
    name: "Qwen3.8-Flash",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: true,
    supportsEffort: true,
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
  {
    id: "GLM-5.3",
    upstreamKey: "gmodel",
    name: "GLM-5.3",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: true,
    supportsEffort: true,
    thinkingLevelMap: {
      off: null,
      minimal: null,
      low: "low",
      medium: null,
      high: "high",
      xhigh: null,
      max: "max",
    },
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
  {
    id: "GLM-5.3-Flash",
    upstreamKey: "gfmodel",
    name: "GLM-5.3-Flash",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: true,
    supportsEffort: true,
    thinkingLevelMap: {
      off: null,
      minimal: null,
      low: null,
      medium: null,
      high: "high",
      xhigh: null,
      max: "max",
    },
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
  {
    id: "Qwen3.7-Max",
    upstreamKey: "qmodel_latest",
    name: "Qwen3.7-Max",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: false,
    supportsEffort: false,
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
  {
    id: "DeepSeek-V4-Pro",
    upstreamKey: "dmodel",
    name: "DeepSeek-V4-Pro",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: true,
    supportsEffort: true,
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
  {
    id: "DeepSeek-V4-Flash",
    upstreamKey: "dfmodel",
    name: "DeepSeek-V4-Flash",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: true,
    supportsEffort: true,
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
  {
    id: "GLM-5.2",
    upstreamKey: "gm51model",
    name: "GLM-5.2",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: true,
    supportsEffort: true,
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
  {
    id: "Kimi-K2.7-Code",
    upstreamKey: "kmodel",
    name: "Kimi-K2.7-Code",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: false,
    supportsEffort: false,
    input: ["text", "image"],
    cost: ZERO_COST,
    // Catalog advertises 256K; not included in the 1M live test in issue #13.
    contextWindow: 256000,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
  {
    id: "Kimi-K3",
    upstreamKey: "kmodel_latest",
    name: "Kimi-K3",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: false,
    supportsEffort: false,
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
  {
    id: "MiniMax-M3",
    upstreamKey: "mmodel",
    name: "MiniMax-M3",
    api: "qoder-api",
    provider: "qoder",
    baseUrl: getQoderBaseUrl("global"),
    reasoning: false,
    supportsEffort: false,
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
  },
];

export const staticCnModels: QoderModelDef[] = [
  {
    id: "Auto",
    upstreamKey: "auto",
    name: "Auto",
    api: "qoder-api",
    provider: "qoder-cn",
    baseUrl: getQoderBaseUrl("cn"),
    reasoning: true,
    supportsEffort: false,
    input: ["text", "image"],
    cost: ZERO_COST,
    // CN Auto has not been live-tested at 1M; keep the conservative 200K
    // fallback until the CN catalog advertises a larger option.
    contextWindow: 200000,
    maxTokens: MAX_OUTPUT_TOKENS,
    description: "Qoder CN smart routing; fallback context window of 200K.",
  },
  {
    id: "Qwen3.8-Max",
    upstreamKey: "qmodel_38max",
    name: "Qwen3.8-Max",
    api: "qoder-api",
    provider: "qoder-cn",
    baseUrl: getQoderBaseUrl("cn"),
    reasoning: true,
    supportsEffort: true,
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
    description: "Qoder CN qmodel_38max; context options 200K/400K/1M.",
  },
  {
    id: "Qwen3.8-Flash",
    upstreamKey: "qfmodel",
    name: "Qwen3.8-Flash",
    api: "qoder-api",
    provider: "qoder-cn",
    baseUrl: getQoderBaseUrl("cn"),
    reasoning: true,
    supportsEffort: true,
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
    description: "Qoder CN qfmodel; context options 200K/400K/1M.",
  },
  {
    id: "GLM-5.3",
    upstreamKey: "gmodel",
    name: "GLM-5.3",
    api: "qoder-api",
    provider: "qoder-cn",
    baseUrl: getQoderBaseUrl("cn"),
    reasoning: true,
    supportsEffort: true,
    thinkingLevelMap: {
      off: null,
      minimal: null,
      low: "low",
      medium: null,
      high: "high",
      xhigh: null,
      max: "max",
    },
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
    description: "Qoder CN gmodel; context options 200K/400K/1M.",
  },
  {
    id: "GLM-5.3-Flash",
    upstreamKey: "gfmodel",
    name: "GLM-5.3-Flash",
    api: "qoder-api",
    provider: "qoder-cn",
    baseUrl: getQoderBaseUrl("cn"),
    reasoning: true,
    supportsEffort: true,
    thinkingLevelMap: {
      off: null,
      minimal: null,
      low: null,
      medium: null,
      high: "high",
      xhigh: null,
      max: "max",
    },
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
    description: "Qoder CN gfmodel; always-on reasoning, efforts high/max.",
  },
  {
    id: "Qwen3.7-Max",
    upstreamKey: "qmodel_latest",
    name: "Qwen3.7-Max",
    api: "qoder-api",
    provider: "qoder-cn",
    baseUrl: getQoderBaseUrl("cn"),
    reasoning: true,
    supportsEffort: false,
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
    description: "Qoder CN qmodel_latest; context options 200K/400K/1M.",
  },
  {
    id: "Qwen3.7-Plus",
    upstreamKey: "qmodel",
    name: "Qwen3.7-Plus",
    api: "qoder-api",
    provider: "qoder-cn",
    baseUrl: getQoderBaseUrl("cn"),
    reasoning: true,
    supportsEffort: false,
    input: ["text"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
    description: "Qoder CN qmodel; context options 200K/400K/1M.",
  },
  {
    id: "Qwen3.6-Flash",
    upstreamKey: "q36fmodel",
    name: "Qwen3.6-Flash",
    api: "qoder-api",
    provider: "qoder-cn",
    baseUrl: getQoderBaseUrl("cn"),
    reasoning: true,
    supportsEffort: false,
    input: ["text"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
    description: "Qoder CN q36fmodel; context options 200K/400K/1M.",
  },
  {
    id: "DeepSeek-V4-Pro",
    upstreamKey: "dmodel",
    name: "DeepSeek-V4-Pro",
    api: "qoder-api",
    provider: "qoder-cn",
    baseUrl: getQoderBaseUrl("cn"),
    reasoning: true,
    supportsEffort: false,
    input: ["text"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
    description: "Qoder CN dmodel; context options 200K/400K/1M.",
  },
  {
    id: "DeepSeek-V4-Flash",
    upstreamKey: "dfmodel",
    name: "DeepSeek-V4-Flash",
    api: "qoder-api",
    provider: "qoder-cn",
    baseUrl: getQoderBaseUrl("cn"),
    reasoning: false,
    supportsEffort: false,
    input: ["text"],
    cost: ZERO_COST,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: MAX_OUTPUT_TOKENS,
    description: "Qoder CN dfmodel; context options 200K/400K/1M.",
  },
  {
    id: "GLM-5.2",
    upstreamKey: "gm51model",
    name: "GLM-5.2",
    api: "qoder-api",
    provider: "qoder-cn",
    baseUrl: getQoderBaseUrl("cn"),
    reasoning: true,
    supportsEffort: false,
    input: ["text", "image"],
    cost: ZERO_COST,
    // Live CN catalog currently displays 200K; do not copy global gm51model's 1M.
    contextWindow: 200000,
    maxTokens: MAX_OUTPUT_TOKENS,
    description: "Qoder CN gm51model; live catalog currently displays GLM-5.2 with 200K context.",
  },
  {
    id: "Kimi-K2.7-Code",
    upstreamKey: "kmodel",
    name: "Kimi-K2.7-Code",
    api: "qoder-api",
    provider: "qoder-cn",
    baseUrl: getQoderBaseUrl("cn"),
    reasoning: true,
    supportsEffort: false,
    input: ["text", "image"],
    cost: ZERO_COST,
    // Catalog advertises 256K; same as global kmodel.
    contextWindow: 256000,
    maxTokens: MAX_OUTPUT_TOKENS,
    description: "Qoder CN kmodel; context option 256K.",
  },
  {
    id: "MiniMax-M2.7",
    upstreamKey: "mmodel",
    name: "MiniMax-M2.7",
    api: "qoder-api",
    provider: "qoder-cn",
    baseUrl: getQoderBaseUrl("cn"),
    reasoning: false,
    supportsEffort: false,
    input: ["text"],
    cost: ZERO_COST,
    // Live CN catalog reports 200K; not confirmed at 1M.
    contextWindow: 200000,
    maxTokens: MAX_OUTPUT_TOKENS,
    description: "Qoder CN mmodel; live catalog reports 200K context.",
  },
];

/** pi thinking levels in display order (matches the pi-ai SDK this build targets). */
const PI_THINKING_LEVELS: readonly ThinkingLevel[] = ["minimal", "low", "medium", "high", "xhigh", "max"];

/**
 * Map Qoder's `thinking_config` to pi's `thinkingLevelMap` so the TUI exposes
 * the levels the upstream model actually supports.
 *
 * Qoder has two shapes:
 *   - effort-based: `thinking_config.enabled.efforts = { low, medium, xhigh, ... }`
 *     Each effort key is already a pi level name, so supported levels map to
 *     themselves and the rest are pinned to null (hidden in the picker).
 *     `xhigh`/`max` are only shown when the map carries them, otherwise the
 *     picker tops out at `high`.
 *   - toggle-based: `thinking_config.enabled` without `efforts` (only on/off).
 *     Every pi level is exposed and maps to "enabled" so a user picking any
 *     level turns thinking on; the exact effort sent upstream is decided at
 *     request time.
 * Returns undefined for models that do not support thinking, so pi falls back
 * to `reasoning: false`-style behavior (only `off`).
 */
export function buildThinkingLevelMap(entry: QoderModelEntry): ThinkingLevelMap | undefined {
  const tc = entry.thinking_config;
  if (!tc) return undefined;
  const efforts = tc.enabled?.efforts;
  if (efforts && typeof efforts === "object") {
    const supported = new Set(Object.keys(efforts));
    // `off` (disable thinking) is selectable when the catalog advertises a
    // `disabled` option; otherwise pin it to null to hide it.
    const map: ThinkingLevelMap = { off: tc.disabled ? "disabled" : null };
    for (const level of PI_THINKING_LEVELS) {
      map[level] = supported.has(level) ? level : null;
    }
    return map;
  }
  // toggle-only (enabled/disabled, no efforts) — expose every level as "on".
  // `off` is selectable when the catalog advertises `disabled`.
  if (tc.enabled) {
    const map: ThinkingLevelMap = { off: tc.disabled ? "disabled" : null };
    for (const level of PI_THINKING_LEVELS) {
      map[level] = "enabled";
    }
    return map;
  }
  return undefined;
}

/**
 * Map Qoder's `thinking_config` to omp's `thinking` metadata so omp's effort
 * picker lists the actual supported levels (it reads `thinking.efforts`, not
 * `thinkingLevelMap`). Only effort-based models have a meaningful ladder;
 * toggle-only models (no `efforts`) return undefined so omp falls back to its
 * generic behavior. earendil pi ignores this field entirely.
 */
export function buildThinkingConfig(entry: QoderModelEntry): QoderThinkingConfig | undefined {
  const tc = entry.thinking_config;
  const efforts = tc?.enabled?.efforts;
  if (!efforts || typeof efforts !== "object") return undefined;

  const supported = new Set(Object.keys(efforts));
  const ordered = PI_THINKING_LEVELS.filter((level) => supported.has(level));
  if (ordered.length === 0) return undefined;

  const defaultLevel = PI_THINKING_LEVELS.find((level) => efforts[level]?.is_default);

  return {
    mode: "effort",
    efforts: ordered,
    ...(defaultLevel ? { defaultLevel } : {}),
    // Qoder's catalog marks a `disabled` branch when thinking can be turned
    // off. Without it reasoning is mandatory upstream, so omp clamps "off" to
    // the lowest supported effort instead of disabling.
    requiresEffort: !tc?.disabled,
  };
}

/**
 * Whether the upstream refuses to run a model with thinking disabled.
 *
 * Qoder's catalogue shape cannot express this: an always-on model such as
 * GLM-5.3-Flash and a merely effort-limited model such as Sonus both ship
 * `thinking_config.enabled.efforts` with no `disabled` branch. Only the curated
 * seed table records which is which, by pinning `off` to null. Sending
 * `enable_thinking: false` to an always-on model fails upstream with 1210, while
 * an effort-limited model accepts it (it just rejects `reasoning_effort: "none"`).
 *
 * Exported for the request path, which needs the distinction to pick between
 * "keep thinking on with the default effort" and "disable, without an effort
 * signal".
 */
// fork: `off: null` marks a model the upstream refuses to run without
// reasoning. Upstream infers this from the catalogue shape instead, which cannot
// distinguish an always-on model (GLM-5.3-Flash, error 1210) from a merely
// effort-limited one (Sonus, rejects reasoning_effort:"none").
export function isAlwaysThinkingModel(modelId: string, mode: QoderMode): boolean {
  const seed = (mode === "cn" ? staticCnModels : staticModels).find((model) => model.id === modelId);
  return seed?.reasoning === true && seed.thinkingLevelMap?.off === null;
}

/** Look up a catalogue entry by public model id, tolerating v1 raw-key shapes. */
function findConfigEntry(
  configs: Record<string, QoderModelEntry> | undefined,
  modelId: string,
): QoderModelEntry | undefined {
  if (!configs) return undefined;
  const direct = configs[modelId];
  if (direct) return direct;
  return Object.values(configs).find(
    (entry) =>
      entry && typeof entry === "object" && toQoderModelId((entry as QoderModelEntry).display_name) === modelId,
  ) as QoderModelEntry | undefined;
}

/** Map a stored model record onto the public picker shape. */
function normalizeCachedModel(
  model: QoderModelDef,
  configs: Record<string, QoderModelEntry> | undefined,
  mode: QoderMode,
): QoderModelDef {
  const config = findConfigEntry(configs, model.id);
  const display = config?.display_name;
  const staticModel = (mode === "cn" ? staticCnModels : staticModels).find((seed) => seed.upstreamKey === model.id);
  const thinkingLevelMap = model.thinkingLevelMap ?? (config ? buildThinkingLevelMap(config) : undefined);
  const thinking = model.thinking ?? (config ? buildThinkingConfig(config) : undefined);
  const baseModel = display
    ? { ...model, id: toQoderModelId(display), name: display, thinkingLevelMap, thinking }
    : staticModel
      ? { ...model, id: staticModel.id, name: staticModel.name, thinkingLevelMap, thinking }
      : model.name
        ? { ...model, id: toQoderModelId(model.name), thinkingLevelMap, thinking }
        : { ...model, thinkingLevelMap, thinking };
  const priceFactor = model.priceFactor ?? getPriceFactor(config?.price_factor);
  return withPriceFactor(baseModel, priceFactor);
}

/** Higher is richer; used to pick between two records sharing a public id. */
function modelRichness(model: QoderModelDef): number {
  return (model.reasoning ? 4 : 0) + (model.thinkingLevelMap ? 2 : 0) + (model.input.includes("image") ? 1 : 0);
}

function mergeCatalogModels(a: QoderModelDef, b: QoderModelDef): QoderModelDef {
  const winner = modelRichness(b) > modelRichness(a) ? b : a;
  return withPriceFactor(winner, winner.priceFactor ?? a.priceFactor ?? b.priceFactor);
}

/**
 * Read every stored slot. Degraded slots sort last so a healthy account's
 * definition, ordering and price factor win when they advertise the same model.
 */
function collectCatalogSlots(data: ParsedModelCache): CatalogSlot[] {
  const accounts = data.accounts;
  if (accounts && typeof accounts === "object") {
    const slots = Object.entries(accounts)
      .map(([key, value]) => ({ key, data: value as QoderAccountCatalog }))
      .filter((slot) => slot.data && Array.isArray(slot.data.models) && slot.data.models.length > 0);
    if (slots.length > 0) return sortCatalogSlots(slots);
  }
  // Pre-v2 snapshot: attribute it to an unknown account so it still contributes.
  if (Array.isArray(data.models) && data.models.length > 0) {
    return [
      {
        key: UNKNOWN_ACCOUNT_KEY,
        data: { updatedAt: data.updatedAt ?? 0, identity: {}, models: data.models, configs: data.configs },
      },
    ];
  }
  return [];
}

function sortCatalogSlots(slots: CatalogSlot[]): CatalogSlot[] {
  return [...slots].sort((a, b) => {
    const degraded = Number(!!a.data.degraded) - Number(!!b.data.degraded);
    if (degraded !== 0) return degraded;
    return (b.data.updatedAt ?? 0) - (a.data.updatedAt ?? 0);
  });
}

/**
 * Union the account slots into one picker catalogue.
 *
 * The union is the whole point of v2: a quota-exhausted account contributes
 * only its free models, and they can no longer shrink another account's list.
 */
function mergeCatalogSlots(
  mode: QoderMode,
  slots: CatalogSlot[],
): { models: QoderModelDef[]; configs: Record<string, QoderModelEntry> } {
  const byId = new Map<string, QoderModelDef>();
  const configs: Record<string, QoderModelEntry> = {};

  for (const slot of slots) {
    for (const raw of slot.data.models) {
      const model = normalizeCachedModel(raw, slot.data.configs, mode);
      const existing = byId.get(model.id);
      if (existing) {
        byId.set(model.id, mergeCatalogModels(existing, model));
        continue;
      }
      byId.set(model.id, model);
      const entry = findConfigEntry(slot.data.configs, raw.id) ?? findConfigEntry(slot.data.configs, model.id);
      if (entry) configs[model.id] = entry;
    }
  }

  // Older releases injected `auto` without a corresponding service config.
  // Keep an explicitly enabled service model, but drop the legacy fallback.
  const hasConfigsMap = slots.some((slot) => slot.data.configs && typeof slot.data.configs === "object");
  // Detect a real service `auto` entry (its upstream key), not the legacy
  // injected placeholder: configs are keyed by display name, so a live `Auto`
  // model appears as `Auto`, never as `auto`.
  const hasAuto = slots.some((slot) =>
    Object.values(slot.data.configs ?? {}).some((entry) => (entry as QoderModelEntry)?.key === "auto"),
  );
  const merged = [...byId.values()];
  const models = hasConfigsMap && !hasAuto ? merged.filter((model) => model.id.toLowerCase() !== "auto") : merged;
  return { models, configs };
}

function staticModelsFor(mode: QoderMode): QoderModelDef[] {
  return mode === "cn" ? staticCnModels : staticModels;
}

function buildCacheDocument(mode: QoderMode, accounts: Record<string, QoderAccountCatalog>): ParsedModelCache {
  const slots = sortCatalogSlots(Object.entries(accounts).map(([key, data]) => ({ key, data })));
  const { models, configs } = mergeCatalogSlots(mode, slots);
  const updatedAt = Object.values(accounts).reduce((max, slot) => Math.max(max, slot.updatedAt ?? 0), 0);
  return { version: CACHE_VERSION, updatedAt, models, configs, accounts };
}

export function getCachedModels(mode: QoderMode): QoderModelDef[] {
  const data = readParsedModelCache(mode);
  const slots = data ? collectCatalogSlots(data) : [];
  if (slots.length === 0) return staticModelsFor(mode);
  const { models } = mergeCatalogSlots(mode, slots);
  return models.length > 0 ? models : staticModelsFor(mode);
}

export function getCachedModelConfig(modelId: string, mode: QoderMode): QoderModelEntry | null {
  const data = readParsedModelCache(mode);
  if (data) {
    // Healthy accounts first, so a degraded slot cannot override their entry.
    for (const slot of collectCatalogSlots(data)) {
      const direct = slot.data.configs?.[modelId] as QoderModelEntry | undefined;
      if (direct && toQoderModelId(direct.display_name) === modelId) {
        return withMaxContextAsDefault(direct);
      }

      // Read old cache shapes without preserving their raw-key aliases.
      const legacyEntry = Object.values(slot.data.configs || {}).find(
        (entry) =>
          entry && typeof entry === "object" && toQoderModelId((entry as QoderModelEntry).display_name) === modelId,
      ) as QoderModelEntry | undefined;
      if (legacyEntry) {
        return withMaxContextAsDefault(legacyEntry);
      }
    }
  }

  const staticModel = (mode === "cn" ? staticCnModels : staticModels).find((model) => model.id === modelId);
  if (staticModel) {
    const thinkingConfig = staticModel.thinkingLevelMap
      ? {
          enabled: {
            efforts: Object.fromEntries(
              Object.entries(staticModel.thinkingLevelMap)
                .filter(([k, v]) => k !== "off" && typeof v === "string")
                .map(([_, v]) => [v, { is_default: v === "max" || v === "high" }]),
            ),
            is_default: true,
          },
          disabled: staticModel.thinkingLevelMap.off !== null ? {} : undefined,
        }
      : staticModel.reasoning
        ? {
            enabled: { is_default: true },
            // A static seed with `reasoning: true` but no explicit level map only
            // tells us the model reasons, not that thinking is mandatory. Keep
            // `disabled` present so "off" stays selectable: without it
            // buildThinkingLevelMap pins `off` to null and the request path would
            // read the entry as an always-on model it cannot disable.
            disabled: {},
          }
        : undefined;

    return {
      key: staticModel.upstreamKey || modelId,
      is_reasoning: staticModel.reasoning,
      source: "system",
      thinking_config: thinkingConfig,
    };
  }

  return null;
}

/**
 * Resolve model config with fallback for raw upstream keys or legacy aliases
 * (e.g. gfmodel -> GLM-5.3-Flash, dfmodel -> DeepSeek-V4-Flash).
 */
export function resolveModelConfig(modelId: string, mode: QoderMode): QoderModelEntry | null {
  const direct = getCachedModelConfig(modelId, mode);
  if (direct) return direct;

  const staticModel = (mode === "cn" ? staticCnModels : staticModels).find((model) => model.upstreamKey === modelId);
  if (staticModel) {
    return getCachedModelConfig(staticModel.id, mode);
  }

  return null;
}

/** Resolve contextWindow from a catalog entry. Exported for tests. */
export function contextWindowFromCatalog(entry: QoderModelEntry): number {
  const contextConfig = entry.context_config;
  if (contextConfig && typeof contextConfig === "object") {
    let advertised = 0;
    for (const configVal of Object.values(contextConfig)) {
      if (configVal && typeof configVal === "object" && typeof configVal.token_count === "number") {
        if (configVal.token_count > advertised) advertised = configVal.token_count;
      }
    }
    if (advertised > 0) return advertised;
  }
  return DEFAULT_CONTEXT_WINDOW;
}

/** Prefer the largest context option when Qoder exposes selectable contexts. */
function withMaxContextAsDefault(entry: QoderModelEntry): QoderModelEntry {
  const contextConfig = entry.context_config;
  if (!contextConfig || typeof contextConfig !== "object") return entry;

  const maxTokenCount = Math.max(
    ...Object.values(contextConfig).map((config) => (typeof config?.token_count === "number" ? config.token_count : 0)),
  );
  if (maxTokenCount <= 0) return entry;

  return {
    ...entry,
    context_config: Object.fromEntries(
      Object.entries(contextConfig).map(([name, config]) => [
        name,
        { ...config, is_default: config.token_count === maxTokenCount },
      ]),
    ),
  };
}

export function isCacheStale(mode: QoderMode): boolean {
  const data = readParsedModelCache(mode);
  if (!data) return true;
  const slots = collectCatalogSlots(data);
  if (slots.length === 0) return true;
  return slots.some((slot) => isSlotStale(slot.data));
}

function isSlotStale(slot: QoderAccountCatalog | undefined): boolean {
  if (!slot || typeof slot.updatedAt !== "number") return true;
  // Stale if older than 1 hour
  return Date.now() - slot.updatedAt > 3600_000;
}

/**
 * Per-account staleness. One degraded account must not force every other
 * account to re-fetch on every session start.
 */
export function isAccountCatalogStale(mode: QoderMode, accountKey: string): boolean {
  const data = readParsedModelCache(mode);
  return isSlotStale(data?.accounts?.[accountKey]);
}

/** True when any catalogue (v1 or v2) is already on disk. */
export function hasCachedCatalog(mode: QoderMode): boolean {
  const data = readParsedModelCache(mode);
  return !!data && collectCatalogSlots(data).length > 0;
}

/** Account keys currently stored for a region (used for diagnostics/tests). */
export function getCachedCatalogAccounts(mode: QoderMode): string[] {
  const data = readParsedModelCache(mode);
  if (!data?.accounts || typeof data.accounts !== "object") return [];
  return Object.keys(data.accounts);
}

/**
 * A free-only answer from an account that previously returned more models is
 * Qoder's quota-exhaustion signal (`isQuotaExceeded`), not a real catalogue
 * change. Detecting it lets the slot keep its last known-good list.
 */
function isFreeOnlyCatalog(configs: Record<string, QoderModelEntry>): boolean {
  const entries = Object.values(configs);
  if (entries.length === 0) return false;
  return entries.every((entry) => (entry as { is_free?: boolean })?.is_free === true);
}

/** How long an account's own catalogue is trusted for entitlement decisions. */
const ENTITLEMENT_TRUST_WINDOW_MS = 3600_000;

/** Account-scoped view of what a credential can serve, used before dispatch. */
export interface QoderAccountServedModels {
  key: string;
  modelIds: string[];
  updatedAt: number;
  /** True when the account's own catalogue is fresh enough to trust. */
  fresh: boolean;
}

/**
 * Model ids the account's credential can serve, from its own cached catalogue.
 * Returns null when we know nothing about the account (callers fail open).
 */
export function getAccountServedModels(mode: QoderMode, userID: string): QoderAccountServedModels | null {
  const trimmed = userID?.trim();
  if (!trimmed) return null;
  const data = readParsedModelCache(mode);
  const key = qoderAccountKey({ userID: trimmed });
  const slot = data?.accounts?.[key];
  if (!slot) return null;
  const updatedAt = typeof slot.updatedAt === "number" ? slot.updatedAt : 0;
  return {
    key,
    modelIds: [...(slot.servedModelIds ?? slot.models.map((model) => model.id))],
    updatedAt,
    fresh: updatedAt > 0 && Date.now() - updatedAt <= ENTITLEMENT_TRUST_WINDOW_MS,
  };
}

export type QoderEntitlementVerdict =
  | { checked: false }
  | { checked: true; served: boolean; modelIds: readonly string[] };

/**
 * Decide whether an account may call a model *before* spending a request on it.
 *
 * Qoder answers a model the account is not entitled to with `403 code 112`
 * only after holding the SSE connection open for ~3 minutes, so one misrouted
 * request costs minutes. The per-account catalogue we already cache matches
 * that entitlement exactly (verified live for plan-restricted models), so use
 * it to fail in milliseconds instead.
 *
 * The veto is deliberately narrow:
 *   - unknown, blank or stale account catalogue -> `checked: false` (fail open);
 *   - the model is listed by the account          -> served;
 *   - nobody in the region lists the model        -> `checked: false`, because
 *     that is more likely a brand-new catalogue entry than a restriction.
 */
export function checkAccountEntitlement(mode: QoderMode, userID: string, modelId: string): QoderEntitlementVerdict {
  const served = getAccountServedModels(mode, userID);
  if (!served?.fresh) return { checked: false };
  const wanted = toQoderModelId(modelId).toLowerCase();
  const matches = (id: string): boolean => toQoderModelId(id).toLowerCase() === wanted;
  if (served.modelIds.some(matches)) return { checked: true, served: true, modelIds: served.modelIds };
  if (!getRegionServedModelIds(mode, served.key).some(matches)) return { checked: false };
  return { checked: true, served: false, modelIds: served.modelIds };
}

/**
 * Model ids served by at least one *fresh* account catalogue in this region.
 * Used to tell a plan restriction apart from a brand-new model.
 */
export function getRegionServedModelIds(mode: QoderMode, excludeKey?: string): string[] {
  const data = readParsedModelCache(mode);
  if (!data?.accounts) return [];
  const seen = new Set<string>();
  for (const [key, slot] of Object.entries(data.accounts)) {
    if (excludeKey !== undefined && key === excludeKey) continue;
    const updatedAt = typeof slot.updatedAt === "number" ? slot.updatedAt : 0;
    if (updatedAt === 0 || Date.now() - updatedAt > ENTITLEMENT_TRUST_WINDOW_MS) continue;
    for (const id of slot.servedModelIds ?? slot.models.map((model) => model.id)) seen.add(id);
  }
  return [...seen];
}

export async function updateQoderModelsCache(
  authToken: string,
  userID: string,
  name: string,
  email: string,
  mode: QoderMode,
): Promise<boolean> {
  const modelListURL = getQoderModelListURL(mode);
  try {
    // Refresh the dynamic identity before signing the account-scoped catalog
    // request, otherwise Qoder may return and cache a reduced model list.
    if (!process.env.VITEST) await refreshGatewayCosyVersion();
    const headers = buildAuthHeaders(null, modelListURL, {
      userID,
      authToken,
      name,
      email,
    });

    const response = await fetch(modelListURL, {
      method: "GET",
      headers: {
        Accept: "application/json",
        ...headers,
      },
      signal: AbortSignal.timeout(8000),
    });

    if (!response.ok) {
      return false;
    }

    const resData = (await response.json()) as { chat?: QoderModelEntry[] };
    const chatModels = resData.chat || [];
    if (chatModels.length === 0) return false;

    const newModels: QoderModelDef[] = [];
    const configs: Record<string, QoderModelEntry> = {};

    for (const entry of chatModels) {
      const key = entry.key;
      if (!key || !entry.enable || !entry.display_name) continue;

      const display = entry.display_name;
      const priceFactor = getPriceFactor(entry.price_factor);
      // Prefer the largest selectable context option the catalog advertises
      // (e.g. 1M when 200K/400K/1M are offered). If none is advertised, use
      // DEFAULT_CONTEXT_WINDOW rather than the stale 180K `max_input_tokens`
      // floor. Do not seed from DEFAULT_CONTEXT_WINDOW before scanning
      // context_config: that would inflate models that only advertise 200K.
      const ctxLen = contextWindowFromCatalog(entry);
      const isVL = !!entry.is_vl;
      const isReasoning = !!entry.is_reasoning || !!entry.thinking_config;
      const supportsEffort = !!entry.thinking_config?.enabled?.efforts;
      const thinkingLevelMap = buildThinkingLevelMap(entry);
      // Both regions expose display_name (whitespace-stripped) as the sole
      // pi-visible id. The config stores the upstream key under that id for
      // request-time use.
      const modelInfo = { id: toQoderModelId(display), name: display };

      configs[modelInfo.id] = entry;

      newModels.push({
        id: modelInfo.id,
        name: addPriceFactorToName(modelInfo.name, priceFactor),
        priceFactor,
        api: "qoder-api",
        provider: getQoderRegionConfig(mode).providerID,
        baseUrl: getQoderBaseUrl(mode),
        reasoning: isReasoning,
        supportsEffort,
        thinkingLevelMap,
        input: isVL ? ["text", "image"] : ["text"],
        cost: ZERO_COST,
        contextWindow: ctxLen,
        maxTokens: MAX_OUTPUT_TOKENS,
      });
    }

    if (newModels.length === 0) return false;

    const key = qoderAccountKey({ userID, email });
    const previous = readParsedModelCache(mode);
    const previousAccounts = previous?.accounts ?? {};
    const previousSlot = previousAccounts[key];
    // A pre-v2 snapshot is un-attributed: let the first refreshed account
    // inherit it so the upgrade cannot lose models that are still valid.
    const inheritedSlot =
      previousSlot ??
      (previous ? collectCatalogSlots(previous).find((slot) => slot.key === UNKNOWN_ACCOUNT_KEY)?.data : undefined);

    const freeOnly = isFreeOnlyCatalog(configs);
    const shrank = (inheritedSlot?.models.length ?? 0) > newModels.length;
    const quotaDegraded = freeOnly && shrank;
    const degraded = freeOnly && (quotaDegraded || previousSlot?.degraded === true);

    const accounts: Record<string, QoderAccountCatalog> = { ...previousAccounts };
    accounts[key] = {
      updatedAt: Date.now(),
      identity: { userID, email, name },
      ...(degraded ? { degraded: true } : {}),
      // Keep the last known-good list for a quota-exhausted account instead of
      // erasing models its plan still lists once the quota resets.
      models: quotaDegraded && inheritedSlot ? inheritedSlot.models : newModels,
      configs: quotaDegraded && inheritedSlot?.configs ? { ...inheritedSlot.configs, ...configs } : configs,
      // Routing signal: what this account's credential can serve right now.
      servedModelIds: newModels.map((model) => model.id),
    };
    delete accounts[UNKNOWN_ACCOUNT_KEY];

    writeParsedModelCache(mode, buildCacheDocument(mode, accounts));

    if (quotaDegraded) {
      console.error(
        `[pi-provider-qoder] ${mode} account ${email || userID} returned a free-only catalog ` +
          `(${newModels.length} of ${inheritedSlot?.models.length ?? 0} models); keeping the last known-good list.`,
      );
    }
    return true;
  } catch {
    return false;
  }
}
