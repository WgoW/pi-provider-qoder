import { existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  checkAccountEntitlement,
  clearQoderModelsMemCache,
  getAccountServedModels,
  getCachedCatalogAccounts,
  getCachedModelConfig,
  getCachedModels,
  getRegionServedModelIds,
  hasCachedCatalog,
  isAccountCatalogStale,
  updateQoderModelsCache,
} from "../catalog.js";
import { loadLiveFixture, responseFromFixture } from "./live-fixture.js";

function testHome(): string {
  return process.env.HOME || process.env.USERPROFILE || homedir();
}

const CACHE_PATHS = {
  global: join(testHome(), ".pi", "agent", "qoder-models-cache.json"),
  cn: join(testHome(), ".pi", "agent", "qoder-cn-models-cache.json"),
};
let originalCaches: Record<keyof typeof CACHE_PATHS, string | undefined>;

beforeEach(() => {
  clearQoderModelsMemCache();
  originalCaches = {
    global: existsSync(CACHE_PATHS.global) ? readFileSync(CACHE_PATHS.global, "utf8") : undefined,
    cn: existsSync(CACHE_PATHS.cn) ? readFileSync(CACHE_PATHS.cn, "utf8") : undefined,
  };
  for (const path of Object.values(CACHE_PATHS)) rmSync(path, { force: true });
  clearQoderModelsMemCache();
});

afterEach(() => {
  vi.unstubAllGlobals();
  for (const region of Object.keys(CACHE_PATHS) as Array<keyof typeof CACHE_PATHS>) {
    const path = CACHE_PATHS[region];
    const original = originalCaches[region];
    if (original === undefined) rmSync(path, { force: true });
    else writeFileSync(path, original, "utf8");
  }
  clearQoderModelsMemCache();
});

describe("Qoder model cache", () => {
  it.each([
    ["global", ["Lite", "GLM5.2"]],
    ["cn", ["Qwen3.7Plus"]],
  ] as const)("maps the %s recorded-format catalog to friendly picker ids", async (region, expectedIds) => {
    const interaction = loadLiveFixture(region).interactions.modelList;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(responseFromFixture(interaction)));

    await updateQoderModelsCache("access-token", "user-id", "Test User", "test@example.com", region);

    const cache = JSON.parse(readFileSync(CACHE_PATHS[region], "utf8"));
    expect(cache.models.map((model: { id: string }) => model.id)).toEqual(expectedIds);
    const catalog = interaction.response.body as { chat: Array<{ key: string; display_name: string }> };
    for (const entry of catalog.chat) {
      const friendlyId = entry.display_name.replace(/\s+/g, "");
      expect(cache.configs[friendlyId]?.key).toBe(entry.key);
      expect(cache.configs[entry.key]).toBeUndefined();
      expect(getCachedModelConfig(friendlyId, region)?.key).toBe(entry.key);
      expect(getCachedModelConfig(entry.key, region)).toBeNull();
    }
  });

  it("does not register raw live-catalog keys as public model ids", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            chat: [
              { key: "lite", enable: true, display_name: "Lite" },
              { key: "qfmodel", enable: true, display_name: "Qwen3.8-Flash" },
            ],
          }),
      }),
    );

    await updateQoderModelsCache("access-token", "user-id", "Test User", "test@example.com", "global");

    expect(getCachedModels("global").map((model) => model.id)).toEqual(["Lite", "Qwen3.8-Flash"]);
    expect(getCachedModelConfig("Lite", "global")?.key).toBe("lite");
    expect(getCachedModelConfig("Qwen3.8-Flash", "global")?.key).toBe("qfmodel");
    expect(getCachedModelConfig("lite", "global")).toBeNull();
    expect(getCachedModelConfig("qfmodel", "global")).toBeNull();
  });

  it("omits catalog entries without a friendly display name", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            chat: [
              { key: "q37fmodel", enable: true },
              { key: "qfmodel", enable: true, display_name: "Qwen3.8-Flash" },
            ],
          }),
      }),
    );

    await updateQoderModelsCache("access-token", "user-id", "Test User", "test@example.com", "global");

    expect(getCachedModels("global").map((model) => model.id)).toEqual(["Qwen3.8-Flash"]);
    expect(getCachedModelConfig("q37fmodel", "global")).toBeNull();
  });

  it("keeps only enabled service models without adding auto as a fallback", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            chat: [
              { key: "auto", enable: false, display_name: "Auto" },
              { key: "ultimate", enable: true, display_name: "Ultimate", is_reasoning: true },
              { key: "lite", enable: true, display_name: "Lite" },
              { key: "performance", enable: false, display_name: "Performance" },
            ],
          }),
      }),
    );

    await updateQoderModelsCache("access-token", "user-id", "Test User", "test@example.com", "global");

    const cache = JSON.parse(readFileSync(CACHE_PATHS.global, "utf8"));
    expect(cache.models.map((model: { id: string }) => model.id)).toEqual(["Ultimate", "Lite"]);
    expect(cache.models.some((model: { id: string }) => model.id === "auto")).toBe(false);
  });

  // Upstream #43: configs are keyed by display id since the refactor, so a
  // service-enabled Auto must survive the legacy-fallback cleanup. Checking the
  // raw upstream key alone dropped an explicitly enabled model.
  it("keeps Auto when the service explicitly enables it (display-keyed configs)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            chat: [
              { key: "auto", enable: true, display_name: "Auto" },
              { key: "lite", enable: true, display_name: "Lite" },
            ],
          }),
      }),
    );

    await updateQoderModelsCache("access-token", "user-id", "Test User", "test@example.com", "global");

    expect(getCachedModels("global").map((model) => model.id)).toEqual(["Auto", "Lite"]);
  });

  it("includes the dynamic Qoder price factor in the model name", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            chat: [{ key: "qmodel_38max", enable: true, display_name: "Qwen3.8-Max", price_factor: 0.5 }],
          }),
      }),
    );

    await updateQoderModelsCache("access-token", "user-id", "Test User", "test@example.com", "global");

    const cache = JSON.parse(readFileSync(CACHE_PATHS.global, "utf8"));
    expect(cache.models[0]).toMatchObject({
      id: "Qwen3.8-Max",
      name: "Qwen3.8-Max (0.5x)",
      priceFactor: 0.5,
    });
    expect(getCachedModels("global")[0]?.name).toBe("Qwen3.8-Max (0.5x)");
  });

  it("keeps the Cantus model returned by the current catalog", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ chat: [{ key: "cmodel", enable: true, display_name: "Cantus" }] }),
      }),
    );

    await updateQoderModelsCache("access-token", "user-id", "Test User", "test@example.com", "global");

    const cache = JSON.parse(readFileSync(CACHE_PATHS.global, "utf8"));
    expect(cache.models.map((model: { id: string }) => model.id)).toEqual(["Cantus"]);
  });

  it("filters auto from a legacy fallback cache when the service did not enable it", () => {
    writeFileSync(
      CACHE_PATHS.global,
      JSON.stringify({
        updatedAt: Date.now(),
        models: [{ id: "auto" }, { id: "ultimate" }],
        configs: { ultimate: { key: "ultimate", enable: true } },
      }),
      "utf8",
    );
    clearQoderModelsMemCache();

    expect(getCachedModels("global").map((model) => model.id)).toEqual(["Ultimate"]);
  });

  it("records a 1M context window when the catalog omits context_config", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            chat: [{ key: "lite", enable: true, display_name: "Lite", max_input_tokens: 180000 }],
          }),
      }),
    );

    await updateQoderModelsCache("access-token", "user-id", "Test User", "test@example.com", "global");

    const cache = JSON.parse(readFileSync(CACHE_PATHS.global, "utf8"));
    expect(cache.models[0].contextWindow).toBe(1_000_000);
  });

  it("records the advertised context_config max, even when it is below 1M", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            chat: [
              {
                key: "gm51model",
                enable: true,
                display_name: "GLM 5.2",
                context_config: { default: { token_count: 200000, is_default: true } },
              },
            ],
          }),
      }),
    );

    await updateQoderModelsCache("access-token", "user-id", "Test User", "test@example.com", "global");

    const cache = JSON.parse(readFileSync(CACHE_PATHS.global, "utf8"));
    expect(cache.models[0].contextWindow).toBe(200000);
  });

  it("serves getCachedModelConfig from memory after the cache file is removed", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            chat: [{ key: "lite", enable: true, display_name: "Lite" }],
          }),
      }),
    );

    await updateQoderModelsCache("access-token", "user-id", "Test User", "test@example.com", "global");
    expect(getCachedModelConfig("Lite", "global")?.key).toBe("lite");

    // Hot-path mem cache must keep serving without touching disk again.
    rmSync(CACHE_PATHS.global, { force: true });
    expect(existsSync(CACHE_PATHS.global)).toBe(false);
    expect(getCachedModelConfig("Lite", "global")?.key).toBe("lite");
    expect(getCachedModels("global").map((m) => m.id)).toEqual(["Lite"]);
  });
});

/**
 * Qoder answers /model/list per account: a quota-exhausted account only ships
 * its `is_free` models. One shared cache slot let that degraded answer erase a
 * funded account's catalogue, which showed up as "only Qwen models".
 */
describe("per-account catalog slots", () => {
  const HEALTHY_CHAT = [
    { key: "efficient", enable: true, display_name: "Efficient", is_free: true },
    { key: "qmodel_38max", enable: true, display_name: "Qwen3.8-Max", is_free: true },
    { key: "gmodel", enable: true, display_name: "GLM-5.3", is_reasoning: true },
    { key: "kmodel", enable: true, display_name: "Kimi-K2.7-Code" },
  ];
  const FREE_ONLY_CHAT = [
    { key: "efficient", enable: true, display_name: "Efficient", is_free: true },
    { key: "qmodel_38max", enable: true, display_name: "Qwen3.8-Max", is_free: true },
    { key: "qfmodel", enable: true, display_name: "Qwen3.8-Flash", is_free: true },
  ];

  function jsonResponse(chat: unknown[]) {
    return { ok: true, json: () => Promise.resolve({ chat }) };
  }

  function modelIdSet(): string[] {
    return getCachedModels("global")
      .map((model) => model.id)
      .sort();
  }

  it("unions a funded account's catalogue with a free-only account's catalogue", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValueOnce(jsonResponse(HEALTHY_CHAT)).mockResolvedValueOnce(jsonResponse(FREE_ONLY_CHAT)),
    );

    await updateQoderModelsCache("token-a", "user-a", "A", "a@example.com", "global");
    await updateQoderModelsCache("token-b", "user-b", "B", "b@example.com", "global");

    expect(getCachedCatalogAccounts("global").sort()).toEqual(["user-a", "user-b"]);
    expect(modelIdSet()).toEqual(["Efficient", "GLM-5.3", "Kimi-K2.7-Code", "Qwen3.8-Flash", "Qwen3.8-Max"].sort());
    // A model only the funded account advertises still resolves for requests.
    expect(getCachedModelConfig("GLM-5.3", "global")?.key).toBe("gmodel");
    expect(getCachedModelConfig("Qwen3.8-Flash", "global")?.key).toBe("qfmodel");
  });

  it("does not let a later degraded refresh shrink an earlier healthy catalogue", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValueOnce(jsonResponse(HEALTHY_CHAT)).mockResolvedValueOnce(jsonResponse(FREE_ONLY_CHAT)),
    );

    await updateQoderModelsCache("token-a", "user-a", "A", "a@example.com", "global");
    await updateQoderModelsCache("token-b", "user-b", "B", "b@example.com", "global");

    const cache = JSON.parse(readFileSync(CACHE_PATHS.global, "utf8"));
    expect(cache.accounts["user-a"].models.length).toBe(4);
    expect(cache.accounts["user-a"].degraded).toBeUndefined();
  });

  it("keeps the last known-good list for a quota-degraded account and flags it", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValueOnce(jsonResponse(HEALTHY_CHAT)).mockResolvedValueOnce(jsonResponse(FREE_ONLY_CHAT)),
    );

    await updateQoderModelsCache("token-a", "user-a", "A", "a@example.com", "global");
    clearQoderModelsMemCache();
    // Same account, now quota-exhausted: Qoder answers with the free subset.
    vi.spyOn(Date, "now").mockReturnValue(Date.now() + 3600_001);
    await updateQoderModelsCache("token-a", "user-a", "A", "a@example.com", "global");

    const cache = JSON.parse(readFileSync(CACHE_PATHS.global, "utf8"));
    expect(cache.accounts["user-a"].degraded).toBe(true);
    expect(cache.accounts["user-a"].models.map((model: { id: string }) => model.id)).toEqual([
      "Efficient",
      "Qwen3.8-Max",
      "GLM-5.3",
      "Kimi-K2.7-Code",
    ]);
    expect(modelIdSet()).toEqual(["Efficient", "GLM-5.3", "Kimi-K2.7-Code", "Qwen3.8-Max"].sort());
  });

  it("inherits a pre-v2 snapshot, then replaces it once a real account is stored", async () => {
    writeFileSync(
      CACHE_PATHS.global,
      JSON.stringify({
        updatedAt: Date.now(),
        models: [{ id: "Efficient" }, { id: "Qwen3.8-Max" }, { id: "GLM-5.3" }, { id: "Kimi-K2.7-Code" }],
        configs: {
          Efficient: { key: "efficient", enable: true, display_name: "Efficient", is_free: true },
          "Qwen3.8-Max": { key: "qmodel_38max", enable: true, display_name: "Qwen3.8-Max", is_free: true },
          "GLM-5.3": { key: "gmodel", enable: true, display_name: "GLM-5.3", is_reasoning: true },
          "Kimi-K2.7-Code": { key: "kmodel", enable: true, display_name: "Kimi-K2.7-Code" },
        },
      }),
      "utf8",
    );
    clearQoderModelsMemCache();

    expect(getCachedCatalogAccounts("global")).toEqual([]);
    expect(modelIdSet()).toEqual(["Efficient", "GLM-5.3", "Kimi-K2.7-Code", "Qwen3.8-Max"].sort());

    vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(jsonResponse(FREE_ONLY_CHAT)));
    await updateQoderModelsCache("token-a", "user-a", "A", "a@example.com", "global");

    const cache = JSON.parse(readFileSync(CACHE_PATHS.global, "utf8"));
    expect(Object.keys(cache.accounts)).toEqual(["user-a"]);
    // The free-only answer was smaller than the inherited snapshot, so the
    // snapshot is kept (flagged) instead of being replaced by 3 free models.
    expect(cache.accounts["user-a"].degraded).toBe(true);
    expect(cache.accounts["user-a"].models.map((model: { id: string }) => model.id)).toEqual([
      "Efficient",
      "Qwen3.8-Max",
      "GLM-5.3",
      "Kimi-K2.7-Code",
    ]);
  });

  it("tracks staleness per account", async () => {
    const now = Date.now();
    vi.spyOn(Date, "now").mockReturnValue(now);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(jsonResponse(HEALTHY_CHAT)));

    await updateQoderModelsCache("token-a", "user-a", "A", "a@example.com", "global");

    expect(isAccountCatalogStale("global", "user-a")).toBe(false);
    expect(isAccountCatalogStale("global", "user-b")).toBe(true);

    vi.spyOn(Date, "now").mockReturnValue(now + 3600_001);
    expect(isAccountCatalogStale("global", "user-a")).toBe(true);
  });

  it("re-reads the file when another process rewrites it", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(jsonResponse(HEALTHY_CHAT)));
    await updateQoderModelsCache("token-a", "user-a", "A", "a@example.com", "global");
    expect(modelIdSet()).toEqual(["Efficient", "GLM-5.3", "Kimi-K2.7-Code", "Qwen3.8-Max"].sort());

    // Simulate a sibling pane writing a different catalogue (no mem-cache clear).
    writeFileSync(
      CACHE_PATHS.global,
      JSON.stringify({
        version: 2,
        updatedAt: Date.now(),
        models: [{ id: "Sonus" }],
        configs: { Sonus: { key: "sonus", enable: true, display_name: "Sonus" } },
        accounts: {
          "user-z": {
            updatedAt: Date.now(),
            identity: { userID: "user-z" },
            models: [{ id: "Sonus" }],
            configs: { Sonus: { key: "sonus", enable: true, display_name: "Sonus" } },
          },
        },
      }),
      "utf8",
    );

    expect(hasCachedCatalog("global")).toBe(true);
    expect(getCachedModels("global").map((model) => model.id)).toEqual(["Sonus"]);
  });

  it("records the raw served models separately from a downgrade-guarded list", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValueOnce(jsonResponse(HEALTHY_CHAT)).mockResolvedValueOnce(jsonResponse(FREE_ONLY_CHAT)),
    );

    await updateQoderModelsCache("token-a", "user-a", "A", "a@example.com", "global");
    await updateQoderModelsCache("token-a", "user-a", "A", "a@example.com", "global");

    const cache = JSON.parse(readFileSync(CACHE_PATHS.global, "utf8"));
    const slot = cache.accounts["user-a"];
    // The display list kept the richer (guarded) answer...
    expect(slot.models.map((model: { id: string }) => model.id)).toContain("GLM-5.3");
    // ...but routing must use the raw answer of the last successful fetch.
    expect(slot.servedModelIds).toEqual(["Efficient", "Qwen3.8-Max", "Qwen3.8-Flash"]);
  });

  it("answers entitlement from the account's own fresh catalogue only", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(jsonResponse(HEALTHY_CHAT))
        .mockResolvedValueOnce(jsonResponse([{ key: "kmodel_latest", enable: true, display_name: "Kimi-K3" }])),
    );
    await updateQoderModelsCache("token-a", "user-a", "A", "a@example.com", "global");
    await updateQoderModelsCache("token-b", "user-b", "B", "b@example.com", "global");

    expect(getAccountServedModels("global", "user-a")?.modelIds).toContain("GLM-5.3");
    expect(checkAccountEntitlement("global", "user-a", "GLM-5.3")).toMatchObject({
      checked: true,
      served: true,
    });
    // Listed by another account, missing from this one: the misroute we veto.
    expect(getRegionServedModelIds("global", "user-a")).toEqual(["Kimi-K3"]);
    expect(checkAccountEntitlement("global", "user-a", "Kimi-K3")).toMatchObject({
      checked: true,
      served: false,
    });
    // Nobody lists it: more likely a brand-new model than a restriction.
    expect(checkAccountEntitlement("global", "user-a", "Kimi-K9")).toEqual({ checked: false });
    // Unknown accounts and blank identities stay fail-open.
    expect(checkAccountEntitlement("global", "user-unknown", "Kimi-K3")).toEqual({ checked: false });
    expect(checkAccountEntitlement("global", "", "Kimi-K3")).toEqual({ checked: false });

    // A stale catalogue is not trusted for entitlement either.
    vi.spyOn(Date, "now").mockReturnValue(Date.now() + 3600_001);
    expect(checkAccountEntitlement("global", "user-a", "Kimi-K3")).toEqual({ checked: false });
  });
});
