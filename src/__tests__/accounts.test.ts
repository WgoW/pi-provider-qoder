import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clearQoderAuthMemCache, listQoderAccounts } from "../auth/oauth.js";
import { clearQoderModelsMemCache } from "../catalog.js";

const AGENT_DIR = join(process.env.HOME || process.env.USERPROFILE || "", ".pi", "agent");
const AUTH_FILE = join(AGENT_DIR, "auth.json");
const MULTI_FILE = join(AGENT_DIR, "multiprovider-auth.json");

const EXPIRES = Date.now() + 3_600_000;

const HEALTHY_CHAT = [
  { key: "efficient", enable: true, display_name: "Efficient", is_free: true },
  { key: "qmodel_38max", enable: true, display_name: "Qwen3.8-Max", is_free: true },
  { key: "gmodel", enable: true, display_name: "GLM-5.3", is_reasoning: true, is_vl: true },
  { key: "kmodel", enable: true, display_name: "Kimi-K2.7-Code" },
];

const FREE_ONLY_CHAT = [
  { key: "efficient", enable: true, display_name: "Efficient", is_free: true },
  { key: "qmodel_38max", enable: true, display_name: "Qwen3.8-Max", is_free: true },
  { key: "qfmodel", enable: true, display_name: "Qwen3.8-Flash", is_free: true },
];

function credential(userID: string, access: string) {
  return {
    type: "oauth",
    access,
    refresh: "rt",
    expires: EXPIRES,
    userID,
    email: `${userID}@example.com`,
    name: userID,
  };
}

beforeEach(() => {
  clearQoderAuthMemCache();
  clearQoderModelsMemCache();
  mkdirSync(AGENT_DIR, { recursive: true });
  rmSync(AUTH_FILE, { force: true });
  rmSync(MULTI_FILE, { force: true });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  clearQoderAuthMemCache();
  clearQoderModelsMemCache();
  rmSync(AUTH_FILE, { force: true });
  rmSync(MULTI_FILE, { force: true });
});

describe("listQoderAccounts", () => {
  it("merges pi auth.json slots with the multiprovider pool and prefers auth entries", () => {
    writeFileSync(
      AUTH_FILE,
      JSON.stringify({ qoder: credential("user-a", "token-a"), "qoder-2": credential("user-b", "token-b") }),
      "utf8",
    );
    writeFileSync(
      MULTI_FILE,
      JSON.stringify({
        version: 1,
        providers: {
          qoder: {
            policy: "round-robin",
            accounts: [
              { credential: credential("user-b", "pool-token-b") },
              { credential: credential("user-c", "token-c") },
            ],
          },
        },
      }),
      "utf8",
    );
    clearQoderAuthMemCache();

    const accounts = listQoderAccounts("global");
    expect(accounts.map((account) => account.key).sort()).toEqual(["user-a", "user-b", "user-c"]);
    expect(accounts.find((account) => account.key === "user-b")).toMatchObject({
      access: "token-b",
      providerID: "qoder-2",
      source: "auth",
    });
    expect(accounts.find((account) => account.key === "user-c")).toMatchObject({
      access: "token-c",
      source: "multiprovider",
    });
    // CN credentials never leak into the global list.
    expect(listQoderAccounts("cn")).toEqual([]);
  });

  it("keeps regions apart when both are configured", () => {
    // The global provider id ("qoder") is a strict prefix of the CN one
    // ("qoder-cn"), so a plain `id.startsWith(`${prefix}-`)` test matched CN
    // slots when listing global accounts. That made a global catalogue refresh
    // call /model/list with CN credentials (against the GLOBAL endpoint) and
    // write the CN account into the global cache.
    writeFileSync(
      AUTH_FILE,
      JSON.stringify({
        qoder: credential("global-user", "global-token"),
        "qoder-2": credential("global-user-2", "global-token-2"),
        "qoder-cn": credential("cn-user", "cn-token"),
        "qoder-cn-2": credential("cn-user-2", "cn-token-2"),
      }),
      "utf8",
    );
    writeFileSync(MULTI_FILE, JSON.stringify({ version: 1, providers: {} }), "utf8");
    clearQoderAuthMemCache();

    expect(
      listQoderAccounts("global")
        .map((a) => a.providerID)
        .sort(),
    ).toEqual(["qoder", "qoder-2"]);
    expect(
      listQoderAccounts("cn")
        .map((a) => a.providerID)
        .sort(),
    ).toEqual(["qoder-cn", "qoder-cn-2"]);
  });
});

describe("account catalogue refresh", () => {
  it("registers the union of every account's catalogue even when auth.json's account is degraded", async () => {
    // Mirrors the reported setup: the account pi stores directly is out of
    // quota (free-only catalogue), the funded account only lives in the pool.
    writeFileSync(AUTH_FILE, JSON.stringify({ qoder: credential("user-a", "token-a") }), "utf8");
    writeFileSync(
      MULTI_FILE,
      JSON.stringify({
        version: 1,
        providers: { qoder: { policy: "round-robin", accounts: [{ credential: credential("user-b", "token-b") }] } },
      }),
      "utf8",
    );

    const fetchMock = vi
      .fn()
      .mockImplementation(async (_url: string, options: { headers?: Record<string, string> }) => {
        const user = options?.headers?.["Cosy-User"];
        return { ok: true, json: () => Promise.resolve({ chat: user === "user-b" ? HEALTHY_CHAT : FREE_ONLY_CHAT }) };
      });
    vi.stubGlobal("fetch", fetchMock);

    const providers = new Map<string, { models: Array<{ id: string }> }>();
    const pi = {
      registerProvider(providerID: string, config: { models: Array<{ id: string }> }) {
        providers.set(providerID, config);
      },
      on: vi.fn(),
    };

    const { default: registerProviders } = await import("../index.js");
    await registerProviders(pi as never);

    const ids = providers.get("qoder")?.models.map((model) => model.id) ?? [];
    expect(new Set(ids)).toEqual(new Set(["Efficient", "Qwen3.8-Max", "Qwen3.8-Flash", "GLM-5.3", "Kimi-K2.7-Code"]));
  });
});
