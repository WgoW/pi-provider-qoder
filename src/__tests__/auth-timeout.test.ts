import { afterEach, describe, expect, it, vi } from "vitest";
import { authFetchSignal } from "../auth/pat.js";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/**
 * The extension's default export awaits automatic login, so every auth call on
 * that path has to be bounded. An upstream that accepts the connection and then
 * never answers otherwise blocks pi from finishing load: no models, no UI, and
 * no error to explain it.
 */
describe("authFetchSignal", () => {
  it("returns a signal that aborts on its own", () => {
    const signal = authFetchSignal();
    expect(signal).toBeInstanceOf(AbortSignal);
    expect(signal.aborted).toBe(false);
  });

  it("combines a caller signal with the internal timeout", () => {
    const controller = new AbortController();
    const signal = authFetchSignal(controller.signal);

    expect(signal.aborted).toBe(false);
    controller.abort();
    // The caller's abort must still win: a cancelled login must stop promptly
    // instead of waiting out the 15s ceiling.
    expect(signal.aborted).toBe(true);
  });

  it("is applied to the PAT exchange that runs during startup", async () => {
    // Without a signal this fetch has no deadline; assert the request carries one
    // so a hung exchange cannot stall extension load.
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(init?.signal).toBeInstanceOf(AbortSignal);
      return Response.json({ token: "jt-1", refresh_token: "rt-1", expires_in: 3600, user_id: "u" });
    });
    vi.stubGlobal("fetch", fetchMock);

    const { exchangeJobToken } = await import("../auth/pat.js");
    await exchangeJobToken("pt-1", "global");

    expect(fetchMock).toHaveBeenCalled();
  });
});
