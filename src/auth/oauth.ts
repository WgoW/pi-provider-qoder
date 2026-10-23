import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import type { OAuthCredentials, OAuthLoginCallbacks } from "@earendil-works/pi-ai";
import * as PiCodingAgent from "@earendil-works/pi-coding-agent";
import { qoderAccountKey, updateQoderModelsCache } from "../catalog.js";
import { getMachineId } from "../cosy.js";
import { getQoderRefreshURL, getQoderRegionConfig, isProviderOfMode, type QoderMode } from "../region.js";
import { interactiveLogin } from "./login.js";
import { authFetchSignal, credentialsFromPat, decodePatRefresh, fetchUserInfo, isPatRefresh } from "./pat.js";

export interface QoderCredentials extends OAuthCredentials {
  userID: string;
  email: string;
  name: string;
  machineID: string;
}

/**
 * `AuthStorage` is not part of every pi-coding-agent release's public exports,
 * so it is read off the module namespace instead of imported by name: a missing
 * export must degrade to the auth-file fallback below, not break the build.
 */
const AuthStorage = (
  PiCodingAgent as unknown as {
    AuthStorage?: { create?: () => { set: (providerID: string, credentials: unknown) => void } };
  }
).AuthStorage;

const identityCache = new Map<string, QoderCredentials>();

function getHomeDir(): string {
  return process.env.HOME || process.env.USERPROFILE || homedir();
}

function getAuthFilePath(): string {
  return join(getHomeDir(), ".pi", "agent", "auth.json");
}

function getMultiAuthFilePath(): string {
  return join(getHomeDir(), ".pi", "agent", "multiprovider-auth.json");
}

interface MultiAuthAccountEntry {
  label?: string;
  credential?: Record<string, unknown>;
}

/** Best-effort read of pi-multiprovider's account pool for one provider. */
function readMultiAuthPool(providerID: string): MultiAuthAccountEntry[] {
  const multiAuthPath = getMultiAuthFilePath();
  if (!existsSync(multiAuthPath)) return [];
  try {
    const raw = readFileSync(multiAuthPath, "utf-8");
    if (!raw.trim()) return [];
    const data = JSON.parse(raw) as {
      providers?: Record<string, { accounts?: MultiAuthAccountEntry[] }>;
    };
    const accounts = data?.providers?.[providerID]?.accounts;
    return Array.isArray(accounts) ? accounts : [];
  } catch {}
  return [];
}

function readMultiAuthCredentials(accessToken?: string, providerID = "qoder"): QoderCredentials | null {
  const accounts = readMultiAuthPool(providerID);
  if (accounts.length === 0) return null;

  if (accessToken) {
    const match = accounts.find((acc) => acc.credential?.access === accessToken);
    if (match?.credential?.userID) {
      return match.credential as unknown as QoderCredentials;
    }
  } else {
    const first = accounts.find((acc) => acc.credential?.userID || acc.credential?.access);
    if (first?.credential) {
      return first.credential as unknown as QoderCredentials;
    }
  }
  return null;
}

/** One catalogue-addressable Qoder account (pi auth.json entry or pool account). */
export interface QoderAccountCredential {
  /** Stable catalogue slot key (Qoder userID, email digest as fallback). */
  key: string;
  /** Logical pi provider id this credential belongs to (qoder, qoder-2, ...). */
  providerID: string;
  access: string;
  userID: string;
  email: string;
  name: string;
  expires?: number;
  source: "auth" | "multiprovider";
}

/**
 * Enumerate every Qoder account we can see for a region.
 *
 * A single provider id (e.g. `qoder`) can be backed by several accounts: pi's
 * own auth.json slots (`qoder`, `qoder-2`, ...) plus the accounts pooled by
 * pi-multiprovider. `/model/list` answers per account, so the catalogue refresh
 * must walk all of them — refreshing only auth.json's account is what pinned
 * the visible catalogue to one account's entitlement.
 */
export function listQoderAccounts(mode: QoderMode): QoderAccountCredential[] {
  const prefix = getQoderRegionConfig(mode).providerID;
  const found = new Map<string, QoderAccountCredential>();

  const add = (
    credential: Record<string, unknown> | undefined,
    providerID: string,
    source: QoderAccountCredential["source"],
  ): void => {
    const access = typeof credential?.access === "string" ? credential.access : "";
    if (!access) return;
    const userID = typeof credential?.userID === "string" ? credential.userID : "";
    const email = typeof credential?.email === "string" ? credential.email : "";
    const name = typeof credential?.name === "string" ? credential.name : "";
    const key = qoderAccountKey({ userID, email });
    const existing = found.get(key);
    // pi's own auth.json entry wins over the same account inside a pool.
    if (existing && !(existing.source === "multiprovider" && source === "auth")) return;
    found.set(key, {
      key,
      providerID,
      access,
      userID,
      email,
      name,
      ...(typeof credential?.expires === "number" ? { expires: credential.expires } : {}),
      source,
    });
  };

  const auth = readAuthFileCached();
  for (const [providerID, credential] of Object.entries(auth ?? {})) {
    if (!isProviderOfMode(providerID, mode)) continue;
    add(credential as Record<string, unknown>, providerID, "auth");
  }

  for (const entry of readMultiAuthPool(prefix)) {
    add(entry.credential, prefix, "multiprovider");
  }

  return [...found.values()];
}

/** Memoized parse of auth.json; invalidated on save. null = not loaded. */
let authFileMem: { path: string; data: Record<string, unknown> } | null = null;

/** Clear process-memory auth caches (used by tests that mutate auth.json). */
export function clearQoderAuthMemCache(): void {
  authFileMem = null;
  identityCache.clear();
}

function readAuthFileCached(): Record<string, unknown> | null {
  const authPath = getAuthFilePath();
  if (!existsSync(authPath)) {
    return null;
  }
  try {
    const raw = readFileSync(authPath, "utf-8");
    if (!raw.trim()) return authFileMem?.data ?? null;
    const data = JSON.parse(raw) as Record<string, unknown>;
    if (data && typeof data === "object") {
      authFileMem = { path: authPath, data };
      return data;
    }
    return authFileMem?.data ?? null;
  } catch {
    return authFileMem?.data ?? null;
  }
}

/** Return the PAT exposed through the environment for a provider mode. */
export function getQoderPatForMode(mode: QoderMode, providerID = "qoder"): string {
  const accountMatch = /-(\d+)$/.exec(providerID);
  const accountNumber = accountMatch ? Number(accountMatch[1]) : 1;
  const suffix = accountNumber > 1 ? `_${accountNumber}` : "";

  for (const envName of getQoderRegionConfig(mode).patEnvNames) {
    const value = process.env[`${envName}${suffix}`];
    if (value) return value;
  }
  return "";
}

function saveCredentialsToAuthFile(providerID: string, credentials: OAuthCredentials): void {
  try {
    const authPath = getAuthFilePath();
    const dir = dirname(authPath);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true, mode: 0o700 });
    }
    const existing = readAuthFileCached();
    const auth: Record<string, unknown> = existing ? { ...existing } : {};
    auth[providerID] = { type: "oauth", ...credentials };
    const temporaryPath = `${authPath}.${process.pid}.${randomUUID()}.tmp`;
    try {
      writeFileSync(temporaryPath, JSON.stringify(auth, null, 2), { encoding: "utf-8", mode: 0o600 });
      renameSync(temporaryPath, authPath);
    } finally {
      rmSync(temporaryPath, { force: true });
    }
    authFileMem = { path: authPath, data: auth };
    const q = credentials as QoderCredentials;
    if (q.access && q.userID) {
      identityCache.set(`${providerID}:${q.access}`, q);
    }
  } catch (err) {
    console.error(`[pi-provider-qoder] Failed to write auth storage for ${providerID}:`, err);
  }
}

/** Exchange an environment PAT before pi resolves its initial model. */
export async function autoLoginQoderFromEnvironment(providerID: string, mode: QoderMode): Promise<void> {
  const pat = getQoderPatForMode(mode, providerID);
  if (!pat) return;

  // An explicitly supplied PAT is authoritative. The auth file only stores
  // the exchanged job token, so it cannot tell us whether the environment
  // token changed. Re-exchange it on startup to avoid silently using an old
  // account's credentials.
  const credentials = await credentialsFromPat(pat, mode);

  if (typeof AuthStorage?.create === "function") {
    try {
      const authStorage = AuthStorage.create();
      authStorage.set(providerID, { type: "oauth", ...credentials });
    } catch {
      saveCredentialsToAuthFile(providerID, credentials);
    }
  } else {
    saveCredentialsToAuthFile(providerID, credentials);
  }

  const qCreds = credentials as QoderCredentials;
  // Wait for the model cache before the provider is registered. This matters
  // for `pi --list-models`, which can exit before background work completes.
  await updateQoderModelsCache(qCreds.access, qCreds.userID, qCreds.name, qCreds.email, mode);
}

/**
 * Read the Qoder identity (userID/email/name/machineID) from pi's own auth
 * store or multiprovider-auth.json. pi persists the full OAuthCredentials there
 * on login/refresh and keeps it up to date.
 *
 * When accessToken is specified, it strictly checks that the cached access
 * matches to prevent identity confusion across pooled accounts.
 *
 * Note: the auth.json path/shape is a pi internal convention, not a public API.
 * This is best-effort and falls back to null so callers can use placeholders.
 */
export function getCachedCredentials(accessToken?: string, providerID = "qoder"): QoderCredentials | null {
  if (accessToken) {
    const mem = identityCache.get(`${providerID}:${accessToken}`);
    if (mem?.userID) return mem;
  }

  const auth = readAuthFileCached();
  if (auth) {
    const creds = (auth[providerID] || (providerID === "qoder" ? auth.qoder : null)) as QoderCredentials | null;
    if (creds?.userID || creds?.access) {
      if (creds.access && creds.userID) {
        identityCache.set(`${providerID}:${creds.access}`, creds);
      }
      if (!accessToken || creds.access === accessToken) {
        return creds;
      }
    }
  }

  const multiCreds = readMultiAuthCredentials(accessToken, providerID);
  if (multiCreds) {
    if (multiCreds.access && multiCreds.userID) {
      identityCache.set(`${providerID}:${multiCreds.access}`, multiCreds);
    }
    return multiCreds;
  }

  return null;
}

export async function resolveQoderIdentity(
  accessToken: string,
  providerID: string,
  mode: QoderMode,
): Promise<QoderCredentials> {
  const region = getQoderRegionConfig(mode);
  const cacheKey = `${providerID}:${accessToken}`;
  const mem = identityCache.get(cacheKey);
  if (mem?.userID) return mem;

  const cached = getCachedCredentials(accessToken, providerID);
  if (cached?.userID) {
    identityCache.set(cacheKey, cached);
    return cached;
  }

  const info = await fetchUserInfo(accessToken, mode);
  const machineID = getMachineId();
  const creds: QoderCredentials = {
    access: accessToken,
    userID: info.userID || "qoder-user",
    email: info.email || region.userEmailFallback,
    name: info.name || region.userNameFallback,
    machineID,
    refresh: "",
    expires: 0,
  };
  identityCache.set(cacheKey, creds);

  // Only persist to auth.json if the entry matches this account or there is no
  // other valid account stored, avoiding clobbering main account credentials.
  const existing = readAuthFileCached();
  const currentSaved = (existing?.[providerID] ||
    (providerID === "qoder" ? existing?.qoder : null)) as QoderCredentials | null;
  if (!currentSaved || currentSaved.access === accessToken) {
    saveCredentialsToAuthFile(providerID, creds);
  }

  return creds;
}

export async function loginQoderForMode(
  callbacks: OAuthLoginCallbacks,
  mode: QoderMode,
  providerID: string = getQoderRegionConfig(mode).providerID,
  onLogin?: (providerID: string) => void,
): Promise<OAuthCredentials> {
  // 1. Try environment variables first (PAT). A PAT (pt-...) must be exchanged
  //    for a short-lived job token before it can be used — credentialsFromPat
  //    handles the exchange + identity resolution.
  const pat = getQoderPatForMode(mode, providerID);
  if (pat) {
    try {
      const creds = await credentialsFromPat(pat, mode);
      const qCreds = creds as QoderCredentials;
      // Persist the resolved identity locally so chat requests can resolve the real uid.
      // Cache models in background
      updateQoderModelsCache(qCreds.access, qCreds.userID, qCreds.name, qCreds.email, mode).catch(() => {});
      // Persist the resolved identity locally. OMP (17.x) stores login
      // credentials in its own agent.db, not in ~/.pi/agent/auth.json, so
      // without this the chat COSY payload would fall back to uid "qoder-user"
      // and Qoder CN rejects it with "Login expired" (105).
      saveCredentialsToAuthFile(providerID, creds);
      try {
        onLogin?.(providerID);
      } catch {}
      return creds;
    } catch {
      // Fall through to interactive login if PAT exchange fails.
    }
  }

  // 2. Interactive login (CN only supports PAT prompt here; global supports device flow fallback)
  const creds = await interactiveLogin(callbacks, mode);

  // Cache models in background.
  try {
    const qCreds = creds as QoderCredentials;
    updateQoderModelsCache(qCreds.access, qCreds.userID, qCreds.name, qCreds.email, mode).catch(() => {});
  } catch {}

  // Persist the resolved identity locally (see note above).
  saveCredentialsToAuthFile(providerID, creds);
  try {
    onLogin?.(providerID);
  } catch {}
  return creds;
}

export async function loginQoderForProvider(
  callbacks: OAuthLoginCallbacks,
  providerID: string,
  mode: QoderMode,
  onLogin?: (providerID: string) => void,
): Promise<OAuthCredentials> {
  return loginQoderForMode(callbacks, mode, providerID, onLogin);
}

export async function refreshQoderTokenForMode(
  credentials: OAuthCredentials,
  mode: QoderMode,
): Promise<OAuthCredentials> {
  // PAT-based credentials: re-exchange the stored PAT for a fresh job token.
  if (isPatRefresh(credentials.refresh)) {
    const { pat } = decodePatRefresh(credentials.refresh);
    if (pat) {
      try {
        const refreshed = await credentialsFromPat(pat, mode);
        const qCreds = refreshed as QoderCredentials;
        updateQoderModelsCache(qCreds.access, qCreds.userID, qCreds.name, qCreds.email, mode).catch(() => {});
        return refreshed;
      } catch {
        // Fall through to validity extension below.
      }
    }
    return {
      ...credentials,
      expires: Date.now() + 60 * 60 * 1000, // extend 1 hour to retry later
    };
  }

  const parts = credentials.refresh.split("|");
  const refreshToken = parts[0] || "";
  const userID = parts[1] || "";
  const machineID = parts[2] || getMachineId();
  const prev = credentials as Partial<QoderCredentials>;
  const prevName = prev.name || "";
  const prevEmail = prev.email || "";

  const refreshURL = getQoderRefreshURL(mode);
  try {
    const response = await fetch(refreshURL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${credentials.access}`,
        Accept: "application/json",
        "User-Agent": "pi-provider-qoder",
      },
      body: JSON.stringify({ refreshToken }),
      // Refresh also runs on the startup path (and mid-turn on 401); bound it so
      // an unresponsive center host cannot stall login or a request forever.
      signal: authFetchSignal(),
    });

    if (response.ok) {
      const data = (await response.json()) as {
        token: string;
        refresh_token?: string;
        expires_at?: string;
        expires_in?: number;
      };

      const newAccess = data.token;
      const newRefresh = data.refresh_token || refreshToken;

      let expireMs = Date.now() + 30 * 24 * 60 * 60 * 1000;
      if (data.expires_at) {
        const parsed = Date.parse(data.expires_at);
        if (!Number.isNaN(parsed)) expireMs = parsed;
      } else if (data.expires_in) {
        expireMs = Date.now() + data.expires_in * 1000;
      }

      const refreshed = {
        ...credentials,
        refresh: `${newRefresh}|${userID}|${machineID}`,
        access: newAccess,
        expires: expireMs - 5 * 60 * 1000,
        userID,
        email: prevEmail,
        name: prevName,
        machineID,
      };

      // pi persists the refreshed credentials in auth.json itself.
      // Cache models in background
      updateQoderModelsCache(newAccess, userID, prevName, prevEmail, mode).catch(() => {});

      return refreshed;
    }
  } catch {}

  // Fallback: Extend validity slightly to buy time, as Qoder tokens are long-lived
  const refreshedFallback = {
    ...credentials,
    expires: Date.now() + 60 * 60 * 1000, // extend for 1 hour
  };
  return refreshedFallback;
}
