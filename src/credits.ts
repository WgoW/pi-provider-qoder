import type { Usage } from "@earendil-works/pi-ai";

export type QoderProvider = "qoder" | "qoder-cn";

/** Request-level fields supplied by Qoder, preserved in serialized assistant usage. */
export interface QoderUsage extends Usage {
  qoder_provider?: QoderProvider;
  credits?: number;
  original_credits?: number;
  billable?: boolean;
  /** Actual deduction: a non-billable request costs zero even when credits is positive. */
  charged_credits?: number;
  /** Reference value at $20 / 1,500 Credits; not a dollar charge. */
  usd_equivalent?: number;
}

export function creditNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined;
}

export function creditsToUsd(credits: number): number {
  return (credits * 20) / 1500;
}

export function chargedCredits(usage: QoderUsage): number | undefined {
  if (usage.billable === false) return 0;
  return usage.billable === true ? creditNumber(usage.credits) : undefined;
}

export function normalizeQoderProvider(provider?: string): QoderProvider | undefined {
  if (!provider) return undefined;
  if (provider === "qoder-cn" || provider.startsWith("qoder-cn-")) return "qoder-cn";
  if (provider === "qoder" || provider.startsWith("qoder-")) return "qoder";
  return undefined;
}

/** Usage chunks are snapshots of one request; replace fields instead of summing them. */
export function applyQoderCredits(usage: QoderUsage, raw: Record<string, unknown>, provider?: string): void {
  const normProvider = normalizeQoderProvider(provider);
  if (normProvider) usage.qoder_provider = normProvider;
  if ("credits" in raw) usage.credits = creditNumber(raw.credits);
  if ("original_credits" in raw) usage.original_credits = creditNumber(raw.original_credits);
  if ("billable" in raw) usage.billable = typeof raw.billable === "boolean" ? raw.billable : undefined;
  usage.charged_credits = chargedCredits(usage);
  usage.usd_equivalent = usage.charged_credits === undefined ? undefined : creditsToUsd(usage.charged_credits);
}
