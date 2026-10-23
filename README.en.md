# @jischeng/pi-provider-qoder

[中文文档](./README.md)

A [Pi](https://pi.dev/) extension that connects Pi to Qoder AI (both Global and CN regions), exposing available models in Pi's model picker.

## Key Features

- **Dual-Region Support**: Full support for both Qoder Global (`qoder`) and Qoder China (`qoder-cn`).
- **Progressive Multi-Account**: Support for multiple accounts per region (`qoder`, `qoder-2`... and `qoder-cn`, `qoder-cn-2`...). Next account slot automatically becomes available after logging in.
- **Thinking & Reasoning Effort Forwarding**: Transparently forwards Pi reasoning levels (`low`/`medium`/`high`/`xhigh`/`max`/`off`) to Qoder upstream, dynamically showing supported thinking levels for each model.
- **128K Output & 1M Context**: Output token limit raised to 128K (131,072) and context window up to 1M tokens, preventing reasoning chain truncation and early compaction.
- **Dynamic Friendly IDs & Latest Models**: Uses upstream `display_name` values (e.g. `Qwen3.8-Max`, `Lite`) and discovers currently available models, including the latest Fast series, from Qoder's catalog.
- **10605 Transient Queue Retry**: Automatically retries requests when Qoder servers return `10605` queue busy (up to 4 retries) with server-recommended backoff.
- **Model Price Factor Display**: Dynamically displays model price multipliers (e.g. `[0.5x]`) in model names.
- **Actual Credits & Account Quota Tracking**: Captures request-level `credits` and `billable` deductions, tracks session spend accurately, and queries period quotas.
- **COSY Version Tracking & Tool Call Compatibility**: Refreshes the signing identity from Qoder CLI release information and rewrites internal tool-call IDs into Qoder-compatible wire IDs.
- **DSML Residue Stripping**: Removes leaked internal DSML tags and replays thinking through out-of-band `reasoning_content`.
- **Optimized Hot-Path Encoding & Zero-Copy Signatures**: Lookup-table Base64 body encoding and chunked MD5 hashing, eliminating memory spikes on large prompts.
- **Pi 0.84+ & OMP Compatibility**: Dynamic `qoder-api` registration and adaptive system prompt normalization.

## Installation

```bash
pi install npm:@jischeng/pi-provider-qoder
# or in OneMorePrompt:
omp install npm:@jischeng/pi-provider-qoder
```

This is an ESM extension. `package.json` publishes both `main` and `exports`, so `import`, `require`, and downstream projects using `moduleResolution: node16` all resolve the bare package name. Requires Node.js >= 22.12 (for `require(esm)`).

## Login & Usage

### 1. Global (`qoder`)
- Login: `/login qoder` (Browser OAuth or PAT)
- PAT generation: https://qoder.com/account/integrations
- Environment variables: `QODER_API_KEY`, `QODER_PERSONAL_ACCESS_TOKEN`, `QODER_PAT` (use `_2`, `_3` for multiple accounts)

### 2. China (`qoder-cn`)
- Login: `/login qoder-cn` (PAT only)
- PAT generation: https://qoder.com.cn/account/integrations
- Environment variables: `QODERCN_API_KEY`, `QODERCN_PERSONAL_ACCESS_TOKEN`, `QODERCN_PAT` (use `_2`, `_3` for multiple accounts)

PAT tokens (`pt-...`) are automatically exchanged for short-lived job tokens at startup or login.

### CLI Launch

```bash
pi --provider qoder --model Lite
pi --provider qoder-cn --model Qwen3.7-Plus
```

In Pi session:

```text
/model Qwen3.8-Max
/model Lite
```

### Multiple Accounts

After authenticating account 1, the next slot appears automatically:

```text
/login qoder-2
/login qoder-cn-2
```

## Endpoints

| Endpoint | Global (`qoder`) | China (`qoder-cn`) |
| --- | --- | --- |
| PAT Exchange | `https://openapi.qoder.sh/api/v1/jobToken/exchange` | `https://openapi.qoder.com.cn/api/v1/jobToken/exchange` |
| User Info | `https://openapi.qoder.sh/api/v1/userinfo` | `https://openapi.qoder.com.cn/api/v1/userinfo` |
| Usage Quota | `https://openapi.qoder.sh/api/v2/quota/usage` | `https://openapi.qoder.com.cn/api/v2/quota/usage` |
| Chat Gateway | `https://api3.qoder.sh/` | `https://gateway.qoder.com.cn/` |

## Request and Account Credits

The footer displays **actual session deductions in Credits**, aggregated from response `usage.credits` and `usage.billable`. Non-billable requests contribute 0.

Account quota displays the **account-wide period balance** (Personal Quota, Add-on Credits, Org Resource Package).

### Commands & Footer Toggle

Inside Pi:

```text
/qoder-usage
/qoder-usage footer on
/qoder-usage footer off
```

Or pass the flag when starting Pi:

```bash
pi --qoder-credit-footer
```

Enabling this replaces the misleading `$0.000 (sub)` cost footer with labelled session Credits and account quota.

### Offline Preview

```bash
npm run demo:credits
```

## License

MIT
