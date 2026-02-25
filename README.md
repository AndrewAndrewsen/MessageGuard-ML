# @andrewlabs/openclaw-messageguard-ml

ML-powered companion plugin to `@andrewlabs/openclaw-messageguard`.

This plugin uses a DistilBERT token classification model via `@huggingface/transformers` (ONNX Runtime under the hood) to detect sensitive content in outgoing messages and replace detected spans with `[REDACTED]`.

## Features

- OpenClaw `before_tool_call` hook — intercepts `message` tool sends before execution
- OpenClaw `message_sending` hook — intercepts agent replies (when wired up by the gateway)
- Automatic model download from Hugging Face on first use (then cached locally)
- Fails open: if model download/inference fails, message passes through unchanged
- Configurable model id, confidence threshold, and redaction token

## Install

```bash
openclaw plugins install @andrewlabs/openclaw-messageguard-ml
openclaw gateway restart
```

Or via npm:

```bash
npm install @andrewlabs/openclaw-messageguard-ml
```

Ensure your OpenClaw plugin loader can discover package extensions via `openclaw.extensions`.

## Manifest

The package includes `openclaw.plugin.json` with plugin id `messageguard-ml` and configuration schema.

## Configuration

`openclaw.plugin.json` supports:

- `enabled` (boolean, default `true`)
- `modelId` (string, default `AndrewAndrewsen/distilbert-secret-masker`)
- `threshold` (number, 0-1, default `0.5`)
- `mask` (string, default `[REDACTED]`)

## How it Works

1. On plugin startup, two hooks are registered:
   - `before_tool_call` — primary enforcement; intercepts `message` tool sends (action `send`/`broadcast`) and redacts sensitive content in the message parameter before the tool executes.
   - `message_sending` — secondary; intercepts agent replies in the delivery pipeline. Note: in OpenClaw 2026.2.x, this hook does not fire for all outbound paths (see [openclaw#XXXX](https://github.com/openclaw/openclaw/issues)).
2. For each outgoing message, the model runs token classification.
3. Spans predicted as sensitive at/above threshold are grouped, reconstructed to character offsets, extended to word boundaries, and merged.
4. Sanitized content is returned to OpenClaw.

If model loading or inference fails (for example, model repo not yet available), the plugin logs a warning and returns without modifying the message.

## Changelog

### 1.2.0

- **Fix: Hook now fires.** Switched primary hook from `message_sending` (not fired for tool sends in 2026.2.x) to `before_tool_call`, which reliably intercepts `message` tool sends. `message_sending` is kept as a secondary hook for future compatibility.
- **Fix: Span reconstruction.** The DistilBERT tokenizer produces subword tokens (`##ia`, `##9`, etc.) and `transformers.js` returns `undefined` for `start`/`end` offsets. The previous `indexOf`-per-subword approach matched fragments at wrong positions, causing partial/broken redaction. New approach: group consecutive sensitive subword tokens into word groups, reconstruct the full text fragment, find it case-insensitively in the original text, and extend to word boundaries to catch trailing characters the tokenizer dropped.

### 1.1.0

- Initial release with `message_sending` hook and basic span reconstruction.

## Exporting/Quantizing ONNX

Use the helper script:

```bash
python3 scripts/export_onnx.py
```

Optional push to Hugging Face (requires token):

```bash
HF_TOKEN=... python3 scripts/export_onnx.py --push
```

If `AndrewAndrewsen/distilbert-secret-masker` does not exist on Hugging Face, the script exits with a clear warning.

## Comparison With Regex Plugin

- `@andrewlabs/openclaw-messageguard`: deterministic regex rules and policy actions
- `@andrewlabs/openclaw-messageguard-ml`: learned token classification for broader, context-sensitive detection

Running both can provide layered defense.

## Development

```bash
npm install
npm run build
```
