# @andrewlabs/openclaw-messageguard-ml

ML-powered companion plugin to `@andrewlabs/openclaw-messageguard`.

This plugin uses a DistilBERT token classification model via `@huggingface/transformers` (ONNX Runtime under the hood) to detect sensitive content in outgoing messages and replace detected spans with `[REDACTED]`.

## Features

- OpenClaw `message_sending` hook integration
- Automatic model download from Hugging Face on first use (then cached locally)
- Fails open: if model download/inference fails, message passes through unchanged
- Configurable model id, confidence threshold, and redaction token

## Install

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

1. On plugin startup, `message_sending` hook is registered.
2. For each outgoing message, the model runs token classification.
3. Spans predicted as sensitive at/above threshold are merged and redacted.
4. Sanitized content is returned to OpenClaw.

If model loading or inference fails (for example, model repo not yet available), the plugin logs a warning and returns without modifying the message.

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
