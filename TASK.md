# Build @andrewlabs/openclaw-messageguard-ml

An OpenClaw plugin that uses a DistilBERT model (via transformers.js / ONNX) to detect secrets and sensitive data in outgoing messages. This is a companion to the regex-based `@andrewlabs/openclaw-messageguard`.

## Architecture

1. **ONNX Model Export Script** (`scripts/export_onnx.py`)
   - Uses `optimum` to export HuggingFace model `AndrewAndrewsen/distilbert-secret-masker` to ONNX
   - Quantizes to int8 for smaller download (~25-30MB)
   - Outputs to `model/` directory
   - The model is a token classification (NER) model that tags tokens as secret/not-secret

2. **OpenClaw Plugin** (TypeScript, ESM)
   - `src/index.ts` — plugin entry, registers `message_sending` hook
   - `src/classifier.ts` — loads ONNX model via `@huggingface/transformers` (formerly `@xenova/transformers`), runs inference
   - On first run, downloads model from HuggingFace Hub automatically (cached locally after)
   - Hook: for each outgoing message, run the model. If tokens are classified as secrets, mask them with `[REDACTED]`

3. **package.json**
   - name: `@andrewlabs/openclaw-messageguard-ml`
   - version: `1.0.0`
   - dependencies: `@huggingface/transformers` (for ONNX inference in Node.js)
   - peerDependencies: `openclaw: >=2026.0.0` (optional)
   - author: AndrewAndersen
   - license: MIT
   - repo: https://github.com/AndrewAndrewsen/MessageGuard

4. **openclaw.plugin.json** — manifest pointing to index.js

5. **README.md** — install instructions, how it works, comparison with regex version

## Key Details

- Use `@huggingface/transformers` (v3+) NOT `@xenova/transformers` (deprecated)
- Model ID on HuggingFace: `AndrewAndrewsen/distilbert-secret-masker`
- If the model doesn't exist on HF yet, the export script should create and upload it
- The plugin should fail gracefully: if model download fails, log warning and pass messages through unfiltered
- TypeScript with ESM (`"type": "module"`)
- The plugin hooks `message_sending` (same hook name as the regex version)
- Plugin ID: `messageguard-ml`

## Reference: Existing regex plugin structure

Look at ~/openclaw-messageguard/ for the structure of the regex-based plugin (types, hook registration pattern, openclaw.plugin.json format).

## Export Script Details

```python
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer
import onnxruntime

model_id = "AndrewAndrewsen/distilbert-secret-masker"
ort_model = ORTModelForTokenClassification.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Quantize
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
quantizer = ORTQuantizer.from_pretrained(ort_model)
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)  # dynamic quant
quantizer.quantize(save_dir="./model_quantized", quantization_config=qconfig)

# Push to hub as ONNX variant
ort_model.push_to_hub(model_id, use_auth_token="...", subfolder="onnx")
```

NOTE: If `AndrewAndrewsen/distilbert-secret-masker` doesn't exist on HuggingFace yet, the export script should note this. The classifier.ts should handle the case where the model isn't available yet by logging a warning.

## Deliverables

1. Complete TypeScript plugin in src/ with build config
2. Export script in scripts/
3. README.md
4. openclaw.plugin.json
5. package.json ready for `npm publish`
6. Build should succeed: `npm install && npm run build`
