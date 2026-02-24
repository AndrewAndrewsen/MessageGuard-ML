import { pipeline } from "@huggingface/transformers";
const classifierByModel = new Map();
function normalizedLabel(entity) {
    return (entity.entity_group ?? entity.entity ?? entity.label ?? "").toUpperCase();
}
function isSensitiveLabel(label) {
    if (!label)
        return false;
    // Common "non-sensitive" labels for token classification tasks.
    if (label === "O" || label === "LABEL_0" || label === "NON_SECRET") {
        return false;
    }
    return true;
}
function mergeSpans(spans) {
    if (spans.length === 0)
        return spans;
    const ordered = [...spans].sort((a, b) => a.start - b.start || a.end - b.end);
    const merged = [ordered[0]];
    for (let i = 1; i < ordered.length; i += 1) {
        const current = ordered[i];
        const previous = merged[merged.length - 1];
        if (current.start <= previous.end) {
            previous.end = Math.max(previous.end, current.end);
            previous.score = Math.max(previous.score, current.score);
            continue;
        }
        merged.push(current);
    }
    return merged;
}
async function getPipeline(modelId) {
    const existing = classifierByModel.get(modelId);
    if (existing) {
        return existing;
    }
    const created = pipeline("token-classification", modelId);
    classifierByModel.set(modelId, created);
    return created;
}
export async function detectSecrets(content, config) {
    const classifier = (await getPipeline(config.modelId));
    const output = await classifier(content, {
        // Keep output per-token spans so we can map to original text indices.
        aggregation_strategy: "none",
    });
    if (!Array.isArray(output)) {
        return [];
    }
    const spans = [];
    for (const item of output) {
        if (typeof item.start !== "number" || typeof item.end !== "number")
            continue;
        const label = normalizedLabel(item);
        const score = typeof item.score === "number" ? item.score : 0;
        if (!isSensitiveLabel(label))
            continue;
        if (score < config.threshold)
            continue;
        spans.push({
            start: item.start,
            end: item.end,
            score,
            label,
        });
    }
    return mergeSpans(spans);
}
export function redactContent(content, spans, mask) {
    if (spans.length === 0)
        return content;
    let redacted = "";
    let cursor = 0;
    for (const span of spans) {
        const start = Math.max(0, Math.min(span.start, content.length));
        const end = Math.max(start, Math.min(span.end, content.length));
        if (start < cursor)
            continue;
        redacted += content.slice(cursor, start);
        redacted += mask;
        cursor = end;
    }
    redacted += content.slice(cursor);
    return redacted;
}
