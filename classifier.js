import { pipeline } from "@huggingface/transformers";
const classifierByModel = new Map();
function normalizedLabel(entity) {
    return (entity.entity_group ?? entity.entity ?? entity.label ?? "").toUpperCase();
}
function isSensitiveLabel(label) {
    if (!label)
        return false;
    if (label === "O" || label === "LABEL_0" || label === "NON_SECRET")
        return false;
    return true;
}
async function getPipeline(modelId) {
    const existing = classifierByModel.get(modelId);
    if (existing)
        return existing;
    const created = pipeline("token-classification", modelId);
    classifierByModel.set(modelId, created);
    return created;
}
function isWordBoundary(ch) {
    return /[\s,;:!?\[\]{}()'"`<>]/.test(ch);
}
/**
 * Reconstruct character offsets from token-level output when start/end are missing.
 *
 * Strategy: group tokens into "word groups" where each group starts with a
 * non-## token followed by zero or more ## continuation tokens. Reconstruct
 * the text fragment for each group, find it in the original text (case-insensitive),
 * then extend to the next word boundary to catch trailing characters the tokenizer
 * may have dropped. Finally merge adjacent/overlapping spans.
 *
 * transformers.js may omit O-labeled tokens from output, so we treat each
 * non-## token as a potential new word boundary and search forward in the text.
 */
function resolveSpans(content, entities, threshold) {
    // If entities have start/end, filter and merge directly
    if (entities.length > 0 &&
        entities.every((e) => typeof e.start === "number" && typeof e.end === "number")) {
        const sensitive = entities.filter((e) => {
            const label = normalizedLabel(e);
            const score = typeof e.score === "number" ? e.score : 0;
            return isSensitiveLabel(label) && score >= threshold;
        });
        if (sensitive.length === 0)
            return [];
        return mergeSpans(sensitive.map((e) => ({
            start: e.start,
            end: e.end,
            score: e.score ?? 0,
            label: normalizedLabel(e),
        })));
    }
    // Filter to sensitive tokens only
    const sensitive = entities.filter((e) => {
        const label = normalizedLabel(e);
        const score = typeof e.score === "number" ? e.score : 0;
        return isSensitiveLabel(label) && score >= threshold;
    });
    if (sensitive.length === 0)
        return [];
    // Group into word groups: each non-## token starts a new group
    const groups = [];
    let currentGroup = null;
    for (const entity of sensitive) {
        const word = entity.word ?? "";
        const isSubword = word.startsWith("##");
        if (!isSubword) {
            // New word boundary — save previous group and start new one
            if (currentGroup)
                groups.push(currentGroup);
            currentGroup = {
                tokens: [word],
                maxScore: entity.score ?? 0,
                label: normalizedLabel(entity),
            };
        }
        else if (currentGroup) {
            // Continuation of current group
            currentGroup.tokens.push(word);
            currentGroup.maxScore = Math.max(currentGroup.maxScore, entity.score ?? 0);
        }
        else {
            // Orphan subword (no preceding word token) — start new group anyway
            currentGroup = {
                tokens: [word],
                maxScore: entity.score ?? 0,
                label: normalizedLabel(entity),
            };
        }
    }
    if (currentGroup)
        groups.push(currentGroup);
    // For each group, reconstruct the text and find it in the original
    const contentLower = content.toLowerCase();
    const spans = [];
    let searchFrom = 0;
    for (const group of groups) {
        let reconstructed = "";
        for (const token of group.tokens) {
            if (token.startsWith("##")) {
                reconstructed += token.slice(2);
            }
            else {
                reconstructed += token;
            }
        }
        const idx = contentLower.indexOf(reconstructed.toLowerCase(), searchFrom);
        if (idx === -1)
            continue;
        // Extend the span to cover any remaining characters in the same "word"
        // (characters that weren't tokenized, e.g. trailing chars the tokenizer dropped)
        let end = idx + reconstructed.length;
        while (end < content.length && !isWordBoundary(content[end])) {
            end++;
        }
        spans.push({
            start: idx,
            end,
            score: group.maxScore,
            label: group.label,
        });
        searchFrom = end;
    }
    return mergeSpans(spans);
}
function mergeSpans(spans) {
    if (spans.length === 0)
        return spans;
    const ordered = [...spans].sort((a, b) => a.start - b.start || a.end - b.end);
    const merged = [ordered[0]];
    for (let i = 1; i < ordered.length; i++) {
        const current = ordered[i];
        const previous = merged[merged.length - 1];
        // Merge if adjacent or overlapping (within 1 char gap for punctuation like - _ .)
        if (current.start <= previous.end + 1) {
            previous.end = Math.max(previous.end, current.end);
            previous.score = Math.max(previous.score, current.score);
            continue;
        }
        merged.push(current);
    }
    return merged;
}
export async function detectSecrets(content, config) {
    const classifier = (await getPipeline(config.modelId));
    const output = await classifier(content, {
        aggregation_strategy: "none",
    });
    if (!Array.isArray(output))
        return [];
    return resolveSpans(content, output, config.threshold);
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
