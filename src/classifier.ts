import { pipeline } from "@huggingface/transformers";

type RawEntity = {
  entity?: string;
  entity_group?: string;
  label?: string;
  score?: number;
  start?: number;
  end?: number;
  index?: number;
  word?: string;
};

export type SecretSpan = {
  start: number;
  end: number;
  score: number;
  label: string;
};

export type ClassifierConfig = {
  modelId: string;
  threshold: number;
};

type WordGroup = {
  tokens: string[];
  maxScore: number;
  label: string;
};

const classifierByModel = new Map<string, Promise<unknown>>();

function normalizedLabel(entity: RawEntity): string {
  return (entity.entity_group ?? entity.entity ?? entity.label ?? "").toUpperCase();
}

function isSensitiveLabel(label: string): boolean {
  if (!label) return false;
  if (label === "O" || label === "LABEL_0" || label === "NON_SECRET") return false;
  return true;
}

async function getPipeline(modelId: string): Promise<unknown> {
  const existing = classifierByModel.get(modelId);
  if (existing) return existing;
  const created = pipeline("token-classification", modelId as never);
  classifierByModel.set(modelId, created);
  return created;
}

function isWordBoundary(ch: string): boolean {
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
function resolveSpans(
  content: string,
  entities: RawEntity[],
  threshold: number
): SecretSpan[] {
  // If entities have start/end, filter and merge directly
  if (
    entities.length > 0 &&
    entities.every((e) => typeof e.start === "number" && typeof e.end === "number")
  ) {
    const sensitive = entities.filter((e) => {
      const label = normalizedLabel(e);
      const score = typeof e.score === "number" ? e.score : 0;
      return isSensitiveLabel(label) && score >= threshold;
    });
    if (sensitive.length === 0) return [];
    return mergeSpans(
      sensitive.map((e) => ({
        start: e.start!,
        end: e.end!,
        score: e.score ?? 0,
        label: normalizedLabel(e),
      }))
    );
  }

  // Filter to sensitive tokens only
  const sensitive = entities.filter((e) => {
    const label = normalizedLabel(e);
    const score = typeof e.score === "number" ? e.score : 0;
    return isSensitiveLabel(label) && score >= threshold;
  });
  if (sensitive.length === 0) return [];

  // Group into word groups: each non-## token starts a new group
  const groups: WordGroup[] = [];
  let currentGroup: WordGroup | null = null;

  for (const entity of sensitive) {
    const word = entity.word ?? "";
    const isSubword = word.startsWith("##");

    if (!isSubword) {
      // New word boundary — save previous group and start new one
      if (currentGroup) groups.push(currentGroup);
      currentGroup = {
        tokens: [word],
        maxScore: entity.score ?? 0,
        label: normalizedLabel(entity),
      };
    } else if (currentGroup) {
      // Continuation of current group
      currentGroup.tokens.push(word);
      currentGroup.maxScore = Math.max(currentGroup.maxScore, entity.score ?? 0);
    } else {
      // Orphan subword (no preceding word token) — start new group anyway
      currentGroup = {
        tokens: [word],
        maxScore: entity.score ?? 0,
        label: normalizedLabel(entity),
      };
    }
  }
  if (currentGroup) groups.push(currentGroup);

  // For each group, reconstruct the text and find it in the original
  const contentLower = content.toLowerCase();
  const spans: SecretSpan[] = [];
  let searchFrom = 0;

  for (const group of groups) {
    let reconstructed = "";
    for (const token of group.tokens) {
      if (token.startsWith("##")) {
        reconstructed += token.slice(2);
      } else {
        reconstructed += token;
      }
    }

    const idx = contentLower.indexOf(reconstructed.toLowerCase(), searchFrom);
    if (idx === -1) continue;

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

function mergeSpans(spans: SecretSpan[]): SecretSpan[] {
  if (spans.length === 0) return spans;

  const ordered = [...spans].sort((a, b) => a.start - b.start || a.end - b.end);
  const merged: SecretSpan[] = [ordered[0]];

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

export async function detectSecrets(
  content: string,
  config: ClassifierConfig
): Promise<SecretSpan[]> {
  const classifier = (await getPipeline(config.modelId)) as (
    input: string,
    options?: Record<string, unknown>
  ) => Promise<unknown>;

  const output = await classifier(content, {
    aggregation_strategy: "none",
  });

  if (!Array.isArray(output)) return [];

  return resolveSpans(content, output as RawEntity[], config.threshold);
}

export function redactContent(
  content: string,
  spans: SecretSpan[],
  mask: string
): string {
  if (spans.length === 0) return content;

  let redacted = "";
  let cursor = 0;

  for (const span of spans) {
    const start = Math.max(0, Math.min(span.start, content.length));
    const end = Math.max(start, Math.min(span.end, content.length));
    if (start < cursor) continue;

    redacted += content.slice(cursor, start);
    redacted += mask;
    cursor = end;
  }

  redacted += content.slice(cursor);
  return redacted;
}
