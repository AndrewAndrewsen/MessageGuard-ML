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

/**
 * Reconstruct character offsets from token-level output when start/end are missing.
 * Uses the original text + word field to find positions.
 */
function resolveSpans(
  content: string,
  entities: RawEntity[],
  threshold: number
): SecretSpan[] {
  const sensitive = entities.filter((e) => {
    const label = normalizedLabel(e);
    const score = typeof e.score === "number" ? e.score : 0;
    return isSensitiveLabel(label) && score >= threshold;
  });

  if (sensitive.length === 0) return [];

  // If entities have start/end, use them directly
  if (
    sensitive.every(
      (e) => typeof e.start === "number" && typeof e.end === "number"
    )
  ) {
    return mergeSpans(
      sensitive.map((e) => ({
        start: e.start!,
        end: e.end!,
        score: e.score ?? 0,
        label: normalizedLabel(e),
      }))
    );
  }

  // Otherwise reconstruct from word tokens by scanning the original text
  const spans: SecretSpan[] = [];
  let cursor = 0;

  for (const entity of sensitive) {
    let word = entity.word ?? "";
    // Remove subword prefix
    if (word.startsWith("##")) word = word.slice(2);

    const idx = content.indexOf(word, cursor);
    if (idx === -1) continue;

    spans.push({
      start: idx,
      end: idx + word.length,
      score: entity.score ?? 0,
      label: normalizedLabel(entity),
    });
    cursor = idx + word.length;
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

    // Merge if adjacent or overlapping (within 0 chars gap for subwords)
    if (current.start <= previous.end) {
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
