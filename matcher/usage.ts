import { generateObject as sdkGenerateObject, generateText as sdkGenerateText } from 'ai';

export interface TokenUsage {
  input: number;
  output: number;
  calls: number;
}

const tokenUsage: TokenUsage = { input: 0, output: 0, calls: 0 };

function record(usage: { inputTokens?: number; outputTokens?: number } | undefined): void {
  tokenUsage.calls += 1;
  tokenUsage.input += usage?.inputTokens ?? 0;
  tokenUsage.output += usage?.outputTokens ?? 0;
}

// Wrapped so every call site accumulates usage without threading a counter
// through each function signature.
export const generateObject = (async (options: any) => {
  const result = await sdkGenerateObject(options);
  record(result.usage);
  return result;
}) as unknown as typeof sdkGenerateObject;

export const generateText = (async (options: any) => {
  const result = await sdkGenerateText(options);
  record(result.usage);
  return result;
}) as unknown as typeof sdkGenerateText;

/**
 * Embeddings bill input tokens only and go through embedMany, which the wrappers
 * above do not cover. Without this the reported cost silently excludes every
 * --prefilter run, while still being labelled the run total.
 */
export function recordEmbedding(tokens: number | undefined): void {
  tokenUsage.calls += 1;
  tokenUsage.input += tokens ?? 0;
}

export function getTokenUsage(): TokenUsage {
  return { ...tokenUsage };
}

// Published per-1M-token rates; unknown models report token counts only.
const PRICE_PER_MTOK: Record<string, { in: number; out: number }> = {
  'gpt-5.6-luna': { in: 0.2, out: 1.2 },
  'gpt-5-mini': { in: 0.25, out: 2 },
  'gpt-5': { in: 1.25, out: 10 },
  'gpt-4o-mini': { in: 0.15, out: 0.6 },
};

export function estimateCostUsd(model: string, usage: TokenUsage): number | null {
  const price = PRICE_PER_MTOK[model];
  if (!price) return null;
  return (usage.input * price.in + usage.output * price.out) / 1_000_000;
}
