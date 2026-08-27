import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { embedMany } from 'ai';
import { recordEmbedding } from './usage.js';
import type { EmbeddingModel } from 'ai';

/**
 * Cosine similarity of two equal-length vectors.
 *
 * A zero vector has no direction, so it scores 0 rather than NaN. A dimension
 * mismatch is a programming error and throws instead of silently scoring 0,
 * which would look like "no match" and hide the bug.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Vector dimension mismatch: ${a.length} vs ${b.length}`);
  }
  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

export interface ScoredKey {
  key: string;
  score: number;
}

/**
 * Rank descending and keep the best k. k <= 0 disables truncation but still
 * ranks, so callers can use the ordering without the cut. Does not mutate input.
 */
export function selectTopK(items: ScoredKey[], k: number): ScoredKey[] {
  const ranked = [...items].sort((x, y) => y.score - x.score);
  return k > 0 ? ranked.slice(0, k) : ranked;
}

const CACHE_FILE = path.join('out', '.embedding-cache.json');

// text-embedding-3-small caps at 8191 tokens. ~4 chars/token with headroom, so a
// long CV is truncated rather than rejecting the batch it travels in.
const MAX_EMBED_CHARS = 24000;

function cacheKey(modelId: string, text: string): string {
  return createHash('sha256').update(`${modelId}\u0000${text}`).digest('hex');
}

/**
 * Indices still needing an embedding call: not already cached, and only the first
 * occurrence of each distinct key. The same resume can appear twice in one batch,
 * and embedding it twice would be paying twice for one vector.
 */
export function pendingEmbedIndices(keys: string[], cached: Set<string>): number[] {
  const seen = new Set<string>();
  const pending: number[] = [];
  for (let i = 0; i < keys.length; i += 1) {
    if (cached.has(keys[i]) || seen.has(keys[i])) continue;
    seen.add(keys[i]);
    pending.push(i);
  }
  return pending;
}

async function readCache(): Promise<Record<string, number[]>> {
  try {
    return JSON.parse(await readFile(CACHE_FILE, 'utf-8'));
  } catch {
    return {};
  }
}

/**
 * Embed texts, reusing anything already embedded under the same model.
 *
 * The cache is keyed on content, not filename, so screening the same candidate
 * against a second job costs nothing. That is the difference between N jobs x M
 * CVs embeddings and just M.
 */
export async function embedWithCache(
  model: EmbeddingModel,
  modelId: string,
  texts: string[],
): Promise<{ vectors: number[][]; embedded: number; reused: number }> {
  const cache = await readCache();
  const keys = texts.map((t) => cacheKey(modelId, t));
  const cached = new Set(Object.keys(cache));

  const missingIndices = pendingEmbedIndices(keys, cached);
  // Counted separately from in-batch duplicates: "reused from cache" should mean
  // exactly that, not "we deduplicated your input".
  const cacheHits = keys.filter((k) => cached.has(k)).length;

  if (missingIndices.length > 0) {
    const { embeddings, usage } = await embedMany({
      model,
      // One over-long resume would reject the whole batch and abort the run, losing
      // every valid candidate. Topical ranking does not need the tail of a CV.
      values: missingIndices.map((i) => texts[i].slice(0, MAX_EMBED_CHARS)),
    });
    recordEmbedding(usage?.tokens);
    missingIndices.forEach((i, n) => {
      // 6dp keeps ~5 significant digits on values around ±0.05, which moves cosine
      // by ~1e-5 (irrelevant for ranking) and roughly halves the cache on disk.
      // At 1000 CVs that is ~15MB instead of ~29MB.
      cache[keys[i]] = embeddings[n].map((v) => Math.round(v * 1e6) / 1e6);
    });
    await mkdir(path.dirname(CACHE_FILE), { recursive: true });
    // Write-then-rename: a crash partway through a direct write leaves invalid JSON,
    // and readCache would swallow that and silently re-embed the entire pool.
    const tmp = `${CACHE_FILE}.${process.pid}.tmp`;
    await writeFile(tmp, JSON.stringify(cache));
    await rename(tmp, CACHE_FILE);
  }

  return {
    vectors: keys.map((k) => cache[k]),
    embedded: missingIndices.length,
    reused: cacheHits,
  };
}
