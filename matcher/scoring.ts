import { CRITERIA } from './criteria.js';
import type { Emphasis } from './schemas.js';

/** Top of the behaviorally anchored scale. Levels run 0..MAX_LEVEL. */
export const MAX_LEVEL = 4;

/**
 * Map an anchored level onto the 0-100 scale the weighted sum expects.
 * Clamped and rounded: the model is not trusted to stay in range.
 */
export function levelToScore(level: number): number {
  const clamped = Math.min(MAX_LEVEL, Math.max(0, Math.round(level)));
  return (clamped / MAX_LEVEL) * 100;
}

export type CriterionLevels = Record<(typeof CRITERIA)[number]['key'], number>;

export interface RedFlags {
  '🚩': string[];
  '📍': string[];
  '⛳': string[];
}

export interface AggregateResult {
  score: number;
  scores: Record<string, number>;
  redFlags: RedFlags;
}

/**
 * Weighted mean of anchored levels, plus red flags for total misses.
 *
 * Still compensatory: strength on one criterion offsets weakness on another.
 * C3 and C4 change that. This function exists so the arithmetic is isolated
 * and testable when they do.
 */
export function aggregateScore(levels: CriterionLevels, emphasis: Emphasis): AggregateResult {
  const redFlags: RedFlags = { '🚩': [], '📍': [], '⛳': [] };
  const scores: Record<string, number> = {};
  let totalScore = 0;
  let totalWeight = 0;

  for (const criterion of CRITERIA) {
    const weight = emphasis[criterion.weightKey];
    const score = levelToScore(levels[criterion.key]);
    scores[criterion.key] = score;
    totalScore += (score * weight) / 100;
    totalWeight += weight;
    if (score === 0) {
      if (weight >= 30) redFlags['🚩'].push(criterion.name);
      else if (weight >= 20) redFlags['📍'].push(criterion.name);
      else redFlags['⛳'].push(criterion.name);
    }
  }

  return {
    score: totalWeight > 0 ? Math.round((totalScore / totalWeight) * 100) : 0,
    scores,
    redFlags,
  };
}
