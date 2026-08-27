import { test } from 'node:test';
import assert from 'node:assert/strict';
import { aggregateScore, levelToScore, MAX_LEVEL, type CriterionLevels } from './scoring.js';
import { DEFAULT_EMPHASIS } from './schemas.js';

const ALL = (level: number): CriterionLevels => ({
  language_proficiency: level,
  education_level: level,
  experience_years: level,
  technical_skills: level,
  certifications: level,
  soft_skills: level,
  location: level,
});

test('levels map onto the 0-100 scale at even steps', () => {
  assert.equal(levelToScore(0), 0);
  assert.equal(levelToScore(1), 25);
  assert.equal(levelToScore(2), 50);
  assert.equal(levelToScore(3), 75);
  assert.equal(levelToScore(4), 100);
});

test('levels outside the scale are clamped, not trusted', () => {
  assert.equal(levelToScore(-3), 0);
  assert.equal(levelToScore(99), 100);
  assert.equal(levelToScore(2.4), 50);
});

test('MAX_LEVEL is the top of the anchor scale', () => {
  assert.equal(levelToScore(MAX_LEVEL), 100);
});

test('all top levels give a perfect weighted score', () => {
  assert.equal(aggregateScore(ALL(4), DEFAULT_EMPHASIS).score, 100);
});

test('all bottom levels give zero', () => {
  assert.equal(aggregateScore(ALL(0), DEFAULT_EMPHASIS).score, 0);
});

test('the weighted mean respects emphasis', () => {
  // DEFAULT_EMPHASIS weights: language 5, education 10, experience 20,
  // technical 50, certifications 5, soft 20, location 50. Total 160.
  // weighted = (50*5 + 50*10 + 50*20 + 100*50 + 50*5 + 50*20 + 0*50)/160
  //          = (250 + 500 + 1000 + 5000 + 250 + 1000 + 0)/160 = 8000/160 = 50
  const levels: CriterionLevels = {
    language_proficiency: 2,
    education_level: 2,
    experience_years: 2,
    technical_skills: 4,
    certifications: 2,
    soft_skills: 2,
    location: 0,
  };
  assert.equal(aggregateScore(levels, DEFAULT_EMPHASIS).score, 50);
});

test('a zero level on a heavily weighted criterion raises the top red flag', () => {
  const levels = { ...ALL(3), technical_skills: 0 };
  const result = aggregateScore(levels, DEFAULT_EMPHASIS);
  assert.deepEqual(result.redFlags['🚩'], ['Technical Skills']);
  assert.deepEqual(result.redFlags['📍'], []);
});

test('a zero level on a mid-weighted criterion raises the middle flag', () => {
  const levels = { ...ALL(3), experience_years: 0 };
  const result = aggregateScore(levels, DEFAULT_EMPHASIS);
  assert.deepEqual(result.redFlags['📍'], ['Years of Experience']);
});

test('a zero level on a lightly weighted criterion raises the low flag', () => {
  const levels = { ...ALL(3), language_proficiency: 0 };
  const result = aggregateScore(levels, DEFAULT_EMPHASIS);
  assert.deepEqual(result.redFlags['⛳'], ['Language Proficiency']);
});

test('level 1 is a weak signal but not a red flag', () => {
  const levels = { ...ALL(3), technical_skills: 1 };
  const result = aggregateScore(levels, DEFAULT_EMPHASIS);
  assert.deepEqual(result.redFlags['🚩'], []);
});

test('per-criterion 0-100 scores are returned for display', () => {
  const levels: CriterionLevels = {
    language_proficiency: 4,
    education_level: 3,
    experience_years: 2,
    technical_skills: 1,
    certifications: 0,
    soft_skills: 4,
    location: 4,
  };
  const result = aggregateScore(levels, DEFAULT_EMPHASIS);
  assert.equal(result.scores.language_proficiency, 100);
  assert.equal(result.scores.experience_years, 50);
  assert.equal(result.scores.certifications, 0);
});

test('zero total weight yields zero rather than NaN', () => {
  const zeroed = {
    technical_skills_weight: 0,
    soft_skills_weight: 0,
    experience_weight: 0,
    education_weight: 0,
    language_proficiency_weight: 0,
    certifications_weight: 0,
    location_weight: 0,
  };
  assert.equal(aggregateScore(ALL(4), zeroed).score, 0);
});
