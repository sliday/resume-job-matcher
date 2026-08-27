import { test } from 'node:test';
import assert from 'node:assert/strict';
import { getScoreDetails, SCORE_RANGES } from './bands.js';

// Ported from PR #14, which hand-picked five bands to check. Asserting the
// invariant over the whole domain is stronger and is what C4 must preserve
// when it collapses the table.

test('every integer score 0-100 lands in a band', () => {
  const unmatched: number[] = [];
  for (let score = 0; score <= 100; score += 1) {
    if (getScoreDetails(score).label === 'Unable to score') unmatched.push(score);
  }
  assert.deepEqual(unmatched, []);
});

test('every integer score 0-100 gets a non-empty label, colour and emoji', () => {
  for (let score = 0; score <= 100; score += 1) {
    const details = getScoreDetails(score);
    assert.ok(details.label.length > 0, `empty label at ${score}`);
    assert.ok(details.color.length > 0, `empty colour at ${score}`);
    assert.ok(details.emoji.length > 0, `empty emoji at ${score}`);
  }
});

test('bands do not overlap: each score matches exactly one range', () => {
  for (let score = 0; score <= 100; score += 1) {
    const hits = SCORE_RANGES.filter((r) => score >= r.min && score < r.max);
    assert.equal(hits.length, 1, `score ${score} matched ${hits.length} ranges`);
  }
});

test('the extremes resolve to the intended ends of the table', () => {
  assert.equal(getScoreDetails(100).label, 'Legendary Unicorn');
  assert.equal(getScoreDetails(0).label, 'No Match');
});

test('band boundaries are half-open, so the lower bound belongs to its own band', () => {
  // 90 is the min of "Great Potential" and the max of "Very Promising".
  assert.equal(getScoreDetails(90).label, 'Great Potential');
  assert.equal(getScoreDetails(89).label, 'Very Promising');
});

test('scores outside 0-100 fall through to the sentinel rather than a wrong band', () => {
  assert.equal(getScoreDetails(-1).label, 'Unable to score');
  assert.equal(getScoreDetails(101).label, 'Unable to score');
});

test('ranges are ordered high to low, which the linear scan relies on', () => {
  for (let i = 1; i < SCORE_RANGES.length; i += 1) {
    assert.ok(
      SCORE_RANGES[i].min < SCORE_RANGES[i - 1].min,
      `range ${i} (min ${SCORE_RANGES[i].min}) is not below range ${i - 1} (min ${SCORE_RANGES[i - 1].min})`,
    );
  }
});
