import { test } from 'node:test';
import assert from 'node:assert/strict';
import { cosineSimilarity, pendingEmbedIndices, selectTopK } from './prefilter.js';

// cosine goes through two sqrt calls, so exact float equality is the wrong
// assertion: [1,1] vs [10,10] lands on 0.9999999999999998.
const near = (actual: number, expected: number, msg: string) =>
  assert.ok(Math.abs(actual - expected) < 1e-12, `${msg}: got ${actual}, want ~${expected}`);

test('identical vectors have similarity 1', () => {
  near(cosineSimilarity([1, 2, 3], [1, 2, 3]), 1, 'identical');
});

test('orthogonal vectors have similarity 0', () => {
  near(cosineSimilarity([1, 0], [0, 1]), 0, 'orthogonal');
});

test('opposed vectors have similarity -1', () => {
  near(cosineSimilarity([1, 0], [-1, 0]), -1, 'opposed');
});

test('magnitude does not affect similarity, only direction', () => {
  near(cosineSimilarity([1, 1], [10, 10]), 1, 'scaled');
});

test('a zero vector yields 0 rather than NaN', () => {
  assert.equal(cosineSimilarity([0, 0], [1, 2]), 0);
  assert.equal(cosineSimilarity([1, 2], [0, 0]), 0);
  assert.equal(cosineSimilarity([0, 0], [0, 0]), 0);
});

test('mismatched dimensions are a programming error, not a silent 0', () => {
  assert.throws(() => cosineSimilarity([1, 2], [1, 2, 3]), /dimension/i);
});

test('selectTopK returns the highest scores first', () => {
  const picked = selectTopK(
    [
      { key: 'a', score: 0.1 },
      { key: 'b', score: 0.9 },
      { key: 'c', score: 0.5 },
    ],
    2,
  );
  assert.deepEqual(
    picked.map((p) => p.key),
    ['b', 'c'],
  );
});

test('k larger than the pool returns the whole pool, still ranked', () => {
  const picked = selectTopK(
    [
      { key: 'a', score: 0.1 },
      { key: 'b', score: 0.9 },
    ],
    99,
  );
  assert.deepEqual(
    picked.map((p) => p.key),
    ['b', 'a'],
  );
});

test('k of zero or less disables the filter and returns everything ranked', () => {
  const items = [
    { key: 'a', score: 0.1 },
    { key: 'b', score: 0.9 },
  ];
  assert.equal(selectTopK(items, 0).length, 2);
  assert.equal(selectTopK(items, -5).length, 2);
});

test('selectTopK does not mutate its input', () => {
  const items = [
    { key: 'a', score: 0.1 },
    { key: 'b', score: 0.9 },
  ];
  selectTopK(items, 1);
  assert.deepEqual(
    items.map((i) => i.key),
    ['a', 'b'],
  );
});

test('an empty pool is not an error', () => {
  assert.deepEqual(selectTopK([], 5), []);
});

test('nothing cached means every text needs embedding', () => {
  assert.deepEqual(pendingEmbedIndices(['a', 'b', 'c'], new Set()), [0, 1, 2]);
});

test('cached keys are skipped', () => {
  assert.deepEqual(pendingEmbedIndices(['a', 'b', 'c'], new Set(['b'])), [0, 2]);
});

test('a duplicate text in one batch is embedded once, at its first position', () => {
  assert.deepEqual(pendingEmbedIndices(['a', 'b', 'a', 'a'], new Set()), [0, 1]);
});

test('a duplicate of an already-cached key is skipped entirely', () => {
  assert.deepEqual(pendingEmbedIndices(['a', 'a', 'b'], new Set(['a'])), [2]);
});

test('everything cached means no embedding call at all', () => {
  assert.deepEqual(pendingEmbedIndices(['a', 'b'], new Set(['a', 'b'])), []);
});

test('an empty batch is not an error', () => {
  assert.deepEqual(pendingEmbedIndices([], new Set(['a'])), []);
});
