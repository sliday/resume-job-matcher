import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BLOCKING_SEVERITY, evaluateGate, type GateQuestion } from './gate.js';

const QUESTIONS: GateQuestion[] = [
  {
    id: 'work-auth',
    question: 'Is the candidate able to work on-site in Germany?',
    severity: 10,
    why: 'Must be located in Germany. On-site in Berlin 4 days/week.',
  },
  {
    id: 'german-b2',
    question: 'Does the candidate have German at B2 or above?',
    severity: 8,
    why: 'Professional working German (B2+)',
  },
  {
    id: 'fintech',
    question: 'Has the candidate worked in payments or fintech?',
    severity: 4,
    why: 'Experience with payment systems or fintech preferred',
  },
];

test('all PASS clears the gate', () => {
  const result = evaluateGate(QUESTIONS, [
    { id: 'work-auth', verdict: 'PASS', evidence: 'Germany / Berlin' },
    { id: 'german-b2', verdict: 'PASS', evidence: 'German (Native)' },
    { id: 'fintech', verdict: 'PASS', evidence: 'Finpay GmbH' },
  ]);
  assert.equal(result.passed, true);
  assert.deepEqual(result.disqualifiers, []);
});

test('FAIL on a blocking question gates the candidate out and keeps the evidence', () => {
  const result = evaluateGate(QUESTIONS, [
    { id: 'work-auth', verdict: 'FAIL', evidence: 'Brazil / Sao Paulo; not willing to relocate' },
    { id: 'german-b2', verdict: 'PASS', evidence: 'German (Native)' },
  ]);
  assert.equal(result.passed, false);
  assert.equal(result.disqualifiers.length, 1);
  assert.equal(result.disqualifiers[0].id, 'work-auth');
  assert.equal(result.disqualifiers[0].severity, 10);
  assert.equal(result.disqualifiers[0].evidence, 'Brazil / Sao Paulo; not willing to relocate');
});

test('FAIL below the blocking severity is advisory, not blocking', () => {
  const result = evaluateGate(QUESTIONS, [
    { id: 'fintech', verdict: 'FAIL', evidence: 'no payments experience' },
  ]);
  assert.equal(result.passed, true);
  assert.deepEqual(result.disqualifiers, []);
});

test('UNCERTAIN never blocks, because silence is not disqualifying', () => {
  const result = evaluateGate(QUESTIONS, [
    { id: 'work-auth', verdict: 'UNCERTAIN', evidence: 'not stated' },
    { id: 'german-b2', verdict: 'UNCERTAIN', evidence: 'not stated' },
  ]);
  assert.equal(result.passed, true);
});

test('a missing answer does not block', () => {
  const result = evaluateGate(QUESTIONS, []);
  assert.equal(result.passed, true);
});

test('an answer for an unknown question id is ignored', () => {
  const result = evaluateGate(QUESTIONS, [
    { id: 'hallucinated-id', verdict: 'FAIL', evidence: 'whatever' },
  ]);
  assert.equal(result.passed, true);
});

test('every blocking failure is reported, not just the first', () => {
  const result = evaluateGate(QUESTIONS, [
    { id: 'work-auth', verdict: 'FAIL', evidence: 'Brazil' },
    { id: 'german-b2', verdict: 'FAIL', evidence: 'No German.' },
  ]);
  assert.equal(result.passed, false);
  assert.deepEqual(
    result.disqualifiers.map((d) => d.id),
    ['work-auth', 'german-b2'],
  );
});

test('severity exactly at the threshold blocks', () => {
  const questions: GateQuestion[] = [
    { id: 'edge', question: 'Boundary?', severity: BLOCKING_SEVERITY, why: 'boundary' },
  ];
  const result = evaluateGate(questions, [{ id: 'edge', verdict: 'FAIL', evidence: 'no' }]);
  assert.equal(result.passed, false);
});

test('severity one below the threshold does not block', () => {
  const questions: GateQuestion[] = [
    { id: 'edge', question: 'Boundary?', severity: BLOCKING_SEVERITY - 1, why: 'boundary' },
  ];
  const result = evaluateGate(questions, [{ id: 'edge', verdict: 'FAIL', evidence: 'no' }]);
  assert.equal(result.passed, true);
});

test('answers are echoed back on the result for auditability', () => {
  const answers = [{ id: 'work-auth', verdict: 'PASS' as const, evidence: 'Berlin' }];
  const result = evaluateGate(QUESTIONS, answers);
  assert.deepEqual(result.answers, answers);
});
