import { test } from 'node:test';
import assert from 'node:assert/strict';
import { jobSlug } from './config.js';

test('a plain filename becomes its stem', () => {
  assert.equal(jobSlug('backend.txt'), 'backend');
});

test('a path is reduced to the filename, not the directory', () => {
  assert.equal(jobSlug('/tmp/jobs/senior-backend.txt'), 'senior-backend');
});

test('spaces and punctuation collapse to single hyphens', () => {
  assert.equal(jobSlug('EXAMPLE job_description.txt'), 'example-job-description');
});

test('leading and trailing separators are trimmed', () => {
  assert.equal(jobSlug('__weird--name__.txt'), 'weird-name');
});

test('two jobs with different names never share a directory', () => {
  assert.notEqual(jobSlug('berlin-backend.txt'), jobSlug('london-frontend.txt'));
});

test('a name with no usable characters still yields a directory', () => {
  assert.equal(jobSlug('!!!.txt'), 'job');
  assert.equal(jobSlug(''), 'job');
});

test('very long names are capped so the path stays usable', () => {
  const slug = jobSlug(`${'a'.repeat(200)}.txt`);
  assert.equal(slug.length, 60);
});

test('case is normalized, so JOB.txt and job.txt land together', () => {
  assert.equal(jobSlug('JOB.txt'), jobSlug('job.txt'));
});
