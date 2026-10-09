// Restore phase: populate a local UV_CACHE_DIR from the shared mount and hand
// its path to uv via GITHUB_ENV. Reads nothing back from the job; stashes what
// the post (save) phase needs into GITHUB_STATE, since inputs are not re-exposed
// to post steps.
'use strict';

const fs = require('fs');
const lib = require('./lib');

const root = lib.getInput('storage-root');
const local = lib.resolveLocal();

// The cache scope is main vs pr-<N>; the PR number comes from the event payload
// (GitHub-assigned, so a fork cannot forge it). push/dispatch have none.
let prNumber = '';
try {
  const event = JSON.parse(fs.readFileSync(process.env.GITHUB_EVENT_PATH, 'utf8'));
  prNumber = event.pull_request ? String(event.pull_request.number) : '';
} catch (_) {
  // no pull_request payload; leave empty.
}

// Carry everything the save needs; post steps see STATE_* but not INPUT_*.
lib.saveState('root', root || '');
lib.saveState('local', local);
lib.saveState('save', lib.getInput('save', 'true'));
lib.saveState('compress', lib.getInput('compress', 'none'));
lib.saveState('keep', lib.getInput('keep', '5'));
lib.saveState('keyfiles', lib.getInput('key-files', 'uv.lock\nspyre-rpms.lock'));
lib.saveState('prnumber', prNumber);

if (!root) {
  lib.warn('storage-root input is empty; uv uses its default cache dir');
  process.exit(0);
}

lib.runHelper([
  'restore',
  '--nfs-root', root,
  '--local-dir', local,
  ...lib.keyFileArgs(),
  '--event-name', process.env.GITHUB_EVENT_NAME || '',
  '--ref', process.env.GITHUB_REF || '',
  '--pr-number', prNumber,
  '--github-env', process.env.GITHUB_ENV || '',
  '--github-output', process.env.GITHUB_OUTPUT || '',
]);
