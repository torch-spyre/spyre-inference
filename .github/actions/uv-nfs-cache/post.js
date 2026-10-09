// Save phase (runs automatically at job end, like actions/cache's post step).
// The helper picks the scope (main vs pr-<N>) and skips if the key is already
// readable, so this only forwards the context it needs.
'use strict';

const lib = require('./lib');

if (lib.getState('save') !== 'true') {
  console.log('uv cache save disabled for this job; skipping');
  process.exit(0);
}

const root = lib.getState('root');
const local = lib.getState('local');
if (!root) {
  process.exit(0);
}

lib.runHelper([
  'save',
  '--nfs-root', root,
  '--local-dir', local,
  ...lib.keyFileArgs(lib.getState('keyfiles')),
  '--compress', lib.getState('compress') || 'none',
  '--keep', lib.getState('keep') || '5',
  '--event-name', process.env.GITHUB_EVENT_NAME || '',
  '--ref', process.env.GITHUB_REF || '',
  '--pr-number', lib.getState('prnumber'),
]);
