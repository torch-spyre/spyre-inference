#!/usr/bin/env bash
# Single source of truth for the unit/integration/regression tier aliases,
# shared by the Makefile TEST_TYPE resolution and _test_matrix.yaml's
# "Decide whether to run" gate step so both entry points apply the same
# mapping.
#
# Usage: resolve_test_type.sh TEST_TYPE
# Empty/no arg resolves to "full" (this repo's own default label; callers
# that want the user-facing "regression" default pass it explicitly).

set -euo pipefail

RAW_TEST_TYPE="${1:-}"

case "$RAW_TEST_TYPE" in
    unit)        echo core ;;
    integration) echo smoke ;;
    regression)  echo full ;;
    '')          echo full ;;
    *)           echo "$RAW_TEST_TYPE" ;;
esac
