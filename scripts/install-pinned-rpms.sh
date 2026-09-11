#!/usr/bin/env bash
# Install the RPMs pinned in spyre-rpms.lock into user space, mirroring the
# spyre-rpm-install action for a dev pod whose image-baked /opt/ibm/spyre
# differs from the locked set.
#
#   bash scripts/install-pinned-rpms.sh
#   source ~/spyre-libs/env.sh
#
# Needs no root and leaves /opt/ibm/spyre alone: payloads are unpacked with
# rpm2cpio and reached by pointing SENTIENT_BASE_INSTALL_DIR at them, so a shell
# that has not sourced env.sh still sees the system libraries.
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 [options]

  -l, --lock FILE     lock file to install (default: spyre-rpms.lock at repo root)
  -p, --prefix DIR    unpack destination (default: \$SPYRE_LIBS or ~/spyre-libs)
  -c, --cache DIR     RPM download cache (default: \$RPM_CACHE_DIR or ~/.cache/spyre-rpms)
  -f, --force         re-unpack even if the destination already matches the lock
  -r, --rebuild       rebuild torch-spyre against the unpacked libraries
  -h, --help          show this message

Environment:
  ARTIFACTORY_TOKEN                            required
  ARTIFACTORY_BASE_URL | ARTIFACTORY_URL       required
  ARTIFACTORY_RPM_PATH | ARTIFACTORY_RPM_REPO  required
  SPYRE_LIBS                                   default for --prefix
  RPM_CACHE_DIR                                default for --cache

--prefix is owned by this script and is wiped when the lock changes, so point it
at a new or previously-installed directory, never at a tree holding anything else.

To install another revision's set, extract its lock file first:

  git show origin/main:spyre-rpms.lock > /tmp/main.lock
  bash $0 --lock /tmp/main.lock
EOF
}

# Not `git rev-parse`: --lock has to work from outside any repo.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="${REPO_ROOT}/spyre-rpms.lock"
PREFIX="${SPYRE_LIBS:-${HOME}/spyre-libs}"
CACHE="${RPM_CACHE_DIR:-${HOME}/.cache/spyre-rpms}"
FORCE=0
REBUILD=0

needs_value() { [[ $2 -ge 2 ]] || { echo "$1 requires a value" >&2; usage >&2; exit 1; }; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--lock) needs_value "$1" $#; LOCK="$2"; shift 2 ;;
        -p|--prefix) needs_value "$1" $#; PREFIX="$2"; shift 2 ;;
        -c|--cache) needs_value "$1" $#; CACHE="$2"; shift 2 ;;
        -f|--force) FORCE=1; shift ;;
        -r|--rebuild) REBUILD=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

: "${ARTIFACTORY_TOKEN:?set ARTIFACTORY_TOKEN (the same secret populate-rpm-cache.yaml uses)}"
BASE_URL="${ARTIFACTORY_BASE_URL:-${ARTIFACTORY_URL:-}}"
RPM_PATH="${ARTIFACTORY_RPM_PATH:-${ARTIFACTORY_RPM_REPO:-}}"
: "${BASE_URL:?set ARTIFACTORY_BASE_URL or ARTIFACTORY_URL}"
: "${RPM_PATH:?set ARTIFACTORY_RPM_PATH or ARTIFACTORY_RPM_REPO}"
LOCATION="${BASE_URL%/}/artifactory/${RPM_PATH}"
RESOLVER="${REPO_ROOT}/.github/scripts/resolve_rpms.py"

ARCH="$(arch)"
[[ "$ARCH" == "arm64" ]] && ARCH="x86_64"

# Arch and location are in the key too, or switching either silently reuses the
# tree built for the other.
STAMP="${PREFIX}/.lock-sha256"
LOCK_SHA="$({ cat "$LOCK"; echo "$ARCH"; echo "$LOCATION"; } | sha256sum | cut -d' ' -f1)"

if [[ $FORCE -eq 0 && -f "$STAMP" && "$(cat "$STAMP")" == "$LOCK_SHA" ]]; then
    echo "${PREFIX} already matches ${LOCK}; pass --force to redo"
    COUNT="$(python3 "$RESOLVER" --lock "$LOCK" names | wc -l)"
else
    # PREFIX is user-supplied and gets wiped below, so refuse anything we did not
    # stamp ourselves.
    if [[ -e "$PREFIX" && ! -f "$STAMP" && -n "$(ls -A "$PREFIX" 2>/dev/null)" ]]; then
        echo "refusing to replace ${PREFIX}: not empty and not installed by this script" >&2
        echo "remove it yourself, or pass --prefix pointing somewhere else" >&2
        exit 1
    fi

    # The lock names commits, not filenames: [overrides.<arch>] wildcards the
    # build number, so resolving needs the same Artifactory query CI uses.
    RESOLVED="$(python3 "$RESOLVER" --lock "$LOCK" resolve --arch "$ARCH")"
    [[ -n "$RESOLVED" ]] || { echo "no packages resolved from $LOCK" >&2; exit 1; }
    mapfile -t RPMS <<< "$RESOLVED"
    COUNT=${#RPMS[@]}

    mkdir -p "$CACHE"
    for relpath in "${RPMS[@]}"; do
        filename="$(basename "$relpath")"
        if [[ -s "${CACHE}/${filename}" ]]; then
            echo "cached   ${filename}"
            continue
        fi
        echo "download ${filename}"
        curl -fsSL -H "Authorization: Bearer ${ARTIFACTORY_TOKEN}" \
            -o "${CACHE}/${filename}.part" "${LOCATION}/${relpath}"
        mv "${CACHE}/${filename}.part" "${CACHE}/${filename}"
    done

    # Unpack into an empty tree so a previous lock's libraries cannot linger.
    rm -rf "$PREFIX"
    mkdir -p "$PREFIX"
    for relpath in "${RPMS[@]}"; do
        rpm2cpio "${CACHE}/$(basename "$relpath")" | (cd "$PREFIX" && cpio -idm 2>/dev/null)
    done
    echo "$LOCK_SHA" > "$STAMP"
fi

BASE_INSTALL_DIR="${PREFIX}/opt/ibm/spyre"
[[ -d "$BASE_INSTALL_DIR" ]] || { echo "unpacked tree has no opt/ibm/spyre: $PREFIX" >&2; exit 1; }

# Clearing _IBM_AIU_SETUP makes ibm-aiu-setup.sh re-derive PATH,
# LD_LIBRARY_PATH and the *_INSTALL_DIR vars from SENTIENT_BASE_INSTALL_DIR
# rather than keeping /opt/ibm/spyre. It reads AIU_SETUP_MULTI_AIU bare, so
# nounset has to come off around it or it aborts any `set -u` caller.
cat > "${PREFIX}/env.sh" <<EOF
export SENTIENT_BASE_INSTALL_DIR="${BASE_INSTALL_DIR}"
export _IBM_AIU_SETUP=
_aiu_nounset="\$(shopt -po nounset || true)"
set +u
source /etc/profile.d/ibm-aiu-setup.sh
eval "\$_aiu_nounset"
unset _aiu_nounset
EOF

# A wheel built against a different RPM set aborts at import with an undefined
# ibm-* symbol. uv keys its wheel cache on the git rev alone, so only the cache
# clean forces a real recompile.
if [[ $REBUILD -eq 1 ]]; then
    echo
    echo "rebuilding torch-spyre against ${BASE_INSTALL_DIR}"
    (
        cd "$REPO_ROOT"
        # shellcheck disable=SC1090
        source "${PREFIX}/env.sh"
        export SEN_COMMON_HEADERS="${SENTIENT_BASE_INSTALL_DIR}/runtime/include"
        uv cache clean torch-spyre
        uv sync --group dev --reinstall-package torch-spyre
    )
fi

cat <<EOF

unpacked ${COUNT} package(s) into ${PREFIX}

  source ${PREFIX}/env.sh

then confirm the pinned libraries are the ones resolved:

  ldd "\$(command -v dxp_standalone)" | grep libdxp
EOF

if [[ $REBUILD -eq 0 ]]; then
    cat <<EOF

torch-spyre must be rebuilt against these libraries or it will fail at import
with an undefined flex::/senlib:: symbol. Re-run with --rebuild, or:

  source ${PREFIX}/env.sh
  export SEN_COMMON_HEADERS="\${SENTIENT_BASE_INSTALL_DIR}/runtime/include"
  uv cache clean torch-spyre
  uv sync --group dev --reinstall-package torch-spyre
EOF
fi
