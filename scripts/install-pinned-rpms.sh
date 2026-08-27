#!/usr/bin/env bash
# Install the RPMs pinned in spyre-rpms.lock into user space, mirroring what the
# spyre-rpm-install action does on CI. Use it in a dev pod whose image-baked
# /opt/ibm/spyre libraries differ from the locked set.
#
#   bash scripts/install-pinned-rpms.sh
#   source ~/spyre-libs/env.sh
#
# Nothing is installed system-wide and /opt/ibm/spyre is untouched: the payloads
# are unpacked with rpm2cpio and picked up by pointing SENTIENT_BASE_INSTALL_DIR
# at the unpacked tree. A shell that has not sourced env.sh sees the system
# libraries, so opening a fresh shell reverts.
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 [options]

  -l, --lock FILE     lock file to install (default: spyre-rpms.lock at repo root)
  -p, --prefix DIR    unpack destination (default: \$SPYRE_LIBS or ~/spyre-libs)
  -c, --cache DIR     RPM download cache (default: ~/.cache/spyre-rpms)
  -f, --force         re-unpack even if the destination already matches the lock
  -h, --help          show this message

Requires ARTIFACTORY_TOKEN plus either ARTIFACTORY_BASE_URL/ARTIFACTORY_RPM_PATH
or ARTIFACTORY_URL/ARTIFACTORY_RPM_REPO in the environment.

To install another revision's set, extract its lock file first:

  git show origin/main:spyre-rpms.lock > /tmp/main.lock
  bash $0 --lock /tmp/main.lock
EOF
}

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
LOCK="${REPO_ROOT}/spyre-rpms.lock"
PREFIX="${SPYRE_LIBS:-${HOME}/spyre-libs}"
CACHE="${RPM_CACHE_DIR:-${HOME}/.cache/spyre-rpms}"
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--lock) LOCK="$2"; shift 2 ;;
        -p|--prefix) PREFIX="$2"; shift 2 ;;
        -c|--cache) CACHE="$2"; shift 2 ;;
        -f|--force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

: "${ARTIFACTORY_TOKEN:?set ARTIFACTORY_TOKEN (the same secret populate-rpm-cache.yaml uses)}"
BASE_URL="${ARTIFACTORY_BASE_URL:-${ARTIFACTORY_URL:-}}"
RPM_PATH="${ARTIFACTORY_RPM_PATH:-${ARTIFACTORY_RPM_REPO:-}}"
: "${BASE_URL:?set ARTIFACTORY_BASE_URL or ARTIFACTORY_URL}"
: "${RPM_PATH:?set ARTIFACTORY_RPM_PATH or ARTIFACTORY_RPM_REPO}"
# next/ is the only tree publishing every package the lock pins; RPM_PREFIX=''
# selects the base <repo>/<arch>/ tree instead.
LOCATION="${BASE_URL%/}/artifactory/${RPM_PATH}/${RPM_PREFIX-next}"
LOCATION="${LOCATION%/}"

ARCH="$(arch)"
[[ "$ARCH" == "arm64" ]] && ARCH="x86_64"

mapfile -t RPMS < <(grep -v '^[[:space:]]*#' "$LOCK" | grep -v '^[[:space:]]*$')
[[ ${#RPMS[@]} -gt 0 ]] || { echo "no packages listed in $LOCK" >&2; exit 1; }

STAMP="${PREFIX}/.lock-sha256"
LOCK_SHA="$(sha256sum < "$LOCK" | cut -d' ' -f1)"
if [[ $FORCE -eq 0 && -f "$STAMP" && "$(cat "$STAMP")" == "$LOCK_SHA" ]]; then
    echo "${PREFIX} already matches ${LOCK}; pass --force to redo"
else
    mkdir -p "$CACHE"
    for name in "${RPMS[@]}"; do
        filename="${name}.${ARCH}.rpm"
        if [[ -s "${CACHE}/${filename}" ]]; then
            echo "cached   ${filename}"
            continue
        fi
        echo "download ${filename}"
        curl -fsSL -H "Authorization: Bearer ${ARTIFACTORY_TOKEN}" \
            -o "${CACHE}/${filename}.part" "${LOCATION}/${ARCH}/${filename}"
        mv "${CACHE}/${filename}.part" "${CACHE}/${filename}"
    done

    # Unpack into an empty tree so a previous lock's libraries cannot linger.
    rm -rf "$PREFIX"
    mkdir -p "$PREFIX"
    for name in "${RPMS[@]}"; do
        rpm2cpio "${CACHE}/${name}.${ARCH}.rpm" | (cd "$PREFIX" && cpio -idm 2>/dev/null)
    done
    echo "$LOCK_SHA" > "$STAMP"
fi

BASE_INSTALL_DIR="${PREFIX}/opt/ibm/spyre"
[[ -d "$BASE_INSTALL_DIR" ]] || { echo "unpacked tree has no opt/ibm/spyre: $PREFIX" >&2; exit 1; }

# Clearing _IBM_AIU_SETUP makes ibm-aiu-setup.sh re-derive DEEPTOOLS_INSTALL_DIR
# and the rest from SENTIENT_BASE_INSTALL_DIR instead of keeping /opt/ibm/spyre.
cat > "${PREFIX}/env.sh" <<EOF
export SENTIENT_BASE_INSTALL_DIR="${BASE_INSTALL_DIR}"
export _IBM_AIU_SETUP=
source /etc/profile.d/ibm-aiu-setup.sh
EOF

cat <<EOF

unpacked ${#RPMS[@]} package(s) into ${PREFIX}

  source ${PREFIX}/env.sh

then confirm the pinned libraries are the ones resolved:

  ldd "\$(command -v dxp_standalone)" | grep libdxp
EOF
