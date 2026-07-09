#!/usr/bin/env bash

function check_env_var() {
    local var_name="$1"
    local var_value="${!var_name}"
    if [[ -z "${var_value}" ]] ; then
        echo "please set the environment variable: $var_name"
        exit 1
    fi
}

function check_command_exists() {
    local command_name="$1"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "${command_name}: command not found"
        exit 1
    fi
}

if [[ -f ".env" ]] ; then
    echo "found a .env file"
    source .env
fi

if [[ -z "$RPM_ARCH" ]] ; then
    echo "RPM_ARCH is not set, detecting..."
    RPM_ARCH="$(arch)"
    if [[ "$RPM_ARCH" == 'arm64' ]] ; then
        echo 'detected arch arm64, defaulting to x86_64'
        RPM_ARCH="x86_64"
    fi
    echo "RPM_ARCH set to $RPM_ARCH"
fi

check_command_exists curl

check_env_var ARTIFACTORY_TOKEN

RPM_NAMES_TXT="${RPM_NAMES_TXT:-rpms.txt}"

if [[ -z "$RPM_NAMES" ]] ; then
    echo 'RPM_NAMES is empty, looking for RPM_NAMES_TXT'
    check_env_var RPM_NAMES_TXT
    if [[ ! -f "$RPM_NAMES_TXT" ]] ; then
        echo "The RPMs file is missing: ${RPM_NAMES_TXT}"
        exit 1
    fi
    RPM_NAMES="$(grep -v '^[[:space:]]*#' "$RPM_NAMES_TXT" | grep -v '^[[:space:]]*$' | tr '\n' ' ')"
fi

check_env_var RPM_NAMES

if [[ -z "$ARTIFACTORY_LOCATION" ]] ; then
    echo 'ARTIFACTORY_LOCATION is empty, looking for ARTIFACTORY_BASE_URL and ARTIFACTORY_RPM_PATH'
    check_env_var ARTIFACTORY_BASE_URL
    check_env_var ARTIFACTORY_RPM_PATH
    ARTIFACTORY_LOCATION="${ARTIFACTORY_BASE_URL}/artifactory/${ARTIFACTORY_RPM_PATH}"
fi

RPMS_DOWNLOAD_DIR="${RPMS_DOWNLOAD_DIR:-rpms}"

mkdir -p "$RPMS_DOWNLOAD_DIR" || exit 1

echo "downloading rpm(s) for arch '${RPM_ARCH}' from '${ARTIFACTORY_LOCATION}' to '${RPMS_DOWNLOAD_DIR}'..."

for rpm_name in $RPM_NAMES; do
    filename="${rpm_name}.${RPM_ARCH}.rpm"
    url="${ARTIFACTORY_LOCATION}/${RPM_ARCH}/${filename}"
    echo "  ${url}"
    curl -fSL \
        -H "Authorization: Bearer ${ARTIFACTORY_TOKEN}" \
        -o "${RPMS_DOWNLOAD_DIR}/${filename}" \
        "${url}" || exit 1
done

echo 'done'
