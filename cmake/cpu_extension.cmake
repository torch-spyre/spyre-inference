# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions adapted from vLLM (https://github.com/vllm-project/vllm),
# Copyright contributors to the vLLM project, Apache-2.0.

# Compiles only SPYRE_EXT_SRC from vLLM's csrc/, for x86_64 (AVX512 + AVX2),
# ppc64le (VSX) and s390x (VXE).

include(FetchContent)

# vLLM's Python calls some of these ops, so build from the rev pyproject.toml installs.
if(NOT SPYRE_VLLM_REV)
  run_python(SPYRE_VLLM_REV
    "
import tomllib
with open('${CMAKE_SOURCE_DIR}/pyproject.toml', 'rb') as f:
    sources = tomllib.load(f)['tool']['uv']['sources']['vllm']
revs = {s['rev'] for s in (sources if isinstance(sources, list) else [sources])}
assert len(revs) == 1, f'expected one vllm rev in [tool.uv.sources], got {revs}'
print(revs.pop())
"
    "Failed to read the vllm rev from pyproject.toml")
endif()
message(STATUS "Building vLLM host kernels from vLLM ${SPYRE_VLLM_REV}")

# A pre-fetched checkout can be passed with -DFETCHCONTENT_SOURCE_DIR_VLLM=<dir>.
FetchContent_Declare(vllm
  URL "https://github.com/vllm-project/vllm/archive/${SPYRE_VLLM_REV}.tar.gz"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  # Nonexistent, so vLLM's own CMake is never configured.
  SOURCE_SUBDIR spyre-no-cmake)
FetchContent_MakeAvailable(vllm)
message(STATUS "vLLM host kernel sources: ${vllm_SOURCE_DIR}")

execute_process(
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/check_schemas.py"
          "${CMAKE_SOURCE_DIR}/csrc/torch_bindings.cpp"
          "${vllm_SOURCE_DIR}/csrc/cpu/torch_bindings.cpp"
  RESULT_VARIABLE SCHEMA_CHECK_RET)
if(NOT SCHEMA_CHECK_RET EQUAL 0)
  message(FATAL_ERROR "csrc/torch_bindings.cpp no longer matches vLLM ${SPYRE_VLLM_REV}.")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Drops utils.cpp's NUMA binding and its libnuma link.
list(APPEND CXX_COMPILE_FLAGS "-fopenmp" "-DVLLM_CPU_EXTENSION" "-DVLLM_NUMA_DISABLED")

# Link the libgomp torch loads, falling back to the toolchain's when torch was
# built from source (Power) or distro-packaged and vendors none.
prepare_torch_gomp_shim(TORCH_GOMP_SHIM_DIR)
if(TORCH_GOMP_SHIM_DIR)
  find_library(OPEN_MP NAMES gomp PATHS "${TORCH_GOMP_SHIM_DIR}" NO_DEFAULT_PATH REQUIRED)
else()
  find_library(OPEN_MP NAMES gomp REQUIRED)
endif()

execute_process(COMMAND cat /proc/cpuinfo
                RESULT_VARIABLE CPUINFO_RET
                OUTPUT_VARIABLE CPUINFO)
if (NOT CPUINFO_RET EQUAL 0)
  message(FATAL_ERROR "Failed to check CPU features via /proc/cpuinfo")
endif()

function (find_isa CPUINFO TARGET OUT)
  string(FIND ${CPUINFO} ${TARGET} ISA_FOUND)
  if(NOT ISA_FOUND EQUAL -1)
    set(${OUT} ON PARENT_SCOPE)
  else()
    set(${OUT} OFF PARENT_SCOPE)
  endif()
endfunction()

find_isa(${CPUINFO} "Power11" POWER11_FOUND)
find_isa(${CPUINFO} "POWER10" POWER10_FOUND)
find_isa(${CPUINFO} "S390" S390_FOUND)

set(SPYRE_EXT_SRC
  "${vllm_SOURCE_DIR}/csrc/cpu/sampling_kernels.cpp"
  "${vllm_SOURCE_DIR}/csrc/cpu/utils.cpp"
  "csrc/torch_bindings.cpp")
set(SPYRE_EXT_INCLUDES "${vllm_SOURCE_DIR}/csrc")

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
  if (NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
          CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.3))
    message(FATAL_ERROR "X86 backend requires gcc/g++ >= 12.3")
  endif()
  list(APPEND CXX_COMPILE_FLAGS "-mf16c")
  set(CXX_COMPILE_FLAGS_AVX512 ${CXX_COMPILE_FLAGS}
    "-mavx512f"
    "-mavx512vl"
    "-mavx512bw"
    "-mavx512dq")
  set(CXX_COMPILE_FLAGS_AVX2 ${CXX_COMPILE_FLAGS} "-mavx2")

  # Both are installed; spyre_inference picks one at import from the host ISA.
  define_extension_target(
    _C
    DESTINATION spyre_inference
    SOURCES ${SPYRE_EXT_SRC}
    INCLUDE_DIRECTORIES ${SPYRE_EXT_INCLUDES}
    LIBRARIES ${OPEN_MP}
    COMPILE_FLAGS ${CXX_COMPILE_FLAGS_AVX512})
  define_extension_target(
    _C_AVX2
    DESTINATION spyre_inference
    SOURCES ${SPYRE_EXT_SRC}
    INCLUDE_DIRECTORIES ${SPYRE_EXT_INCLUDES}
    LIBRARIES ${OPEN_MP}
    COMPILE_FLAGS ${CXX_COMPILE_FLAGS_AVX2})
  return()
endif()

if (POWER10_FOUND OR POWER11_FOUND)
  list(APPEND CXX_COMPILE_FLAGS "-mvsx" "-mcpu=power10" "-mtune=power10")
elseif (S390_FOUND)
  list(APPEND CXX_COMPILE_FLAGS "-mvx" "-mzvector" "-march=z15" "-mtune=z15")
else()
  message(FATAL_ERROR "spyre-inference CPU kernels require x86_64, Power10+ or s390x.")
endif()

define_extension_target(
  _C
  DESTINATION spyre_inference
  SOURCES ${SPYRE_EXT_SRC}
  INCLUDE_DIRECTORIES ${SPYRE_EXT_INCLUDES}
  LIBRARIES ${OPEN_MP}
  COMPILE_FLAGS ${CXX_COMPILE_FLAGS})
