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

# Host CPU kernels, ported from vLLM's cmake/cpu_extension.cmake for the ISAs
# spyre-inference ships on: x86_64 (AVX512 + AVX2 fallback), ppc64le (VSX) and
# s390x (VXE). Keep the flags in sync with vLLM when resyncing csrc/cpu.

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CXX_COMPILE_FLAGS "-fopenmp")

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
  "csrc/cpu/sampling_kernels.cpp"
  "csrc/cpu/torch_bindings.cpp")

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
    LIBRARIES ${OPEN_MP}
    COMPILE_FLAGS ${CXX_COMPILE_FLAGS_AVX512})
  define_extension_target(
    _C_AVX2
    DESTINATION spyre_inference
    SOURCES ${SPYRE_EXT_SRC}
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
  LIBRARIES ${OPEN_MP}
  COMPILE_FLAGS ${CXX_COMPILE_FLAGS})
