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

# Helpers ported from vLLM's cmake/utils.cmake (CPU-only subset).

macro (find_python_from_executable EXECUTABLE)
  file(REAL_PATH ${EXECUTABLE} EXECUTABLE)
  set(Python_EXECUTABLE ${EXECUTABLE})
  find_package(Python COMPONENTS Interpreter Development.Module)
  if (NOT Python_FOUND)
    message(FATAL_ERROR "Unable to find python matching: ${EXECUTABLE}.")
  endif()
  message(STATUS "Found python matching: ${EXECUTABLE}.")
endmacro()

#
# Run `EXPR` in python. The standard output of python is stored in `OUT` with
# trailing whitespace stripped. A python error is a fatal `ERR_MSG`.
#
function (run_python OUT EXPR ERR_MSG)
  execute_process(
    COMMAND
    "${Python_EXECUTABLE}" "-c" "${EXPR}"
    OUTPUT_VARIABLE PYTHON_OUT
    RESULT_VARIABLE PYTHON_ERROR_CODE
    ERROR_VARIABLE PYTHON_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT PYTHON_ERROR_CODE EQUAL 0)
    message(FATAL_ERROR "${ERR_MSG}: ${PYTHON_STDERR}")
  endif()
  set(${OUT} ${PYTHON_OUT} PARENT_SCOPE)
endfunction()

# Extend `CMAKE_PREFIX_PATH` with `EXPR` evaluated after importing `PKG`, so
# the torch cmake configuration can be found.
macro (append_cmake_prefix_path PKG EXPR)
  run_python(_PREFIX_PATH
    "import ${PKG}; print(${EXPR})" "Failed to locate ${PKG} path")
  list(APPEND CMAKE_PREFIX_PATH ${_PREFIX_PATH})
endmacro()

# Find the libgomp shipped with the PyTorch wheel and create a shim dir with
#   libgomp.so    -> libgomp-<hash>.so...
#   libgomp.so.1  -> libgomp-<hash>.so...
# so the extension links the same OpenMP runtime torch loads.
# OUTPUT: TORCH_GOMP_SHIM_DIR ("" if torch vendors none, e.g. a source build)
function(prepare_torch_gomp_shim TORCH_GOMP_SHIM_DIR)
  set(${TORCH_GOMP_SHIM_DIR} "" PARENT_SCOPE)

  run_python(_TORCH_GOMP_PATH
    "
import os, glob
import torch
torch_pkg = os.path.dirname(torch.__file__)
site_root = os.path.dirname(torch_pkg)
roots = [os.path.join(site_root, 'torch.libs'), os.path.join(torch_pkg, 'lib')]
candidates = []
for root in roots:
    if os.path.isdir(root):
        candidates.extend(glob.glob(os.path.join(root, 'libgomp*.so*')))
print(candidates[0] if candidates else '')
"
    "failed to probe for libgomp")

  if(_TORCH_GOMP_PATH STREQUAL "" OR NOT EXISTS "${_TORCH_GOMP_PATH}")
    return()
  endif()

  set(_shim "${CMAKE_BINARY_DIR}/gomp_shim")
  file(MAKE_DIRECTORY "${_shim}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E rm -f "${_shim}/libgomp.so" "${_shim}/libgomp.so.1")
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${_TORCH_GOMP_PATH}" "${_shim}/libgomp.so")
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${_TORCH_GOMP_PATH}" "${_shim}/libgomp.so.1")

  set(${TORCH_GOMP_SHIM_DIR} "${_shim}" PARENT_SCOPE)
endfunction()

#
# Define a python extension target `MOD_NAME` linked against torch.
#
# DESTINATION <dest>   - Install destination, relative to the install prefix.
# SOURCES <sources>    - Source files.
# COMPILE_FLAGS <flags>
# LIBRARIES <libraries>
#
function (define_extension_target MOD_NAME)
  cmake_parse_arguments(PARSE_ARGV 1
    ARG
    ""
    "DESTINATION"
    "SOURCES;COMPILE_FLAGS;LIBRARIES")

  Python_add_library(${MOD_NAME} MODULE WITH_SOABI "${ARG_SOURCES}")
  target_include_directories(${MOD_NAME} PRIVATE csrc)
  target_compile_options(${MOD_NAME} PRIVATE ${ARG_COMPILE_FLAGS})
  target_compile_definitions(${MOD_NAME} PRIVATE "-DTORCH_EXTENSION_NAME=${MOD_NAME}")
  target_link_libraries(${MOD_NAME} PRIVATE torch ${TORCH_LIBRARIES} ${ARG_LIBRARIES})

  install(TARGETS ${MOD_NAME} LIBRARY DESTINATION ${ARG_DESTINATION} COMPONENT ${MOD_NAME})
endfunction()
