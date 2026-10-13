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

cmake_minimum_required(VERSION 3.26)

# Host-side CPU kernels (sampling) for spyre-inference.
project(spyre_inference_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# PyTorch's C++20 headers need full C++20 support.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.3")
  message(FATAL_ERROR "GCC >= 11.3 is required (found ${CMAKE_CXX_COMPILER_VERSION}).")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables.
set(ignoreMe "${SPYRE_PYTHON_PATH}")

install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

if (SPYRE_PYTHON_EXECUTABLE)
  find_python_from_executable(${SPYRE_PYTHON_EXECUTABLE})
else()
  message(FATAL_ERROR "Please set SPYRE_PYTHON_EXECUTABLE to the python used for the build.")
endif()

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
find_package(Torch REQUIRED)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/cpu_extension.cmake)
