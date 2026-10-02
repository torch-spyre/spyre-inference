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

"""Builds the host CPU kernels (csrc/) with CMake, as vLLM's setup.py does.

Metadata lives in pyproject.toml. Supported hosts: x86_64, ppc64le, s390x.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).parent.resolve()


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class cmake_build_ext(build_ext):
    def build_extensions(self) -> None:
        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        cfg = os.environ.get("CMAKE_BUILD_TYPE", "Debug" if self.debug else "RelWithDebInfo")
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DSPYRE_PYTHON_EXECUTABLE={sys.executable}",
            f"-DSPYRE_PYTHON_PATH={':'.join(sys.path)}",
        ]
        if extra := os.environ.get("CMAKE_ARGS"):
            cmake_args += extra.split()
        subprocess.check_call(["cmake", str(ROOT), *cmake_args], cwd=build_temp)

        targets = [ext.name.removeprefix("spyre_inference.") for ext in self.extensions]
        subprocess.check_call(
            ["cmake", "--build", ".", f"-j={os.cpu_count() or 1}"]
            + [f"--target={t}" for t in targets],
            cwd=build_temp,
        )

        for ext, target in zip(self.extensions, targets):
            # CMake appends DESTINATION (the package dir) to the prefix.
            prefix = Path(self.get_ext_fullpath(ext.name)).parent.parent.resolve()
            subprocess.check_call(
                ["cmake", "--install", ".", "--prefix", str(prefix), "--component", target],
                cwd=build_temp,
            )


ext_modules = [CMakeExtension("spyre_inference._C")]
if platform.machine() == "x86_64":
    ext_modules.append(CMakeExtension("spyre_inference._C_AVX2"))

setup(ext_modules=ext_modules, cmdclass={"build_ext": cmake_build_ext})
