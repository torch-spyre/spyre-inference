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
import zipfile
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).parent.resolve()


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class cmake_build_ext(build_ext):
    def build_extensions(self) -> None:
        if self._extract_from_prebuilt_wheel():
            return
        # The kernels are optional: without them the samplers fall back to
        # PyTorch ops, so a failed build must not fail the install.
        try:
            self._build_with_cmake()
        except (OSError, subprocess.CalledProcessError) as e:
            # A previous build's .so would import as stale kernels; fall back instead.
            for stale in [*self._outputs(), *(ROOT / "spyre_inference").glob("_C*.so")]:
                stale.unlink(missing_ok=True)
            cmake_failed = isinstance(e, subprocess.CalledProcessError)
            why = "see the CMake output above" if cmake_failed else e
            print(
                f"WARNING: spyre-inference sampling kernels failed to build ({why}). "
                "Installing without them: host sampling falls back to PyTorch ops "
                "and gets no kernel speedup.",
                file=sys.stderr,
            )
            # setuptools copies and lists outputs from this list; nothing was built.
            self.extensions = []

    def _outputs(self) -> list[Path]:
        return [Path(self.get_ext_fullpath(ext.name)) for ext in self.extensions]

    # CI builds the kernels once per run (the build_kernels job) and hands its wheel to every
    # test job via SPYRE_KERNELS_WHEEL_DIR. Only the compiled extensions are taken from it; the
    # Python code comes from the checkout. A wheel without matching extensions (another
    # Python ABI, say) falls back to building.
    def _extract_from_prebuilt_wheel(self) -> bool:
        wheel_dir = os.environ.get("SPYRE_KERNELS_WHEEL_DIR")
        if not wheel_dir:
            return False
        try:
            wheels = sorted(Path(wheel_dir).glob("spyre_inference-*.whl"))
            if not wheels:
                raise FileNotFoundError(f"no spyre_inference wheel in {wheel_dir}")
            with zipfile.ZipFile(wheels[-1]) as whl:
                members = {f"spyre_inference/{out.name}": out for out in self._outputs()}
                if missing := set(members) - set(whl.namelist()):
                    raise FileNotFoundError(f"{wheels[-1].name} lacks {sorted(missing)}")
                for member, out in members.items():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(whl.read(member))
                    out.chmod(0o755)
        except (OSError, zipfile.BadZipFile) as e:
            print(f"WARNING: not using the prebuilt kernels ({e}); building.", file=sys.stderr)
            return False
        print(f"Using the prebuilt spyre-inference kernels from {wheels[-1].name}")
        return True

    def _build_with_cmake(self) -> None:
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
        # Two sources per target, so more jobs only oversubscribe a shared runner.
        num_jobs = os.environ.get("MAX_JOBS") or str(2 * len(targets))
        subprocess.check_call(
            ["cmake", "--build", ".", f"-j={num_jobs}"] + [f"--target={t}" for t in targets],
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
