import os
import re
import sys
import sysconfig
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand
from shutil import copyfile, copymode
import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from pathlib import Path


# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())
        # print("sourcedir", self.sourcedir)
        # raise Exception("sourcedir", self.sourcedir)
        # sys.exit()


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python
        # 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        print("ext dir", extdir)

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"
        # cfg = "Debug"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need
        # to.
        # cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward
            # compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or
            # PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        print("cmake_args", cmake_args)
        print("build_temp", build_temp)
        # cmake_args += ["-DBUILD_PYRRT=ON"]

        # cmake_args += ["-DCMAKE_BUILD_TYPE=Release",
        #                "-DBUILD_PYRRT=1",
        #                # "-DBoost_USE_STATIC_LIBS=1",
        #                # "-DBoost_CHRONO_LIBRARY_RELEASE=/usr/local/lib/libboost_chrono.a"
        #                ]

        # cmake_args += [
        #     "-DCMAKE_BUILD_TYPE=Release",
        #     "-DBUILD_PYRRT=1",
        #     "-DBoost_USE_STATIC_LIBS=1",
        #     "-DBUILD_SHARED_LIBS=0",
        #     # "-DPYTHON_EXECUTABLE=/opt/python/cp38-cp38/bin/python",
        #     "-DBoost_CHRONO_LIBRARY_RELEASE=/usr/local/lib/libboost_chrono.a",
        #     "-DUSE_ZLIBSTATIC=1",
        # ]

        cmake_args += [
            # "-DCMAKE_BUILD_TYPE=Release",
            # "-DCMAKE_CXX_COMPILER=clang++-13",
            "-DBUILD_DYNOBENCH_TOOLS=0",
            "-DBUILD_DYNOBENCH_PYBINDINGS=1",
            "-DCMAKE_PREFIX_PATH=/home/quim/local/",
        ]

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        build_args += ["-j4"]
        print("build_args", build_args)
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in
# a file.


# this_directory = Path(__file__).parent
# long_description = (this_directory / "README.md").read_text()

datadir1 = Path(__file__).parent / "envs"
files1 = ["envs/" + str(p.relative_to(datadir1)) for p in datadir1.rglob("*.yaml")]

datadir2 = Path(__file__).parent / "models"
files2 = ["models/" + str(p.relative_to(datadir2)) for p in datadir2.rglob("*.yaml")]

files = files1 + files2
print("files is", files)
# setup(
#     ...,
#     packages=['mypkg'],
#     package_data={'mypkg': files},
# )


# SOURCE:
# https://stackoverflow.com/questions/63131139/python-setuptools-pip-packing-data-files-into-your-package

setup(
    name="dynobench",
    version="0.0.4",
    author="Joaquim Ortiz-Haro",
    author_email="quimortiz21@gmail.com",
    description="C++/Python Dynamics Models",
    long_description="",
    # packages=find_packages(mypkg
    #     "."
    # ),  # where the folder dynobench is. Inside the dynobench I should have a __init__.py
    # packages=["dynobench","models", "envs"],
    packages=["dynobench", "dynobench.models", "dynobench.envs"],
    # "dynobench.test", "dynobench.utils"],
    # package_dir={"": "."},
    package_dir={"dynobench.models": "models", "dynobench.envs": "envs"},
    ext_modules=[CMakeExtension("dynobench/dynobench")],
    # if ext_modules=[CMakeExtension("bar/foo")],
    # build directory will be bar/foo
    # The ext module will be copied to bar/name_of_the_module (as defined in CMakeLists.txt)
    # it is important that first name matches the name of the package
    cmdclass=dict(build_ext=CMakeBuild),
    # package_data={"pydynobench": ["pydynobench.pyi"]},
    # test_suite="tests",
    include_package_data=True,
    # package_data={
    #     "models": ["*.yaml"]
    # },  # NOTE: i have to find files by hand because this does not support recursive globs
    # ["models/*.yaml"] + files},
    # "envs/*.yaml",
    # ]},
    zip_safe=False,
    # package_data={'': [
    #     'models/*.yaml',
    # ]}
)
