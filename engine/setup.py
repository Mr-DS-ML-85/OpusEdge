"""setup.py — build & install the opusedge_cpp Python package.

Layout:
    opusedge_cpp/           <- Python package
      __init__.py           <- pure-python facade
      _core.<abi>.so        <- built from wrapper.cpp (contents of _opusedge_cpp)

The C extension is compiled *into the package directory* as `opusedge_cpp._core`,
so users just do `import opusedge_cpp`. No sys.path hacks, no leading-underscore
package names, no rename shenanigans.
"""
from setuptools import setup, Extension, find_packages
import os, sys

HERE = os.path.abspath(os.path.dirname(__file__))

# Try a few common Eigen install paths so this works cross-distro.
def _find_eigen():
    for p in ("/usr/include/eigen3", "/usr/local/include/eigen3",
              "/opt/homebrew/include/eigen3", "/opt/local/include/eigen3"):
        if os.path.exists(os.path.join(p, "Eigen", "Dense")):
            return p
    return "/usr/include/eigen3"  # fallback

ext = Extension(
    "opusedge_cpp._core",                       # <- lives inside the package
    sources=["wrapper.cpp"],
    include_dirs=["include", _find_eigen()],
    extra_compile_args=[
        "-std=c++20", "-O3", "-march=native",
        "-fopenmp", "-DNDEBUG", "-Wno-unused-variable",
    ],
    extra_link_args=["-fopenmp"],
    language="c++",
)

setup(
    name="opusedge_cpp",
    version="1.0.0",
    description="OpusEdge — C++20 signal-driven compute allocation primitives",
    long_description=open(os.path.join(HERE, "README.md")).read()
        if os.path.exists(os.path.join(HERE, "README.md")) else "",
    long_description_content_type="text/markdown",
    author="Irfan Mahir",
    author_email="",
    url="https://github.com/Mr-DS-ML-85/OpusEdge",
    license="PolyForm Noncommercial 1.0.0",
    packages=["opusedge_cpp"],
    package_dir={"opusedge_cpp": "opusedge_cpp"},
    package_data={"opusedge_cpp": ["py.typed"]},   # PEP 561 type-marker
    ext_modules=[ext],
    python_requires=">=3.10",
    keywords="llm inference kv-cache mixture-of-experts state-space-models attention",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Free for non-commercial use",
        "License :: Other/Proprietary License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    project_urls={
        "Source":  "https://github.com/Mr-DS-ML-85/OpusEdge",
        "Paper":   "https://github.com/Mr-DS-ML-85/OpusEdge/blob/main/paper/OpusEdge.pdf",
        "Issues":  "https://github.com/Mr-DS-ML-85/OpusEdge/issues",
        "Docs":    "https://github.com/Mr-DS-ML-85/OpusEdge/tree/main/docs",
    },
)
