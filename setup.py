from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "neural_cpp",
        [
            "neural_binding.cpp",
            "neural_network.cpp",
        ],
        include_dirs=[
            ".",
        ],
        cxx_std=17,
    ),
]

setup(
    name="neural_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)