import pathlib
from setuptools import find_packages, setup


# shamelessly stolen from https://github.com/huggingface/huggingface_hub/blob/main/setup.py
def get_version() -> str:
    rel_path = "src/pxia/__init__.py"
    with open(rel_path, "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


with open("requirements.txt", "r", encoding="utf-8") as f:
    required = f.read().splitlines()

setup(
    name="pxia",
    version=get_version(),
    description="a repository for PXIA models",
    long_description=pathlib.Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    Homepage="https://github.com/not-lain/pxia",
    url="https://github.com/not-lain/pxia",
    Issues="https://github.com/not-lain/pxia/issues",
    authors=[{"name": "hafedh hichri", "email": "hhichri60@gmail.com"}],
    license="Apache 2.0 License",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    install_requires=required,
)
