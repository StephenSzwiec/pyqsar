import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyqsar",
    version="0.8.0",
    author="Stephen Szwiec",
    author_email="Stephen.Szwiec@ndus.edu",
    description="Refactor of pyqsar to Python3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stephenszwiec/pyqsar",
    project_urls={
        "Bug Tracker": "https://github.com/stephenszwiec/pyqsar/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
