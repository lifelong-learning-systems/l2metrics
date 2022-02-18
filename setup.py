"""
Copyright © 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import pathlib
from setuptools import setup, find_packages

# Get the directory of this file
HERE = pathlib.Path(__file__).parent

setup(
    name="l2metrics",
    version="3.0.0",
    description="Metrics for Lifelong Learning",
    long_description=(HERE / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Eric Nguyen, Megan Baker",
    author_email="Eric.Nguyen@jhuapl.edu, Megan.Baker@jhuapl.edu",
    license="MIT",
    python_requires=">=3.7",
    url="https://github.com/darpa-l2m/l2metrics",
    download_url="https://github.com/darpa-l2m/l2metrics/archive/v3.0.0.tar.gz",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "l2logger>=1.7.0",
        "ipykernel",
        "ipympl",
        "matplotlib",
        "numpy",
        "pandas",
        "pyarrow",
        "python-dateutil",
        "pytz",
        "scipy",
        "seaborn",
        "six",
        "tabulate",
        "tqdm",
    ],
)
