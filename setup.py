import pathlib
from setuptools import setup, find_packages

# Get the directory of this file
HERE = pathlib.Path(__file__).parent

setup(
    name='l2metrics',
    version='0.1.0',
    description='Metrics for Lifelong Learning',
    long_description=(HERE / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    author='Megan Baker',
    author_email='megan.baker@jhuapl.edu',
    license='UNLICENSED',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'learnkit',
        'pandas',
        'numpy'
    ]
)
