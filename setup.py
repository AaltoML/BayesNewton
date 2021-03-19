from setuptools import setup, find_packages

__version__ = "0.0.0"

setup(
    name='newt',
    version=__version__,
    packages=find_packages(exclude=["examples"]),
    python_requires='>=3.6',
    url='https://github.com/AaltoML/Newt',
    license='Apache-2.0',
)
