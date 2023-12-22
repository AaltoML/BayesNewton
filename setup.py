from setuptools import setup, find_packages

__version__ = "1.3.4"

setup(
    name='bayesnewton',
    version=__version__,
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        "jax==0.4.14",
        "jaxlib==0.4.14",
        "objax==1.7.0",
        "tensorflow_probability==0.21",
        "numpy>=1.22"
    ],
    url='https://github.com/AaltoML/BayesNewton',
    license='Apache-2.0',
)
