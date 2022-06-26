from setuptools import setup

__version__ = "0.2.0"

setup(
    name="dmc-memory-maze",
    version=__version__,
    author="Jurgis Pasukonis",
    author_email="jurgisp@gmail.com",
    url="https://github.com/jurgisp/dmc-memory-maze",
    description="Python wrapper for DMLab maze generator",
    zip_safe=False,
    python_requires=">=3",
    packages=["dmc_memory_maze"],
    install_requires=[
        'dm_control'
    ],
)
