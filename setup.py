from setuptools import setup
import pathlib

__version__ = "1.0.1"

setup(
    name="memory-maze",
    version=__version__,
    author="Jurgis Pasukonis",
    author_email="jurgisp@gmail.com",
    url="https://github.com/jurgisp/memory-maze",
    description="Memory Maze is an environment to benchmark memory abilities of RL agents",
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    zip_safe=False,
    python_requires=">=3",
    packages=["memory_maze"],
    install_requires=[
        'dm_control'
    ],
)
