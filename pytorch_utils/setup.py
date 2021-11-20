from setuptools import setup

setup(
    name = "pytorch_utils",
    version = "0.0.1",
    author = "Chris COtter",
    author_email = "cotter@sciencesundries.com",
    description = ("Collection of code common to training most pytorch models"),
    license = "The Unlicense",
    packages=['pytorch_utils'],
    install_requires=[
          'matplotlib',
          'numpy',
          'torch',
      ],
)
