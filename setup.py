from setuptools import setup

setup(
    name = "pp2ts",
    version = "0.0.1",
    author = "Mike Dewar",
    author_email = "mikedewar@gmail.com",
    description = ("calculates the rate of a point process and returns the result as a time series",),
    keywords = "point process, pandas",
    py_modules=['pp2ts'],
    requires=['pandas','scipy'],
)
