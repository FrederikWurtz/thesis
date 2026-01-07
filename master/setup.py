from setuptools import setup, find_packages

setup(
    name="master",
    version="0.0.0",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    description="Editable install helper for local 'master' package",
)
