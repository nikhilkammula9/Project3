from setuptools import find_packages, setup

setup(
    name="tweets_dag",
    packages=find_packages(exclude=["tweets_dag_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagit", "pytest"]},
)
