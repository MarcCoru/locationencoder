from setuptools import setup, find_packages

with open('requirements.txt') as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]

setup(
    name='locationencoder',
    version='1.0',
    author="Marc Rußwurm",
    author_email="marc.russwurm@wur.nl",
    packages=find_packages(include=['locationencoder']),
    install_requires=requirements
)