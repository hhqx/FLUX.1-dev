from setuptools import setup, find_packages

def read_requirements(filename='requirements.txt'):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="FLUX1dev",
    version="0.1.0",
    description="FLUX Temp development package",
    author="HQX",
    author_email="hqx@example.com",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.8",
)
