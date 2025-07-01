from setuptools import setup, find_packages

setup(
    name="FLUX1dev",
    version="0.1.0",
    description="FLUX Temp development package",
    author="HQX",
    author_email="hqx@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "diffusers>=0.18.0",
        "numpy<2.0",
    ],
    python_requires=">=3.8",
)
