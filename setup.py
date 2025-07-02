from setuptools import setup, find_packages

def read_requirements(filename='requirements.txt'):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="FLUX1dev",
    version="0.1.1",
    description="FLUX Temp development package",
    author="HQX",
    author_email="hqx@example.com",
    # 修改包含逻辑：只包含必要的包
    packages=['FLUX1dev', 'tests'],  # 明确列出需要的包
    install_requires=read_requirements(),
    python_requires=">=3.8",
    include_package_data=True,
)
