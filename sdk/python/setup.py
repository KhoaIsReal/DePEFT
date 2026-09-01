from setuptools import setup, find_packages

setup(
    name="depeft",
    version="0.1.0",
    description="Python SDK for DePEFT Decentralized Parameter-Efficient Fine-Tuning Network",
    author="DePEFT Contributors",
    packages=find_packages(),
    install_requires=[
        "requests>=2.28.0",
    ],
    python_requires=">=3.8",
)
