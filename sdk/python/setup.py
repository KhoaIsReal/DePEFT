from setuptools import setup, find_packages

setup(
    name="depeft",
    version="0.1.0",
    description="Python SDK for DePEFT Decentralized Parameter-Efficient Fine-Tuning Network",
    author="DePEFT Contributors",
    license="Apache-2.0 OR MIT",
    packages=find_packages(),
    install_requires=[
        "requests>=2.28.0",
    ],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)

