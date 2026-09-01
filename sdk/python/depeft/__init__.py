"""
DePEFT Python SDK: Client library for interacting with DePEFT App-Chain Nodes.
Enables AI Engineers & Miners to create fine-tuning tasks, upload datasets, and query chain status.
"""

from .client import DePeftClient
from .crypto import generate_keypair, sign_message

__version__ = "0.1.0"
__all__ = ["DePeftClient", "generate_keypair", "sign_message"]
