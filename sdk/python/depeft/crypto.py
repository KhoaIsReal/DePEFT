import os
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

def generate_keypair():
    """Generates an Ed25519 compatible 32-byte secret key and derived account ID."""
    secret = os.urandom(32)
    secret_hex = "0x" + secret.hex()
    private_key = Ed25519PrivateKey.from_private_bytes(secret)
    public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    account_id = f"0x{public_key.hex()}"
    return {
        "secret_key": secret_hex,
        "account_id": account_id,
        "public_key": account_id,
    }

def sign_message(secret_hex: str, message_bytes: bytes) -> str:
    """Create a 64-byte Ed25519 signature compatible with the Rust node."""
    clean_secret = secret_hex[2:] if secret_hex.startswith("0x") else secret_hex
    secret = bytes.fromhex(clean_secret)
    if len(secret) != 32:
        raise ValueError("secret_hex must contain exactly 32 bytes")
    return "0x" + Ed25519PrivateKey.from_private_bytes(secret).sign(message_bytes).hex()
