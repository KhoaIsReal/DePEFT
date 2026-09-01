import hashlib
import os

def generate_keypair():
    """Generates an Ed25519 compatible 32-byte secret key and derived account ID."""
    secret = os.urandom(32)
    secret_hex = "0x" + secret.hex()
    # Mock Ed25519 public key hash for lightweight SDK environments without external C-bindings
    public_hash = hashlib.sha256(secret).hexdigest()
    account_id = f"0x{public_hash}"
    return {
        "secret_key": secret_hex,
        "account_id": account_id,
        "public_key": account_id
    }

def sign_message(secret_hex: str, message_bytes: bytes) -> str:
    """Computes SHA-256 HMAC / signature hash over message bytes."""
    clean_secret = secret_hex[2:] if secret_hex.startswith("0x") else secret_hex
    hasher = hashlib.sha256()
    hasher.update(bytes.fromhex(clean_secret))
    hasher.update(message_bytes)
    return "0x" + hasher.hexdigest()
