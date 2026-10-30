"""Seal an evaluation result with an RSA public key; no plaintext enters CI logs."""
import base64
import gzip
import json
import os
import sys
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

pub=serialization.load_pem_public_key(Path(sys.argv[1]).read_bytes())
payload=gzip.compress(Path(sys.argv[2]).read_bytes())
key=os.urandom(32); nonce=os.urandom(12)
enc=AESGCM(key).encrypt(nonce,payload,None)
wrapped=pub.encrypt(key,padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),algorithm=hashes.SHA256(),label=None))
result={'key':base64.b64encode(wrapped).decode(),'nonce':base64.b64encode(nonce).decode(),'data':base64.b64encode(enc).decode()}
print(base64.b64encode(json.dumps(result,separators=(',',':')).encode()).decode())
