"""Protect private review data passed through a public Actions repository."""
import base64
import json
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import padding,rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def derive(token,salt):
    if not token: raise ValueError('Missing encryption secret')
    return HKDF(algorithm=hashes.SHA256(),length=32,salt=salt,info=b'badini-private-review-v1').derive(token.encode())


def create(token,folder):
    root=Path(folder);root.mkdir(parents=True,exist_ok=False)
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    private=key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption())
    public=key.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo)
    salt,nonce=os.urandom(32),os.urandom(12)
    encrypted=AESGCM(derive(token,salt)).encrypt(nonce,private,None)
    data={'salt':base64.b64encode(salt).decode(),'nonce':base64.b64encode(nonce).decode(),'data':base64.b64encode(encrypted).decode()}
    (root/'sealed-key.json').write_text(json.dumps(data,separators=(',',':')))
    return public


def unseal(token,key_path,envelope_path,output):
    sealed=json.loads(Path(key_path).read_text())
    raw=AESGCM(derive(token,base64.b64decode(sealed['salt']))).decrypt(base64.b64decode(sealed['nonce']),base64.b64decode(sealed['data']),None)
    key=serialization.load_pem_private_key(raw,password=None)
    import gzip
    obj=json.loads(base64.b64decode(Path(envelope_path).read_text().strip()))
    aes=key.decrypt(base64.b64decode(obj['key']),padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),algorithm=hashes.SHA256(),label=None))
    plaintext=gzip.decompress(AESGCM(aes).decrypt(base64.b64decode(obj['nonce']),base64.b64decode(obj['data']),None))
    Path(output).write_bytes(plaintext)


if __name__=='__main__':
    token=os.environ.get('HF_TOKEN','')
    if sys.argv[1]=='create':
        pub=create(token,sys.argv[2]); print('::notice title=Private review channel::'+base64.b64encode(pub).decode())
    elif sys.argv[1]=='unseal': unseal(token,*sys.argv[2:])
    else: raise ValueError('Unknown command')
