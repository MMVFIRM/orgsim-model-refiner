"""Verify every sealed v0.9 delivery file; hashes do not authenticate publisher."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'DELIVERY_MANIFEST.json').read_text('utf-8'))
if manifest.get('format')!='orgsim-hardened-delivery-manifest-v1':
    raise SystemExit('Invalid delivery manifest')
fail=[]
for name,expect in manifest['files'].items():
    path=Path(name)
    if path.is_absolute() or '..' in path.parts:
        fail.append(name+': path traversal');continue
    p=root/path
    if not p.is_file(): fail.append(name+': missing');continue
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expect['sha256'] or len(raw)!=expect['bytes']:
        fail.append(name+': digest/size mismatch')
if fail:
    print('Verification FAILED:\n'+'\n'.join(fail));sys.exit(1)
print(f"Verified {len(manifest['files'])} delivery files. This is local integrity, not publisher authentication.")
