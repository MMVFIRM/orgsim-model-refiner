"""Verify the v0.9 delivery source/artifact manifest, not live ERP accuracy."""
from __future__ import annotations
import json,hashlib
from pathlib import Path

def verify(root:Path):
    root=Path(root);m=json.loads((root/'RELEASE_MANIFEST.json').read_text('utf-8'))
    if m.get('format')!='orgsim-refiner-integrated-v0.9-release-manifest':raise ValueError('Wrong release manifest')
    for name,h in m['files_sha256'].items():
        r=Path(name)
        if r.is_absolute() or '..' in r.parts or not (root/r).is_file():raise ValueError('Invalid/missing manifest path: '+name)
        if hashlib.sha256((root/r).read_bytes()).hexdigest()!=h:raise ValueError('Release file differs: '+name)
    return {'verified':True,'files_verified':len(m['files_sha256']),
            'scope':'Bundled file integrity only; no independent attestation, provider efficacy or live ERP authorization.'}
if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parents[1]),indent=2))
