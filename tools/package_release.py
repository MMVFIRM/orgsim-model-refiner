"""Build a source ZIP and byte-level manifest. Run from a reviewed source tree."""
from __future__ import annotations
import argparse,hashlib,json,zipfile
from pathlib import Path

EXCLUDE={'__pycache__','.pytest_cache','.venv','build','dist','.git'}
def eligible(p,root):
    rel=p.relative_to(root)
    return p.is_file() and not any(x in EXCLUDE or x.endswith('.egg-info') for x in rel.parts) and p.suffix not in ('.pyc','.pyo') and not p.name.endswith(('-journal','-wal','-shm')) and p.name not in ('.env',)

def package(root,out):
    root=Path(root);out=Path(out)
    files=sorted(p for p in root.rglob('*') if eligible(p,root) and p.name!='RELEASE_MANIFEST.json')
    manifest={'format':'orgsim-refiner-integrated-v0.9-release-manifest','version':'0.9.0',
        'files_sha256':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'scope':'Source and bundled research artifacts, not real ERP certification.'}
    mp=root/'RELEASE_MANIFEST.json';mp.write_bytes((json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode('utf-8'))
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[mp]:z.write(p,root.name+'/'+p.relative_to(root).as_posix())
    with zipfile.ZipFile(out) as z:
        bad=z.testzip()
        if bad:raise ValueError('ZIP CRC failure: '+bad)
    return {'file':str(out),'manifest_files':len(files),'zip_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    print(json.dumps(package(Path(__file__).resolve().parents[1],a.out),indent=2))
