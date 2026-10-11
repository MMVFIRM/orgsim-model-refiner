"""Portable content-addressing and bounded strict JSON I/O; no pickle/eval."""
from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path
from typing import Any

def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',',':'))

def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()

def file_digest(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def _pairs(items):
    out={}
    for k,v in items:
        if k in out: raise ValueError(f'Duplicate JSON key: {k}')
        out[k]=v
    return out

def loads(text: str):
    return json.loads(text, object_pairs_hook=_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def read_json(path: str | Path, max_bytes: int=32_000_000):
    p=Path(path)
    if p.stat().st_size>max_bytes: raise ValueError('JSON exceeds file limit')
    return loads(p.read_text('utf-8'))

def write_json(path: str | Path, value, *, overwrite: bool=False):
    """Write canonical UTF-8/LF bytes. Refuse to replace evidence by default."""
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    data=(canonical(value)+'\n').encode('utf-8')
    if not overwrite:
        with p.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    else:
        fd, tmp=tempfile.mkstemp(prefix=p.name+'.',dir=p.parent)
        try:
            with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,p)
        finally:
            Path(tmp).unlink(missing_ok=True)
    return digest(value)

def flatten(obj, prefix=''):
    out={}
    if not isinstance(obj,dict): raise ValueError('Expected JSON object')
    for k,v in obj.items():
        key=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict): out.update(flatten(v,key))
        else: out[key]=v
    return out
