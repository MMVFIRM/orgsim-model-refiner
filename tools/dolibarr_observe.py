"""Production-safe narrow Dolibarr GET observation CLI. Never mutates records."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from orgsim_refiner.erp.safety import DolibarrReadOnly


def main(argv=None):
    p=argparse.ArgumentParser(description='Read-only production Dolibarr observer; no mutations')
    p.add_argument('--url',required=True,help='Origin only, e.g. https://erp.example.com')
    p.add_argument('--allow-remote-read',action='store_true',help='Explicitly opt into HTTPS remote reads')
    p.add_argument('--family',choices=['orders','products','thirdparties','invoices'],required=True)
    p.add_argument('--id',type=int,required=True)
    p.add_argument('--api-key-env',default='DOLIBARR_READ_KEY')
    p.add_argument('--out',help='Private output; contains full record data, do not share')
    a=p.parse_args(argv)
    key=os.environ.get(a.api_key_env)
    if not key:
        p.error('Read-only API key variable is unset')
    c=DolibarrReadOnly(a.url,key,allow_remote_read=a.allow_remote_read)
    rec=c.get(a.family,a.id)
    data=json.dumps(rec,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    public={'mode':'GET_ONLY','origin':c.origin,'family':a.family,'record_id':a.id,
            'response_sha256':hashlib.sha256(data).hexdigest(), 'bytes':len(data),
            'automatic_writes':0,'raw_data_exported':bool(a.out)}
    if a.out:
        dst=Path(a.out)
        if dst.exists():
            p.error('Private observation output already exists; refusing overwrite')
        dst.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(dst,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,'wb') as f:f.write(data+b'\n')
    print(json.dumps(public,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
