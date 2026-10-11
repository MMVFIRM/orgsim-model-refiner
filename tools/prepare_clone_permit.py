"""Create an expiring LOCAL permit. The marker product must be created in a clone.

This utility never contacts Dolibarr, provisions identities, or enables writes.
"""
from __future__ import annotations
import argparse,datetime as dt,json,os,secrets,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from orgsim_refiner.erp.safety import _origin,ACKNOWLEDGMENT

def main(argv=None):
    p=argparse.ArgumentParser(description='Prepare local expiring disposable-clone authorization')
    p.add_argument('--url',required=True,help='Loopback origin of disposable clone')
    p.add_argument('--expires-hours',type=int,default=8)
    p.add_argument('--max-mutations',type=int,default=30000)
    p.add_argument('--ack',required=True,help=f'Exact string: {ACKNOWLEDGMENT}')
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.ack!=ACKNOWLEDGMENT:p.error('Explicit disposable-clone acknowledgment mismatch')
    if not 1<=a.expires_hours<=24:p.error('Expiration must be between 1 and 24 hours')
    if not 1<=a.max_mutations<=100000:p.error('Mutation budget must be between 1 and 100000')
    origin=_origin(a.url,write=True)
    marker_ref='ORGSIM-CLONE-'+secrets.token_hex(12).upper()
    marker_label='ORGSIM isolated disposable lab '+secrets.token_urlsafe(30)
    permit={'format':'orgsim-disposable-clone-permit-v1',
      'acknowledgment':ACKNOWLEDGMENT,'target_origin':origin,
      'marker_ref':marker_ref,'marker_label':marker_label,
      'expires_at':(dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=a.expires_hours)).isoformat(),
      'max_mutations':a.max_mutations}
    out=Path(a.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(out,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(permit,f,indent=2);f.write('\n')
    print('Created private permit at',out)
    print('Manually create this exact product marker ONLY in the disposable clone:')
    print(' product.ref:',marker_ref)
    print(' product.label:',marker_label)
    print('The writer verifies both over GET before any mutation. Never place marker on production.')
    return 0
if __name__=='__main__':raise SystemExit(main())
