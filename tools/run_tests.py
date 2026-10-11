"""Run all tests with project-local temporary storage; no live service calls."""
from __future__ import annotations
import os,sys,subprocess,uuid
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    temp=root/'build'/('test_temp_'+uuid.uuid4().hex[:12]);temp.mkdir(parents=True)
    env=dict(os.environ)
    env.update(TMP=str(temp),TEMP=str(temp),TMPDIR=str(temp),PYTHONPATH=str(root/'src'),
               OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    args=sys.argv[1:] or ['tests']
    return subprocess.call([sys.executable,'-m','pytest','-q',*args,'--basetemp',str(temp/'pytest')],cwd=root,env=env)
if __name__=='__main__':raise SystemExit(main())
