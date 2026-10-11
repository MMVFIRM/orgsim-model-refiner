"""Run the complete regression suite in project-local temp, without API keys."""
from pathlib import Path
import os,subprocess,sys,tempfile
root=Path(__file__).resolve().parents[1]
(root/'.test-work').mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=root/'.test-work',prefix='run-') as temp:
    env=os.environ.copy()
    for key in list(env):
        if any(text in key.upper() for text in ('API_KEY','DOLAPIKEY','WRITER_KEY','DOLIBARR_READ_KEY','DOLIBARR_CLONE_KEY')):
            env.pop(key,None)
    env.update(PYTHONPATH=str(root/'src'),TMP=temp,TEMP=temp,TMPDIR=temp,
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    raise SystemExit(subprocess.call([sys.executable,'-m','pytest','-q','tests','--basetemp',str(Path(temp)/'pytest')],cwd=root,env=env))
