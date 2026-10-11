"""Read-only compatibility utilities. The v0.6.1 study design is unchanged."""
from __future__ import annotations
from pathlib import Path
from .util import digest,write_json

def check_instrumentation(receipt:Path,diagnostic:Path,reason:str):
    from simlearn.dolibarr_factorial import triage_500
    result=triage_500(Path(receipt),Path(diagnostic),reason=reason)
    return {'format':'orgsim-refiner-calibration-check-v1','evidence_check':result,'automatic_erp_writes':0,
            'logging_observed_for_this_saved_attempt_only':True,'current_logging_configuration_not_verified':True,
            'manual_correlation_not_independently_authenticated':True,
            'warning':'This does not fill missing logs for an earlier action. Reuse no receipt across interventions.'}

def inspect_workspace(root:Path):
    from simlearn.dolibarr_factorial import workspace,status
    _,p,_,_=workspace(Path(root))
    return {'format':'orgsim-refiner-legacy-workspace-readonly-status-v1','plan_sha256':digest(p),
            'status':status(Path(root)),'changed_existing_files':False,'automatic_erp_writes':0,
            'design_preserved':True,'note':'No reclassification, rerun, fit, network call, or scoring change is performed.'}
