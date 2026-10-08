"""Submit one bounded smoke -> matched array -> JSON summary chain exactly once."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
from run_pair import digest,verify

SBATCH='/apps/slurm/current/bin/sbatch'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--submit',action='store_true')
    args=ap.parse_args()
    root=args.campaign.resolve()
    sha=digest(root/'manifest.json')
    m=verify(root,sha)
    ledger=root/'ledger/submissions.json'
    if ledger.exists():
        raise SystemExit('This campaign already has a submission ledger. Inspect it; do not duplicate the queue.')
    smoke=[SBATCH,'--parsable','--mail-type=NONE','--job-name=lh-horizon-smoke',
           f'--output={root}/logs/smoke-%j.out',str(root/'slurm/pair.sbatch'),str(root),sha,'smoke']
    array=[SBATCH,'--parsable','--mail-type=NONE','--job-name=lh-horizon-dev',
           f'--array=0-{len(m["pairs"])-1}%1','--kill-on-invalid-dep=yes',
           f'--output={root}/logs/pair-%A_%a.out','--dependency=afterok:SMOKE_ID',
           str(root/'slurm/pair.sbatch'),str(root),sha,'evaluation']
    report=[SBATCH,'--parsable','--mail-type=NONE','--job-name=lh-horizon-summary',
            f'--output={root}/logs/summary-%j.out','--dependency=afterany:ARRAY_ID',
            str(root/'slurm/summarize.sbatch'),str(root)]
    if not args.submit:
        print(json.dumps({'manifest_sha256':sha,'source_verification':'passed',
                          'smoke':smoke,'array':array,'summary':report},indent=2))
        return
    (root/'ledger').mkdir(exist_ok=True)
    with (root/'ledger/submission.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if ledger.exists():
            raise SystemExit('Another process submitted this campaign')
        names=subprocess.check_output(['squeue','-u','sanchej7','-h','-o','%j'],text=True).splitlines()
        if any(name.strip() in ('lh-horizon-smoke','lh-horizon-dev','lh-horizon-summary') for name in names):
            raise SystemExit('A folding horizon campaign is already queued; inspect it instead of duplicating it')
        state={'manifest_sha256':sha,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
               'state':'submitting','jobs':{}}
        ledger.write_text(json.dumps(state,indent=2)+'\n')
        try:
            for stage,cmd in [('smoke',smoke),('development',array),('summary',report)]:
                cmd=[x.replace('SMOKE_ID',state['jobs'].get('smoke',{}).get('id','SMOKE_ID'))
                      .replace('ARRAY_ID',state['jobs'].get('development',{}).get('id','ARRAY_ID')) for x in cmd]
                output=subprocess.check_output(cmd,text=True).strip()
                job=output.split(';')[0]
                if not job.isdigit():
                    raise ValueError(f'Unexpected sbatch output: {output!r}')
                state['jobs'][stage]={'id':job,'command':cmd}
                ledger.write_text(json.dumps(state,indent=2)+'\n')
            state['state']='queued'
        except Exception as exc:
            state.update(state='submission_error',error=repr(exc))
            ledger.write_text(json.dumps(state,indent=2)+'\n')
            raise
        ledger.write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state,indent=2))

if __name__=='__main__':
    main()
