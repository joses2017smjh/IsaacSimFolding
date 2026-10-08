"""Execute a matched H50/H10 pair with unchanged model and source-verified scoring."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()

def verify(root, expected):
    path=root/'manifest.json'
    if digest(path)!=expected:
        raise ValueError('Manifest changed after submission')
    m=json.loads(path.read_text())
    changed=[path for path,sha in m['executed_sources'].items()
             if not Path(path).is_file() or digest(path)!=sha]
    cp=Path(m['baseline_checkpoint']['path'])
    changed.extend(str(cp/name) for name,sha in m['baseline_checkpoint']['sha256'].items()
                   if not (cp/name).is_file() or digest(cp/name)!=sha)
    for label,record in m.get('environment_digests',{}).items():
        tree=Path(record['path'])
        files=sorted(p for p in tree.rglob(record['glob']) if p.is_file() and '__pycache__' not in p.parts)
        h=hashlib.sha256()
        for path in files:
            h.update(f'{path.relative_to(tree)}\0{digest(path)}\n'.encode())
        if len(files)!=record['files'] or h.hexdigest()!=record['sha256']:
            changed.append(f'environment:{label}')
    if changed:
        raise ValueError(f'Pinned source/checkpoint drift: {changed}')
    return m

def validate(result,row,horizon):
    for key,value in {'garment':row['garment'],'seed':row['seed'],'steps':row['steps'],
                      'effective_n_action_steps':horizon,'prediction_chunk_size':50,
                      'terminal_settle_steps':60}.items():
        if result.get(key)!=value:
            raise ValueError(f'{key}={result.get(key)!r}, expected {value!r}')
    if not result.get('physics_finite') or not result.get('robot_finite'):
        raise ValueError('Nonfinite simulator state')
    if result.get('terminal_success') != result.get('terminal_checker',{}).get('success'):
        raise ValueError('Terminal checker disagreement')
    if not isinstance(result.get('terminal_success'),bool):
        raise ValueError('Terminal success is not boolean')

def command(root,m,row,horizon,dest):
    return [sys.executable,'-u',str(root/'scripts/guarded_runner.py'),m['runner'],
            '--lehome',m['lehome'],'--policy_path',m['baseline_checkpoint']['path'],
            '--garment',row['garment'],'--garment_dir',str(Path(row['asset_config']).parent),
            '--assets',m['assets'],'--steps',str(row['steps']),
            '--frames_out',str(dest/'frames'),'--match_pose='+row['match_pose'],
            '--match_scale',str(row['match_scale']),'--settle_steps','60',
            '--terminal_settle_steps','60','--seed',str(row['seed']),
            '--n_action_steps',str(horizon),'--policy_variant',f'runtime_h{horizon}',
            '--task',m['task_prompt'],'--gif_every','12','--result_out',str(dest/'rollout.json')]

def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--manifest-sha256',required=True)
    ap.add_argument('--index',type=int,default=0)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    root=args.campaign.resolve()
    m=verify(root,args.manifest_sha256)
    if not args.smoke and not 0<=args.index<len(m['pairs']):
        raise ValueError('Pair index out of bounds')
    row=m['smoke'] if args.smoke else m['pairs'][args.index]
    base=root/('smoke' if args.smoke else 'evaluation')/row['id']
    if args.dry_run:
        print(json.dumps({'row':row,'commands':[command(root,m,row,h,base/f'h{h}') for h in row['horizons']],
                          'source_verification':'passed','checkpoint_verification':'passed'}))
        return 0
    names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()
    if set(names)!={m['gpu']['required_name']}:
        raise ValueError(f'Wrong allocated hardware: {names}')
    base.mkdir(parents=True,exist_ok=False)
    pair={'state':'running','row':row,'gpu_names':names,'host':socket.gethostname(),
          'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
          'slurm_array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),
          'manifest_sha256':args.manifest_sha256,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          'arms':{}}
    write(base/'status.json',pair)
    for horizon in row['horizons']:
        dest=base/f'h{horizon}'
        (dest/'frames').mkdir(parents=True)
        cmd=command(root,m,row,horizon,dest)
        write(dest/'request.json',dict(row=row,horizon=horizon,command=cmd,
                                      checkpoint=m['baseline_checkpoint'],manifest_sha256=args.manifest_sha256))
        started=time.monotonic()
        status={'state':'running','horizon':horizon}
        with (dest/'rollout.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=m['lehome'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                proc.wait(timeout=m['budget']['per_episode_timeout_seconds'])
                if proc.returncode:
                    raise RuntimeError(f'Isaac exited {proc.returncode}')
                result=json.loads((dest/'rollout.json').read_text())
                validate(result,row,horizon)
                if m['gpu']['required_name'] not in (dest/'rollout.log').read_text():
                    raise ValueError('Isaac log did not confirm allocated GPU model')
                status.update(state='completed',terminal_success=result['terminal_success'],
                              ever_success=result['success'],terminal_checker=result['terminal_checker'])
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()
                status.update(state='infrastructure_timeout',error='Bounded episode timeout exceeded')
            except Exception as exc:
                status.update(state='infrastructure_error',error=repr(exc))
        status['wall_seconds']=time.monotonic()-started
        write(dest/'status.json',status)
        pair['arms'][str(horizon)]=status
        if status['state']!='completed':
            pair.update(state='infrastructure_error',completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            write(base/'status.json',pair)
            print(json.dumps(pair),flush=True)
            return 3
        write(base/'status.json',pair)
    pair.update(state='completed',completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write(base/'status.json',pair)
    print(json.dumps(pair),flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
