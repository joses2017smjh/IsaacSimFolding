"""Freeze a paired runtime-horizon intervention without changing model weights."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-fold-repro')
V9 = REPO / 'campaigns/20260928-fresh-seed-remeasure-v9'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def seed_scan():
    seeds, scanned, skipped = set(), 0, 0
    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if 'seed' in key.lower():
                    for item in value if isinstance(value, list) else [value]:
                        if isinstance(item, int) and not isinstance(item, bool):
                            seeds.add(item)
                walk(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if item == '--seed' and i + 1 < len(obj) and str(obj[i+1]).isdigit():
                    seeds.add(int(obj[i+1]))
                walk(item)
    files = set()
    for pattern in ('*/manifest.json', '*/manifests/*.json', '*/plans/*.json',
                    '*/**/request.json', '*/**/status.json', '*/analysis/**/*.json',
                    '*/audit/**/*.json', '*/ledger/*.json'):
        files.update((REPO / 'campaigns').glob(pattern))
    for path in sorted(files):
        if path.stat().st_size > 8 << 20:
            skipped += 1
            continue
        try:
            walk(json.loads(path.read_text()))
            scanned += 1
        except (ValueError, UnicodeError):
            skipped += 1
    return seeds, {'files_scanned': scanned, 'files_skipped': skipped,
                   'distinct_prior_seeds': len(seeds)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', action='store_true')
    args = ap.parse_args()
    old = json.loads((V9 / 'manifest.json').read_text())
    rows = []
    bases = old['remeasure'][:8]
    for replicate in range(6):
        for pose, base in enumerate(bases):
            rows.append({k:v for k,v in base.items() if k not in
                         ('id', 'seed', 'horizon', 'seed_block', 'source_row')})
            rows[-1].update(id=f'dev{pose:02d}_s{98000+replicate*100+pose}',
                            seed=98000+replicate*100+pose, development_pose_slot=pose,
                            replicate=replicate, horizons=[50,10] if (pose+replicate)%2 == 0 else [10,50])
    smoke = dict(rows[0], id='smoke_dev00_s97900', seed=97900, horizons=[50,10])
    used, scan = seed_scan()
    collisions = sorted(({row['seed'] for row in rows} | {smoke['seed']}) & used)
    if collisions:
        raise SystemExit(f'Seeds already used: {collisions}')
    source_pins = {str(REPO / key):value for key,value in old['executed_sources'].items()}
    for row in rows:
        source_pins[row['asset_config']]=digest(row['asset_config'])
    if old.get('pose_metadata'):
        source_pins[old['pose_metadata']['path']]=old['pose_metadata']['sha256']
    for path in sorted((ROOT / 'scripts').glob('*.py')) + sorted((ROOT / 'slurm').glob('*')):
        if path.is_file():
            source_pins[str(path)] = digest(path)
    manifest = {
        'schema_version':1, 'campaign':'20261006-folding-runtime-horizon',
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'authorization':'User requested actual stronger project results and overnight queues on 2026-10-06.',
        'question':'Does deploying the immutable baseline with 10-action replanning improve settled folds over its shipped 50-action execution queue?',
        'intervention':{'control_n_action_steps':50,'candidate_n_action_steps':10,
                        'prediction_chunk_size':50,'weights_changed':False,
                        'policy_inputs':'three rasterized RGB views and twelve observed robot joints',
                        'checker_used_for_control':False},
        'historical_rationale':'v9 baseline descriptively settled H10 16/48 versus H50 9/48; this is a new runtime intervention comparison, not a checkpoint improvement claim.',
        'baseline_checkpoint':old['baseline_checkpoint'],
        'lehome':old['lehome'],'assets':old['assets'],'task_prompt':old['task_prompt'],
        'runner':str(V9 / 'scripts/render/policy_rollout51.py'),
        'repo_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        'executed_sources':source_pins,'environment_digests':old.get('environment_digests',{}),
        'smoke':smoke,'pairs':rows,
        'seeds':dict(scan, rule='98000 + 100*replicate + development pose; smoke=97900; same seed for both runtime horizons'),
        'gpu':{'constraint':'a40','required_name':'NVIDIA A40','partition':'ampere','concurrency':1,
               'design':'Both arms of each pair execute consecutively on the same allocated GPU. Order is counterbalanced by pose and replicate.'},
        'protocol':{'primary_metric':'terminal_success after 600 policy actions plus 60 fixed hold-settle physics steps',
                    'primary_test':'one-sided exact paired McNemar across 48 matched pairs',
                    'improvement_rule':'all 48 pairs valid AND candidate wins at least 5 additional settled folds AND one-sided paired p <= 0.05',
                    'secondary':'reached versus settled, per-pose counts, control latency and policy plan counts; descriptive only',
                    'promotion':'A development screen cannot establish held-out generalization. No automatic checkpoint replacement or final-set execution.',
                    'invalid_policy':'Infrastructure failures are missing data, not failed folds; no outcome-driven retries.',
                    'stop_rule':'One full-episode smoke pair, then exactly 48 development pairs; no extra seeds, no retries, no checkpoint search.',
                    'scope':'Pant_Short on the eight reused development poses; rasterized simulator results only; no hardware deployment claim.'},
        'budget':{'gpu_episodes':98,'expected_gpu_hours':7.0,'max_pair_seconds':3000,
                  'per_episode_timeout_seconds':1500,'per_task_wall_minutes':55},
        'outputs':'JSON, JSONL, logs, PNG, MP4 only; prohibit creating new .md and .txt files.',
    }
    if args.freeze:
        path=ROOT/'manifest.json'
        if path.exists():
            raise SystemExit('Refusing to overwrite frozen manifest')
        path.write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({'manifest':str(path),'sha256':digest(path),'pairs':len(rows),'seed_scan':scan}))
    else:
        print(json.dumps({'pairs':len(rows),'smoke':smoke,'seed_scan':scan,'sources':len(source_pins)}))

if __name__ == '__main__':
    main()
