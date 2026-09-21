"""Submit exactly one replacement gate. Never edit or cancel existing jobs."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from preflight_assets import validate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--campaign', type=Path, required=True)
parser.add_argument('--submit', action='store_true')
args = parser.parse_args()
root = args.campaign.resolve()
receipt = root / 'gate_submission.json'
if receipt.exists():
    print(receipt.read_text())
    raise SystemExit(0)
validation = validate(root)
subprocess.run([sys.executable, str(root / 'scripts/run_media_task.py'), '--campaign', str(root), '--index', '0', '--dry-run'], check=True)
cmd = ['sbatch', '--parsable', '--chdir=' + str(root),
       '--output=' + str(root / 'logs/gate-%j.out'), '--error=' + str(root / 'logs/gate-%j.err'),
       str(root / 'slurm/media.sbatch'), str(root), '0']
if not args.submit:
    print(json.dumps({'command': cmd, 'submitted': False}, indent=2))
    raise SystemExit(0)
(root / 'logs').mkdir(exist_ok=True)
lock = root / '.gate-submission.lock'
with lock.open('x') as stream:
    stream.write('Submission started. If interrupted, inspect squeue before any retry.\n')
# Do not inherit a compute allocation or sbatch defaults from the invoking shell.
env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'SBATCH_', 'SRUN_'))}
result = subprocess.run(cmd, env=env, check=True, text=True, capture_output=True)
job_id = result.stdout.strip().split(';')[0]
if not job_id.isdigit():
    raise RuntimeError(f'Unexpected sbatch response; inspect scheduler before retry: {result.stdout!r}')
record = {'job_id': job_id, 'submitted_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'command': cmd, 'validation': validation,
          'replaces_gate': '21367674', 'existing_jobs_modified': False, 'additional_array_submitted': False}
with receipt.open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
lock.unlink()
print(json.dumps(record, indent=2))
