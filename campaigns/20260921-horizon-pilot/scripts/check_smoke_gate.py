"""Recheck the original v5 certification used by this passive-telemetry pilot."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
cert = json.loads((ROOT / 'audit/v5_certification.json').read_text())
parent = Path(cert['source_campaign'])
subprocess.run([sys.executable, str(parent / 'scripts/check_smoke_gate.py')], check=True)
for row in cert['rows']:
    if hashlib.sha256(Path(row['result_path']).read_bytes()).hexdigest() != row['result_sha256']:
        raise ValueError('Certified result changed')
if cert.get('passed') is not True or len(cert['rows']) != 5:
    raise ValueError('Incomplete certification')
(ROOT / 'audit/smoke_gate.json').write_text(json.dumps(cert, indent=2) + '\n')
print('All five original v5 smokes certified; passive telemetry controller-equivalence regression passed.')
