"""Append verified runtime results to existing docs without overwriting other edits."""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

ROOT=Path('/nfs/hpc/share/sanchej7/Computer_Vision/project_results_upgrade/folding')
REPO=Path('/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-fold-repro')
MANIFEST_SHA='5b29da9f8deb14cc65c587e286df0d72da5b81d7b91909b5ca0f98f8c91db98b'
SUMMARY_JOB='21601325'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def verified_result(root=ROOT):
    result_path=root/'analysis/results.json'
    output={'classification':'incomplete','source_summary_sha256':None,
            'source_summary':str(result_path),'manifest_sha256':MANIFEST_SHA,
            'validation_errors':[],'result':None}
    try:
        raw=(root/'manifest.json').read_bytes()
        if digest(raw)!=MANIFEST_SHA:
            raise ValueError('The frozen manifest changed')
        manifest=json.loads(raw)
        summary_script=root/'scripts/summarize.py'
        if digest(summary_script.read_bytes())!=manifest['executed_sources'][str(summary_script)]:
            raise ValueError('The frozen summarizer changed')
        raw_result=result_path.read_bytes()
        output['source_summary_sha256']=digest(raw_result)
        recorded=json.loads(raw_result)
        spec=importlib.util.spec_from_file_location('frozen_horizon_summary',summary_script)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        computed=module.summary(root)
        compared=('campaign','state','valid_pairs','expected_pairs','invalid_or_missing',
                  'counts','h10_minus_h50_settled','exact_paired_one_sided_p',
                  'preregistered_screen_passed','per_pose','pairs','protocol','observed_gpu_hours')
        mismatches=[key for key in compared if recorded.get(key)!=computed.get(key)]
        if mismatches:
            raise ValueError(f'Saved summary differs from retained records: {mismatches}')
        for row in manifest['pairs']:
            status_path=root/'evaluation'/row['id']/'status.json'
            if not status_path.is_file():
                continue
            status=json.loads(status_path.read_text())
            if status.get('state')=='completed':
                if status.get('manifest_sha256')!=MANIFEST_SHA or status.get('row')!=row:
                    raise ValueError(f'Completed pair has invalid provenance: {row["id"]}')
        output['result']=recorded
        if recorded['state']=='complete':
            output['classification']='pass' if recorded['preregistered_screen_passed'] else 'negative'
    except (OSError,ValueError,KeyError,TypeError,ImportError) as exc:
        output['validation_errors'].append(str(exc))
    return output

def result_block(record):
    result=record.get('result')
    if record['classification']=='pass':
        title='PASS on the eight development poses'
    elif record['classification']=='negative':
        title='No improvement under the preregistered development rule'
    else:
        title='Incomplete or unverifiable measurement; no improvement claim'
    text=f'**Runtime execution-horizon result: {title}.**'
    if result is not None:
        counts=result['counts']
        n=result['valid_pairs']
        text+=f' H10 settled **{counts["h10_settled"]}/{n}** versus H50 **{counts["h50_settled"]}/{n}** across {n}/{result["expected_pairs"]} valid matched pairs.'
        if record['classification'] in ('pass','negative'):
            text+=f' Difference: {result["h10_minus_h50_settled"]:+d} folds; one-sided exact paired p = {result["exact_paired_one_sided_p"]:.6g}.'
    if record['validation_errors']:
        text+=' Validation did not establish a complete result; the receipt records the missing input or provenance error.'
    text+=' The unchanged baseline weights remain the deliverable. A development-screen pass requires separately preregistered held-out confirmation before promotion; this update establishes no hardware-deployment or generalization result.'
    text+=' All results retain the 600-action policy budget and 60-step terminal settling protocol.'
    sha=record['source_summary_sha256'] or 'unavailable'
    text+=f' Source JSON SHA-256: `{sha}`. [Publication receipt](campaigns/20261006-runtime-horizon/analysis/publication_receipt.json).'
    if result is not None:
        text+=' [Measured result JSON](campaigns/20261006-runtime-horizon/analysis/results.json).'
    identity=digest(json.dumps({k:record[k] for k in ('classification','source_summary_sha256','manifest_sha256','validation_errors')},sort_keys=True).encode())
    return f'\n\n<!-- folding-runtime-result:{identity}:begin -->\n\n{text}\n\n<!-- folding-runtime-result:{identity}:end -->\n',identity

def append_existing(path,block,identity):
    """Use one O_APPEND write; preserve concurrent edits and avoid new docs."""
    marker=f'<!-- folding-runtime-result:{identity}:begin -->'
    for _ in range(3):
        # Deliberately omit O_CREAT: a removed document must never be recreated.
        fd=os.open(path,os.O_RDWR|os.O_APPEND)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX)
            inode=os.fstat(fd)
            chunks=[]
            while True:
                data=os.read(fd,1<<20)
                if not data:
                    break
                chunks.append(data)
            current=b''.join(chunks).decode('utf-8')
            live=os.stat(path)
            if (inode.st_dev,inode.st_ino)!=(live.st_dev,live.st_ino):
                continue
            if marker in current:
                return 'already_recorded'
            encoded=block.encode('utf-8')
            if os.write(fd,encoded)!=len(encoded):
                raise OSError('Short documentation append')
            os.fsync(fd)
        finally:
            os.close(fd)
        # An editor may replace the inode while it is locked. If so, append
        # the block to the current file rather than overwriting that edit.
        if marker in Path(path).read_text():
            return 'appended'
    raise RuntimeError(f'Document changed repeatedly during publication: {path}')

def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='publication-',suffix='.json',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--publication-manifest-sha256')
    args=ap.parse_args()
    if not args.dry_run:
        publication_path=ROOT/'runtime_publication/manifest.json'
        if not args.publication_manifest_sha256 or digest(publication_path.read_bytes())!=args.publication_manifest_sha256:
            raise SystemExit('Publication manifest differs from the submitted pin')
        publication=json.loads(publication_path.read_text())
        if publication['upstream_manifest_sha256']!=MANIFEST_SHA:
            raise SystemExit('Publication points to a different experiment')
        for path,expected in publication['source_sha256'].items():
            if digest(Path(path).read_bytes())!=expected:
                raise SystemExit(f'Publication source changed: {path}')
    record=verified_result()
    record.update(summary_job_id=SUMMARY_JOB,
                  publisher_job_id=os.environ.get('SLURM_JOB_ID'),
                  publication_manifest_sha256=args.publication_manifest_sha256,
                  updated_local=datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(timespec='seconds'))
    block,identity=result_block(record)
    docs=[REPO/'README.md',REPO/'PENDING.md']
    if not all(path.is_file() for path in docs):
        raise SystemExit('Publication requires both existing documents')
    if args.dry_run:
        print(json.dumps(dict(record,existing_docs=[str(p) for p in docs],proposed_block=block),indent=2))
        return
    target=REPO/'campaigns/20261006-runtime-horizon/analysis'
    # Preserve the exact current source bytes so the README's digest is
    # independently checkable; no new Markdown or text artifact is created.
    if record['result'] is not None:
        raw=Path(record['source_summary']).read_bytes()
        if digest(raw)!=record['source_summary_sha256']:
            raise RuntimeError('Summary changed during publication')
        target.mkdir(parents=True,exist_ok=True)
        (target/'results.json').write_bytes(raw)
    record['document_updates']={str(path):append_existing(path,block,identity) for path in docs}
    atomic_json(target/'publication_receipt.json',record)
    atomic_json(ROOT/'runtime_publication/last_receipt.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='result'},indent=2))

if __name__=='__main__':
    main()
