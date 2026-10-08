"""Summarize the frozen paired screen, retaining invalid or missing episodes."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

def exact_p(wins,losses):
    n=wins+losses
    return sum(math.comb(n,i) for i in range(wins,n+1))/2**n if n else 1.0

def summary(root):
    manifest=json.loads((root/'manifest.json').read_text())
    pairs=[]
    invalid={}
    per_pose=defaultdict(lambda:{'n':0,'h10_settled':0,'h50_settled':0})
    counts={'h10_settled':0,'h50_settled':0,'h10_ever':0,'h50_ever':0,
            'h10_only':0,'h50_only':0,'both':0,'neither':0}
    for row in manifest['pairs']:
        base=root/'evaluation'/row['id']
        try:
            status=json.loads((base/'status.json').read_text())
            if status['state']!='completed':
                raise ValueError(status['state'])
            for horizon in ('10','50'):
                if status['arms'][horizon]['state']!='completed':
                    raise ValueError('Incomplete arm')
            a,b=(status['arms'][str(h)] for h in (10,50))
            av,bv=a['terminal_success'],b['terminal_success']
            if not isinstance(av,bool) or not isinstance(bv,bool):
                raise ValueError('Malformed terminal verdict')
            pairs.append({'id':row['id'],'seed':row['seed'],'pose':row['development_pose_slot'],
                          'h10_settled':av,'h50_settled':bv,
                          'h10_wall_seconds':a['wall_seconds'],'h50_wall_seconds':b['wall_seconds']})
            counts['h10_settled']+=av
            counts['h50_settled']+=bv
            counts['h10_ever']+=a['ever_success']
            counts['h50_ever']+=b['ever_success']
            counts['both' if av and bv else 'h10_only' if av else 'h50_only' if bv else 'neither']+=1
            cell=per_pose[str(row['development_pose_slot'])]
            cell['n']+=1
            cell['h10_settled']+=av
            cell['h50_settled']+=bv
        except (OSError,ValueError,KeyError) as exc:
            invalid[row['id']]=str(exc)
    n=len(pairs)
    p=exact_p(counts['h10_only'],counts['h50_only'])
    gain=counts['h10_settled']-counts['h50_settled']
    complete=n==len(manifest['pairs'])
    passed=complete and gain>=5 and p<=.05
    return {'campaign':manifest['campaign'],'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'state':'complete' if complete else 'incomplete','valid_pairs':n,
            'expected_pairs':len(manifest['pairs']),'invalid_or_missing':invalid,
            'counts':counts,'h10_minus_h50_settled':gain,'exact_paired_one_sided_p':p,
            'preregistered_screen_passed':passed,
            'interpretation':('H10 runtime intervention passed the development screen; held-out confirmation is still required.'
                              if passed else 'Development screen did not pass; retain the baseline checkpoint and avoid an improvement claim.'
                              if complete else 'Incomplete measurement; no improvement claim.'),
            'per_pose':dict(per_pose),'pairs':pairs,'protocol':manifest['protocol'],
            'observed_gpu_hours':sum(x['h10_wall_seconds']+x['h50_wall_seconds'] for x in pairs)/3600}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--write',action='store_true')
    args=ap.parse_args()
    result=summary(args.campaign.resolve())
    if args.write:
        (args.campaign/'analysis').mkdir(exist_ok=True)
        (args.campaign/'analysis/results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','invalid_or_missing','protocol','per_pose')},indent=2))

if __name__=='__main__':
    main()
