"""CPU checks for failure accounting, exact paired inference, and output rules."""
import json
from pathlib import Path
import tempfile

from guarded_runner import deny_new_doc, install_guard
from run_pair import validate
from summarize import exact_p, summary

def main():
    assert exact_p(5,0)==1/32
    assert exact_p(0,5)==1
    assert exact_p(0,0)==1
    assert abs(exact_p(3,2)-0.5)<1e-12
    row={'garment':'Pant_Short_Seen_0','seed':1,'steps':600}
    result=dict(garment=row['garment'],seed=1,steps=600,effective_n_action_steps=10,
                prediction_chunk_size=50,terminal_settle_steps=60,physics_finite=True,
                robot_finite=True,terminal_success=False,terminal_checker={'success':False})
    validate(result,row,10)
    for key,bad in [('seed',2),('steps',599),('effective_n_action_steps',50),
                    ('terminal_settle_steps',0),('physics_finite',False),('terminal_success',True)]:
        try:
            validate(dict(result,**{key:bad}),row,10)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid {key} was accepted')
    with tempfile.TemporaryDirectory(prefix='folding-protocol-',dir='/tmp') as temporary:
        root=Path(temporary)
        rows=[{'id':f'pair{i}','seed':i,'development_pose_slot':i%8} for i in range(48)]
        (root/'manifest.json').write_text(json.dumps({'campaign':'test','pairs':rows,'protocol':{}}))
        incomplete=summary(root)
        assert incomplete['state']=='incomplete' and incomplete['valid_pairs']==0
        assert not incomplete['preregistered_screen_passed']
        for i,row in enumerate(rows):
            dest=root/'evaluation'/row['id']
            dest.mkdir(parents=True)
            arms={str(h):dict(state='completed',terminal_success=(i<5 if h==10 else False),
                             ever_success=True,wall_seconds=100) for h in (10,50)}
            (dest/'status.json').write_text(json.dumps({'state':'completed','arms':arms}))
        passing=summary(root)
        assert passing['valid_pairs']==48 and passing['preregistered_screen_passed']
        assert passing['counts']['h10_only']==5 and passing['counts']['h50_only']==0
        assert passing['exact_paired_one_sided_p']==1/32
        (root/'evaluation'/rows[0]['id']/'status.json').write_text(json.dumps({'state':'infrastructure_error'}))
        broken=summary(root)
        assert broken['valid_pairs']==47 and not broken['preregistered_screen_passed']
        assert broken['counts']['h10_settled']==4
        install_guard()
        for name in ('blocked.md','blocked.txt'):
            try:
                (root/name).write_text('should never write')
            except PermissionError:
                assert not (root/name).exists()
            else:
                raise AssertionError(f'New {name} was allowed')
        (root/'permitted.json').write_text('{}')
        assert (root/'permitted.json').exists()
        deny_new_doc(root/'readable.txt',False)
    print(json.dumps({'passed':True,'checks':['exact paired tails','episode invariants',
                    'missing and invalid data not counted as folds','complete-screen gate',
                    'new documentation writes denied; JSON retained']}))

if __name__=='__main__':
    main()
