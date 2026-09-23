"""Offline, matched analysis; never changes policy behavior or submits training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]

def stats(values):
    values = sorted(values)
    if not values:
        return {'n': 0, 'mean': None, 'max': None, 'p95': None}
    return {'n': len(values), 'mean': statistics.mean(values), 'max': max(values),
            'p95': values[min(len(values)-1, int(.95 * (len(values)-1)))]}


def summarize(result, row, status, directory, output):
    geometry = result['geometry_trajectory']
    policy = [r for r in geometry if r['phase'] == 'policy']
    terminal = result['terminal_checker']
    behavior = result.get('behavior_telemetry')
    trace = []
    if behavior:
        trace = [json.loads(x) for x in Path(behavior['path']).read_text().splitlines()]
    margins = {key: v['margin_cm'] for key, v in terminal['details'].items()}
    worst = min(margins, key=margins.get)
    closest = min(margins, key=lambda key: abs(margins[key]))
    first = result.get('first_success_step')
    dt = result['physics_dt'] * result['decimation']
    nearby = behavior['minimum_gripper_link_origin_distance_m'] if behavior else {
        side: min(r['last_link_distance_m'][side] for r in result['manipulation_trajectory'])
        for side in ('left', 'right')}
    if terminal['success']:
        failure = None
    elif result['geometric_ever_success'] or result['official_ever_success']:
        failure = 'success_then_unfolded'
    elif min(nearby.values()) > .05:
        failure = 'never_approached_cloth_proxy'
    else:
        failure = f"partial_fold_{terminal['conditions_passed']}_of_{terminal['conditions_total']}"
    commands = None
    boundary, ordinary = [], []
    camera = {'left_rgb': [], 'right_rgb': [], 'top_rgb': []}
    camera_manipulation = {key: [] for key in camera}
    if behavior:
        boundary = [x['l2_rad'] for x in behavior['replan_boundary_jumps']]
        ordinary = [r['all_step_jump_l2_rad'] for r in trace if not r['replan_boundary'] and r['all_step_jump_l2_rad'] is not None]
        for r in trace:
            changes = r['camera_change_since_previous_replan']
            if changes:
                for key in camera:
                    camera[key].append(changes[key]['mean_abs_rgb_0_255'])
                    if r['p95_particle_lift_m'] > .02:
                        camera_manipulation[key].append(changes[key]['mean_abs_rgb_0_255'])
        commands = {'near': behavior['first_decreasing_command_near_cloth_proxy'],
                    'away': behavior['first_decreasing_command_away_from_cloth_proxy']}
    item = {'id': row['id'], 'pose': row.get('development_pose_slot'),
        'garment': row['garment'], 'seed': row['seed'], 'horizon': row['horizon'],
        'infrastructure_valid': True, 'status': status['state'],
        'official_ever_success': result['official_ever_success'],
        'terminal_success': terminal['success'],
        'first_official_success_action': first,
        'first_official_success_simulation_seconds': None if first is None else first * dt,
        'fresh_geometric_ever_success': result['geometric_ever_success'],
        'geometric_first_success_action': result['geometric_first_success_action'],
        'geometric_success_persisted': result['geometric_success_persisted'],
        'best_policy_conditions': max(r['conditions_passed'] for r in policy),
        'best_including_settle_conditions': max(r['conditions_passed'] for r in geometry),
        'terminal_conditions': terminal['conditions_passed'], 'conditions_total': terminal['conditions_total'],
        'terminal_margins_cm': margins,
        'worst_condition': worst, 'worst_margin_cm': margins[worst],
        'closest_threshold_condition': closest, 'closest_threshold_margin_cm': margins[closest],
        'cloth_motion': result['cloth_motion'],
        'minimum_gripper_link_origin_distance_m': nearby,
        'distance_basis': 'named jaw/gripper origins' if behavior else 'legacy last-link origin',
        'maximum_particle_lift_m': behavior['maximum_particle_lift_m'] if behavior else None,
        'first_contact': None, 'first_grasp': None, 'failed_grasp_count': None,
        'contact_status': 'unmeasured; proximity does not establish contact or grasp',
        'decreasing_command_events_proxy': commands,
        'failure_category': failure,
        'failure_basis': 'earliest identifiable lack of approach from origin proximity, otherwise geometric outcome; no grasp claims',
        'replans': result['replanning_count'], 'inference_seconds': stats(result['inference_seconds']),
        'total_inference_seconds': sum(result['inference_seconds']),
        'boundary_action_jump_l2_rad': stats(boundary), 'nonboundary_action_jump_l2_rad': stats(ordinary),
        'all_action_jump_max_l2_rad': result['max_action_discontinuity_l2'],
        'camera_change_mean_abs_rgb': {key: stats(values) for key, values in camera.items()},
        'camera_change_during_p95_lift_over_2cm_proxy': {key: stats(values) for key, values in camera_manipulation.items()},
        'rollout_wall_seconds': result['wall_seconds'], 'worker_wall_seconds': status['wall_seconds'],
        'telemetry_wall_seconds': behavior['measurement_wall_seconds'] if behavior else None,
        'fresh_camera_acquisitions': len(result['render_integrity']['acquisitions']),
        'minimum_garment_pixels': min(r['top_garment_pixels'] for r in result['render_integrity']['acquisitions']),
        'result': str(directory/'rollout.json'),
        'result_sha256': hashlib.sha256((directory/'rollout.json').read_bytes()).hexdigest(),
        'policy_mp4': result['media']['mp4']['path']}
    with (output / (row['id'] + '_conditions.csv')).open('w') as f:
        keys = ['phase', 'step', 'conditions_passed', 'conditions_total', 'success']
        conditions = list(terminal['details'])
        writer = csv.DictWriter(f, fieldnames=keys+[k+'_margin_cm' for k in conditions])
        writer.writeheader()
        for record in geometry:
            writer.writerow({**{key: record[key] for key in keys},
                             **{key+'_margin_cm': record['details'][key]['margin_cm'] for key in conditions}})
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        x = [r['step'] for r in geometry]
        axes[0].plot(x, [r['conditions_passed'] for r in geometry])
        axes[0].set(ylabel='Conditions passed', ylim=(-.1, terminal['conditions_total']+.1), title=row['id'])
        for key in terminal['details']:
            axes[1].plot(x, [r['details'][key]['margin_cm'] for r in geometry], label=key)
        axes[1].axhline(0, color='black', linewidth=.7)
        for ax in axes:
            ax.axvline(result['steps'], color='gray', linestyle='--')
        axes[1].set(xlabel='Policy action / settling step', ylabel='Signed margin (cm)')
        axes[1].legend(fontsize=7)
        fig.tight_layout();fig.savefig(output/(row['id']+'_conditions.png'), dpi=140);plt.close(fig)
    except ImportError:
        item['plot_status'] = 'matplotlib unavailable; complete CSV timeline saved'
    return item


def analyze(root, mode, output):
    manifest = json.loads((root/'manifest.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root/'scripts'))
    from run_improvement_task import validate_result
    tasks = manifest['pilot_tasks'] if mode == 'pilot' else manifest['smoke_tasks'][:5]
    rows=[]
    for row in tasks:
        directory = root/'outputs'/row['id']
        try:
            status=json.loads((directory/'status.json').read_text())
            if status['state']!='completed':
                raise ValueError(status['state'])
            result=json.loads((directory/'rollout.json').read_text())
            validate_result(result,row)
            if status['success']!=result['success'] or status['terminal_success']!=result['terminal_success']:
                raise ValueError('status/result disagreement')
            rows.append(summarize(result,row,status,directory,output))
        except (OSError,ValueError,KeyError) as exc:
            rows.append({'id':row['id'],'pose':row.get('development_pose_slot'),'garment':row['garment'],
                         'horizon':row['horizon'],'infrastructure_valid':False,'reason':str(exc),
                         'official_ever_success':None,'terminal_success':None})
    valid=[r for r in rows if r['infrastructure_valid']]
    aggregate={}
    for horizon in sorted({r['horizon'] for r in rows}, reverse=True):
        group=[r for r in valid if r['horizon']==horizon]
        aggregate[str(horizon)]={'valid_episodes':len(group),
            'invalid_or_missing':sum(r['horizon']==horizon and not r['infrastructure_valid'] for r in rows),
            'ever_successes':sum(r['official_ever_success'] for r in group),
            'terminal_successes':sum(r['terminal_success'] for r in group),
            'mean_best_conditions':statistics.mean(r['best_policy_conditions'] for r in group) if group else None,
            'mean_terminal_conditions':statistics.mean(r['terminal_conditions'] for r in group) if group else None}
    comparisons={}
    for horizon in [10,5] if mode=='pilot' else []:
        pairs=[]
        for pose in range(8):
            a=next((r for r in valid if r['pose']==pose and r['horizon']==50),None)
            b=next((r for r in valid if r['pose']==pose and r['horizon']==horizon),None)
            if a and b:
                pairs.append({'pose':pose,'ever_delta':int(b['official_ever_success'])-int(a['official_ever_success']),
                    'terminal_delta':int(b['terminal_success'])-int(a['terminal_success']),
                    'terminal_conditions_delta':b['terminal_conditions']-a['terminal_conditions'],
                    'best_conditions_delta':b['best_policy_conditions']-a['best_policy_conditions'],
                    'worst_margin_delta_cm':b['worst_margin_cm']-a['worst_margin_cm'],
                    'closest_threshold_margin_delta_cm':b['closest_threshold_margin_cm']-a['closest_threshold_margin_cm'],
                    'terminal_mean_cloth_displacement_delta_m':b['cloth_motion']['terminal_mean_particle_displacement']-a['cloth_motion']['terminal_mean_particle_displacement'],
                    'minimum_gripper_origin_distance_delta_m':min(b['minimum_gripper_link_origin_distance_m'].values())-min(a['minimum_gripper_link_origin_distance_m'].values()),
                    'replans_delta':b['replans']-a['replans'],
                    'total_inference_seconds_delta':b['total_inference_seconds']-a['total_inference_seconds'],
                    'boundary_jump_mean_delta_rad':None if b['boundary_action_jump_l2_rad']['mean'] is None or a['boundary_action_jump_l2_rad']['mean'] is None else b['boundary_action_jump_l2_rad']['mean']-a['boundary_action_jump_l2_rad']['mean'],
                    'worker_wall_seconds_delta':b['worker_wall_seconds']-a['worker_wall_seconds']})
        comparisons[str(horizon)]={'complete_pairs':len(pairs),'pairs':pairs,
            'terminal_wins':sum(p['terminal_delta']>0 for p in pairs),'terminal_losses':sum(p['terminal_delta']<0 for p in pairs),
            'ever_wins':sum(p['ever_delta']>0 for p in pairs),'ever_losses':sum(p['ever_delta']<0 for p in pairs)}
    complete=len(valid)==len(tasks)
    evidence={'mode':mode,'complete_valid_protocol':complete,'valid':len(valid),'expected':len(tasks),
        'aggregate':aggregate,'matched_comparisons_vs_50':comparisons,'rows':rows,
        'historical_8_of_24_comparison_valid':False,
        'contact_ground_truth_measured':False,
        'interpretation':'No inference from missing/invalid episodes. Pilot is exploratory; no public improvement claim or training release.',
        'next_step':'Inspect matched terminal/ever wins and margins, action boundaries and footage before freezing any formal evaluation horizon.' if complete else 'Wait for all results or repair infrastructure; do not select a horizon from incomplete pairs.'}
    (output/'summary.json').write_text(json.dumps(evidence,indent=2)+'\n')
    fields=['id','pose','garment','horizon','infrastructure_valid','official_ever_success','terminal_success','best_policy_conditions','terminal_conditions','conditions_total','worst_margin_cm','closest_threshold_margin_cm','replans','worker_wall_seconds','failure_category']
    with (output/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    lines=['# Internal folding analysis', '', f'Infrastructure-valid results: **{len(valid)}/{len(tasks)}**. Missing and invalid episodes are not policy failures.', '',
        'No direct comparison to historical 8/24; contacts/acquisition remain unmeasured. No public improvement claim.', '',
        '| Horizon | Valid | Invalid/missing | Ever-success | Settled terminal |', '|---|---:|---:|---:|---:|']
    for h,a in aggregate.items():lines.append(f"| {h} | {a['valid_episodes']} | {a['invalid_or_missing']} | {a['ever_successes']} | {a['terminal_successes']} |")
    if mode=='pilot':
        lines+=['','Cells: ever / terminal; best → terminal conditions; worst margin cm; replans; wall seconds. Full displacement, proximity, all margins, camera/boundary metrics and paired deltas are in summary.json.','','| Pose | Garment / local pose | H50 | H10 | H5 |','|---|---|---|---|---|']
        for pose in range(8):
            task=next(r for r in tasks if r['development_pose_slot']==pose)
            cells=[]
            for h in [50,10,5]:
                r=next(r for r in rows if r['pose']==pose and r['horizon']==h)
                cells.append(f"{int(r['official_ever_success'])}/{int(r['terminal_success'])}; {r['best_policy_conditions']}→{r['terminal_conditions']}/{r['conditions_total']}; {r['worst_margin_cm']:.2f}; {r['replans']}; {r['worker_wall_seconds']:.1f}" if r['infrastructure_valid'] else 'INVALID / pending')
            lines.append(f"| {pose} | {task['garment']} / {task['pose_source_local_episode_key']} | "+' | '.join(cells)+' |')
    if mode=='pilot':
        lines+=['','Matched behavior cells: closest-threshold margin cm; mean terminal cloth displacement m; minimum gripper-origin distance m; boundary jump mean rad; total inference seconds. Contacts and grasps remain unknown.','','| Pose | H50 | H10 | H5 |','|---|---|---|---|']
        for pose in range(8):
            cells=[]
            for h in [50,10,5]:
                r=next(r for r in rows if r['pose']==pose and r['horizon']==h)
                if r['infrastructure_valid']:
                    jump=r['boundary_action_jump_l2_rad']['mean']
                    jump_text='unknown' if jump is None else f'{jump:.3f}'
                    cells.append(f"{r['closest_threshold_margin_cm']:.2f}; {r['cloth_motion']['terminal_mean_particle_displacement']:.4f}; {min(r['minimum_gripper_link_origin_distance_m'].values()):.4f}; {jump_text}; {r['total_inference_seconds']:.1f}")
                else:
                    cells.append('INVALID / pending')
            lines.append(f'| {pose} | '+' | '.join(cells)+' |')
    lines+=['',evidence['next_step'],'','Condition CSVs/PNGs include the terminal settling period. Proximity and decreasing command events are proxies, not contact, grasp or failed-close ground truth.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'mode':mode,'valid':len(valid),'expected':len(tasks),'complete':complete,'output':str(output)}))

if __name__=='__main__':
    source_hashes = json.loads((ROOT/'analysis/source_sha256.json').read_text())
    for relative, expected in source_hashes.items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen analysis source changed: '+relative)
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--mode',choices=['pilot','smoke'],default='pilot');p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.root.resolve(),a.mode,a.output.resolve())
