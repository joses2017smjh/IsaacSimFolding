# Failure instrumentation and matched trajectory analysis

This directory analyzes the completed 24-rollout pilot without replaying or changing any episode. `events.jsonl` contains one record per executed action (14,400 records): the raw simulator-state/action telemetry is nested under `raw`; explicitly derived proximity, motion-coupling and stage fields are under `derived`. `matched-comparison.csv` and `matched-comparison.json` compare each H10/H5 pose against its same-pose H50 trajectory. `episode-summary.json` contains episode-level labels and measurement limits. `plots/pose{1,3,5,7}_comparison.svg` are compact synchronized diagnostics with replan lines, first-divergence lines, proximity markers, gripper commands, condition counts and particle-displacement traces. `make_plots.py` regenerates them offline.

The pilot did not enable a validated Isaac Lab `ContactSensor` or particle-to-rigid contact-pair stream. The particle cloth view exposes positions and velocities, but that is not a contact report. Consequently `native_contact` is null, contact start/end, contacting gripper, acquisition, retention and release/drop remain unknown. A named jaw/gripper-origin distance <=3 cm plus particle lift >=2 cm is retained only as a `proxy_coupling` event; it is not called contact or grasp.

The raw fields include action targets, pre-action measured joints, simulator time, replan boundary, boundary jump, named jaw/gripper origin and nearest particle coordinates, particle lift summaries, condition values/margins, camera-change samples and command deltas. Finite-difference gripper/particle velocities and velocity alignment are derived from those raw simulator positions. Derived stage labels are conservative:

- `A_never_reached_cloth_proxy`: no named jaw/gripper origin reached 5 cm.
- `B_reached_cloth_no_verified_contact`: reached 5 cm but no coupling proxy.
- `C_contact_or_coupling_unknown_proxy`: five consecutive proximity/motion-coupling proxy samples occurred; true contact remains unknown.
- `F_partial_condition_regression` or `F_local_success_then_terminal_failure`: geometry changed after a best intermediate state.
- `G_official_task_success`: official ever and settled success.

The analysis does not retroactively turn proximity into contact. It reports simulator facts, derived proxies and hypotheses separately in the report.
