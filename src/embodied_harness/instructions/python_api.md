# Public Python runtime contract

The JSON representation in the model message is serialization only. At execution time public tools return typed Python objects, NOT dictionaries. Use the attributes below. Only your skill state/parameters/options are dictionaries.

Names: every variable, helper function and attribute must avoid a leading underscore, including `_` loop variables. Do not use reflection, getattr/setattr, file/network/OS functions, classes or private/native fields. Only `import math` is allowed. NumPy is not importable; public numeric arrays support indexing, iteration and .tolist(). Builtins include abs, all, any, bool, dict, enumerate, float, int, len, list, max, min, print, range, reversed, round, set, sorted, str, sum, tuple, zip, isinstance and Exception/RuntimeError/ValueError/TypeError.

## Objects and return values
- get_observation() -> Observation: `.source`, `.instruction`, `.cameras` (mapping), `.robot_state`. Never call obs.get(...) or obs['source'].
- get_robot_state() -> RobotState: `.embodiment`, `.arms` mapping. Each ArmState has `.ee_pose`, `.joint_positions`, `.joint_names`, `.gripper_position`.
- ObservationSource: `.episode_id`, `.snapshot_id`, `.reset_epoch`, `.state_version`, `.native_action_count`.
- Pose(position, quaternion_wxyz, frame): ALL three arguments required. World targets use frame='world'. Access `.position` and `.quaternion_wxyz`.
- ArmCommand(gripper_position, ee_target=pose) and RobotAction(source, arms) are injected constructors. Prefer the public servo_action tool for bounded pose increments.
- servo_action(observation, arm, target, opening=None, translation_m=.01, rotation_rad=.1) -> RobotAction. It does not move. Submit it with step(action), returning that StepReceipt from skill_tick. Pose targets must derive from public observations; the other arms, if any, are held at their measured pose.
- step(action) -> StepReceipt: `.observation`, `.terminated`, `.truncated`. Do not read private rewards or success fields. The host independently checks native success.
- segment_text(episode_id,snapshot_id,camera_id,query) / segment_point(...,point_xy) -> Result with `.ok`, `.data`, `.error`. Data is a tuple of Segmentation objects; each has `.mask`, `.box_xyxy`, `.score`, `.source`. If false, `.error.code.value` and `.error.message` explain the failure.
- localize_object(episode_id,snapshot_id,camera_id,query,candidate_index=None) -> Result[Localization]. On success, result.data.geometry is already ObjectGeometry: `.center`, `.extents`, `.axes`, `.frame`, `.source`, `.orientation_ambiguous`. Do not pass a Localization/Result to mask_to_point_cloud or estimate_geometry.
- On ambiguous localization, result.ok is false but result.data.candidates can contain candidates. Inspect their public boxes/masks; choose an index only from observed identity evidence. Do not choose by largest mask alone.
- mask_to_point_cloud(segmentation,target_frame='world') -> Result[Projection]; successful result.data.cloud is PointCloud.
- estimate_geometry(cloud) -> Result[ObjectGeometry]. A PointCloud has `.points`, `.frame`, `.source`.
- Cameras have `.rgb`, `.depth_m`, `.intrinsics`, `.T_world_from_camera_optical`. Select an existing name from observation.cameras; do not invent a head camera when only agentview and robot0_eye_in_hand exist.

Example of typed localization access (an interface example, not a task solution):
```python
obs = get_observation()
result = localize_object(obs.source.episode_id, obs.source.snapshot_id, 'agentview', query)
if result.ok and result.data.geometry is not None:
    center = result.data.geometry.center
    target = Pose([float(center[0]), float(center[1]), float(center[2]) + clearance],
                  obs.robot_state.arms['right'].ee_pose.quaternion_wxyz.tolist(), 'world')
```
Handle errors explicitly; never swallow all tool exceptions and continue issuing motion.

## Hook contracts and numeric menu
skill_init(parameters) returns a state dict, initializes runtime_parameters, and takes no action.
skill_observe(state) calls get_observation(), updates state, and returns {'source': obs.source}. It may use public perception, but takes no action.
skill_context(state) returns status='active'/'done'/'unknown', optionally typed target_pose and active_arm, plus parameter_scope while active.
skill_options(state) returns named options matching adjustments; skill_tick(state, option) receives 'advance', reads state['runtime_parameters'], submits at most ONE action, and returns its receipt. Do not mutate the host-assigned runtime parameters inside skill_tick.

An adaptive parameter_scope has exactly these keys:
```python
{'stage': 'approach', 'local_mode': 'adaptive', 'contact_mode': 'free_space',
 'evidence_valid': True,
 'parameters': {'translation_step_m': {'unit': 'm', 'min': .001, 'max': .01,
                                     'value': state['runtime_parameters']['translation_step_m']}},
 'adjustments': {'fine': {'parameter': 'translation_step_m', 'value': .001},
                 'normal': {'parameter': 'translation_step_m', 'value': .01}}}
```
Options then are {'fine':'1 mm increment','normal':'10 mm increment'}. Adaptive mode requires 2–12 distinct actual numeric choices; stage transitions stay in code, not the menu. Other supported parameters: rotation_step_rad (.0001–.15 rad), target_dx_m/target_dy_m/target_dz_m (-.03–.03 m), opening_step (.001–.05 normalized). Translation bounds are .0001–.03 m. Each declaration uses unit/min/max/value; each assignment uses parameter/value.
Contact modes: free_space, pregrasp, contact, release, recovery. Contact/pregrasp/release needs fresh successful SAM3 localization tied to the current physical state. Code-owned routine mode is ONLY for home/return_home; it is not a bypass for normal task actions.

Each delegated skill is bounded to 100 actions, 150 decisions and 180 seconds; the session also has its submission and native budgets. These are ceilings, not a guarantee of successful manipulation. Preserve failures and hand back unknown when evidence is insufficient.

## System 2 prior with optional Jev override
When public.parameter_selection.mode is `system2_prior_jev_override`, every active adaptive `skill_context` MUST also return `parameter_prior = {'option': <one of skill_options keys>, 'reason': <nonempty reason based on current public tool measurements>, 'source': <the current Observation.source>}`. Compute this prior in the System 2-authored program from each fresh observation, before Jev is queried. Do not call a live System 2 model within frozen evaluation. A missing, invalid, or stale prior stops execution; never choose an arbitrary first option as a fallback.
The host sends Jev only structured tool-derived measurements and bounded numeric options, without images, image paths, or the prior's free-text reason. Jev overrides the prior ONLY if its validated confidence is strictly greater than the configured threshold (0.8 for this condition). At or below the threshold, or if the read-only Jev request fails or returns invalid data, retain the validated System 2 prior. Both candidates, confidence, threshold and final choice are audited. The host still enforces all physical bounds, fresh visual evidence requirements and native-success checks; low Jev confidence is not permission to bypass those gates.
