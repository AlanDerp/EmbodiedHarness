# RoboDojo ASPIRE fix-loop role contract

Adapted from ASPIRE iterative-debugging, revision fa00f0ef2bc1d91d594b9aa502c22133c9f538f7.
Apache-2.0 provenance is recorded in THIRD_PARTY_NOTICES.md. This adaptation uses RoboDojo layouts, not LIBERO seeds.

All roles: solve only the public operator goal. Use supplied RGB/depth/calibration and SAM3 via the public API.
Never inspect simulator state, rewards, native object poses, layout files, hidden scoring code, external baseline programs,
previous solutions or filesystem/network. Do not invent evidence or claim native task success from visual review.
Public API schema is in context.public.public_api (or context.public_api for interactive Relay).

## generate
Return JSON {goal, subtasks:[{task_id,goal,source,parameters,relation}]}.
Start from public observations and existing verified library. Plan subtasks in order. Source defines skill_init(parameters),
skill_observe(state), skill_context(state), skill_options(state), skill_tick(state, option). Only import math; constructors Pose,
ArmCommand, RobotAction and public tools are injected. No classes, private attributes, reflection, file/network access.
Use SAM3 localize_object for object segmentation and RGB-D geometry; never substitute ground-truth masks or fixed object coordinates.
Never infer object identity just by largest mask: ambiguities require explicit visible evidence or hand back to System 2.

System 1 gets a host-validated parameter_scope with stage/local_mode/contact_mode/evidence_valid/parameters/adjustments.
Its options assign physical numeric parameters (translation_step_m, rotation_step_rad, target_dx_m/target_dy_m/target_dz_m,
opening_step) within host limits. skill_tick receives 'advance'; assigned values are in state['runtime_parameters'].
Exactly one bounded native submission per tick. Observe again after action. Treat failed primitives/unknown/low-confidence
or stale evidence as handoff, never false completion. Contact/pregrasp/release requires fresh SAM3 geometry.
A complete reference hook implementation is shipped in examples/ee_skill.py; use the hook schemas in the public context.

## repair
Read development_reports (campaign), or report + active_index + plan (interactive Relay).
Identify the common failure mode; fix the smallest mechanism rather than memorize a scene, fixed pose or trace.
Return the same Plan schema and preserve goal. Interactive repairs retain completed prefix, current task ID and subgoal.
Campaign retests every dev scene for each revision. Never request unseen/held-out data or tune after evaluation.

## review
Return {snapshot_id, verdict:'true'|'false'|'unknown', evidence: nonempty visible reason} tied to the current public image.
This can verify a subtask relation only. It is never native task success.

## propose
Return {skill_report: [{skill: existing library path, verdict: helpful|unhelpful|unused, evidence: development evidence}], skills: {'skills/name.md': reusable skill description, 'skills/name.py': optional reusable pure helper source}}.
Propose transferable methods and failure lessons, with inputs, expected evidence, limitations and recovery conditions.
No exact solution, task-specific coordinates, layout IDs, replay traces or privileged data. Return {skill_report: [{skill: existing library path, verdict: helpful|unhelpful|unused, evidence: development evidence}], skills:{}} if no useful knowledge.
All Python helpers obey the same source contract. Existing library files cannot be overwritten.

## verify
Independently reconstruct a Plan using only public goal/observation/API and supplied library including proposed skills.
You do not receive the proposer solution or its reports. Return the Plan schema. A fresh execution on dev scenes,
not your confidence or self-report, decides whether the candidate library is promoted. At most two independent attempts.

Hook status is active/done/unknown. skill_observe returns a feedback dict with source; skill_context returns status and parameter_scope; skill_options returns numeric option names/descriptions. Initialize runtime_parameters in skill_init. Never return raw RobotAction from tick: submit it through step(action) and return its receipt.
