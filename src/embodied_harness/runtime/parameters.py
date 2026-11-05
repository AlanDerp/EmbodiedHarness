"""Host-validated execution parameters adapted from EmbodiedDualLoop's parameter policy."""

import math

LIMITS = {
    "translation_step_m": ("m", .0001, .03), "rotation_step_rad": ("rad", .0001, .15),
    "target_dx_m": ("m", -.03, .03), "target_dy_m": ("m", -.03, .03), "target_dz_m": ("m", -.03, .03),
    "opening_step": ("normalized", .001, .05)}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def parameter_menu(context, options):
    scope = context.get("parameter_scope")
    if not isinstance(scope, dict) or set(scope) != {"stage", "local_mode", "contact_mode", "evidence_valid", "parameters", "adjustments"}:
        raise ValueError("parameter_scope requires stage/mode/contact/evidence/parameters/adjustments")
    if scope["local_mode"] not in ("adaptive", "routine", "recovery"):
        raise ValueError("invalid local mode")
    if scope["contact_mode"] not in ("free_space", "pregrasp", "contact", "release", "recovery"):
        raise ValueError("invalid contact mode")
    if type(scope["evidence_valid"]) is not bool:
        raise ValueError("evidence_valid must be boolean")
    specs, assignments = scope["parameters"], scope["adjustments"]
    if not isinstance(specs, dict) or not isinstance(assignments, dict):
        raise ValueError("parameter declarations must be objects")
    for name, spec in specs.items():
        if name not in LIMITS or not isinstance(spec, dict) or set(spec) != {"unit", "min", "max", "value"}:
            raise ValueError("unknown execution parameter or schema")
        unit, low, high = LIMITS[name]
        if (spec["unit"] != unit or not all(finite(spec[key]) for key in ("min", "max", "value"))
                or not low <= spec["min"] <= spec["value"] <= spec["max"] <= high):
            raise ValueError("parameter unit/value/range outside host bounds")
    if scope["local_mode"] != "adaptive":
        if assignments or set(options) != {"advance"}:
            raise ValueError("routine/recovery must use code-owned advance only")
        if scope["local_mode"] == "routine" and scope["stage"] not in ("home", "return_home"):
            raise ValueError("routine mode only permitted for home")
        return scope, {"advance": "Execute code-owned bounded home/recovery step"}
    if not 2 <= len(assignments) <= 12 or set(options) != set(assignments):
        raise ValueError("adaptive mode requires 2..12 actual numeric choices")
    menu = {}
    values = set()
    for option, assignment in assignments.items():
        if not isinstance(option, str) or not option.isidentifier() or not isinstance(assignment, dict) or set(assignment) != {"parameter", "value"}:
            raise ValueError("each option must assign one named parameter")
        name, value = assignment["parameter"], assignment["value"]
        if name not in specs or not finite(value) or not specs[name]["min"] <= value <= specs[name]["max"]:
            raise ValueError("choice outside declared range")
        values.add((name, value))
        menu[option] = f"Set {name} = {value} {specs[name]['unit']} in the current code-owned stage"
    if len(values) < 2:
        raise ValueError("numeric choices must have distinct physical values")
    return scope, menu


def measured_state(observation, localizations=(), *, target_pose=None, active_arm=None):
    """No System 2 free text, option coaching or custom context is sent to System 1."""
    source = observation.source
    objects = []
    for result in localizations:
        if result.ok and result.data.geometry is not None:
            geometry = result.data.geometry
            if geometry.source.same_state(source):
                objects.append({"source_snapshot_id": geometry.source.snapshot_id,
                                "center_world_m": geometry.center.tolist(), "extents_m": geometry.extents.tolist(),
                                "orientation_ambiguous": geometry.orientation_ambiguous})
    state = {"native_action_count": source.native_action_count, "state_version": source.state_version,
            "arms": {name: {"position_m": arm.ee_pose.position.tolist(),
                             "quaternion_wxyz": arm.ee_pose.quaternion_wxyz.tolist(),
                             "gripper_position": arm.gripper_position} for name, arm in observation.robot_state.arms.items()},
            "objects": objects}
    if target_pose is not None:
        from embodied_harness.contracts import Pose
        from embodied_harness.control import pose_error
        if not isinstance(target_pose, Pose) or active_arm not in observation.robot_state.arms:
            raise ValueError("typed target pose and valid arm required")
        measured = observation.robot_state.arms[active_arm].ee_pose
        position, rotation = pose_error(measured, target_pose)
        state.update(target_position_m=target_pose.position.tolist(),
                     position_error_m=position, rotation_error_rad=rotation,
                     target_minus_measured_m=(target_pose.position-measured.position).tolist())
    return state
