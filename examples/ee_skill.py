"""Python-hook example: measured EE arrival only, no claim of object manipulation."""

import math


def pose_error(measured, target):
    distance = math.sqrt(sum((target.position[i] - measured.position[i]) ** 2 for i in range(3)))
    dot = abs(sum(measured.quaternion_wxyz[i] * target.quaternion_wxyz[i] for i in range(4)))
    return distance, 2 * math.acos(max(0, min(1, dot)))


def skill_init(parameters):
    return {"arm": parameters["arm"], "target": Pose(**parameters["target"]), "status": "active",
            "runtime_parameters": {"translation_step_m": .01}}


def skill_observe(state):
    state["observation"] = get_observation()
    measured = state["observation"].robot_state.arms[state["arm"]].ee_pose
    state["position_error"], state["rotation_error"] = pose_error(measured, state["target"])
    state["status"] = "done" if state["position_error"] <= .003 and state["rotation_error"] <= .01 else "active"
    return {"source": state["observation"].source}


def skill_context(state):
    return {"status": state["status"], "position_error_m": state["position_error"],
            "rotation_error_rad": state["rotation_error"], "target_pose": state["target"],
            "active_arm": state["arm"], "parameter_scope": {
                "stage": "servo", "local_mode": "adaptive", "contact_mode": "free_space", "evidence_valid": True,
                "parameters": {"translation_step_m": {"unit": "m", "min": .001, "max": .01,
                    "value": state["runtime_parameters"]["translation_step_m"]}},
                "adjustments": {"fine": {"parameter": "translation_step_m", "value": .001},
                                "normal": {"parameter": "translation_step_m", "value": .01}}}}


def skill_options(state):
    return {} if state["status"] == "done" else {"fine": "1mm / 0.01rad", "normal": "10mm / 0.1rad"}


def skill_tick(state, option):
    action = servo_action(state["observation"], state["arm"], state["target"],
                          translation_m=state["runtime_parameters"]["translation_step_m"], rotation_rad=.1)
    return step(action)
