"""Python-hook example: measured EE arrival only, no claim of object manipulation."""

from embodied_harness.contracts import Pose
from embodied_harness.control import pose_error, servo_action


def skill_init(parameters):
    return {"arm": parameters["arm"], "target": Pose(**parameters["target"]), "status": "active"}


def skill_observe(state):
    state["observation"] = get_observation()
    measured = state["observation"].robot_state.arms[state["arm"]].ee_pose
    state["position_error"], state["rotation_error"] = pose_error(measured, state["target"])
    state["status"] = "done" if state["position_error"] <= .003 and state["rotation_error"] <= .01 else "active"
    return {"source": state["observation"].source}


def skill_context(state):
    return {"status": state["status"], "position_error_m": state["position_error"],
            "rotation_error_rad": state["rotation_error"]}


def skill_options(state):
    return {} if state["status"] == "done" else {"fine": "1mm / 0.01rad", "normal": "10mm / 0.1rad"}


def skill_tick(state, option):
    scale = .1 if option == "fine" else 1
    action = servo_action(state["observation"], state["arm"], state["target"],
                          translation_m=.01 * scale, rotation_rad=.1 * scale)
    return step(action)
