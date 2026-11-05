"""Role-isolated agent service. The deployment supplies the actual model driver."""

from importlib.resources import files
import uuid

from embodied_harness.runtime.relay import Plan


class RemoteCampaignAgent:
    def __init__(self, rpc):
        self.rpc = rpc

    def request(self, role, context):
        return self.rpc.call("respond", role=role, context=context)


class AgentService:
    """factory() must create a fresh model conversation for each call.

    Driver: respond(role, instructions, context) -> JSON reply. It receives only
    allowlisted role context, no runtime handle, layout manifest or credentials.
    """
    def __init__(self, factory, readiness=None):
        self.factory = factory
        self.readiness = readiness

    def respond(self, role, context):
        if role not in ("generate", "repair", "review", "propose", "verify") or not isinstance(context, dict):
            raise ValueError("unknown role/context")
        if role == "verify" and set(context) != {"goal", "public", "library"}:
            raise ValueError("independent verifier may only see public goal/observation/library")
        prompt = files("embodied_harness.instructions").joinpath("fix_loop.md").read_text()
        prompt += "\n" + files("embodied_harness.instructions").joinpath("python_api.md").read_text()
        prompt += "\nActive role: " + role
        driver = self.factory()
        try:
            reply = driver.respond(role, prompt, context)
            if role in ("generate", "repair", "verify"):
                Plan.from_dict(reply)
            return {"agent_session_id": uuid.uuid4().hex, "reply": reply}
        finally:
            close = getattr(driver, "close", None)
            if close:
                close()

    def methods(self):
        methods = {"respond": self.respond, "describe": self.describe}
        # Relay uses the same service but consumes bare replies.
        for role in ("generate", "repair", "review"):
            methods[role] = lambda context, active=role: self.respond(active, context)["reply"]
        return methods

    def describe(self):
        return {"protocol": "aspire-role-agent-v1", "fresh_conversation_per_request": True,
                **(self.readiness() if self.readiness else {})}
