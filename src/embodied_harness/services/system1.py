"""Authenticated numeric System 1 service; provider credentials stay on this host."""

import json


class System1Service:
    def __init__(self, selector, audit=None):
        self.selector, self.audit = selector, audit

    def describe(self):
        return {"protocol": "numeric-system1-v1", "model": self.selector.model,
                "credentials_configured": bool(self.selector.api_key(required=False))}

    def select(self, state, options):
        if (not isinstance(state, dict) or not isinstance(options, dict) or not 1 <= len(options) <= 12
                or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k,v in options.items())):
            raise ValueError("bounded named numeric choices and measured state required")
        if len(json.dumps({"state": state, "options": options}, allow_nan=False)) > 256_000:
            raise ValueError("numeric context too large")
        answer = self.selector.select(state, options)
        answer.validate(options)
        response_model = self.selector.last_response.get("model")
        if response_model != self.selector.model:
            raise ValueError("provider returned an unexpected model version")
        result = {"option": answer.option, "confidence": answer.confidence, "model": response_model}
        if self.audit:
            self.audit.append("system1_selection", state=state, options=options, result=result)
        return result

    def methods(self):
        return {"describe": self.describe, "select": self.select}
