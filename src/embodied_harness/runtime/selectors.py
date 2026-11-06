"""Interchangeable numeric parameter selectors; no implicit physical success judgement."""

from dataclasses import dataclass
import json
import os
import urllib.request

from embodied_harness.contracts.types import _number, _text


@dataclass(frozen=True)
class Selection:
    option: str
    confidence: float

    def validate(self, options):
        if self.option not in options:
            raise ValueError("selector chose unavailable option")
        _number(self.confidence, "confidence")
        if self.confidence > 1:
            raise ValueError("confidence must be <=1")


class FixedSelector:
    def select(self, state, options):
        return Selection("normal" if "normal" in options else sorted(options)[0], 1.0)


def prior_with_jev_override(selector, state, options, prior, source, threshold):
    """Choose an already bounded S2 option; Jev cannot directly mutate state."""
    if (not isinstance(prior, dict) or set(prior) != {"option", "reason", "source"}
            or prior["option"] not in options or not isinstance(prior["reason"], str)
            or not prior["reason"].strip()):
        raise ValueError("System 2 must provide a bounded parameter prior and reason")
    from embodied_harness.contracts import ObservationSource
    if not isinstance(prior["source"], ObservationSource) or not prior["source"].same_state(source):
        raise ValueError("System 2 prior must use current observation evidence")
    decision = {"option": prior["option"], "selected_by": "system2_prior",
                "prior": prior, "threshold": threshold, "jev_option": None, "jev_confidence": None}
    try:
        suggestion = selector.select(state, options)
        suggestion.validate(options)
        decision.update(jev_option=suggestion.option, jev_confidence=suggestion.confidence)
        if suggestion.confidence > threshold:
            decision.update(option=suggestion.option, selected_by="jev_override")
    except Exception as exc:
        # No action has been submitted by this read-only selector; retain the valid prior.
        decision["jev_error_type"] = type(exc).__name__
    return decision


class JevSelector:
    """Numeric-only first implementation, compatible with the existing SystemOne choice shape."""

    def __init__(self, model, *, timeout_s=30, transport=None, credentials_file=None):
        _text(model, "pinned model")
        _number(timeout_s, "timeout_s")
        if timeout_s == 0:
            raise ValueError("timeout must be positive")
        self.model, self.timeout_s, self.transport = model, timeout_s, transport
        self.credentials_file = credentials_file
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": os.environ["TYPESAFE_PROXY_URL"]} if os.environ.get("TYPESAFE_PROXY_URL") else {}))

    def _request(self, payload):
        key = self.api_key()
        if not key:
            raise ValueError("TYPESAFE_API_KEY missing on harness host")
        request = urllib.request.Request("https://api.typesafe.ai/v1/systemone",
            json.dumps(payload, allow_nan=False).encode(), headers={
                "Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with self.opener.open(request, timeout=self.timeout_s) as response:
            raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("Jev response too large")
        return json.loads(raw)

    def api_key(self, *, required=True):
        if self.credentials_file:
            from embodied_harness.credentials import read_key
            return read_key(self.credentials_file, "TYPESAFE_API_KEY", required=required)
        return os.environ.get("TYPESAFE_API_KEY")

    def select(self, state, options):
        payload = {"model": self.model, "state": state, "questions": {"execution_parameter": {
            "type": "choice", "instructions": "Select one bounded numeric servo step. Code owns stages and success.",
            "criteria": options}}}
        answer = (self.transport or self._request)(payload)
        self.last_response = answer
        choice = answer["answers"]["execution_parameter"]
        probabilities = choice["probabilities"]
        if choice.get("type") != "choice" or set(probabilities) != set(options):
            raise ValueError("malformed Jev choice distribution")
        for probability in probabilities.values():
            _number(probability, "probability")
            if probability > 1:
                raise ValueError("probability must be <=1")
        if abs(sum(probabilities.values()) - 1) > 0.03:
            raise ValueError("probabilities must sum to one within API rounding tolerance")
        selected = Selection(choice["choice"], choice["confidence"])
        selected.validate(options)
        return selected


class RemoteSelector:
    def __init__(self, rpc, model):
        _text(model, "pinned System 1 model")
        self.rpc, self.model = rpc, model

    def select(self, state, options):
        result = self.rpc.call("select", state=state, options=options)
        if set(result) != {"option", "confidence", "model"} or result["model"] != self.model:
            raise ValueError("System 1 service model/schema differs from configuration")
        selection = Selection(result["option"], result["confidence"])
        selection.validate(options)
        return selection
