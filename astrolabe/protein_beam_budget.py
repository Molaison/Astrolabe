"""AF3 owner schema adapter to the shared native-call quota authority."""
from astrolabe.design import quotas


def reserve_calls(policy, tasks):
    return quotas.reserve_calls(dict(policy, token_field="af3_call_token"), tasks)


def claim_call(policy, task):
    return quotas.claim_call(dict(policy, token_field="af3_call_token"), task)


__all__ = ["claim_call", "reserve_calls"]
