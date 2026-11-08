"""Host resource admission; independent of model and scientific configuration.

ResourceBroker(config, sampler=None).start() owns physical GPU UUID locks and a
private Unix socket until successful close(); BrokerBusy retains ownership while
GPU work/residency remains live. ResourceClient(socket_path) provides lease(),
register_model(), unregister_model(), snapshot(), and from_env() (None when
ASTROLABE_RESOURCE_SOCKET is absent). See admission for configuration and lifetime
contracts. start() exports that socket to subsequently spawned daemon children.
"""

from .admission import (
    AdmissionError, AdmissionTimeout, BrokerBusy, BrokerClosed, Lease, ResourceBroker,
    ResourceClient,
)

__all__ = [
    "AdmissionError", "AdmissionTimeout", "BrokerBusy", "BrokerClosed", "Lease",
    "ResourceBroker", "ResourceClient",
]
