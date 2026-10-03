from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig
from freetoken.utils import zmq_endpoint, zmq_tcp_ports


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    prefill_decode_interval: int = 4
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)
    # the queue endpoints travel to the workers with this config, so the process that builds
    # it owns the ports; empty on POSIX, where the endpoints are unix sockets
    _zmq_ports: tuple[int, ...] = field(default_factory=lambda: zmq_tcp_ports(5))

    @property
    def zmq_backend_addr(self) -> str:
        return zmq_endpoint(0, self._unique_suffix, self._zmq_ports)

    @property
    def zmq_detokenizer_addr(self) -> str:
        return zmq_endpoint(1, self._unique_suffix, self._zmq_ports)

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return zmq_endpoint(2, self._unique_suffix, self._zmq_ports)

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
