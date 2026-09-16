from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    return f".pid={os.getpid()}"


def zmq_ipc_endpoint(slot: int, suffix: str) -> str:
    """One of the numbered zmq ipc channel endpoints. libzmq's ipc transport needs a real
    filesystem path; ``/tmp`` is hardcoded for the Linux/WSL reference platform and absent
    on Windows, which falls back to the temp dir."""
    base = "/tmp" if os.name != "nt" else tempfile.gettempdir()
    return f"ipc://{base}/freetoken_{slot}{suffix}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return zmq_ipc_endpoint(0, self._unique_suffix)

    @property
    def zmq_detokenizer_addr(self) -> str:
        return zmq_ipc_endpoint(1, self._unique_suffix)

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return zmq_ipc_endpoint(2, self._unique_suffix)

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
