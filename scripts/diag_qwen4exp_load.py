"""CPU/meta reproduction of the qwen4_exp NVFP4 dense-weight load to name the mismatching param.

Run with the launcher venv's python. No CUDA needed: the model is built on the meta device
exactly like Engine.__init__, the dense weights are read to CPU via the same load_weight path,
then shapes/dtypes are compared like BaseOP.load_state_dict does.
"""
import os
import sys
import torch

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/suzuri/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/"
    "snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594"
)
TP = int(sys.argv[2]) if len(sys.argv) > 2 else 2

from freetoken.distributed import set_tp_info
from freetoken.layers.quantization import QuantBackend, set_quant_backend
from freetoken.engine.config import DistributedInfo, EngineConfig
from freetoken.engine.engine import _materialize_loaded_weight_state_dict
from freetoken.layers.rotary import set_rope_device
from freetoken.models import create_model
from freetoken.models.weight import load_weight
from freetoken.utils import torch_dtype

set_rope_device(torch.device("cpu"))
set_tp_info(0, TP)
set_quant_backend(QuantBackend.parse(None))

config = EngineConfig(
    model_path=MODEL_PATH,
    tp_info=DistributedInfo(rank=0, size=TP),
    dtype=torch.bfloat16,
    max_running_req=4,
    attention_backend="auto",
    moe_strategy="offload",
    ple_backend="disk",
    kv_cache_dtype="fp8",
    num_token_override=131072,
)
model_config = config.model_config  # installs the checkpoint QuantConfig
# what Engine/_adjust_config stamps before create_model
object.__setattr__(model_config, "moe_strategy", config.moe_strategy)
object.__setattr__(model_config, "decode_target", "gpu")

with torch.device("meta"), torch_dtype(config.dtype):
    model = create_model(model_config)

model_state = model.state_dict()
print(f"model state_dict: {len(model_state)} keys")

include_vision = bool(config.active_encoders)
print(f"active encoders: {config.active_encoders}, include_vision={include_vision}")

weights = load_weight(
    MODEL_PATH,
    torch.device("cpu"),
    include_moe_experts=False,
    include_vision=include_vision,
)
state = _materialize_loaded_weight_state_dict(model_state, weights, device=torch.device("cpu"))

missing = sorted(set(model_state) - set(state))
unexpected = sorted(set(state) - set(model_state))
shape_bad = []
dtype_bad = []
for k, v in state.items():
    exp = model_state.get(k)
    if exp is None:
        continue
    if tuple(exp.shape) != tuple(v.shape):
        shape_bad.append((k, tuple(exp.shape), tuple(v.shape)))
    if exp.dtype != v.dtype:
        dtype_bad.append((k, exp.dtype, v.dtype))

if os.environ.get("NOVISION"):
    for k in list(state):
        if k.startswith("visual."):
            model_state.pop(k, None)
            state.pop(k)
    shape_bad = [(k, ms, ws) for k, ms, ws in shape_bad if not k.startswith("visual.")]

print(f"\nmissing keys ({len(missing)}):")
for k in missing[:40]:
    print("  ", k, tuple(model_state[k].shape), model_state[k].dtype)
print(f"unexpected keys ({len(unexpected)}):")
for k in unexpected[:40]:
    print("  ", k, tuple(state[k].shape), state[k].dtype)
import collections
print(f"shape mismatch groups:", collections.Counter(__import__("re").sub(r"\d+","N",k) for k,_,_ in shape_bad))
print(f"shape mismatches ({len(shape_bad)}):")
for k, ms, ws in shape_bad[:200]:
    print(f"   {k}: model {ms} vs ckpt {ws}")
print(f"dtype mismatches ({len(dtype_bad)}):")
for k, md, wd in dtype_bad[:40]:
    print(f"   {k}: model {md} vs ckpt {wd}")
