"""Speculative decoding: SpecRunner (the target side) and one Proposer (the draft side) per method."""

from freetoken.utils import Registry

# method name -> "module:Class" of its Proposer, loaded on use so this package imports no method's code
PROPOSERS = Registry[str]("Speculative proposer")
PROPOSERS.register("mtp")("freetoken.spec.mtp.proposer:MTPProposer")
PROPOSERS.register("dflash")("freetoken.spec.dflash.proposer:DFlashProposer")
PROPOSERS.register("dfly")("freetoken.spec.dfly.proposer:DFlyProposer")

__all__ = ["PROPOSERS"]
