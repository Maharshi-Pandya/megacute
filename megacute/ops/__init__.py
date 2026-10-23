"""Device implementations of graph ops. Each node class provides:

  view_specs        [(View, mode)]  tensors the runtime must pass (mode: "2d" | "sbc" | "csb")
  host(views)       -> tuple of assets (TMA atoms, tensors, layouts), built inside the jit
  load(ctx, a, slot, pslot, pph, qph)                                 -> (pslot, pph, qph)
  mma(ctx, a, slot, pslot, pph, accidx, accph, qfph, sph, pfph)       -> same 7 values
  compute(ctx, a, slot, pslot, pph, accidx, accph, sfph, pvph)        -> same 6 values

Role methods of ops that do nothing in a role still advance the page ring by the number of
pages the instruction uses, so all roles agree on the FIFO position.
"""
