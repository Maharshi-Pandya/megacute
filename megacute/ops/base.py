import dataclasses

import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16
from cutlass.tensor_utils import LayoutEnum

from .. import config as C


def epi_layout(cols=C.TILE, dtype=BFloat16):
    """128 x cols row-major swizzled tile (TMA box for elementwise tiles and output staging)."""
    return cute.select(sm100_utils.make_smem_layout_epi(dtype, LayoutEnum.ROW_MAJOR, (C.TILE, cols), 1),
                       mode=[0, 1])


class Node:
    """Default role behaviour: do nothing and leave every ring position untouched.

    Page rule: pages come only from the loader's `fill_page`, which hands each one to exactly one
    consumer role's FIFO; that role takes it with `take_page` and releases it on page_empty.
    A role never touches a FIFO it does not consume (see the page-pool note in config.py)."""

    stage_out = False   # output tile goes through the staging buffer and the storer warp
    stage_scratch = False   # compute uses the staging buffer as scratch and writes gmem directly
    early_loads = False     # load() issues dependency-free loads (weights), then calls wait_deps

    def __init__(self, prog, op, members=None):
        self.op = op
        self.prog = prog
        self.members = members or [op]   # the same op in each merged layer (megacute.layers)
        self._delta = {}

    def view(self, v, mode="2d", **kw):
        """Kernel tensor index for operand `v` of self.op. For a merged node it spans the operand
        of every layer; layer_off() is the per-layer row offset into it."""
        f = next(f.name for f in dataclasses.fields(self.op) if getattr(self.op, f.name) is v)
        vs = [getattr(m, f) for m in self.members]
        d = vs[1].row0 - vs[0].row0 if len(vs) > 1 else 0
        span = type(v)(v.buf, v.row0, v.col0, d * (len(vs) - 1) + v.rows, v.cols) if d else v
        idx = self.prog.view_index(span, mode, **kw)
        assert self._delta.setdefault(idx, d) == d, "one kernel tensor, two layer strides"
        return idx

    def layer_off(self, ctx, slot, idx, unit=1):
        """Row offset (in units of `unit` rows) of this instruction's layer in tensor `idx`."""
        d = self._delta.get(idx, 0)
        if d == 0:
            return 0
        return ctx.arg(slot, C.LAYER_ARG) * (d // unit)

    def host(self, views):
        return ()

    def load(self, ctx, a, slot, pmask, pcur, mq, mqph, cq, cqph):
        return pmask, pcur, mq, mqph, cq, cqph

    def mma(self, ctx, a, slot, mq, mqph, accidx, accph, sph, pfph):
        return mq, mqph, accidx, accph, sph, pfph

    def compute(self, ctx, a, slot, cq, cqph, accidx, accph, sfph, pvph):
        return cq, cqph, accidx, accph, sfph, pvph

    def store(self, ctx, a, slot):
        """Storer warp (stage_out ops): issue the TMA store of the staging buffer."""
