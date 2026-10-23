"""The persistent megakernel: one CTA per SM interpreting the instruction stream.

Warp roles (8 warps, 256 threads):
  warps 0-3  compute     epilogues, elementwise math, norms, softmax, all global stores
  warp 4     mma         tcgen05 MMA issue (one elected lane)
  warp 5     loader      dependency waits, page allocation, TMA loads into pool pages
  warp 6     controller  fetches this CTA's instructions (static round robin) into the smem ring
  warp 7     storer      TMA-stores staged output tiles, waits for the writes, publishes counters

Output hand-off: compute writes a tile into the staging buffer and arrives stage_full, then moves
on; the storer frees the stage (stage_empty) as soon as the TMA has *read* it, and only then
waits for the global write and publishes. Compute never waits on global-write latency.

Protocol rules (violations deadlock silently):
  * TMA copies are warp-collective -- the DSL wraps the issue in elect.sync over the full
    warp -- so they are always called from a whole, converged warp.
  * Every mbarrier is initialised with the exact number of arrivals it receives; single-arrival
    barriers are arrived by one elected thread.

  * Every mbarrier is waited, in order, by every role that waits on it -- no role skips a
    ring position. (Parity waits alias once a waiter is a full lap ahead of the producer.)

Shared memory (bytes): Storage (barriers, instruction slots, page-record FIFOs) |
POOL_PAGES x 32 KB page pool | 32 KB output staging (= attention P).
TMEM (columns): GEMM accumulator stages [0, 256) | attention S [256, 384) | attention O [384, 512).

Page pool (see config.py): the loader allocates any free page (`fill_page`), the consumer role
learns which page from its FIFO (`take_page`) and releases it on its empty barrier. Pages are
claimed in instruction order, so a page is only ever waited on by instructions older than the
one waiting for it: no allocation cycle can form.
"""

import cutlass
import cutlass.cute as cute
import cutlass.memory as cmem
import cutlass.pipeline as pipeline
from cutlass import Boolean, Int32, Int64

from . import config as C

BUF_BYTES = 32768
TMEM_ACC = 0
TMEM_S = 256
TMEM_O = 384


@cute.struct
class Storage:
    inst: cute.struct.MemRange[Int32, C.NSLOT * C.IW]
    inst_full: cute.struct.MemRange[Int64, C.NSLOT]
    inst_empty: cute.struct.MemRange[Int64, C.NSLOT]
    dep_ok: cute.struct.MemRange[Int64, C.NSLOT]
    page_full: cute.struct.MemRange[Int64, C.POOL_PAGES]
    page_empty: cute.struct.MemRange[Int64, C.POOL_PAGES]
    mq_full: cute.struct.MemRange[Int64, C.QDEPTH]       # page-record FIFO -> mma warp
    mq_empty: cute.struct.MemRange[Int64, C.QDEPTH]
    cq_full: cute.struct.MemRange[Int64, C.QDEPTH]       # page-record FIFO -> compute warpgroup
    cq_empty: cute.struct.MemRange[Int64, C.QDEPTH]
    mq_rec: cute.struct.MemRange[Int32, C.QDEPTH]
    cq_rec: cute.struct.MemRange[Int32, C.QDEPTH]
    acc_full: cute.struct.MemRange[Int64, 2]
    acc_empty: cute.struct.MemRange[Int64, 2]
    stage_full: Int64
    stage_empty: Int64
    s_full: Int64
    s_empty: Int64
    p_full: Int64
    pv_done: Int64
    tmem_holding: Int32


def ring_next(slot, phase, n):
    """Advance a ring position; the phase bit flips when the slot wraps."""
    nxt = slot + 1
    wrap = nxt // n
    return nxt - wrap * n, phase ^ wrap


def page_tensor(pool, page, byte_off, layout, dtype):
    # a dynamic page index erases the pointer's alignment; vector access needs it back
    return buf_tensor(pool + cute.assume(page * C.PAGE_BYTES + byte_off, divby=128), layout, dtype)


def buf_tensor(base, layout, dtype):
    """Typed smem tensor at `base`; `layout` is a swizzled ComposedLayout or a plain Layout."""
    if isinstance(layout, cute.ComposedLayout):
        return cute.make_tensor(cute.recast_ptr(base, layout.inner, dtype=dtype), layout.outer)
    return cute.make_tensor(cute.recast_ptr(base, dtype=dtype), layout)


WATCHDOG_SPINS = 1 << 22

# Tracing: per CTA and role, records of (instruction seq, t0, t1, t2) in globaltimer ns.
#   loader:  t0 instruction visible, t1 dependencies satisfied, t2 loads issued
#   mma:     t0 instruction visible, t2 MMAs issued
#   compute: t0 instruction visible, t1 dep_ok, t2 outputs written and published
ROLE_LOADER, ROLE_MMA, ROLE_COMPUTE = 0, 1, 2


def now():
    return cute.arch.globaltimer()


@cute.jit
def trace_rec(ctx, role: cutlass.Constexpr, ev, slot, t0, t1, t2):
    """Write one trace record (caller elects the thread); returns the next record index."""
    if ev < ctx.trace_cap:
        bx, _, _ = cute.arch.block_idx()
        buf = ctx.trace_buf
        buf[bx, role, ev, 0] = Int64(ctx.sInst[slot * C.IW + C.W_SEQ])
        buf[bx, role, ev, 1] = t0
        buf[bx, role, ev, 2] = t1
        buf[bx, role, ev, 3] = t2
    return ev + 1


@cute.jit
def wait_bar(ctx, bar, phase, slot, tag: cutlass.Constexpr, extra=-1):
    """mbarrier wait. With ctx.debug, spin on try_wait and report (once) a wait that looks
    stuck: CTA, warp, which barrier, the phase wanted and the instruction's global index."""
    if cutlass.const_expr(ctx.debug):
        done = cute.arch.mbarrier_try_wait(bar, phase)
        n = Int32(0)
        while not done:
            done = cute.arch.mbarrier_try_wait(bar, phase)
            n = n + 1
            if n == WATCHDOG_SPINS:
                bx, _, _ = cute.arch.block_idx()
                cute.printf("[watchdog] cta %d warp %d stuck on " + tag + "[%d] phase %d inst %d icode %d\n",
                            bx, ctx.warp, Int32(extra), phase, ctx.sInst[slot * C.IW + C.W_SEQ],
                            ctx.sInst[slot * C.IW + C.W_ICODE])
    else:
        # Not cute.arch.mbarrier_wait: its intrinsic (expanded retry loop) intermittently never
        # returned (attention, 2026-10-22); this explicit loop is equivalent and reliable.
        done = cute.arch.mbarrier_try_wait(bar, phase)
        while not done:
            done = cute.arch.mbarrier_try_wait(bar, phase)


@cute.jit
def wait_deps(ctx, slot):
    """Loader warp: lane 0 spins on the instruction's dependency counters (acquire), then the
    warp may issue loads of dependent data; compute is released through dep_ok."""
    # lane-parallel: lane i checks waits i, i+32, ... -- one L2 round trip however many counters
    nw = ctx.sInst[slot * C.IW + C.W_NWAIT]
    woff = ctx.sInst[slot * C.IW + C.W_WAIT_OFF]
    for wi in cutlass.range(ctx.lane, nw, 32, unroll=1):
        cnt = ctx.waits[2 * (woff + wi)]
        tgt = ctx.waits[2 * (woff + wi) + 1] * ctx.epoch
        v = cute.arch.atomic_add(ctx.counters.iterator + cnt, Int32(0), sem="acquire", scope="gpu")
        spins = Int32(0)
        while v < tgt:
            v = cute.arch.atomic_add(ctx.counters.iterator + cnt, Int32(0), sem="acquire", scope="gpu")
            if cutlass.const_expr(ctx.debug):
                spins = spins + 1
                if spins == WATCHDOG_SPINS:
                    bx, _, _ = cute.arch.block_idx()
                    cute.printf("[watchdog] cta %d loader stuck: inst %d counter %d value %d target %d\n",
                                bx, ctx.sInst[slot * C.IW + C.W_SEQ], cnt, v, tgt)
    cute.arch.sync_warp()
    cute.arch.fence_proxy("async.global")
    if ctx.lane == 0:
        cute.arch.mbarrier_arrive(ctx.dep_ok + slot)


# ------------------------------------------------------------------------------ page pool
MMA_Q, COMPUTE_Q = 0, 1


@cute.jit
def fill_page(ctx, slot, queue: cutlass.Constexpr, nbytes: cutlass.Constexpr, pmask, pcur, qi, qph, skip=-1):
    """Loader warp: claim the next page in round-robin order (waiting until its last use is
    released), arm its full barrier for `nbytes` of TMA, and hand it to the consumer role's FIFO.
    `skip` is a page the current instruction still holds (attention's Q). The order depends only
    on the instruction stream, never on timing. pmask bit p = parity of page p's next use.
    Returns (page, pmask, pcur, qi, qph); the caller issues the TMA onto ctx.page_full + page."""
    p = pcur
    if p == skip:
        p = p + 1
        if p == C.POOL_PAGES:
            p = Int32(0)
    par = (pmask >> p) & 1
    wait_bar(ctx, ctx.page_empty + p, par ^ 1, slot, "page_empty", p)
    pmask = pmask ^ (Int32(1) << p)
    pcur = p + 1
    if pcur == C.POOL_PAGES:
        pcur = Int32(0)
    if ctx.lane == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(ctx.page_full + p, nbytes)
    rec, full, empty = ctx.mq_rec, ctx.mq_full, ctx.mq_empty
    if cutlass.const_expr(queue == COMPUTE_Q):
        rec, full, empty = ctx.cq_rec, ctx.cq_full, ctx.cq_empty
    wait_bar(ctx, empty + qi, qph, slot, "pq_empty", qi)
    if ctx.lane == 0:
        rec[qi] = p | (par << 8)
        cute.arch.mbarrier_arrive(full + qi)
    qi, qph = ring_next(qi, qph, C.QDEPTH)
    return p, pmask, pcur, qi, qph


@cute.jit
def take_page(ctx, slot, queue: cutlass.Constexpr, qi, qph):
    """Consumer role (every thread): pop the next page record and wait until the page's data
    has landed. The consumer later releases the page on ctx.page_empty + page.
    Returns (page, qi, qph)."""
    rec, full, empty = ctx.mq_rec, ctx.mq_full, ctx.mq_empty
    if cutlass.const_expr(queue == COMPUTE_Q):
        rec, full, empty = ctx.cq_rec, ctx.cq_full, ctx.cq_empty
    wait_bar(ctx, full + qi, qph, slot, "pq_full", qi)
    r = rec[qi]
    cute.arch.mbarrier_arrive(empty + qi)      # init count = consumer role's thread count
    p = r & 0xFF
    wait_bar(ctx, ctx.page_full + p, r >> 8, slot, "page_full", p)
    qi, qph = ring_next(qi, qph, C.QDEPTH)
    return p, qi, qph


class Ctx:
    """Per-CTA handles shared by all node code (plain Python object, built at trace time).

    Never mutate a Ctx inside a dynamic branch -- the DSL would try to carry it through the
    branch as a runtime value. Roles that need extra handles take a copy via `with_()`."""

    def arg(self, slot, i):
        return self.sInst[slot * C.IW + C.W_ARG0 + i]

    def with_(self, **kw):
        new = Ctx()
        new.__dict__.update(self.__dict__)
        new.__dict__.update(kw)
        return new


class Megakernel:
    def __init__(self, nodes, num_insts: int, grid: int, debug: bool = False, trace_cap: int = 0):
        self.debug = debug
        self.trace_cap = trace_cap
        self.nodes = nodes
        self.num_insts = num_insts
        self.grid = grid

    @cute.jit
    def __call__(self, views: tuple, insts: cute.Tensor, waits: cute.Tensor, arrives: cute.Tensor,
                 counters: cute.Tensor, trace: cute.Tensor, epoch: Int32, stream):
        assets = []
        for k in cutlass.range_constexpr(len(self.nodes)):
            assets.append(self.nodes[k].host(views))
        self.kernel(tuple(assets), insts, waits, arrives, counters, trace, epoch).launch(
            grid=(self.grid, 1, 1), block=(C.NUM_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, assets: tuple, insts: cute.Tensor, waits: cute.Tensor, arrives: cute.Tensor,
               counters: cute.Tensor, trace: cute.Tensor, epoch: Int32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()

        smem = cmem.SmemAllocator()
        st = smem.allocate(Storage)
        ctx = Ctx()
        ctx.pool = smem.allocate(C.POOL_PAGES * C.PAGE_BYTES, byte_alignment=1024)
        ctx.stage = smem.allocate(BUF_BYTES, byte_alignment=1024)
        # attention P aliases the staging buffer: compute writes P during the KV loop and the
        # output tile only after the last PV has completed, so the two never overlap in time
        ctx.pbuf = ctx.stage
        ctx.sInst = st.inst.get_tensor(cute.make_layout(C.NSLOT * C.IW))
        ctx.inst_full = st.inst_full.data_ptr()
        ctx.inst_empty = st.inst_empty.data_ptr()
        ctx.dep_ok = st.dep_ok.data_ptr()
        ctx.page_full = st.page_full.data_ptr()
        ctx.page_empty = st.page_empty.data_ptr()
        ctx.mq_full = st.mq_full.data_ptr()
        ctx.mq_empty = st.mq_empty.data_ptr()
        ctx.cq_full = st.cq_full.data_ptr()
        ctx.cq_empty = st.cq_empty.data_ptr()
        ctx.mq_rec = st.mq_rec.get_tensor(cute.make_layout(C.QDEPTH))
        ctx.cq_rec = st.cq_rec.get_tensor(cute.make_layout(C.QDEPTH))
        ctx.acc_full = st.acc_full.data_ptr()
        ctx.acc_empty = st.acc_empty.data_ptr()
        ctx.stage_full = st.stage_full.ptr
        ctx.stage_empty = st.stage_empty.ptr
        ctx.s_full = st.s_full.ptr
        ctx.s_empty = st.s_empty.ptr
        ctx.p_full = st.p_full.ptr
        ctx.pv_done = st.pv_done.ptr
        ctx.tidx = tidx
        ctx.lane = lane
        ctx.warp = warp
        ctx.epoch = epoch
        ctx.debug = self.debug
        ctx.trace_cap = self.trace_cap
        ctx.trace_buf = trace

        if warp == 0:
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(C.NSLOT):
                    cute.arch.mbarrier_init(ctx.inst_full + s, 1)
                    cute.arch.mbarrier_init(ctx.inst_empty + s, 4)  # loader, mma, compute, storer
                    cute.arch.mbarrier_init(ctx.dep_ok + s, 1)
                for p in cutlass.range_constexpr(C.POOL_PAGES):
                    cute.arch.mbarrier_init(ctx.page_full + p, 1)
                    cute.arch.mbarrier_init(ctx.page_empty + p, 1)
                for q in cutlass.range_constexpr(C.QDEPTH):
                    cute.arch.mbarrier_init(ctx.mq_full + q, 1)
                    cute.arch.mbarrier_init(ctx.mq_empty + q, 32)                 # mma warp
                    cute.arch.mbarrier_init(ctx.cq_full + q, 1)
                    cute.arch.mbarrier_init(ctx.cq_empty + q, C.COMPUTE_THREADS)  # compute group
                for a in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(ctx.acc_full + a, 1)
                    cute.arch.mbarrier_init(ctx.acc_empty + a, 1)
                for b in (ctx.stage_full, ctx.stage_empty, ctx.s_full, ctx.s_empty, ctx.p_full, ctx.pv_done):
                    cute.arch.mbarrier_init(b, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        tmem = cmem.TmemAllocator(st.tmem_holding.ptr,
                                  barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=2, num_threads=160),
                                  allocator_warp_id=0)
        # Each role runs in its own jit function: its loop state stays local to that function
        # instead of being carried through the role `if` (where Python objects cannot live).
        if warp == C.CONTROLLER_WARP:
            if lane == 0:
                self.controller_role(ctx, insts)
        if warp == C.LOADER_WARP:
            self.loader_role(ctx, assets, waits, counters, epoch)
        # (no method calls on ctx in these branches either: the DSL treats them as mutation)
        if warp == C.MMA_WARP:
            self.mma_role(ctx, assets, tmem)
        if warp < C.COMPUTE_WARPS:
            self.compute_role(ctx, assets, arrives, counters, tmem)
        if warp == C.STORER_WARP:
            self.storer_role(ctx, assets, arrives, counters)

    # ---------------------------------------------------------------------------- roles
    @cute.jit
    def controller_role(self, ctx, insts: cute.Tensor):
        """One lane: copy this CTA's instructions -- global indices bx, bx + grid, bx + 2 grid, ... --
        into the smem ring. Static round robin spreads every op over consecutive SMs (a dynamic
        queue let early-free SMs grab several instructions of one op while others got none), and
        each CTA's sequence is a subsequence of the global order, so dependencies, which all
        point backwards, cannot deadlock."""
        sInst = ctx.sInst
        bx, _, _ = cute.arch.block_idx()
        idx = Int32(bx)
        slot = Int32(0)
        ephase = Int32(1)
        running = Boolean(True)
        while running:
            wait_bar(ctx, ctx.inst_empty + slot, ephase, slot, "inst_empty")
            if idx < self.num_insts:
                for w in cutlass.range_constexpr(C.IW):
                    sInst[slot * C.IW + w] = insts[idx, w]
            else:
                sInst[slot * C.IW + C.W_ICODE] = Int32(0)
                running = Boolean(False)
            cute.arch.mbarrier_arrive(ctx.inst_full + slot)
            slot, ephase = ring_next(slot, ephase, C.NSLOT)
            idx = idx + self.grid

    @cute.jit
    def loader_role(self, ctx0, assets: tuple, waits: cute.Tensor, counters: cute.Tensor, epoch: Int32):
        """Whole warp. Lane 0 spins on the instruction's dependency counters (acquire), then the
        warp issues the op's TMA loads; compute is released through dep_ok."""
        ctx = ctx0.with_(waits=waits, counters=counters)
        nodes = self.nodes
        lane = ctx.lane
        slot = Int32(0)
        fph = Int32(0)
        pmask = Int32(0)   # parity of each pool page's next use
        pcur = Int32(0)    # round-robin allocation cursor
        mq = Int32(0)
        mqph = Int32(1)    # producer side of the mma page-record FIFO
        cq = Int32(0)
        cqph = Int32(1)    # producer side of the compute page-record FIFO
        ev = Int32(0)
        running = Boolean(True)
        while running:
            wait_bar(ctx, ctx.inst_full + slot, fph, slot, "inst_full")
            t0 = Int64(0)
            if cutlass.const_expr(self.trace_cap > 0):
                t0 = now()
            icode = ctx.sInst[slot * C.IW + C.W_ICODE]
            if icode == 0:
                running = Boolean(False)
            else:
                # early_loads nodes (GEMM) issue their weight loads first and call wait_deps
                # themselves; every other node waits here. (Kept inline: routing every node through
                # the wait_deps helper hung graphs mixing GEMM and eltwise nodes, cause not found.)
                early = Boolean(False)
                for k in cutlass.range_constexpr(len(nodes)):
                    if cutlass.const_expr(nodes[k].early_loads):
                        if icode == k + 1:
                            early = Boolean(True)
                if not early:
                    # lane-parallel spin (see wait_deps)
                    nw = ctx.sInst[slot * C.IW + C.W_NWAIT]
                    woff = ctx.sInst[slot * C.IW + C.W_WAIT_OFF]
                    for wi in cutlass.range(lane, nw, 32, unroll=1):
                        cnt = waits[2 * (woff + wi)]
                        tgt = waits[2 * (woff + wi) + 1] * epoch
                        v = cute.arch.atomic_add(counters.iterator + cnt, Int32(0), sem="acquire", scope="gpu")
                        spins = Int32(0)
                        while v < tgt:
                            v = cute.arch.atomic_add(counters.iterator + cnt, Int32(0), sem="acquire", scope="gpu")
                            if cutlass.const_expr(self.debug):
                                spins = spins + 1
                                if spins == WATCHDOG_SPINS:
                                    bx, _, _ = cute.arch.block_idx()
                                    cute.printf("[watchdog] cta %d loader stuck: inst %d counter %d value %d target %d\n",
                                                bx, ctx.sInst[slot * C.IW + C.W_SEQ], cnt, v, tgt)
                    cute.arch.sync_warp()
                    cute.arch.fence_proxy("async.global")
                    if lane == 0:
                        cute.arch.mbarrier_arrive(ctx.dep_ok + slot)
                t1 = Int64(0)
                if cutlass.const_expr(self.trace_cap > 0):
                    t1 = now()
                for k in cutlass.range_constexpr(len(nodes)):
                    if icode == k + 1:
                        pmask, pcur, mq, mqph, cq, cqph = nodes[k].load(ctx, assets[k], slot, pmask, pcur,
                                                                        mq, mqph, cq, cqph)
                if cutlass.const_expr(self.trace_cap > 0):
                    if lane == 0:
                        ev = trace_rec(ctx, ROLE_LOADER, ev, slot, t0, t1, now())
            if lane == 0:
                cute.arch.mbarrier_arrive(ctx.inst_empty + slot)
            slot, fph = ring_next(slot, fph, C.NSLOT)

    @cute.jit
    def mma_role(self, ctx0, assets: tuple, tmem):
        """Whole warp; tcgen05 issue and commits are elected inside the op code."""
        tmem.wait_for_alloc()
        ctx = ctx0.with_(tmem_ptr=tmem.retrieve_ptr(cutlass.Float32))
        nodes = self.nodes
        slot = Int32(0)
        fph = Int32(0)
        mq = Int32(0)
        mqph = Int32(0)
        accidx = Int32(0)
        accph = Int32(1)   # producer side of acc_empty
        sph = Int32(1)     # producer side of s_empty
        pfph = Int32(0)
        ev = Int32(0)
        running = Boolean(True)
        while running:
            wait_bar(ctx, ctx.inst_full + slot, fph, slot, "inst_full")
            t0 = Int64(0)
            if cutlass.const_expr(self.trace_cap > 0):
                t0 = now()
            icode = ctx.sInst[slot * C.IW + C.W_ICODE]
            if icode == 0:
                running = Boolean(False)
            else:
                for k in cutlass.range_constexpr(len(nodes)):
                    if icode == k + 1:
                        mq, mqph, accidx, accph, sph, pfph = nodes[k].mma(
                            ctx, assets[k], slot, mq, mqph, accidx, accph, sph, pfph)
                if cutlass.const_expr(self.trace_cap > 0):
                    if ctx.lane == 0:
                        ev = trace_rec(ctx, ROLE_MMA, ev, slot, t0, t0, now())
            if ctx.lane == 0:
                cute.arch.mbarrier_arrive(ctx.inst_empty + slot)
            slot, fph = ring_next(slot, fph, C.NSLOT)

    @cute.jit
    def publish(self, ctx, slot, arrives: cute.Tensor, counters: cute.Tensor):
        """One thread, after the instruction's global writes are complete: release-increment
        each of its arrival counters."""
        na = ctx.sInst[slot * C.IW + C.W_NARRIVE]
        aoff = ctx.sInst[slot * C.IW + C.W_ARRIVE_OFF]
        for ai in cutlass.range(na, unroll=1):
            cute.arch.red(counters.iterator + arrives[aoff + ai], Int32(1), op="add",
                          dtype=Int32, sem="release", scope="gpu")

    @cute.jit
    def storer_role(self, ctx, assets: tuple, arrives: cute.Tensor, counters: cute.Tensor):
        """Whole warp. For ops that stage their output: TMA-store the staging buffer, free it as
        soon as it has been read, then wait for the global write and publish."""
        nodes = self.nodes
        slot = Int32(0)
        fph = Int32(0)
        sfph = Int32(0)
        running = Boolean(True)
        while running:
            wait_bar(ctx, ctx.inst_full + slot, fph, slot, "inst_full")
            icode = ctx.sInst[slot * C.IW + C.W_ICODE]
            if icode == 0:
                running = Boolean(False)
            else:
                for k in cutlass.range_constexpr(len(nodes)):
                    if cutlass.const_expr(nodes[k].stage_out):
                        if icode == k + 1:
                            wait_bar(ctx, ctx.stage_full, sfph, slot, "stage_full")
                            sfph = sfph ^ 1
                            nodes[k].store(ctx, assets[k], slot)
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                            if ctx.lane == 0:
                                cute.arch.mbarrier_arrive(ctx.stage_empty)
                            cute.arch.cp_async_bulk_wait_group(0, read=False)
                            cute.arch.fence_proxy("async.global")
                            if ctx.lane == 0:
                                self.publish(ctx, slot, arrives, counters)
            if ctx.lane == 0:
                cute.arch.mbarrier_arrive(ctx.inst_empty + slot)
            slot, fph = ring_next(slot, fph, C.NSLOT)

    @cute.jit
    def compute_role(self, ctx0, assets: tuple, arrives: cute.Tensor, counters: cute.Tensor, tmem):
        """Warpgroup. Ops that stage their output (`stage_out`) get the staging buffer before
        they run and hand it to the storer after; the rest (norm) write global memory directly,
        so once they return their writes are complete and one thread publishes them here.
        Warp 0 of this group allocates all 512 TMEM columns and frees them at the end."""
        tmem.allocate(512)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        ctx = ctx0.with_(tmem_ptr=tmem_ptr)
        nodes = self.nodes
        tidx = ctx.tidx
        slot = Int32(0)
        fph = Int32(0)
        cq = Int32(0)
        cqph = Int32(0)
        accidx = Int32(0)
        accph = Int32(0)
        sfph = Int32(0)
        pvph = Int32(0)
        seph = Int32(1)    # producer side of stage_empty
        ev = Int32(0)
        running = Boolean(True)
        while running:
            wait_bar(ctx, ctx.inst_full + slot, fph, slot, "inst_full")
            t0 = Int64(0)
            if cutlass.const_expr(self.trace_cap > 0):
                t0 = now()
            icode = ctx.sInst[slot * C.IW + C.W_ICODE]
            if icode == 0:
                running = Boolean(False)
            else:
                wait_bar(ctx, ctx.dep_ok + slot, fph, slot, "dep_ok")
                t1 = Int64(0)
                if cutlass.const_expr(self.trace_cap > 0):
                    t1 = now()
                for k in cutlass.range_constexpr(len(nodes)):
                    if icode == k + 1:
                        if cutlass.const_expr(nodes[k].stage_out or nodes[k].stage_scratch):
                            wait_bar(ctx, ctx.stage_empty, seph, slot, "stage_empty")
                            seph = seph ^ 1
                        cq, cqph, accidx, accph, sfph, pvph = nodes[k].compute(
                            ctx, assets[k], slot, cq, cqph, accidx, accph, sfph, pvph)
                        if cutlass.const_expr(nodes[k].stage_out):
                            cute.arch.fence_proxy("async.shared", space="cta")   # stage -> TMA store
                        else:
                            cute.arch.fence_acq_rel_gpu()   # direct global writes, before the publish
                        cute.arch.barrier(barrier_id=1, number_of_threads=C.COMPUTE_THREADS)
                        if tidx == 0:
                            if cutlass.const_expr(nodes[k].stage_out):
                                cute.arch.mbarrier_arrive(ctx.stage_full)
                            else:
                                self.publish(ctx, slot, arrives, counters)
                            if cutlass.const_expr(nodes[k].stage_scratch):   # hand the stage back
                                cute.arch.mbarrier_arrive(ctx.stage_empty)
                if tidx == 0:
                    if cutlass.const_expr(self.trace_cap > 0):
                        ev = trace_rec(ctx, ROLE_COMPUTE, ev, slot, t0, t1, now())
            if tidx == 0:
                cute.arch.mbarrier_arrive(ctx.inst_empty + slot)
            slot, fph = ring_next(slot, fph, C.NSLOT)
        cute.arch.barrier(barrier_id=1, number_of_threads=C.COMPUTE_THREADS)
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)
