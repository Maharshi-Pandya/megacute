"""Device configuration shared by the scheduler (tiling) and the kernel (layouts)."""

# CTA shape and warp roles (8 warps). Warps 0-3 form the compute warpgroup: 128 threads, one per
# TMEM lane, so a 128-row accumulator maps one row per thread. (A 12-warp FA4-style variant with a
# second softmax warpgroup was measured slower here -- see README.)
NUM_THREADS = 256
COMPUTE_WARPS = 4
COMPUTE_THREADS = 32 * COMPUTE_WARPS
MMA_WARP = 4
LOADER_WARP = 5
CONTROLLER_WARP = 6
STORER_WARP = 7         # TMA stores out of the staging buffer + publishing dependency counters

# Instruction ring between controller and the role warps.
NSLOT = 4
IW = 16                 # int32 words per instruction

# Shared memory: one pool of 32 KB pages (MegaKittens-style) plus a 32 KB output staging
# buffer (aliased by attention P). The loader is the only allocator: it takes any free page,
# arms the page's full barrier with the TMA and pushes a page record (id, parity) into the FIFO
# of the role that consumes it (mma warp or compute warpgroup). Consumers pop records in
# instruction order, so each page barrier has one producer and one consumer per use and no
# role ever guesses a phase (a guessing role that runs a lap ahead aliases parity: deadlock).
# Any op can use every free page -- the GEMM main loop is latency-bound (2/3/4 stages:
# 737/1023/1181 TFLOP/s at 4096^3), so depth is what it needs.
PAGE_BYTES = 32768      # one GEMM k-stage (A|B 128x64), one Q/K/V tile, one elementwise tile
POOL_PAGES = 6
QDEPTH = 8              # page-record FIFO depth per consumer role
TILE = 128              # cols of an elementwise tile; rows/cols of an output tile
EW_ROWS = 128           # rows of an elementwise tile

# GEMM
GEMM_BM, GEMM_BN, GEMM_BK = 128, 128, 64
SPLIT_BN = 64           # split-K partial tiles: a 128x64 fp32 partial is exactly one 32 KB stage

# Attention
ATTN_BQ, ATTN_BK = 128, 128

# Instruction word layout
W_ICODE = 0             # 0 = stop; else 1 + node index
W_ARG0 = 1              # op-specific args in words 1..8
LAYER_ARG = 8           # arg 8 (word 9): layer index of an instruction of a merged node
W_NWAIT = 10
W_WAIT_OFF = 11         # offset into the wait table (pairs: counter, target)
W_NARRIVE = 12
W_ARRIVE_OFF = 13       # offset into the arrive table (counter ids)
W_SEQ = 14              # global instruction index (for tracing)
