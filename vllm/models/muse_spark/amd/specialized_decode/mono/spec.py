# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The contract every Muse Spark MonoKernel package shares: shapes, numerics,
rows, grid, the K1 / K2 placements, mailbox slots, the build keys, the argument
lists and the scratch / peer layouts.

Pure Python (``dataclasses`` only): the runtime sizes its buffers and the CPU
tests check the layouts without torch.cuda or FlyDSL. ``pair_layout`` is the
K3 helper (``mono.common.plan.pair_layout``) re-implemented here for that
reason.

Every count is per TP rank of Muse Spark Open at TP8. Every scratch and peer
offset is sized by ``MAX_TOKENS`` rows (``MAX_PICKS`` picks), never by a
launch's width, so every width of a step shares one layout and one parity
half.
"""

import os
from dataclasses import dataclass

# --------------------------------------------------------------------------
# The model (Muse Spark Open at TP8)

# Tensor-parallel degree. TP8 is the model's native shard (MI355X); TP4 (for a
# 4-GPU TP island under PP) doubles every per-rank count below. SUPPORTED_TP is
# the set the shapes and placement are validated for. Like BLOCKS this is a
# per-deployment constant; the runtime picks it from tensor_parallel_size.
SUPPORTED_TP = (8, 4)
TP = int(os.environ.get("VLLM_MUSE_SPARK_TP", "8"))
assert TP in SUPPORTED_TP, f"TP {TP} not in {SUPPORTED_TP}"
HIDDEN = 8192  # H: the residual stream
MOE_HIDDEN = 4096  # the pooled width the routed experts see
EXPERTS = 256
TOPK = 8
HEAD_DIM = 64
# The model's full (pre-shard) counts; the per-rank shapes are these // TP.
N_HEADS = 128  # num_attention_heads
N_KV_HEADS = 16  # num_key_value_heads
MOE_INTER = 4096  # moe_intermediate_size
assert N_HEADS % TP == 0 and N_KV_HEADS % TP == 0 and MOE_INTER % TP == 0
assert EXPERTS % TP == 0
INTER = MOE_INTER // TP  # 512 (TP8) / 1024 (TP4): an expert's intermediate a rank
EXPERTS_RANK = EXPERTS // TP  # 32 (TP8) / 64 (TP4) router rows a rank
HEADS = N_HEADS // TP  # 16 (TP8) / 32 (TP4) query heads a rank
KV_HEADS = N_KV_HEADS // TP  # 2 (TP8) / 4 (TP4) KV heads a rank
CORE = HEADS * HEAD_DIM  # 1024 (TP8) / 2048 (TP4): o_proj's K a rank

# runtime.text_shapes(text_config) must equal MUSE_SHAPES, field by field.
MUSE_SHAPE_FIELDS = (
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "num_local_experts",
    "num_experts_per_tok",
    "moe_hidden_size",
    "moe_intermediate_size",
    "num_shared_experts",
    "sliding_window",
)
MUSE_SHAPES = (8192, 128, 16, 64, 256, 8, 4096, 4096, 0, 2048)

# --------------------------------------------------------------------------
# Numerics (refuse_model asserts the loaded modules carry these)

POST_ATTN_EPS = 1e-8  # weightless post_attn_norm
PRE_FFN_EPS = 1e-5
PRE_EXP_EPS = 1e-5
POST_EXP_EPS = 1e-8
GAIN = 1.0  # fused_gain_center of pre_ffn / pre_expert / post_expert norms
ROUTE_SCALE = 1.0
ROUTE_EPS = 1e-15  # router_score_division_eps
AMAX_FLOOR = 1e-10  # MXFP4 RoundUp: amax = max(amax, AMAX_FLOOR)
LOG2E = 1.4426950408889634

# Phases B and C (DESIGN_BC section 1.1): the attention front and core.
RMS_EPS = 1e-5  # pre_qkv_proj_norm, the weightless qk_norm and o_norm
POST_FFN_EPS = 1e-8  # the previous layer's post_ffn_norm (weight first)
QK_SCALE_FACTOR = 5.4730064863875  # config.qk_scale_factor
QK_SCALE = HEAD_DIM**-0.5 * QK_SCALE_FACTOR  # == float(impl.scale)
QK_SCALE_LOG2E = QK_SCALE * LOG2E  # the split's base-2 logit scale (f32 in-kernel)
ROPE_THETA = 5e5  # sliding layers' GPT-J RoPE over all 64 dims; full layers: NoPE
WINDOW = 2048  # sliding layers attend keys L - WINDOW <= k < L
KV_BLOCK = 16  # tokens a KV page (the kernel block)
KV_C = 2 * HEAD_DIM  # a (block, head, token) row: K at [0, 64), V at [64, 128)
V_OFF = HEAD_DIM  # V's element offset in the row
Q_PER_KV = HEADS // KV_HEADS  # 8 query heads a KV head

# --------------------------------------------------------------------------
# Rows

MAX_ROWS = 8  # rows a served step has at most (the K3 thread maps)
MAX_TOKENS = 8  # rows scratch and peer memory are sized for
MAX_PICKS = MAX_TOKENS * TOPK  # 64 (row, k) picks
MAX_U = MAX_PICKS  # experts in a step's union

# --------------------------------------------------------------------------
# Grid: one CTA per CU, all resident

# The grid is one co-resident CTA per CU, so its width is the part's CU count.
# MI355X has 256, MI350P 128; both are gfx950, so the per-CTA budget (LDS, VGPR,
# occupancy) is identical and only the CTA count -- and thus the per-CTA task
# fan-out -- differs. Every task count below is fixed by the model shape, not by
# BLOCKS; a narrower grid runs more rounds per CTA (``*_ROUNDS``), never fewer
# tasks. SUPPORTED_WIDTHS is the set the static placement is validated for.
SUPPORTED_WIDTHS = (256, 128)
BLOCKS = int(os.environ.get("VLLM_MUSE_SPARK_BLOCKS", "256"))
assert BLOCKS in SUPPORTED_WIDTHS, f"BLOCKS {BLOCKS} not in {SUPPORTED_WIDTHS}"
THREADS = 512
WAVES = THREADS // 64  # 8
LDS_BYTES = 160 * 1024  # a CU's LDS: one CTA's whole
VGPR_MAX = 256  # arch + acc VGPRs a lane at 2 waves a SIMD
ROWS = 16  # output rows of a GEMV task (one MFMA tile)
COLS = 128  # columns of a seam / gather CTA


def cdiv(a: int, b: int) -> int:
    """ceil(a / b) for a >= 0, b > 0."""
    return (a + b - 1) // b


def rounds(tasks: int, blocks: int = BLOCKS) -> int:
    """CTA b runs tasks b, b + blocks, ... b + (rounds - 1) * blocks: the number
    of grid-striped passes a stage of ``tasks`` tasks takes on a ``blocks``-wide
    grid. One per CTA when tasks <= blocks."""
    return cdiv(tasks, blocks)

# --------------------------------------------------------------------------
# K2 placement (static; section 5.2 of the design)

O_TASKS = HIDDEN // ROWS  # 512 o_proj tasks: CTA b runs b, b + BLOCKS, ...
NA = HIDDEN // COLS  # 64 seam CTAs [0, NA): 128 columns each
R0 = 64  # router CTAs [R0, R0 + R_TASKS): one expert row each
R_TASKS = EXPERTS_RANK  # 32
P_GROUPS = MOE_HIDDEN // TP // ROWS  # 32 pooler row groups a rank
P_PARTS = 4  # K quarters of a pooler row group
P_K = HIDDEN // P_PARTS  # 2048
P0 = 96  # pooler CTAs [P0, P0 + P_TASKS): g = t // P_PARTS, q = t % P_PARTS
P_TASKS = P_GROUPS * P_PARTS  # 128
UG_GROUPS = INTER // 32  # 16 ug tasks a union slot (32 columns each)
W13_K = MOE_HIDDEN  # 4096
W13_STEPS = W13_K // 128  # 32 (16 a K half)
W2_STEPS = INTER // 128  # 4
DOWN_TASKS = MOE_HIDDEN // ROWS  # 256: CTA b owns down rows 16b .. 16b + 16, ...
DOWN_BATCH = 4  # slots of w2 in flight a wave
OWN_COLS = MOE_HIDDEN // TP  # 512 pooled columns a rank owns (reduce-scatter)
RED_TASKS = OWN_COLS // ROWS  # 32 owner CTAs [32 rank, 32 rank + 32)
U_GROUPS = HIDDEN // TP // ROWS  # 64 unpooler row groups a rank
U_PARTS = 4  # K quarters of an unpooler row group
U_K = MOE_HIDDEN // U_PARTS  # 1024
U_TASKS = U_GROUPS * U_PARTS  # 256: CTA b: g = b // U_PARTS, q = b % U_PARTS
G_TASKS = HIDDEN // COLS  # 64 gather CTAs [0, G_TASKS)

# Per-CTA task fan-out of the grid-striped K2 stages (CTA b runs b, b + BLOCKS,
# ...). At BLOCKS = 256 every round count is today's value (o_proj 2, down /
# unpool 1); at BLOCKS = 128 a CTA runs twice the rounds. The phase-2 roles
# (seam / router / pooler) keep their fixed CTA ranges [0, NA), [R0, R0 +
# R_TASKS) and [P0, ...); only the pooler, which owns the grid tail [P0, BLOCKS),
# stripes its P_TASKS over the POOL_CTAS it has (one round when they fit).
O_ROUNDS = rounds(O_TASKS)  # 2 (256) / 4 (128)
DOWN_ROUNDS = rounds(DOWN_TASKS)  # 1 (256) / 2 (128)
U_ROUNDS = rounds(U_TASKS)  # 1 (256) / 2 (128)
POOL_CTAS = BLOCKS - P0  # pooler CTAs: 160 (256) / 32 (128)
P_ROUNDS = rounds(P_TASKS, POOL_CTAS)  # 1 (256) / 4 (128)


def ug_cta0(s: int) -> int:
    """The first CTA that can get a ug task at S = ``s``: ``run_ug`` gives CTA
    b the tasks 255 - b, + 256, ... of the step's |U| 16, and |U| <= min(8 S,
    MAX_U) (only S 8 picks reach the union), so CTA b has a task only if
    b >= 256 - 16 min(8 S, MAX_U). 128 at S = 1, 0 from S = 2 on."""
    return max(0, BLOCKS - UG_GROUPS * min(s * TOPK, MAX_U))


# MXFP4 activation rows (a1): words a pooled row
XQ_WORDS = MOE_HIDDEN // 8  # 512 int32 words of packed fp4
XSC_WORDS = MOE_HIDDEN // 32  # 128 e8m0 codes, one a word in LDS

# LDS route table (common/route_lds.RouteLds): 32-bit words a field
ROUTE_LDS = dict(
    route=MAX_TOKENS * 16,  # t*16 + k: id; t*16 + 8 + k: fp32 bits of the weight
    flag=EXPERTS,
    expert=MAX_U,
    kof=MAX_U * 8,
    scan=WAVES,
    tot=MAX_TOKENS,
)

# --------------------------------------------------------------------------
# K1 placement (Phase B1; DESIGN_BC 5.1): weight-free boundary CTAs [0, NA),
# the epilogue on CTAs [0, E_UNITS * S), the qkv GEMV on CTAs [NA, BLOCKS)

QKV_ROWS = 2 * CORE + 2 * KV_HEADS * HEAD_DIM  # 2304 a rank: q | k | v | og
QKV_TILES = QKV_ROWS // ROWS  # 144
KPARTS = 16  # K parts of a qkv tile
KPART = HIDDEN // KPARTS  # 512 columns: 4 chunks of 128, one a wave
K1_TASKS = QKV_TILES * KPARTS  # 2304 (tile, K part) tasks: tau = 16 T + kp
K1_GEMV0 = NA  # 64: the first GEMV CTA
K1_GEMV_CTAS = BLOCKS - K1_GEMV0  # 192
K1_TPC = K1_TASKS // K1_GEMV_CTAS  # 12 tasks a GEMV CTA: tau = 12 g + i
K1_ROUNDS = K1_TPC // 2  # 6 rounds: waves 0-3 take task 2r, waves 4-7 2r + 1
K1_MAXC = 2  # GEMV CTAs contributing to a tile
# The qkv GEMV streams its K1_TPC tasks through LDS a chunk at a time so ``xa``
# (sized by the chunk, not K1_TPC) and the weight rounds in flight stay within a
# CU's budget at any TP / grid width. K1_CHUNK = 12 is the TP8/256 K1_TPC (so
# that config is one chunk -- unchanged) and divides K1_TPC in every supported
# config (12/1, 36/3, 24/2, 72/6).
K1_CHUNK = 12  # tasks staged a pass (even; <= the LDS-bound ~18 at s=8)
assert K1_CHUNK % 2 == 0 and K1_TPC % K1_CHUNK == 0
K1_NCHUNK = K1_TPC // K1_CHUNK  # LDS passes a GEMV CTA: 1 (TP8/256) .. 6 (TP4/128)
K1_CHUNK_ROUNDS = K1_CHUNK // 2  # 6 MFMA rounds a chunk
ROW_Q = 0  # the qkv row blocks: q heads, k heads, v heads, o_gate heads
ROW_K = CORE
ROW_V = ROW_K + KV_HEADS * HEAD_DIM
ROW_G = ROW_V + KV_HEADS * HEAD_DIM
E_UNITS = 5  # epilogue units a token: q 0-7, q 8-15, k + v, og 0-7, og 8-15


def k1_contributor0(tile: int) -> int:
    """The first of the ``K1_MAXC`` GEMV CTAs (g = b - K1_GEMV0) whose tasks
    cover ``tile``: contributor c = g - k1_contributor0(tile) holds a run of the
    tile's K parts, c = 0 the lower ones."""
    return (KPARTS * tile) // K1_TPC


assert K1_TASKS == K1_TPC * K1_GEMV_CTAS and ROWS * QKV_TILES == QKV_ROWS
assert ROW_G + CORE == QKV_ROWS and KPART * KPARTS == HIDDEN and KPART == 4 * COLS
assert E_UNITS * MAX_ROWS <= NA and K1_ROUNDS * 2 == K1_TPC


def _validate_k1_placement(blocks: int) -> None:
    """The K1 qkv GEMV placement fits a ``blocks``-wide grid: the GEMV CTAs
    [K1_GEMV0, blocks) tile the (tile, K part) tasks exactly, each CTA's task
    count is even (two waves take 2 r / 2 r + 1 a round), and no qkv tile is
    split across more than K1_MAXC GEMV CTAs (``qpart`` holds K1_MAXC partials a
    tile). The boundary / epilogue CTAs [0, NA) are width-independent."""
    gemv_ctas = blocks - K1_GEMV0
    assert gemv_ctas > 0 and K1_TASKS % gemv_ctas == 0, blocks
    tpc = K1_TASKS // gemv_ctas
    assert tpc % 2 == 0, (blocks, tpc)
    for t in range(QKV_TILES):  # a tile's KPARTS tau values span <= K1_MAXC CTAs
        first = (KPARTS * t) // tpc
        last = (KPARTS * t + KPARTS - 1) // tpc
        assert last - first + 1 <= K1_MAXC, (blocks, t)


for _w in SUPPORTED_WIDTHS:
    _validate_k1_placement(_w)

# --------------------------------------------------------------------------
# K2 split-KV core placement (Phase C, q = 1 widths; DESIGN_BC 7.1). Split
# unit u = (t * 2 + kvh) * N + p runs on CTA u; combine (t, h) on CTA
# C0(S) + 16 t + h. A CTA holding both runs its unit first.

NSPLIT_SW = 9  # sliding: <= 129 pages a window -> <= 9 live units
MIN_PER = 16  # pages a live unit at least: two a wave (two in flight)


def nsplit(S: int, window: bool, blocks: int = BLOCKS) -> int:
    """N: units a (token, KV head) (DESIGN_BC 7.1). Capped so the launch's
    2 * S * N units fit a ``blocks``-wide grid: ``split.pages`` sizes a unit at
    ``max(cdiv(nb, N), MIN_PER)`` pages, so a smaller N just gives larger (still
    correct) units -- the window's NSPLIT_SW is an upper bound, not a floor.
    At 256 every S keeps today's N (blocks // (2 S) >= NSPLIT_SW for a window)."""
    full = blocks // (2 * S)
    return min(NSPLIT_SW, full) if window else full


def n_units(S: int, window: bool, blocks: int = BLOCKS) -> int:
    """Split units of a launch (CTAs [0, n_units))."""
    return 2 * S * nsplit(S, window, blocks)


def n_comb(S: int) -> int:
    """Combine CTAs of a launch: one a (token, q head), 16 S."""
    return HEADS * S


def C0(S: int, blocks: int = BLOCKS) -> int:
    """The first combine CTA: [C0, blocks) runs one (token, q head) each."""
    return blocks - n_comb(S)


def _validate_split_placement(blocks: int) -> None:
    """The split-KV core placement fits a ``blocks``-wide grid at every width S:
    the split units [0, n_units) and the combine CTAs [C0, blocks) both fit, and
    a unit has at least one page part. The two ranges may overlap (they already
    do at 256, S = 8); a CTA holding both runs its unit first, so overlap needs
    no separation -- only C0 >= 0 (n_comb <= blocks)."""
    for s in range(1, MAX_ROWS + 1):
        assert n_comb(s) <= blocks, (blocks, s)  # C0 >= 0
        for w in (False, True):
            assert nsplit(s, w, blocks) >= 1, (blocks, s, w)
            assert n_units(s, w, blocks) <= blocks, (blocks, s, w)


for _w in SUPPORTED_WIDTHS:
    _validate_split_placement(_w)

# --------------------------------------------------------------------------
# Mailbox tags: tag = (epoch & EPOCH_MASK) * TAG_SLOTS + slot + 1

EPOCH_MASK = (1 << 22) - 1  # kernels use e & EPOCH_MASK: the int32 epoch may wrap
TAG_SLOTS = 256
K1_SLOT0 = 0  # Phase B's K1 launches
K2_SLOT0 = 128  # K2 launches


def k1_slot(layer_idx: int) -> int:
    """The tag slot of a layer's K1 launch (Phase B)."""
    if not 0 <= layer_idx < K2_SLOT0 - K1_SLOT0:
        raise ValueError(f"layer {layer_idx} has no K1 slot")
    return K1_SLOT0 + layer_idx


def k2_slot(layer_idx: int) -> int:
    """The tag slot of a layer's K2 launch."""
    if not 0 <= layer_idx < TAG_SLOTS - K2_SLOT0:
        raise ValueError(f"layer {layer_idx} has no K2 slot")
    return K2_SLOT0 + layer_idx


def step_tag(epoch: int, slot: int) -> int:
    """The host mirror of ``sync.step_tag``: the tag every pair a launch of
    ``slot`` writes at ``epoch`` carries (two's complement ``&``, so a wrapped,
    negative int32 epoch gives the kernel's tag too). Never 0: a zeroed pair is
    never current."""
    return (epoch & EPOCH_MASK) * TAG_SLOTS + slot + 1


def _validate_k2_placement(blocks: int) -> None:
    """The static K2 MoE / o_proj placement holds on a ``blocks``-wide grid: the
    phase-2 roles pack into disjoint CTA ranges and every grid-striped stage
    tiles its model-fixed task count exactly. Run for every SUPPORTED_WIDTHS at
    import, so a narrower grid (MI350P's 128) is proven before any kernel builds.
    The split-KV core and K1 GEMV placement are validated separately (they need
    their own width rework; windowed split overflows a 128-wide grid)."""
    pool_ctas = blocks - P0
    # phase-2 roles seam [0, NA), router [R0, R0 + R_TASKS), pooler [P0, blocks)
    # are disjoint and within the grid
    assert NA <= R0 and R0 + R_TASKS <= P0 and blocks > P0, blocks
    assert blocks >= G_TASKS and pool_ctas > 0, blocks
    # grid-striped stages (CTA b: b, b + blocks, ...) tile their tasks exactly
    assert O_TASKS % blocks == 0 and DOWN_TASKS % blocks == 0, blocks
    assert U_TASKS % blocks == 0, blocks
    # the pooler tail covers its P_TASKS over the CTAs it has
    assert cdiv(P_TASKS, pool_ctas) * pool_ctas >= P_TASKS, blocks


for _w in SUPPORTED_WIDTHS:
    _validate_k2_placement(_w)
assert RED_TASKS * TP == DOWN_TASKS and MAX_ROWS <= MAX_TOKENS <= 8
assert MAX_U <= EXPERTS and TOPK <= 8
assert step_tag(EPOCH_MASK, TAG_SLOTS - 1) < 2**31  # a tag is a positive int32

# --------------------------------------------------------------------------
# The build keys and the argument lists

K2_CORES = ("tensor", "gated", "split")  # K2Build.core
_CORE_SUFFIX = {"tensor": "", "gated": "_g"}  # split: "_sw" / "_full"


def _check_key(key, ints: tuple[str, ...], bools: tuple[str, ...]) -> None:
    """The fields ``ints`` / ``bools`` typed, the width in range, and the fault
    hook complete and able to fire in a K2 launch."""
    cls = type(key).__name__
    for f in ints:
        v = getattr(key, f)
        if type(v) is not int:  # bool too: it would name the symbol "sTrue"
            raise TypeError(f"{cls}.{f} = {v!r}: an int")
    for f in bools:
        v = getattr(key, f)
        if type(v) is not bool:
            raise TypeError(f"{cls}.{f} = {v!r}: a bool")
    if not 1 <= key.tokens <= MAX_ROWS:
        raise ValueError(f"{cls}.tokens {key.tokens} (1 .. {MAX_ROWS})")
    if isinstance(key, K1Build):
        return
    if not -1 <= key.fault_rank < TP:
        raise ValueError(f"{cls}.fault_rank {key.fault_rank} (-1 .. {TP - 1})")
    if not key.faulty:
        # One build, one key: the fault fields of a build without the hook
        # would key a second binary under the same kernel name.
        if (key.fault_epoch, key.fault_slot, key.fault_ticks) != (0, 0, 0):
            raise ValueError(f"{cls}.fault_* set without fault_rank")
        return
    # sync.stall compares these with the int32 epoch word and the slot
    # argument: an epoch past int32 or a slot this kernel never runs never fires.
    if not 0 <= key.fault_epoch < 2**31:
        raise ValueError(f"{cls}.fault_epoch {key.fault_epoch} (0 .. 2^31 - 1)")
    if not K2_SLOT0 <= key.fault_slot < TAG_SLOTS:
        raise ValueError(
            f"{cls}.fault_slot {key.fault_slot}: a K2 slot "
            f"({K2_SLOT0} .. {TAG_SLOTS - 1}, spec.k2_slot)"
        )
    if not 0 < key.fault_ticks < 2**63:
        raise ValueError(f"{cls}.fault_ticks {key.fault_ticks} (> 0)")


def _fault_suffix(key) -> str:
    if not key.faulty:
        return ""
    return f"_f{key.fault_rank}_{key.fault_epoch}_{key.fault_slot}_{key.fault_ticks}"


@dataclass(frozen=True)
class K2Build:
    """Every K2 build parameter, scalars only (``plan.key_tuple``): two builds
    differing in any field never share a binary. ``rank`` is a runtime argument,
    never a field (every rank runs one binary)."""

    tokens: int  # S, the launch's rows: 1 .. MAX_ROWS
    # The fault hook (sync.stall, tests only): rank fault_rank stalls fault_ticks
    # once, in the launch of slot fault_slot of the step at epoch fault_epoch.
    fault_rank: int = -1
    fault_epoch: int = 0
    fault_slot: int = 0
    fault_ticks: int = 0
    # The attention core K2 reads (K2_CORES): "tensor" o_proj's input from
    # vLLM (Phase A), "gated" vLLM's attention output and the qkv rows, the
    # gated norm in the core load (B2), "split" the split-KV core in-kernel
    # (C, q = 1 widths).
    core: str = "tensor"
    window: bool = False  # split only: a sliding layer's key window

    def __post_init__(self):
        _check_key(
            self,
            ("tokens", "fault_rank", "fault_epoch", "fault_slot", "fault_ticks"),
            ("window",),
        )
        if type(self.core) is not str:
            raise TypeError(f"K2Build.core = {self.core!r}: a str")
        if self.core not in K2_CORES:
            raise ValueError(f"K2Build.core {self.core!r} (one of {K2_CORES})")
        if self.window and self.core != "split":
            raise ValueError(f"K2Build.window with core {self.core!r} (split only)")

    @property
    def faulty(self) -> bool:
        """The build compiles ``sync.stall`` into its prologue."""
        return self.fault_rank >= 0

    def nsplit(self) -> int:
        """A split build's N: units a (token, KV head)."""
        return nsplit(self.tokens, self.window)

    def n_units(self) -> int:
        """A split build's units (CTAs [0, n_units))."""
        return n_units(self.tokens, self.window)

    def c0(self) -> int:
        """A split build's first combine CTA."""
        return C0(self.tokens)

    def kernel_name(self) -> str:
        """The kernel symbol: one per build. A Phase A build ("tensor") keeps
        its Phase A name."""
        if self.core == "split":
            core = "_sw" if self.window else "_full"
        else:
            core = _CORE_SUFFIX[self.core]
        return f"muse_mono_k2_s{self.tokens}{core}" + _fault_suffix(self)


@dataclass(frozen=True)
class K1Build:
    """Every K1 build parameter (Phase B1: the residual boundary, the qkv GEMV,
    QK-norm / RoPE and the KV insert of one layer), scalars only. One template;
    ``rope`` is the only difference between the two attention kinds."""

    tokens: int  # S, the launch's rows: 1 .. MAX_ROWS
    rope: bool  # a sliding layer's GPT-J RoPE (full layers: NoPE)

    def __post_init__(self):
        _check_key(self, ("tokens",), ("rope",))

    def kernel_name(self) -> str:
        """The kernel symbol: one per build."""
        kind = "rope" if self.rope else "nope"
        return f"muse_mono_k1_s{self.tokens}_{kind}"


K2_ARGS = (
    "core",  # bf16 [S, CORE]: o_proj's input
    "stream",  # fp32 [S, HIDDEN]: the residual stream
    "pad",  # bool [>= S]: pad rows
    "w_o",
    "attn_alpha",
    "attn_beta",
    "w_preffn",
    "w_router",
    "bias",
    "w_pool",
    "w_preexp",
    "w13",
    "w13s",
    "w2",
    "w2s",
    "w_postexp",
    "w_unpool",
    "stream_out",  # fp32 [S, HIDDEN]
    "branch_out",  # bf16 [S, HIDDEN]
    "scratch",
    "peers",  # int64 [TP]: every rank's peer buffer past PEER_HEAD
    "rank",
    "epoch",
    "slot",
    "health",
    "alarm",
)

# K2's arguments by K2Build.core: "gated" reads vLLM's attention output and the
# qkv rows (og at ROW_G) in place of ``core``; "split" reads the qkv rows, this
# layer's KV cache and its KV group's block table and sequence lengths. The
# launch state (the last 7) is every core's.
K2_KV_ARGS = (
    "kv",  # bf16 [blocks, KV_HEADS, KV_BLOCK, KV_C]: the layer's cache
    "kv_sb",  # byte strides of a block, a head, a token (Int64)
    "kv_sh",
    "kv_st",
    "kv_blocks",  # Int32: the cache's blocks (block ids are clamped below it)
)
K2_ARGS_BY_CORE = {
    "tensor": K2_ARGS,
    "gated": (
        "attn",  # bf16 [S, CORE]: the stock core's output, contiguous
        "qkvo",  # bf16 [S, QKV_ROWS]: the qkv rows (og at ROW_G)
        *K2_ARGS[1:],
    ),
    "split": (
        "qkvo",  # bf16 [S, QKV_ROWS]: q normalized and roped, og raw
        *K2_KV_ARGS,
        "block_table",  # int32 [bt_rows, bt_stride]: the KV group's (persistent)
        "bt_rows",  # Int32: the table's rows (its buffer's size)
        "bt_stride",  # Int32: the table's row stride in entries
        "seq_lens",  # int32 [>= S]: context lengths, this step's token included
        *K2_ARGS[1:],
    ),
}

K1_ARGS = (
    "branch",  # bf16 [S, HIDDEN]: the previous layer's deferred branch
    "stream",  # fp32 [S, HIDDEN]: the residual stream
    "pad",  # bool [>= S]: pad rows
    "positions",  # int64 [>= S]
    "slots",  # int64 [>= S]: this layer's slot mapping (< 0: no insert)
    "w_postffn",  # layer i - 1: post_ffn_norm.weight bf16
    "ffn_alpha",  # layer i - 1: add_ffn alpha / beta fp32
    "ffn_beta",
    "w_preqkv",  # layer i: pre_qkv_proj_norm.weight bf16
    "w_qkv",  # layer i: qkv_proj.weight bf16 [QKV_ROWS, HIDDEN]
    "cos_sin",  # layer i: rotary cos_sin_cache fp32 [max_pos, 64] (0 if NoPE)
    *K2_KV_ARGS,
    "max_pos",  # Int32: cos_sin's rows (positions are clamped below it)
    "stream_out",  # fp32 [S, HIDDEN]: K2's stream
    "qkvo",  # bf16 [S, QKV_ROWS]: the qkv rows after QK-norm / RoPE
    "scratch",
    "rank",  # no peers: K1 is rank-local; stall() compares the rank
    "epoch",
    "slot",
    "health",
    "alarm",
)

# --------------------------------------------------------------------------
# Scratch and peer layouts (section 5.10 of the design)

PAIR_BYTES = 8  # a (value, tag) int32 pair

# Mailbox regions, in layout order; then debug.region_ids order (wait records).
SCRATCH_REGIONS = (
    "xrdy",
    "ass",
    "bss",
    "ppart",
    "route",
    "inter",
    "isc",
    "upart",
)
PEER_REGIONS = ("ar", "rlog", "pool", "rsx", "sqx", "moe", "branch")
# Plain (untagged) scratch regions, after the mailboxes.
XROW = "xrow"  # bf16 [MAX_TOKENS, HIDDEN]: ffn_in rows (seam -> router, pooler)
XROW_ALIGN = 16  # dwordx4 loads

_MT, _MP = MAX_TOKENS, MAX_PICKS
SCRATCH_PAIRS = (
    ("xrdy", NA),  # j: seam slice j published (flag)
    ("ass", NA * _MT),  # j*MT + t: fp32 sum of attn_out^2 over slice j
    ("bss", NA * _MT),  # j*MT + t: fp32 sum of new_stream^2 over slice j
    ("ppart", P_GROUPS * (P_PARTS - 1) * _MT * ROWS),  # ((g*3 + q-1)*MT + t)*16 + r
    ("route", _MT * 16),  # t*16 + k: id; t*16 + 8 + k: fp32 weight
    ("inter", _MP * 64),  # pick*64 + 4g + w: a2 fp4 words
    ("isc", _MP * UG_GROUPS),  # pick*16 + g: a2 e8m0 code
    (None, RED_TASKS * _MP),  # reserved (the retired rank-local ``pss``)
    ("upart", U_GROUPS * (U_PARTS - 1) * _MT * ROWS),  # ((g*3 + q-1)*MT + t)*16 + r
)
# Reserved bytes before ``xrow`` (a retired debug region): kept so that every
# later offset, compiled into the kernels as an immediate, stays (as the
# reserved pairs of the retired ``pss`` above).
RESERVED_A = _MT * (XQ_WORDS + XSC_WORDS) * PAIR_BYTES
XROW_BYTES = _MT * HIDDEN * 2

# Phases B and C (DESIGN_BC 5.7): mailboxes, then plain regions, laid out past
# ``xrow`` so that every Phase A offset (and every Phase A K2 binary) stays.
SCRATCH_PAIRS_BC = (
    ("fss", NA * _MT),  # j*MT + t: K1 boundary, fp32 sum of x^2 over slice j
    ("nss", NA * _MT),  # j*MT + t: fp32 sum of new_stream^2 over slice j
    ("ardy", NA),  # j: K1 boundary slice j's arow published (flag)
    ("qpart", QKV_TILES * K1_MAXC * _MT * ROWS),  # ((T*2 + c)*MT + t)*16 + r
    ("srdy", BLOCKS),  # u: split unit u's opart / pstat published (flag)
    ("crdy", 16 * _MT),  # 16 t + h: combine (t, h)'s xo row published (flag)
)
SCRATCH_REGIONS_BC = tuple(n for n, _ in SCRATCH_PAIRS_BC)
SCRATCH_REGIONS_ALL = SCRATCH_REGIONS + SCRATCH_REGIONS_BC
# Every region a wait record may name (debug.region_ids; ids are name CRCs, so
# the order is free).
REGIONS = SCRATCH_REGIONS_ALL + PEER_REGIONS

# Plain (untagged) regions past the B/C mailboxes: (name, bytes, alignment).
AROW = "arow"  # bf16 [MT, HIDDEN]: K1's attn_in rows (boundary -> GEMV)
XO = "xo"  # bf16 [MT, CORE]: the split core's gated rows (combine -> load_core)
PSTAT = "pstat"  # fp32 [BLOCKS, 8, 2]: a split unit's (m, l) per q head
OPART = "opart"  # fp32 [BLOCKS, 8, HEAD_DIM]: a split unit's unnormalized rows
SCRATCH_PLAIN_BC = (
    (AROW, _MT * HIDDEN * 2, 16),
    (None, _MT * QKV_ROWS * 2, 16),  # reserved (a retired debug region)
    (XO, _MT * CORE * 2, 16),
    (PSTAT, BLOCKS * Q_PER_KV * 2 * 4, 16),
    (OPART, BLOCKS * Q_PER_KV * HEAD_DIM * 4, 16),
)

# One parity half; pair indexes are in the reader's buffer.
PEER_PAIRS = (
    ("ar", TP * _MT * HIDDEN // 2),  # ((src*MT + t)*8192 + col)/2: bf16 partials
    ("rlog", TP * _MT * EXPERTS_RANK),  # (src*MT + t)*32 + i: fp32 logits
    ("pool", _MT * MOE_HIDDEN // 2),  # (t*4096 + col)/2: bf16 pooled
    ("rsx", TP * _MP * OWN_COLS // 2),  # ((src*MP + pick)*512 + cc)/2: bf16 rows
    ("sqx", TP * _MP * RED_TASKS),  # (src*MP + pick)*32 + j: fp32 sum x^2, 16 cols
    ("moe", _MT * MOE_HIDDEN // 2),  # (t*4096 + col)/2: bf16 MoE output
    ("branch", _MT * HIDDEN // 2),  # (t*8192 + col)/2: bf16 unpooler output
)


def pair_layout(items, align: int = PAIR_BYTES, start: int = 0) -> dict:
    """``items`` (name, pairs) laid out in order: name -> (byte offset, bytes),
    each region from an ``align``-byte boundary at or past ``start``; a None
    name reserves its pairs."""
    out, off = {}, start
    for name, pairs in items:
        off = (off + align - 1) // align * align
        if name is not None:
            out[name] = (off, pairs * PAIR_BYTES)
        off += pairs * PAIR_BYTES
    return out


def _end(layout: dict) -> int:
    return max(off + size for off, size in layout.values())


def _align(off: int, align: int) -> int:
    return (off + align - 1) // align * align


def scratch_layout() -> dict[str, tuple[int, int]]:
    """A rank's scratch (one buffer shared by every launch of a step; the tags
    tell launches apart): the mailboxes, ``RESERVED_A`` bytes, then ``xrow``
    (16-byte aligned); then Phases B / C's mailboxes and plain regions.
    name -> (byte offset, bytes)."""
    out = pair_layout(SCRATCH_PAIRS)
    out[XROW] = (_align(_end(out) + RESERVED_A, XROW_ALIGN), XROW_BYTES)
    out.update(pair_layout(SCRATCH_PAIRS_BC, start=_end(out)))
    off = _end(out)
    for name, nbytes, align in SCRATCH_PLAIN_BC:
        off = _align(off, align)
        if name is not None:
            out[name] = (off, nbytes)
        off += nbytes
    return out


def scratch_bytes() -> int:
    return _end(scratch_layout())


def peer_layout() -> dict[str, tuple[int, int]]:
    """One parity half of a rank's peer buffer: name -> (byte offset, bytes).
    The kernels see the buffer past ``abi.PEER_HEAD``; parity ``e & 1`` adds
    ``e & 1`` halves."""
    return pair_layout(PEER_PAIRS)


def peer_half_bytes() -> int:
    return _end(peer_layout())


def peer_bytes() -> int:
    """Both parity halves; ``abi.PEER_HEAD`` is not included."""
    return 2 * peer_half_bytes()


# --------------------------------------------------------------------------
# Pair indexes (section 5.10): pair units for put / put_words / poll; put_bf
# takes the element index 2 * pair. Plain integer arithmetic, so the host tests
# and the kernels' traced Int32 math read the same formulas.


def ar_pair(src, t, col):
    return ((src * _MT + t) * HIDDEN + col) // 2


def rlog_pair(src, t, i):
    return (src * _MT + t) * EXPERTS_RANK + i


def pool_pair(t, col):
    return (t * MOE_HIDDEN + col) // 2


def rsx_pair(src, pick, cc):
    return ((src * _MP + pick) * OWN_COLS + cc) // 2


def sqx_pair(src, j, pick):
    """Owner CTA j of rank src's 16-column sum of squares of ``pick``: a
    rank's 32 slices of a pick adjacent (a lane loads 4 as two 2-pair specs)."""
    return (src * _MP + pick) * RED_TASKS + j


def moe_pair(t, col):
    return (t * MOE_HIDDEN + col) // 2


def branch_pair(t, col):
    return (t * HIDDEN + col) // 2


def xrdy_pair(j):
    return j


def ss_pair(j, t):
    """``ass`` and ``bss`` (K2's seam); ``fss`` and ``nss`` (K1's boundary)."""
    return j * _MT + t


def ppart_pair(g, q, t, r):
    return ((g * (P_PARTS - 1) + q - 1) * _MT + t) * ROWS + r


def route_id_pair(t, k):
    return t * 16 + k


def route_weight_pair(t, k):
    return t * 16 + 8 + k


def upart_pair(g, q, t, r):
    return ((g * (U_PARTS - 1) + q - 1) * _MT + t) * ROWS + r


def ardy_pair(j):
    return j


def qpart_pair(T, c, t, r):
    """K1: contributor c's partial of qkv tile T's row r for token t."""
    return ((T * K1_MAXC + c) * _MT + t) * ROWS + r


def srdy_pair(u):
    return u


def crdy_pair(t, h):
    return 16 * t + h


# Plain (untagged) regions: word / element indexes from the region's base.


def xrow_elem(t, col):
    """``xrow`` bf16 element of row t, column col."""
    return t * HIDDEN + col


def arow_elem(t, col):
    """``arow`` bf16 element of row t, column col."""
    return t * HIDDEN + col


def xo_elem(t, h, d):
    """``xo`` bf16 element of token t, q head h, dim d."""
    return t * CORE + HEAD_DIM * h + d


def pstat_word(u, j, k):
    """``pstat`` fp32 word of split unit u, its q head j (of 8), k = 0: the
    running max m (base 2), k = 1: the sum l."""
    return (u * Q_PER_KV + j) * 2 + k


def opart_word(u, j, d):
    """``opart`` fp32 word of split unit u, its q head j, dim d."""
    return (u * Q_PER_KV + j) * HEAD_DIM + d
