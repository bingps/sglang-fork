from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE = {
    "bfloat16": {"flashmla_sparse"},
    "fp8_e4m3": {"flashmla_kv"},
}
HISPARSE_ROCM_DSA_BACKENDS = {"tilelang", "aiter"}
HISPARSE_KV_CACHE_DTYPES = ("bfloat16", "fp8_e4m3")


def _is_hip() -> bool:
    from sglang.srt.server_args import is_hip

    return is_hip()


def _hisparse_default_backend(kv_cache_dtype: str) -> str:
    if _is_hip():
        return "tilelang"
    return "flashmla_kv" if kv_cache_dtype == "fp8_e4m3" else "flashmla_sparse"


def _hisparse_allowed_backends(kv_cache_dtype: str) -> set[str]:
    if _is_hip():
        return HISPARSE_ROCM_DSA_BACKENDS
    return HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE.get(
        kv_cache_dtype, {"flashmla_sparse", "flashmla_kv"}
    )


# The hisparse DSA backend defaults moved to the resolution pipeline
# (arg_groups/overrides.py: _dsa_split_backend_resolution, hisparse arm).


def validate_hisparse_dsa_backend(
    server_args: ServerArgs, attr: str, label: str
) -> None:
    from sglang.srt.arg_groups.overrides import resolved_view

    # Invoked after the DSA kv-cache-dtype / split-backend declarations:
    # read the resolving state through the view.
    view = resolved_view(server_args)
    backend = getattr(view, attr)
    kv_cache_dtype = view.kv_cache_dtype
    allowed_backends = _hisparse_allowed_backends(kv_cache_dtype)
    if backend is not None and backend not in allowed_backends:
        raise ValueError(
            f"HiSparse supports DSA {label} backend(s) {sorted(allowed_backends)} "
            f"on this platform with --kv-cache-dtype={kv_cache_dtype}, "
            f"but got --dsa-{label}-backend={backend}. "
            f"Please use --dsa-{label}-backend="
            f"{_hisparse_default_backend(kv_cache_dtype)} "
            "or omit it."
        )


def validate_hisparse_kv_cache_dtype(server_args: ServerArgs) -> None:
    from sglang.srt.arg_groups.overrides import resolved_view

    kv_cache_dtype = resolved_view(server_args).kv_cache_dtype
    if kv_cache_dtype in HISPARSE_KV_CACHE_DTYPES:
        return

    choices = " or ".join(
        f"--kv-cache-dtype={dtype}" for dtype in HISPARSE_KV_CACHE_DTYPES
    )
    raise ValueError(
        f"HiSparse requires one of {HISPARSE_KV_CACHE_DTYPES} KV cache dtypes, "
        f"but got --kv-cache-dtype={kv_cache_dtype}. Please use {choices}."
    )


# Speculative algorithms whose worker runs EAGLEWorkerV2.forward_batch_generation
# unmodified. That method carries the HiSparse admit hook (staging DMA + device
# buffer + spec staging ring allocation right after prefill); workers that
# override or bypass it (multi-layer EAGLE, FROZEN_KV_MTP, NGRAM, DFLASH,
# DSPARK, custom plugins) never admit requests, so the first decode round
# would fail on the unallocated staging ring — or worse, silently skip ring
# recycling. NEXTN is aliased to EAGLE before validation runs.
# STANDALONE is excluded: its draft model may be non-DSA, and the draft KV
# pool's flat-logical sizing (kv_cache_configurator._build_dsa_kv_pool) is
# only implemented for DSA drafts — a non-DSA draft pool would be sized to
# the physical space while addressed with the target allocator's logical
# ids, the exact out-of-bounds class fixed for the NextN draft.
HISPARSE_SPEC_ALGORITHMS = {"EAGLE", "EAGLE3"}

# DSA backends whose target-verify path implements the HiSparse page swap-in.
# This startup check is the sole enforcement: a backend outside this set has no
# verify swap-in path (dsa_backend routes verify through swap_in_verify_pages
# only for these impls).
HISPARSE_SPEC_VERIFY_BACKENDS = {"flashmla_sparse", "flashmla_kv", "tilelang"}


def _validate_hisparse_speculative_algorithm(server_args: ServerArgs) -> None:
    algo = server_args.speculative_algorithm
    if algo is None:
        return
    if algo not in HISPARSE_SPEC_ALGORITHMS:
        raise ValueError(
            "--enable-hisparse with speculative decoding is only supported for "
            f"{sorted(HISPARSE_SPEC_ALGORITHMS)} (workers that inherit "
            "EAGLEWorkerV2.forward_batch_generation, which performs the "
            "HiSparse staging/buffer/ring admission after prefill), "
            f"but got --speculative-algorithm={algo}."
        )
    if server_args.enable_multi_layer_eagle:
        raise ValueError(
            "--enable-hisparse is not supported with --enable-multi-layer-eagle: "
            "MultiLayerEagleWorkerV2 overrides forward_batch_generation without "
            "the HiSparse admission hook."
        )

    # Tree drafting (topk > 1) moves accepted KV via move_kv_cache() with
    # LOGICAL ids; HiSparseDSATokenToKVPool does not override that method, so
    # the base implementation would index the physical buffer directly —
    # wrong slots, and out of bounds once logical ids exceed the physical
    # pool (host_to_device_ratio > 1). Only the linear chain (topk in
    # {None, 1}) is supported until the move path translates both sides.
    if server_args.speculative_eagle_topk not in (None, 1):
        raise ValueError(
            "--enable-hisparse with speculative decoding requires a linear "
            "draft chain (--speculative-eagle-topk in {None, 1}): the tree "
            "accept path moves KV with untranslated logical ids. Got "
            f"--speculative-eagle-topk={server_args.speculative_eagle_topk!r}."
        )

    # PD disaggregation + HiSparse + MTP is unvalidated and known-unsafe: the
    # decode-side KV offload path hands untranslated logical ids to the plain
    # KV backup and bypasses hisparse_coordinator.request_finished (leaking
    # device buffer / staging ring / host pool). Reject the combination until
    # a coordinator-aware offload lands. Plain HiSparse (no spec) keeps its
    # existing direct-to-host PD path.
    if server_args.disaggregation_mode != "null":
        raise ValueError(
            "--enable-hisparse with speculative decoding is not supported "
            "under PD disaggregation (--disaggregation-mode="
            f"{server_args.disaggregation_mode!r}): the decode-side KV "
            "offload path neither translates logical->physical locations nor "
            "releases HiSparse per-request resources. Drop "
            "--speculative-algorithm or run without disaggregation."
        )

    # The plain-hisparse whitelist admits backends (e.g. ROCm aiter) whose
    # target-verify path has no swap-in implementation; without this check the
    # server boots and dies with NotImplementedError on the first verify.
    from sglang.srt.arg_groups.overrides import resolved_view

    decode_backend = resolved_view(server_args).dsa_decode_backend
    if (
        decode_backend is not None
        and decode_backend not in HISPARSE_SPEC_VERIFY_BACKENDS
    ):
        raise ValueError(
            "--enable-hisparse with speculative decoding requires a DSA decode "
            f"backend with a verify swap-in path {sorted(HISPARSE_SPEC_VERIFY_BACKENDS)}, "
            f"but got --dsa-decode-backend={decode_backend}. Use one of the "
            "supported backends or drop --speculative-algorithm."
        )


def _validate_hisparse_device_buffer_size(server_args: ServerArgs) -> None:
    """The verify swap-in needs top_k buffer slots for every draft token.

    Checked here rather than at coordinator init so an undersized buffer fails
    at launch instead of part-way through memory-pool initialization.
    """
    from sglang.srt.mem_cache.sparsity import parse_hisparse_config

    hisparse_cfg = parse_hisparse_config(server_args)
    # index_topk is an optional HF field; fall back to the hisparse config.
    top_k = getattr(
        server_args.get_model_config().hf_text_config,
        "index_topk",
        hisparse_cfg.top_k,
    )
    num_draft_tokens = server_args.max_speculative_num_draft_tokens or 1
    required = top_k * num_draft_tokens
    if hisparse_cfg.device_buffer_size < required:
        raise ValueError(
            f"HiSparse requires device_buffer_size "
            f"({hisparse_cfg.device_buffer_size}) >= top_k * num_draft_tokens "
            f"({required} = {top_k} * {num_draft_tokens}). Raise "
            '"device_buffer_size" in --hisparse-config, or reduce '
            "--speculative-num-draft-tokens."
        )


def validate_hisparse(server_args: ServerArgs) -> None:
    """Validate --enable-hisparse constraints (model class, radix cache, DSA backend)."""
    if not server_args.enable_hisparse:
        return

    from sglang.srt.configs.model_config import (
        is_deepseek_dsa,
        is_deepseek_v4,
    )

    hf_config = server_args.get_model_config().hf_config
    is_v4_hisparse = is_deepseek_v4(hf_config)
    is_hip = _is_hip()
    assert is_deepseek_dsa(hf_config) or is_v4_hisparse, (
        "--enable-hisparse is only supported for DSA (DeepSeek Sparse Attention) "
        "models (e.g., DeepSeek V3.2, GLM-5) and DeepSeek V4 now. "
    )

    assert (
        server_args.disable_radix_cache
    ), "Hierarchical sparse attention currently requires --disable-radix-cache."

    # Decode-side KV offload hands req_to_token LOGICAL ids straight to the plain
    # KV backup without the HiSparse logical->physical translation, and its async
    # completion bypasses hisparse_coordinator.request_finished (leaking the
    # per-request device buffer / host pool / staging ring). This breaks plain
    # HiSparse and HiSparse+MTP alike, so reject the combination until a
    # coordinator-aware offload path lands.
    if server_args.disaggregation_decode_enable_offload_kvcache:
        raise ValueError(
            "--enable-hisparse is not supported with "
            "--disaggregation-decode-enable-offload-kvcache: the offload path "
            "neither translates logical->physical KV locations nor releases "
            "HiSparse per-request resources. Drop the offload flag or run "
            "without --enable-hisparse."
        )

    if is_v4_hisparse and server_args.speculative_algorithm is not None:
        raise ValueError(
            "--enable-hisparse with speculative decoding is not supported for "
            "DeepSeek V4 models: the MTP verify swap-in path and the "
            "speculative staging ring are DSV3.2-only (the coordinator "
            "rejects DSV4 at initialization). Run DSV4 HiSparse without "
            "--speculative-algorithm, or disable --enable-hisparse."
        )

    _validate_hisparse_speculative_algorithm(server_args)
    _validate_hisparse_device_buffer_size(server_args)

    # DSv4 hisparse handles its own dtype/backend pairing elsewhere; the dtype-
    # aware checks below only apply to the DSA hisparse path.
    if is_hip and is_v4_hisparse:
        # TEMPORARY GUARD: DSv4 HiSparse is not supported on the unified-KV path.
        # In unified-KV mode c4_kv_pool is None, so DeepSeekV4HiSparseTokenToKVPoolAllocator
        # cannot attach and pool init dies with a cryptic AssertionError. Fail fast
        # at startup with a clear message instead. Remove once unified-KV HiSparse lands.
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if is_unified_kv_triton():
            raise ValueError(
                "--enable-hisparse is not supported with the unified-KV path on ROCm"
                "(SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton) for DeepSeek-V4: "
                "HiSparse currently requires the separate packed KV layout. "
                "Either set SGLANG_HACK_FLASHMLA_BACKEND=triton, or run without "
                "--enable-hisparse."
            )
        return

    from sglang.srt.arg_groups.overrides import resolved_view

    if resolved_view(server_args).kv_cache_dtype not in (
        "bfloat16",
        "auto",
        "fp8_e4m3",
    ):
        validate_hisparse_kv_cache_dtype(server_args)

    for attr, label in [
        ("dsa_prefill_backend", "prefill"),
        ("dsa_decode_backend", "decode"),
    ]:
        validate_hisparse_dsa_backend(server_args, attr, label)
