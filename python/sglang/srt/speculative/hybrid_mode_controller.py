"""Runtime controller for pure-MTP <-> HiSparse+MTP switching.

The server always boots HiSparse (the pool family, host mirror and coordinator
are built once at startup). A request is either:

- **resident** (pure MTP): its KV stays fully device-resident, the decode batch
  carries no coordinator, and attention runs the plain fused-topk + translate
  path. This is exactly a request that finished prefill but was never admitted
  into HiSparse staging.
- **offloaded** (HiSparse+MTP): the existing HiSparse flow -- device buffer +
  spec staging ring + host backup + per-step swap-in.

Switching is *global and batch-level*: the whole running batch is in one mode
at a time, so there are no mixed-mode batches. When the mode flips, in-flight
requests are migrated (KV backed up to host / restored to device) rather than
drained. Data movement (Phase 4/5) is separated from the cheap metadata flip so
the old path stays valid until the flip; the transient PENDING_* states model
that window.

This module (Phase 1) provides the decision logic and state enum. The migration
driving and scheduler wiring land in later phases; see the seams marked
``TODO(hybrid)`` below.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import TYPE_CHECKING, List, Optional

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class DecodeMode(IntEnum):
    """Global decode mode for the hybrid controller.

    MTP and HISPARSE are the two *stable* modes. PENDING_OFFLOAD and
    PENDING_RESTORE are transient windows during which requests are being
    migrated but the *old* attention path is still the valid one (data has not
    finished moving), so the batch keeps running the source mode's path.
    """

    MTP = 0
    PENDING_OFFLOAD = 1
    HISPARSE = 2
    PENDING_RESTORE = 3

    @property
    def attaches_coordinator(self) -> bool:
        """Whether the decode batch should carry the HiSparse coordinator.

        True for the offloaded path. During PENDING_RESTORE the device buffer
        and host backup are still live, so the batch keeps running the HiSparse
        path until the restore flip completes.
        """
        return self in (DecodeMode.HISPARSE, DecodeMode.PENDING_RESTORE)

    @property
    def admits_offloaded(self) -> bool:
        """Whether a freshly prefilled request should be admitted into HiSparse.

        True once we are heading into / are in the offloaded mode. During
        PENDING_RESTORE we are leaving HiSparse, so new requests stay resident.
        """
        return self in (DecodeMode.HISPARSE, DecodeMode.PENDING_OFFLOAD)

    @property
    def is_migrating(self) -> bool:
        return self in (DecodeMode.PENDING_OFFLOAD, DecodeMode.PENDING_RESTORE)


def _parse_force_mode() -> Optional[DecodeMode]:
    """Optional debug pin so Phase 0 can measure a fixed path.

    ``SGLANG_FORCE_HISPARSE_MTP_MODE=mtp`` pins every request resident (pure
    MTP on a HiSparse-booted server); ``=hisparse`` pins the existing
    HiSparse+MTP behavior. Unset leaves the controller in charge.
    """
    raw = (envs.SGLANG_FORCE_HISPARSE_MTP_MODE.get() or "").strip().lower()
    if not raw:
        return None
    if raw == "mtp":
        return DecodeMode.MTP
    if raw == "hisparse":
        return DecodeMode.HISPARSE
    raise ValueError(
        f"SGLANG_FORCE_HISPARSE_MTP_MODE must be 'mtp' or 'hisparse', got {raw!r}"
    )


class HybridModeController:
    """Decide the global decode mode from KV-pool pressure and batch size.

    Decision signals (both already computed every scheduler step):
      - primary: ``kv_pool_usage`` = used KV tokens / total KV capacity
      - secondary: ``batch_size`` (guards small-batch swap-in inefficiency and
        latency-sensitive low-concurrency cases)

    Hysteresis: an asymmetric up/down usage band plus a post-switch cooldown
    prevents ping-pong. This mirrors the AdaptiveController's philosophy but the
    state here is a single global mode rather than per-request spec steps.
    """

    def __init__(
        self,
        hisparse_coordinator: "HiSparseCoordinator",
        token_to_kv_pool_allocator,
        *,
        usage_threshold_up: float = 0.6,
        usage_threshold_down: float = 0.3,
        min_bsz_for_hisparse: int = 8,
        max_bsz_for_mtp: int = 4,
        cooldown_steps: int = 10,
        tp_group=None,
    ):
        if not usage_threshold_down < usage_threshold_up:
            raise ValueError(
                "hisparse-mtp-usage-down must be < hisparse-mtp-usage-up, got "
                f"down={usage_threshold_down}, up={usage_threshold_up}"
            )
        self.hisparse_coordinator = hisparse_coordinator
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.usage_threshold_up = usage_threshold_up
        self.usage_threshold_down = usage_threshold_down
        self.min_bsz_for_hisparse = min_bsz_for_hisparse
        self.max_bsz_for_mtp = max_bsz_for_mtp
        self.cooldown_steps = cooldown_steps
        self.tp_group = tp_group

        self._forced_stable = _parse_force_mode()
        self.current_mode = (
            self._forced_stable if self._forced_stable is not None else DecodeMode.MTP
        )
        self._cooldown_remaining = 0

        if self._forced_stable is not None:
            logger.info(
                "HybridModeController: mode pinned to %s via "
                "SGLANG_FORCE_HISPARSE_MTP_MODE",
                self.current_mode.name,
            )
        else:
            logger.info(
                "HybridModeController: usage band [%.2f, %.2f], bsz band "
                "[<=%d MTP, >=%d HiSparse], cooldown=%d steps",
                self.usage_threshold_down,
                self.usage_threshold_up,
                self.max_bsz_for_mtp,
                self.min_bsz_for_hisparse,
                self.cooldown_steps,
            )

    # -- Decision --------------------------------------------------------

    def decide_stable_target(self, batch_size: int, kv_pool_usage: float) -> DecodeMode:
        """Return the desired *stable* mode (MTP or HISPARSE) for this step.

        Applies hysteresis and cooldown. The result is a stable target, not a
        transient PENDING_* state; the caller decides whether/how to migrate.
        """
        if self._forced_stable is not None:
            return self._forced_stable

        stable = (
            DecodeMode.HISPARSE
            if self.current_mode.attaches_coordinator
            else DecodeMode.MTP
        )

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return stable

        if stable == DecodeMode.MTP:
            if (
                kv_pool_usage >= self.usage_threshold_up
                and batch_size >= self.min_bsz_for_hisparse
            ):
                return DecodeMode.HISPARSE
        else:
            if (
                kv_pool_usage <= self.usage_threshold_down
                or batch_size <= self.max_bsz_for_mtp
            ):
                return DecodeMode.MTP
        return stable

    def note_switch(self) -> None:
        """Record that a switch just happened; arms the cooldown window."""
        self._cooldown_remaining = self.cooldown_steps

    # -- Per-step entry point -------------------------------------------

    def _device_pool_usage(self) -> float:
        """Device-pool pressure driving the mode decision.

        Reads the PHYSICAL hisparse device pool rather than the scheduler's
        full_token_usage: the latter is measured against the logical pool,
        which is host_to_device_ratio times larger, so it badly understates
        the pressure that actually constrains keeping requests device-resident.
        """
        return self.hisparse_coordinator.get_token_stats().device_token_usage

    def on_step(self, batch_size: int, reqs: List["Req"]) -> DecodeMode:
        """Advance the state machine for one decode step; return the mode to run.

        Must be called before the scheduler attaches (or withholds) the
        coordinator for this step, so the batch and its requests agree on the
        layout. Migration is all-or-nothing per switch: after it returns, every
        live request is in the layout the returned mode implies, so batches are
        never mixed.
        """
        if self._forced_stable is not None:
            return self.current_mode

        kv_pool_usage = self._device_pool_usage()
        stable = (
            DecodeMode.HISPARSE
            if self.current_mode.attaches_coordinator
            else DecodeMode.MTP
        )
        target = self.decide_stable_target(batch_size, kv_pool_usage)
        if target == stable:
            return self.current_mode

        live = [r for r in reqs if not r.finished()]
        if target == DecodeMode.HISPARSE:
            self._switch_to_hisparse(live, batch_size, kv_pool_usage)
        else:
            self._switch_to_mtp(live, batch_size, kv_pool_usage)
        return self.current_mode

    def _offload_headroom_ok(self, live: List["Req"]) -> bool:
        """Simulated transient-capacity check, in migration order.

        A long request largely funds its own migration: alloc_device_buffer
        keeps the buffer out of pages the request already owns and defer-frees
        the surplus, which _reclaim_deferred_staging_pages hands back to the
        ring allocation within the same admission
        (test_admit_long_prompt_reclaims_deferred_pages_for_ring). So the gate
        credits each request's own reclaimable surplus instead of demanding
        every ring and buffer shortfall from free space up front -- the
        pessimistic form permanently pinned the hybrid to resident exactly
        under the memory pressure it exists to relieve. ``live`` must already
        be in migration order (longest first, see _switch_to_hisparse), so
        net-freeing requests fund the net-allocating short ones behind them.
        """
        coord = self.hisparse_coordinator
        allocator = coord.token_to_kv_pool_allocator.hisparse_attn_allocator
        page = coord.mem_pool_device.page_size

        def feasible() -> bool:
            avail = allocator.available_size()
            for req in live:
                if int(coord.req_device_buffer_size[req.req_pool_idx]) > 0:
                    continue  # already offloaded, no transient cost
                held = req.kv.kv_allocated_len
                cost = coord.spec_ring_capacity + page  # ring + alignment slack
                cost += max(0, coord.padded_buffer_size - held)
                # Surplus its own offload defer-frees; one page shaved off for
                # the buffer-boundary page that free excludes.
                gain = max(0, held - coord.padded_buffer_size - page)
                if avail + gain < cost:
                    return False
                avail += gain - cost
            return True

        if feasible():
            return True
        coord._reclaim_deferred_staging_pages()
        return feasible()

    def _switch_to_hisparse(
        self, live: List["Req"], batch_size: int, usage: float
    ) -> None:
        """MTP -> HiSparse: offload every live request, then run the HiSparse path.

        Longest first: their defer-freed surplus funds the ring and buffer
        shortfall of the short (net-allocating) requests behind them, matching
        the order the headroom simulation assumes. The switch is postponed and
        retried on a later step when it cannot proceed atomically.
        """
        live = sorted(live, key=lambda r: r.kv.kv_allocated_len, reverse=True)
        if not self._offload_headroom_ok(live):
            logger.debug(
                "Hybrid: postponing MTP -> HiSparse, transient headroom "
                "insufficient (bsz=%d usage=%.3f)",
                batch_size,
                usage,
            )
            return
        migrated = 0
        for req in live:
            self.hisparse_coordinator.offload_running_request(req)
            migrated += 1
        self.current_mode = DecodeMode.HISPARSE
        self.note_switch()
        logger.info(
            "Hybrid: MTP -> HiSparse (bsz=%d usage=%.3f, offloaded %d)",
            batch_size,
            usage,
            migrated,
        )

    def _projected_resident_usage(self, live: List["Req"]) -> float:
        """Device-pool usage that would result from restoring ``live`` requests.

        Offloading is self-defeating as a *restore* signal: moving KV to host is
        exactly what drops device usage, so the post-offload usage is always far
        below the down-threshold and comparing against it flip-flops forever.
        Project what residency would actually cost instead -- each offloaded
        request would grow from its fixed footprint (device buffer + staging
        ring) back to a slot per allocated position.
        """
        coord = self.hisparse_coordinator
        allocator = coord.token_to_kv_pool_allocator.hisparse_attn_allocator
        capacity = allocator.size
        if capacity <= 0:
            return 1.0
        used = capacity - allocator.available_size()
        extra = 0
        for req in live:
            footprint = int(coord.req_device_buffer_size[req.req_pool_idx])
            if footprint > 0:
                footprint += coord.spec_ring_capacity
            extra += max(0, req.kv.kv_allocated_len - footprint)
        return (used + extra) / capacity

    def _switch_to_mtp(self, live: List["Req"], batch_size: int, usage: float) -> None:
        """HiSparse -> MTP: restore every live request, or stay offloaded.

        Restore needs a dedicated device slot for every allocated position, so
        it can genuinely not fit. restore_running_request allocates before it
        frees and reports failure without side effects, so a partial restore
        leaves the already-restored requests resident and the rest offloaded --
        which would be a mixed batch. Probe capacity by restoring in order and,
        on the first failure, roll the restored ones back by offloading them
        again (cheap and always succeeds), keeping the batch homogeneous.
        """
        # Refuse to restore into a state that would immediately re-offload.
        projected = self._projected_resident_usage(live)
        if projected >= self.usage_threshold_up:
            self.note_switch()
            logger.debug(
                "Hybrid: staying on HiSparse, restoring would put usage at "
                "%.3f >= %.2f (bsz=%d, current usage=%.3f)",
                projected,
                self.usage_threshold_up,
                batch_size,
                usage,
            )
            return

        restored: List["Req"] = []
        for req in live:
            if self.hisparse_coordinator.restore_running_request(req):
                restored.append(req)
                continue
            # Out of device capacity: undo and stay offloaded.
            for done in restored:
                self.hisparse_coordinator.offload_running_request(done)
            self.note_switch()
            logger.info(
                "Hybrid: HiSparse -> MTP aborted, device pool cannot hold %d "
                "requests resident (bsz=%d usage=%.3f); rolled back %d",
                len(live),
                batch_size,
                usage,
                len(restored),
            )
            return

        self.current_mode = DecodeMode.MTP
        self.note_switch()
        logger.info(
            "Hybrid: HiSparse -> MTP (bsz=%d usage=%.3f, restored %d)",
            batch_size,
            usage,
            len(restored),
        )
