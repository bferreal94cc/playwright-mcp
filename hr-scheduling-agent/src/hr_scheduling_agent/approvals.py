"""Multi-stage approval routing with timeout escalation.

Two stage modes cover the workflows the design calls for:

* ``SEQUENTIAL`` -- named approvers sign off in order. Used for offers and
  anything with a chain of command.
* ``QUORUM``     -- any *k* of *n* approvers suffice. This is the
  Byzantine-tolerant path: it reaches a decision while up to ``n - k``
  approvers are unreachable, asleep, or disagreeing, which is the normal state
  of affairs at 03:00 in a hospital.

Stages carry a deadline. When it lapses the stage resolves itself by policy --
escalate, auto-approve, or reject -- so a schedule is never silently stuck
waiting on someone who has gone home. Every transition is written to the audit
log with its reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .audit import AuditLog


class Decision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


class StageMode(str, Enum):
    SEQUENTIAL = "sequential"
    QUORUM = "quorum"


class TimeoutPolicy(str, Enum):
    ESCALATE = "escalate"
    AUTO_APPROVE = "auto_approve"
    REJECT = "reject"


class RequestState(str, Enum):
    PENDING = "pending"
    ESCALATED = "escalated"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ApprovalStage:
    """One gate a request must pass."""

    name: str
    approvers: tuple[str, ...]
    mode: StageMode = StageMode.SEQUENTIAL
    quorum: int = 1
    timeout_hours: float = 4.0
    on_timeout: TimeoutPolicy = TimeoutPolicy.ESCALATE
    escalate_to: str | None = None

    def required_approvals(self) -> int:
        if self.mode is StageMode.QUORUM:
            return max(1, min(self.quorum, len(self.approvers)))
        return len(self.approvers)


@dataclass(frozen=True)
class RecordedDecision:
    """A single approver's answer at a single stage."""

    stage_index: int
    approver: str
    decision: Decision
    reason: str
    at: datetime


@dataclass
class ApprovalRequest:
    """A schedule, exception cover or offer awaiting sign-off."""

    id: str
    subject: str
    summary: str
    stages: tuple[ApprovalStage, ...]
    created_at: datetime
    stage_started_at: datetime
    current_stage: int = 0
    state: RequestState = RequestState.PENDING
    decisions: list[RecordedDecision] = field(default_factory=list)
    resolution_note: str = ""

    @property
    def is_open(self) -> bool:
        return self.state in (RequestState.PENDING, RequestState.ESCALATED)

    @property
    def stage(self) -> ApprovalStage | None:
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return None

    def decisions_at(self, stage_index: int) -> list[RecordedDecision]:
        return [d for d in self.decisions if d.stage_index == stage_index]

    def pending_approvers(self) -> list[str]:
        """Who the request is currently waiting on."""
        stage = self.stage
        if stage is None or not self.is_open:
            return []
        decided = {d.approver for d in self.decisions_at(self.current_stage)}
        if self.state is RequestState.ESCALATED:
            return [stage.escalate_to] if stage.escalate_to else []
        outstanding = [a for a in stage.approvers if a not in decided]
        if stage.mode is StageMode.SEQUENTIAL:
            return outstanding[:1]
        return outstanding

    def deadline(self) -> datetime | None:
        stage = self.stage
        if stage is None or not self.is_open:
            return None
        return self.stage_started_at + timedelta(hours=stage.timeout_hours)


class ApprovalRouter:
    """Drives requests through their stages and records every transition."""

    def __init__(self, audit_log: AuditLog | None = None) -> None:
        self.audit = audit_log or AuditLog()

    # -- lifecycle --------------------------------------------------------

    def submit(
        self,
        request_id: str,
        subject: str,
        summary: str,
        stages: tuple[ApprovalStage, ...],
        at: datetime | None = None,
    ) -> ApprovalRequest:
        if not stages:
            raise ValueError("an approval workflow needs at least one stage")
        now = at or datetime.now()
        request = ApprovalRequest(
            id=request_id,
            subject=subject,
            summary=summary,
            stages=stages,
            created_at=now,
            stage_started_at=now,
        )
        self.audit.record(
            actor="approval_router",
            action="approval_requested",
            subject=subject,
            reasoning=summary,
            metadata={
                "request_id": request_id,
                "stages": [s.name for s in stages],
                "first_stage_approvers": list(stages[0].approvers),
            },
            at=now,
        )
        return request

    def decide(
        self,
        request: ApprovalRequest,
        approver: str,
        decision: Decision,
        reason: str = "",
        at: datetime | None = None,
    ) -> ApprovalRequest:
        """Record one approver's answer and re-evaluate the current stage."""
        now = at or datetime.now()
        if not request.is_open:
            raise ValueError(f"request '{request.id}' is already {request.state.value}")
        stage = request.stage
        if stage is None:
            raise ValueError(f"request '{request.id}' has no active stage")

        permitted = request.pending_approvers()
        if approver not in permitted:
            raise ValueError(
                f"'{approver}' cannot act on stage '{stage.name}' right now; "
                f"waiting on {permitted or 'nobody'}"
            )

        request.decisions.append(
            RecordedDecision(
                stage_index=request.current_stage,
                approver=approver,
                decision=decision,
                reason=reason,
                at=now,
            )
        )
        self.audit.record(
            actor=approver,
            action=f"stage_{decision.value}",
            subject=request.subject,
            reasoning=reason or f"{decision.value} at stage '{stage.name}'",
            metadata={"request_id": request.id, "stage": stage.name},
            at=now,
        )

        if request.state is RequestState.ESCALATED:
            # The escalation target's word settles the stage outright.
            if decision is Decision.APPROVE:
                self._advance(request, now, f"escalation approved by {approver}")
            else:
                self._reject(request, now, f"escalation rejected by {approver}")
            return request

        self._evaluate(request, stage, now)
        return request

    def tick(self, request: ApprovalRequest, now: datetime) -> ApprovalRequest:
        """Apply the timeout policy if the current stage's deadline has passed."""
        deadline = request.deadline()
        if deadline is None or now < deadline:
            return request
        stage = request.stage
        if stage is None:
            return request

        if request.state is RequestState.ESCALATED:
            # An escalation that also times out is rejected -- failing closed
            # is the safe default for a schedule nobody has looked at.
            self._reject(request, now, f"escalation to '{stage.escalate_to}' also timed out")
            return request

        if stage.on_timeout is TimeoutPolicy.AUTO_APPROVE:
            self._advance(request, now, f"stage '{stage.name}' auto-approved after timeout")
        elif stage.on_timeout is TimeoutPolicy.REJECT:
            self._reject(request, now, f"stage '{stage.name}' rejected after timeout")
        else:
            request.state = RequestState.ESCALATED
            request.stage_started_at = now
            self.audit.record(
                actor="approval_router",
                action="stage_escalated",
                subject=request.subject,
                reasoning=(
                    f"stage '{stage.name}' passed its {stage.timeout_hours:g}h deadline; "
                    f"escalated to {stage.escalate_to or 'unassigned'}"
                ),
                metadata={"request_id": request.id, "stage": stage.name},
                at=now,
            )
        return request

    # -- internals --------------------------------------------------------

    def _evaluate(self, request: ApprovalRequest, stage: ApprovalStage, now: datetime) -> None:
        decisions = request.decisions_at(request.current_stage)
        approvals = sum(1 for d in decisions if d.decision is Decision.APPROVE)
        rejections = sum(1 for d in decisions if d.decision is Decision.REJECT)

        if stage.mode is StageMode.SEQUENTIAL:
            if rejections:
                self._reject(request, now, f"rejected at stage '{stage.name}'")
            elif approvals == len(stage.approvers):
                self._advance(request, now, f"stage '{stage.name}' fully approved")
            return

        required = stage.required_approvals()
        if approvals >= required:
            self._advance(request, now, f"stage '{stage.name}' reached quorum {approvals}/{required}")
        elif rejections > len(stage.approvers) - required:
            self._reject(request, now, f"stage '{stage.name}' can no longer reach quorum")

    def _advance(self, request: ApprovalRequest, now: datetime, reasoning: str) -> None:
        request.current_stage += 1
        request.stage_started_at = now
        request.state = RequestState.PENDING
        if request.current_stage >= len(request.stages):
            request.state = RequestState.APPROVED
            request.resolution_note = reasoning
            self.audit.record(
                actor="approval_router",
                action="request_approved",
                subject=request.subject,
                reasoning=reasoning,
                metadata={"request_id": request.id},
                at=now,
            )
        else:
            self.audit.record(
                actor="approval_router",
                action="stage_advanced",
                subject=request.subject,
                reasoning=reasoning,
                metadata={
                    "request_id": request.id,
                    "next_stage": request.stages[request.current_stage].name,
                },
                at=now,
            )

    def _reject(self, request: ApprovalRequest, now: datetime, reasoning: str) -> None:
        request.state = RequestState.REJECTED
        request.resolution_note = reasoning
        self.audit.record(
            actor="approval_router",
            action="request_rejected",
            subject=request.subject,
            reasoning=reasoning,
            metadata={"request_id": request.id},
            at=now,
        )


# -- ready-made workflows -------------------------------------------------


def schedule_approval_workflow(
    manager: str, finance: str = "finance-lead", director: str = "ops-director"
) -> tuple[ApprovalStage, ...]:
    """Weekly schedule: the manager signs, then finance checks the budget."""
    return (
        ApprovalStage(
            name="manager-review",
            approvers=(manager,),
            mode=StageMode.SEQUENTIAL,
            timeout_hours=4.0,
            on_timeout=TimeoutPolicy.ESCALATE,
            escalate_to=director,
        ),
        ApprovalStage(
            name="budget-check",
            approvers=(finance,),
            mode=StageMode.SEQUENTIAL,
            timeout_hours=8.0,
            on_timeout=TimeoutPolicy.AUTO_APPROVE,
        ),
    )


def exception_approval_workflow(
    on_shift_leads: tuple[str, ...], quorum: int = 1, director: str = "ops-director"
) -> tuple[ApprovalStage, ...]:
    """Urgent cover: any one of the leads on duty can clear it within the hour."""
    return (
        ApprovalStage(
            name="urgent-cover",
            approvers=on_shift_leads,
            mode=StageMode.QUORUM,
            quorum=quorum,
            timeout_hours=1.0,
            on_timeout=TimeoutPolicy.ESCALATE,
            escalate_to=director,
        ),
    )
