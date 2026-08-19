"""Approval routing: sequential chains, quorum tolerance, and deadlines."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from hr_scheduling_agent.approvals import (
    ApprovalRouter,
    ApprovalStage,
    Decision,
    RequestState,
    StageMode,
    TimeoutPolicy,
    exception_approval_workflow,
    schedule_approval_workflow,
)
from hr_scheduling_agent.audit import AuditLog

T0 = datetime(2026, 8, 17, 9, 0)


@pytest.fixture
def router() -> ApprovalRouter:
    return ApprovalRouter(AuditLog())


def submit(router: ApprovalRouter, stages):
    return router.submit("req-1", "schedule", "test request", stages, at=T0)


class TestSequential:
    def test_single_approver_approves_outright(self, router):
        stages = (ApprovalStage("only", ("mgr",)),)
        request = submit(router, stages)
        router.decide(request, "mgr", Decision.APPROVE, "fine", at=T0)
        assert request.state is RequestState.APPROVED

    def test_all_approvers_must_sign(self, router):
        stages = (ApprovalStage("both", ("mgr", "director")),)
        request = submit(router, stages)
        router.decide(request, "mgr", Decision.APPROVE, at=T0)
        assert request.state is RequestState.PENDING
        router.decide(request, "director", Decision.APPROVE, at=T0)
        assert request.state is RequestState.APPROVED

    def test_order_is_enforced(self, router):
        stages = (ApprovalStage("chain", ("first", "second")),)
        request = submit(router, stages)
        assert request.pending_approvers() == ["first"]
        with pytest.raises(ValueError, match="cannot act"):
            router.decide(request, "second", Decision.APPROVE, at=T0)

    def test_any_rejection_fails_the_request(self, router):
        stages = (ApprovalStage("chain", ("first", "second")),)
        request = submit(router, stages)
        router.decide(request, "first", Decision.REJECT, "over budget", at=T0)
        assert request.state is RequestState.REJECTED
        assert "rejected at stage" in request.resolution_note

    def test_multiple_stages_advance_in_order(self, router):
        stages = (ApprovalStage("one", ("a",)), ApprovalStage("two", ("b",)))
        request = submit(router, stages)
        router.decide(request, "a", Decision.APPROVE, at=T0)
        assert request.state is RequestState.PENDING
        assert request.stage.name == "two"
        router.decide(request, "b", Decision.APPROVE, at=T0)
        assert request.state is RequestState.APPROVED


class TestQuorum:
    def test_reaches_quorum_without_everyone(self, router):
        stages = (ApprovalStage("q", ("a", "b", "c"), StageMode.QUORUM, quorum=2),)
        request = submit(router, stages)
        router.decide(request, "a", Decision.APPROVE, at=T0)
        assert request.state is RequestState.PENDING
        router.decide(request, "b", Decision.APPROVE, at=T0)
        assert request.state is RequestState.APPROVED

    def test_tolerates_a_dissenter(self, router):
        """The Byzantine case: one rejects, quorum still carries."""
        stages = (ApprovalStage("q", ("a", "b", "c"), StageMode.QUORUM, quorum=2),)
        request = submit(router, stages)
        router.decide(request, "a", Decision.REJECT, "disagree", at=T0)
        assert request.state is RequestState.PENDING
        router.decide(request, "b", Decision.APPROVE, at=T0)
        router.decide(request, "c", Decision.APPROVE, at=T0)
        assert request.state is RequestState.APPROVED

    def test_fails_once_quorum_is_unreachable(self, router):
        stages = (ApprovalStage("q", ("a", "b", "c"), StageMode.QUORUM, quorum=2),)
        request = submit(router, stages)
        router.decide(request, "a", Decision.REJECT, at=T0)
        router.decide(request, "b", Decision.REJECT, at=T0)
        assert request.state is RequestState.REJECTED
        assert "no longer reach quorum" in request.resolution_note

    def test_any_pending_approver_may_act(self, router):
        stages = (ApprovalStage("q", ("a", "b", "c"), StageMode.QUORUM, quorum=2),)
        request = submit(router, stages)
        assert set(request.pending_approvers()) == {"a", "b", "c"}

    def test_quorum_is_clamped_to_the_panel_size(self):
        stage = ApprovalStage("q", ("a", "b"), StageMode.QUORUM, quorum=9)
        assert stage.required_approvals() == 2


class TestTimeouts:
    def test_escalates_after_the_deadline(self, router):
        stages = (
            ApprovalStage(
                "s", ("mgr",), timeout_hours=4.0,
                on_timeout=TimeoutPolicy.ESCALATE, escalate_to="director",
            ),
        )
        request = submit(router, stages)
        router.tick(request, T0 + timedelta(hours=1))
        assert request.state is RequestState.PENDING, "must not fire early"
        router.tick(request, T0 + timedelta(hours=5))
        assert request.state is RequestState.ESCALATED
        assert request.pending_approvers() == ["director"]

    def test_escalation_target_can_approve(self, router):
        stages = (
            ApprovalStage(
                "s", ("mgr",), timeout_hours=1.0,
                on_timeout=TimeoutPolicy.ESCALATE, escalate_to="director",
            ),
        )
        request = submit(router, stages)
        router.tick(request, T0 + timedelta(hours=2))
        router.decide(request, "director", Decision.APPROVE, "signed", at=T0 + timedelta(hours=2))
        assert request.state is RequestState.APPROVED

    def test_escalation_target_can_reject(self, router):
        stages = (
            ApprovalStage(
                "s", ("mgr",), timeout_hours=1.0,
                on_timeout=TimeoutPolicy.ESCALATE, escalate_to="director",
            ),
        )
        request = submit(router, stages)
        router.tick(request, T0 + timedelta(hours=2))
        router.decide(request, "director", Decision.REJECT, "no", at=T0 + timedelta(hours=2))
        assert request.state is RequestState.REJECTED

    def test_stalled_escalation_fails_closed(self, router):
        stages = (
            ApprovalStage(
                "s", ("mgr",), timeout_hours=1.0,
                on_timeout=TimeoutPolicy.ESCALATE, escalate_to="director",
            ),
        )
        request = submit(router, stages)
        router.tick(request, T0 + timedelta(hours=2))
        router.tick(request, T0 + timedelta(hours=4))
        assert request.state is RequestState.REJECTED
        assert "also timed out" in request.resolution_note

    def test_auto_approve_policy(self, router):
        stages = (
            ApprovalStage("s", ("finance",), timeout_hours=8.0,
                          on_timeout=TimeoutPolicy.AUTO_APPROVE),
        )
        request = submit(router, stages)
        router.tick(request, T0 + timedelta(hours=9))
        assert request.state is RequestState.APPROVED
        assert "auto-approved" in request.resolution_note

    def test_reject_policy(self, router):
        stages = (
            ApprovalStage("s", ("mgr",), timeout_hours=2.0, on_timeout=TimeoutPolicy.REJECT),
        )
        request = submit(router, stages)
        router.tick(request, T0 + timedelta(hours=3))
        assert request.state is RequestState.REJECTED

    def test_deadline_is_reported(self, router):
        stages = (ApprovalStage("s", ("mgr",), timeout_hours=4.0),)
        request = submit(router, stages)
        assert request.deadline() == T0 + timedelta(hours=4)


class TestGuards:
    def test_workflow_needs_a_stage(self, router):
        with pytest.raises(ValueError):
            router.submit("r", "s", "summary", (), at=T0)

    def test_settled_request_rejects_further_decisions(self, router):
        stages = (ApprovalStage("only", ("mgr",)),)
        request = submit(router, stages)
        router.decide(request, "mgr", Decision.APPROVE, at=T0)
        with pytest.raises(ValueError, match="already approved"):
            router.decide(request, "mgr", Decision.APPROVE, at=T0)

    def test_stranger_cannot_approve(self, router):
        stages = (ApprovalStage("only", ("mgr",)),)
        request = submit(router, stages)
        with pytest.raises(ValueError):
            router.decide(request, "random-person", Decision.APPROVE, at=T0)


class TestAuditIntegration:
    def test_every_transition_is_recorded(self, router):
        stages = (ApprovalStage("only", ("mgr",)),)
        request = submit(router, stages)
        router.decide(request, "mgr", Decision.APPROVE, "looks right", at=T0)
        actions = [entry.action for entry in router.audit.entries]
        assert actions == ["approval_requested", "stage_approve", "request_approved"]
        assert router.audit.verify().valid


class TestPrebuiltWorkflows:
    def test_schedule_workflow_shape(self):
        stages = schedule_approval_workflow("mgr-1")
        assert [s.name for s in stages] == ["manager-review", "budget-check"]
        assert stages[1].on_timeout is TimeoutPolicy.AUTO_APPROVE

    def test_exception_workflow_is_quorum_based(self):
        stages = exception_approval_workflow(("lead-a", "lead-b"))
        assert stages[0].mode is StageMode.QUORUM
        assert stages[0].timeout_hours == 1.0
