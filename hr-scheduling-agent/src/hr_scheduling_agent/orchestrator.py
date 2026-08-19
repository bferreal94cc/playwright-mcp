"""The orchestrator: the queen agent that routes work to the specialists.

Callers talk to this class, not to the engines underneath. It owns the live
schedule, decides which specialist handles an incoming request, and makes sure
every consequential action reaches the audit log.

The eight agents from the architecture design map onto this module as follows:

===========================  ==================================================
Orchestrator (queen)         :class:`Orchestrator`
Schedule Optimizer           :class:`~hr_scheduling_agent.optimizer.ScheduleOptimizer`
Exception Handler            :class:`~hr_scheduling_agent.exception_handler.ExceptionHandler`
Approval Router              :class:`~hr_scheduling_agent.approvals.ApprovalRouter`
Interview Agent              :class:`~hr_scheduling_agent.hiring.InterviewAgent`
Compliance Auditor           :class:`~hr_scheduling_agent.compliance.ComplianceAuditor`
Notification Engine          :class:`~hr_scheduling_agent.integrations.NotificationEngine`
Legacy Bridge                :class:`~hr_scheduling_agent.integrations.HRISConnector`
===========================  ==================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .approvals import ApprovalRequest, ApprovalRouter, schedule_approval_workflow
from .audit import AuditLog
from .compliance import ComplianceAuditor, ComplianceReport
from .domain import Schedule, SchedulingContext
from .exception_handler import ExceptionEvent, ExceptionHandler, ExceptionResolution, ExceptionType
from .hiring import InterviewAgent
from .integrations import (
    CalendarConnector,
    HRISConnector,
    InMemoryCalendar,
    InMemoryHRIS,
    NotificationEngine,
    Urgency,
)
from .nlu import Intent, NLUPipeline, ParsedRequest, parser_for_context
from .optimizer import ScheduleOptimizer
from .providers import LLMProvider
from .scoring import ObjectiveWeights, compute_metrics


@dataclass
class AgentResult:
    """Uniform envelope for whatever a specialist did."""

    agent: str
    action: str
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        marker = "ok" if self.ok else "!!"
        return f"[{marker}] {self.agent}/{self.action}: {self.summary}"


class Orchestrator:
    """Routes requests to the right specialist and holds the live schedule."""

    def __init__(
        self,
        context: SchedulingContext,
        weights: ObjectiveWeights | None = None,
        provider: LLMProvider | None = None,
        audit_log: AuditLog | None = None,
        hris: HRISConnector | None = None,
        calendar: CalendarConnector | None = None,
        notifications: NotificationEngine | None = None,
    ) -> None:
        self.context = context
        self.weights = weights or ObjectiveWeights.balanced()
        self.audit = audit_log or AuditLog()

        self.optimizer = ScheduleOptimizer(context, self.weights)
        self.exceptions = ExceptionHandler(context, self.weights)
        self.approvals = ApprovalRouter(self.audit)
        self.auditor = ComplianceAuditor(context)
        self.interviews = InterviewAgent(provider)
        self.nlu = NLUPipeline(parser_for_context(context), provider)
        self.hris = hris or InMemoryHRIS(context.employees)
        self.calendar = calendar or InMemoryCalendar()
        self.notifications = notifications or NotificationEngine.in_memory()

        self.schedule: Schedule | None = None
        self.pending_approval: ApprovalRequest | None = None

    # -- natural language entry point -------------------------------------

    def handle(
        self, text: str, actor_id: str = "unknown", reference: date | None = None
    ) -> AgentResult:
        """Interpret a message and dispatch it to the responsible agent."""
        parsed = self.nlu.understand(text, reference=reference)
        self.audit.record(
            actor=actor_id,
            action="request_received",
            subject=parsed.intent.value,
            reasoning=text,
            metadata={"confidence": parsed.confidence, "source": parsed.source},
        )

        if parsed.intent is Intent.CREATE_SCHEDULE:
            return self.build_schedule(requested_by=actor_id)
        if parsed.intent is Intent.REPORT_ABSENCE:
            return self.report_absence(actor_id, parsed, reference)
        if parsed.intent is Intent.SWAP_SHIFT:
            return self.report_absence(actor_id, parsed, reference, ExceptionType.AVAILABILITY_CHANGE)
        if parsed.intent is Intent.QUERY_SCHEDULE:
            return self.describe_schedule_for(actor_id)
        if parsed.intent is Intent.POST_JOB:
            return self.post_job(parsed)
        if parsed.intent is Intent.REQUEST_TIME_OFF:
            return AgentResult(
                agent="orchestrator",
                action="time_off_request",
                ok=True,
                summary="Time-off request logged for manager review.",
                data={"dates": parsed.entities.get("dates", []), "employee_id": actor_id},
            )
        return AgentResult(
            agent="orchestrator",
            action="clarify",
            ok=False,
            summary="Could not tell what was being asked. Ask the user to rephrase.",
            data={"confidence": parsed.confidence, "raw_text": text},
        )

    # -- scheduling --------------------------------------------------------

    def build_schedule(self, requested_by: str = "manager") -> AgentResult:
        """Optimize a schedule and audit the result."""
        schedule = self.optimizer.solve()
        self.schedule = schedule
        metrics = compute_metrics(self.context, schedule)
        self.audit.record(
            actor="schedule_optimizer",
            action="schedule_generated",
            subject="schedule",
            reasoning=(
                f"filled {len(schedule.assignments)}/{len(self.context.shifts)} shifts "
                f"at ${schedule.total_cost:,.2f}"
            ),
            metadata={"requested_by": requested_by, **metrics.as_dict()},
        )
        return AgentResult(
            agent="schedule_optimizer",
            action="build_schedule",
            ok=not schedule.unfilled,
            summary=(
                f"Filled {len(schedule.assignments)}/{len(self.context.shifts)} shifts, "
                f"${schedule.total_cost:,.2f} labour, "
                f"{metrics.preference_rate:.0%} preference match."
            ),
            data={"schedule": schedule, "metrics": metrics},
        )

    def audit_schedule(self, groups: dict[str, str] | None = None) -> AgentResult:
        """Run the compliance auditor over the live schedule."""
        if self.schedule is None:
            return self._no_schedule("audit_schedule")
        as_of = min((s.window.start.date() for s in self.context.shifts), default=date.today())
        report: ComplianceReport = self.auditor.audit(self.schedule, as_of=as_of, groups=groups)
        self.audit.record(
            actor="compliance_auditor",
            action="compliance_reviewed",
            subject="schedule",
            reasoning=report.summary(),
            metadata={"compliant": report.is_compliant},
        )
        return AgentResult(
            agent="compliance_auditor",
            action="audit_schedule",
            ok=report.is_compliant,
            summary=report.summary(),
            data={"report": report},
        )

    def submit_for_approval(
        self, manager: str = "manager-01", at: datetime | None = None
    ) -> AgentResult:
        """Send the live schedule into the approval workflow."""
        if self.schedule is None:
            return self._no_schedule("submit_for_approval")
        request = self.approvals.submit(
            request_id=f"sched-{len(self.audit.entries)}",
            subject="schedule",
            summary=(
                f"{len(self.schedule.assignments)} shifts, "
                f"${self.schedule.total_cost:,.2f} labour cost"
            ),
            stages=schedule_approval_workflow(manager),
            at=at,
        )
        self.pending_approval = request
        return AgentResult(
            agent="approval_router",
            action="submit_for_approval",
            ok=True,
            summary=f"Awaiting {', '.join(request.pending_approvers())}.",
            data={"request": request},
        )

    # -- exceptions --------------------------------------------------------

    def report_absence(
        self,
        employee_id: str,
        parsed: ParsedRequest,
        reference: date | None = None,
        kind: ExceptionType = ExceptionType.CALL_OUT,
    ) -> AgentResult:
        """Find the affected shift from the message and re-solve it."""
        if self.schedule is None:
            return self._no_schedule("report_absence")
        target_dates = [date.fromisoformat(d) for d in parsed.entities.get("dates", [])]
        shift_id = self._find_shift(employee_id, target_dates or [reference or date.today()])
        if shift_id is None:
            return AgentResult(
                agent="exception_handler",
                action="report_absence",
                ok=False,
                summary=f"No scheduled shift found for {employee_id} on the stated date.",
                data={"dates": parsed.entities.get("dates", [])},
            )
        event = ExceptionEvent(
            type=kind,
            shift_id=shift_id,
            employee_id=employee_id,
            reported_at=datetime.now(),
            note=parsed.raw_text,
        )
        return self.resolve_exception(event)

    def resolve_exception(self, event: ExceptionEvent) -> AgentResult:
        """Rank cover options for a broken shift and audit the outcome."""
        if self.schedule is None:
            return self._no_schedule("resolve_exception")
        resolution: ExceptionResolution = self.exceptions.resolve(self.schedule, event)
        self.audit.record(
            actor="exception_handler",
            action="exception_triaged",
            subject=event.shift_id,
            reasoning=(
                f"{event.type.value} by {event.employee_id}; "
                f"{len(resolution.options)} cover option(s) in {resolution.resolved_in_ms:.0f}ms"
            ),
            metadata={"coverable": resolution.is_coverable},
        )
        if not resolution.is_coverable:
            return AgentResult(
                agent="exception_handler",
                action="resolve_exception",
                ok=False,
                summary=resolution.escalation_summary(),
                data={"resolution": resolution},
            )
        return AgentResult(
            agent="exception_handler",
            action="resolve_exception",
            ok=True,
            summary=f"Recommend {resolution.recommended.describe()}",
            data={"resolution": resolution},
        )

    def apply_cover(self, event: ExceptionEvent, resolution: ExceptionResolution) -> AgentResult:
        """Commit the recommended cover and notify both people involved."""
        if self.schedule is None:
            return self._no_schedule("apply_cover")
        option = resolution.recommended
        if option is None:
            return AgentResult(
                agent="exception_handler", action="apply_cover", ok=False,
                summary="No cover option to apply.", data={},
            )
        self.exceptions.apply(self.schedule, event, option)
        self.audit.record(
            actor="exception_handler",
            action="cover_applied",
            subject=event.shift_id,
            reasoning=f"{option.employee_id} covers for {event.employee_id}: {option.rationale}",
            metadata={"cost_delta": round(option.cost_delta, 2)},
        )
        shift = self.context.shift_by_id(event.shift_id)
        when = f"{shift.window.start:%a %d %b %H:%M}" if shift else event.shift_id
        self.notifications.notify(
            option.employee_id,
            "Shift cover request confirmed",
            f"You are now covering {when}.",
            urgency=Urgency.URGENT,
        )
        self.notifications.notify(
            event.employee_id,
            "Absence recorded",
            f"Your {when} shift has been covered. No further action needed.",
            urgency=Urgency.IMPORTANT,
        )
        return AgentResult(
            agent="exception_handler",
            action="apply_cover",
            ok=True,
            summary=f"{option.employee_name} now covers {event.shift_id}.",
            data={"option": option},
        )

    # -- hiring ------------------------------------------------------------

    def post_job(self, parsed: ParsedRequest) -> AgentResult:
        """Draft a posting from whatever the manager said."""
        role = str(parsed.entities.get("role") or "team member")
        headcount = int(parsed.entities.get("headcount") or 1)
        location = next(iter({s.location_id for s in self.context.shifts}), "unspecified")
        posting = self.interviews.draft_posting(role, location, headcount)
        self.audit.record(
            actor="interview_agent",
            action="posting_drafted",
            subject=role,
            reasoning=f"{headcount} opening(s) at {location}",
            metadata={"source": posting.source},
        )
        return AgentResult(
            agent="interview_agent",
            action="post_job",
            ok=True,
            summary=f"Drafted posting: {posting.title}",
            data={"posting": posting},
        )

    # -- publication -------------------------------------------------------

    def publish(self) -> AgentResult:
        """Push the approved schedule to HRIS, calendars and everyone's phone."""
        if self.schedule is None:
            return self._no_schedule("publish")
        written = self.hris.push_schedule(self.schedule, self.context)
        events = 0
        for assignment in self.schedule.assignments:
            employee = self.context.employee_by_id(assignment.employee_id)
            shift = self.context.shift_by_id(assignment.shift_id)
            if employee is not None and shift is not None:
                self.calendar.publish(employee, shift)
                events += 1
        reached = self.notifications.broadcast_schedule(self.schedule, self.context)
        self.audit.record(
            actor="legacy_bridge",
            action="schedule_published",
            subject="schedule",
            reasoning=f"{written} HRIS records, {events} calendar events, {reached} people notified",
            metadata={"hris": self.hris.system_name},
        )
        return AgentResult(
            agent="legacy_bridge",
            action="publish",
            ok=True,
            summary=f"Published {written} assignments; notified {reached} staff.",
            data={"hris_records": written, "calendar_events": events, "notified": reached},
        )

    def describe_schedule_for(self, employee_id: str) -> AgentResult:
        """Answer 'when am I working?'."""
        if self.schedule is None:
            return self._no_schedule("describe_schedule_for")
        lines = []
        for assignment in self.schedule.assignments_for(employee_id):
            shift = self.context.shift_by_id(assignment.shift_id)
            if shift is not None:
                lines.append(
                    f"{shift.window.start:%a %d %b %H:%M}-{shift.window.end:%H:%M} "
                    f"{shift.role} @ {shift.location_id}"
                )
        lines.sort()
        return AgentResult(
            agent="orchestrator",
            action="describe_schedule",
            ok=True,
            summary=f"{len(lines)} shift(s) scheduled.",
            data={"shifts": lines},
        )

    # -- helpers -----------------------------------------------------------

    def _find_shift(self, employee_id: str, days: list[date]) -> str | None:
        if self.schedule is None:
            return None
        wanted = set(days)
        for assignment in self.schedule.assignments_for(employee_id):
            shift = self.context.shift_by_id(assignment.shift_id)
            if shift is not None and shift.window.start.date() in wanted:
                return shift.id
        return None

    @staticmethod
    def _no_schedule(action: str) -> AgentResult:
        return AgentResult(
            agent="orchestrator",
            action=action,
            ok=False,
            summary="No schedule has been built yet. Build one first.",
        )
