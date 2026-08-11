"""Command line demo: both launch verticals, end to end.

Runs the full loop the Figma prototype describes -- build, audit, approve,
publish, break a shift, cover it, then verify the audit chain -- against real
engines rather than mocks.

    python -m hr_scheduling_agent.cli --scenario both
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

from .approvals import Decision
from .hiring import Candidate, InterviewSlot
from .domain import TimeWindow
from .exception_handler import ExceptionEvent, ExceptionType
from .orchestrator import Orchestrator
from .providers import provider_from_env
from .scenarios import build_healthcare_context, build_retail_context
from .scoring import ObjectiveWeights

RULE = "=" * 78


def _heading(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def _step(label: str) -> None:
    print(f"\n-- {label}")


def run_scenario(name: str, profit_weight: float, days: int, groups_demo: bool) -> bool:
    """Run one vertical end to end. Returns whether everything came out clean."""
    context = (
        build_retail_context(days=days)
        if name == "retail"
        else build_healthcare_context(days=days)
    )
    _heading(
        f"{name.upper()}  --  {len(context.shifts)} shifts, {len(context.employees)} staff, "
        f"profit/wellbeing dial at {profit_weight:.2f}"
    )

    orchestrator = Orchestrator(
        context,
        weights=ObjectiveWeights.balanced(profit_weight),
        provider=provider_from_env(),
    )

    # 1. Build --------------------------------------------------------------
    _step("Schedule Optimizer")
    built = orchestrator.build_schedule(requested_by="manager-01")
    print(f"   {built.summary}")
    metrics = built.data["metrics"]
    print(
        f"   fairness: hours {metrics.hours_fairness:.2f}, "
        f"unsocial-shift spread {metrics.load_fairness:.2f}, "
        f"minimum-hours met {metrics.min_hours_met:.0%}"
    )
    for gap in built.data["schedule"].unfilled[:3]:
        print(f"   UNFILLED {gap.shift_id}: {gap.summary()}")

    # 2. Compliance ---------------------------------------------------------
    _step("Compliance Auditor")
    groups = None
    if groups_demo:
        # Group labels come from the customer, never inferred. Here we split
        # the roster by seniority to demonstrate the disparity report.
        groups = {
            e.id: ("tenured" if e.seniority_months >= 24 else "newer")
            for e in context.employees
        }
    audited = orchestrator.audit_schedule(groups=groups)
    print(f"   {audited.summary}")
    report = audited.data["report"]
    for violation in report.violations[:3]:
        print(f"   {violation.describe()}")
    for credential in report.expiring[:3]:
        print(f"   CREDENTIAL {credential.describe()}")
    for finding in report.workload_outliers[:2]:
        print(f"   WORKLOAD {finding.describe()}")
    for finding in report.group_disparities[:2]:
        print(f"   DISPARITY {finding.describe()}")

    # 3. Approval -----------------------------------------------------------
    _step("Approval Router")
    t0 = datetime(2026, 8, 14, 9, 0)
    submitted = orchestrator.submit_for_approval(manager="manager-01", at=t0)
    request = submitted.data["request"]
    print(f"   {submitted.summary}")
    orchestrator.approvals.decide(
        request, "manager-01", Decision.APPROVE, "coverage and cost look right", at=t0
    )
    print(f"   after manager sign-off: {request.state.value}, stage '{request.stage.name}'")
    orchestrator.approvals.tick(request, t0 + timedelta(hours=9))
    print(f"   after budget deadline: {request.state.value} -- {request.resolution_note}")

    # 4. Publish ------------------------------------------------------------
    _step("Legacy Bridge + Notification Engine")
    published = orchestrator.publish()
    print(f"   {published.summary}")

    # 5. Live exception -----------------------------------------------------
    _step("Exception Handler (live call-out)")
    schedule = orchestrator.schedule
    assert schedule is not None
    if not schedule.assignments:
        print("   nothing was scheduled, so there is no shift to break")
    else:
        victim = schedule.assignments[len(schedule.assignments) // 2]
        event = ExceptionEvent(
            type=ExceptionType.CALL_OUT,
            shift_id=victim.shift_id,
            employee_id=victim.employee_id,
            reported_at=datetime.now(),
            note="woke up unwell",
        )
        print(f"   {victim.employee_id} calls out of {victim.shift_id}")
        resolved = orchestrator.resolve_exception(event)
        resolution = resolved.data["resolution"]
        print(f"   triaged in {resolution.resolved_in_ms:.1f}ms")
        for option in resolution.options:
            print(f"     option: {option.describe()}")
        if resolved.ok:
            print(f"   {orchestrator.apply_cover(event, resolution).summary}")
        else:
            print(f"   ESCALATED: {resolved.summary}")

    # 6. Natural language ---------------------------------------------------
    _step("NLU + routing")
    speaker = context.employees[0]
    reference = min(s.window.start.date() for s in context.shifts)
    for message in (
        "when am I working this week?",
        f"we need to hire two more {context.shifts[0].role}s",
    ):
        result = orchestrator.handle(message, actor_id=speaker.id, reference=reference)
        print(f'   "{message}" -> {result.agent}: {result.summary}')

    # 7. Hiring -------------------------------------------------------------
    _step("Interview Agent")
    role = context.shifts[0].role
    required = context.shifts[0].required_certifications
    candidates = [
        Candidate("c1", "Robin", 4.0, frozenset({role}), required, frozenset({"s1", "s2"})),
        Candidate("c2", "Sam", 1.0, frozenset({role}), frozenset(), frozenset({"s1"})),
        Candidate("c3", "Wren", 7.0, frozenset({role}), required, frozenset({"s2", "s3"})),
    ]
    scored = orchestrator.interviews.rank(
        [
            orchestrator.interviews.score_screening(
                c, frozenset({role}), required, minimum_years=2.0
            )
            for c in candidates
        ]
    )
    for result in scored:
        print(f"   {result.describe()}")
    base = datetime.combine(reference, datetime.min.time()).replace(hour=10)
    slots = [
        InterviewSlot(f"s{i}", "manager-01", TimeWindow(base + timedelta(hours=i), base + timedelta(hours=i, minutes=45)))
        for i in range(1, 4)
    ]
    plan = orchestrator.interviews.schedule_interviews(candidates, slots)
    for booking in plan.bookings:
        print(f"   booked {booking.candidate_id} into {booking.slot_id} with {booking.interviewer}")
    if plan.unscheduled:
        print(f"   could not place: {', '.join(plan.unscheduled)}")

    # 8. Audit --------------------------------------------------------------
    _step("Audit trail")
    verification = orchestrator.audit.verify()
    print(f"   {len(orchestrator.audit.entries)} entries, chain valid: {verification.valid}")
    for entry in orchestrator.audit.entries[:4]:
        print(f"     {entry.sequence:>2}. {entry.actor}/{entry.action}: {entry.reasoning[:64]}")

    clean = built.ok and audited.ok and verification.valid
    print(f"\n   scenario result: {'CLEAN' if clean else 'ATTENTION NEEDED'}")
    return clean


def _dial(value: str) -> float:
    """Parse the profit/wellbeing dial, rejecting anything outside [0, 1]."""
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number")
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0.0 and 1.0, got {parsed:g}")
    return parsed


def _positive_days(value: str) -> int:
    """Parse a day count. Zero days would leave nothing to schedule."""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number")
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1 day, got {parsed}")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HR scheduling agent demo")
    parser.add_argument(
        "--scenario", choices=("retail", "healthcare", "both"), default="both"
    )
    parser.add_argument(
        "--profit-weight",
        type=_dial,
        default=0.5,
        help="0.0 optimises purely for employee wellbeing, 1.0 purely for labour cost",
    )
    parser.add_argument(
        "--days", type=_positive_days, default=7, help="days of demand to schedule"
    )
    parser.add_argument(
        "--groups", action="store_true", help="demonstrate the group disparity report"
    )
    args = parser.parse_args(argv)

    names = ("retail", "healthcare") if args.scenario == "both" else (args.scenario,)
    results = [run_scenario(n, args.profit_weight, args.days, args.groups) for n in names]

    _heading("SUMMARY")
    for name, clean in zip(names, results):
        print(f"   {name:<12} {'clean' if clean else 'attention needed'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
