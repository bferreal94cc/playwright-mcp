"""Hard constraint checking.

Every rule in here is a veto: if it fails, the assignment is illegal and the
optimizer may not make it at any price. Soft preferences live in
:mod:`hr_scheduling_agent.scoring`.

Each check returns a short human-readable reason on failure. Those reasons are
what surfaces in the manager UI ("why can't Dana take this shift?") and in the
compliance audit log, so they are written to be read by people.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .domain import Employee, SchedulingContext, Shift, week_of


@dataclass(frozen=True)
class Feasibility:
    """Outcome of checking one employee against one shift."""

    ok: bool
    reason: str = ""

    @classmethod
    def allowed(cls) -> "Feasibility":
        return cls(True)

    @classmethod
    def blocked(cls, reason: str) -> "Feasibility":
        return cls(False, reason)


class ConstraintEngine:
    """Evaluates the compliance envelope for a single scheduling context."""

    def __init__(self, context: SchedulingContext) -> None:
        self.context = context

    # -- individual rules -------------------------------------------------

    def _check_role(self, employee: Employee, shift: Shift) -> Feasibility:
        if not employee.has_role(shift.role):
            return Feasibility.blocked(f"not qualified for role '{shift.role}'")
        return Feasibility.allowed()

    def _check_certifications(self, employee: Employee, shift: Shift) -> Feasibility:
        day = shift.window.start.date()
        for required in sorted(shift.required_certifications):
            if not employee.certification_valid(required, day):
                return Feasibility.blocked(f"certification '{required}' missing or expired")
        return Feasibility.allowed()

    def _check_shift_length(self, shift: Shift) -> Feasibility:
        limit = self.context.policy.max_shift_hours
        if shift.hours > limit:
            return Feasibility.blocked(f"shift exceeds the {limit:g}h maximum shift length")
        return Feasibility.allowed()

    def _check_availability(self, employee: Employee, shift: Shift) -> Feasibility:
        if not employee.is_available_for(shift.window):
            return Feasibility.blocked("outside declared availability")
        return Feasibility.allowed()

    def _check_overlap(self, shift: Shift, assigned: list[Shift]) -> Feasibility:
        for other in assigned:
            if other.window.overlaps(shift.window):
                return Feasibility.blocked("already assigned to an overlapping shift")
        return Feasibility.allowed()

    def _check_rest(self, shift: Shift, assigned: list[Shift]) -> Feasibility:
        required = self.context.policy.min_rest_hours_between_shifts
        for other in assigned:
            gap = other.window.gap_hours_to(shift.window)
            if 0 <= gap < required:
                return Feasibility.blocked(f"less than {required:g}h rest after an adjacent shift")
        return Feasibility.allowed()

    def _check_weekly_hours(
        self, employee: Employee, shift: Shift, assigned: list[Shift]
    ) -> Feasibility:
        cap = self.context.max_hours_for(employee, shift.location_id)
        target_week = week_of(shift.window.start.date())
        booked = sum(
            other.hours for other in assigned if week_of(other.window.start.date()) == target_week
        )
        booked += self.context.prior_hours.get(employee.id, 0.0)
        if booked + shift.hours > cap:
            return Feasibility.blocked(f"would exceed the {cap:g}h weekly cap")
        return Feasibility.allowed()

    def _check_consecutive_days(self, shift: Shift, assigned: list[Shift]) -> Feasibility:
        limit = self.context.policy.max_consecutive_days
        worked: set[date] = {other.window.start.date() for other in assigned}
        worked.add(shift.window.start.date())
        run = self._longest_run(worked)
        if run > limit:
            return Feasibility.blocked(f"would create a {run}-day run past the {limit}-day limit")
        return Feasibility.allowed()

    @staticmethod
    def _longest_run(days: set[date]) -> int:
        if not days:
            return 0
        ordered = sorted(days)
        longest = run = 1
        for previous, current in zip(ordered, ordered[1:]):
            run = run + 1 if current - previous == timedelta(days=1) else 1
            longest = max(longest, run)
        return longest

    # -- composite --------------------------------------------------------

    def check(self, employee: Employee, shift: Shift, assigned: list[Shift]) -> Feasibility:
        """Check every hard rule, cheapest and most-explanatory first.

        ``assigned`` is the set of shifts the employee already holds in the
        schedule under construction.
        """
        checks = (
            self._check_role(employee, shift),
            self._check_certifications(employee, shift),
            self._check_shift_length(shift),
            self._check_availability(employee, shift),
            self._check_overlap(shift, assigned),
            self._check_rest(shift, assigned),
            self._check_weekly_hours(employee, shift, assigned),
            self._check_consecutive_days(shift, assigned),
        )
        for result in checks:
            if not result.ok:
                return result
        return Feasibility.allowed()

    def eligible_employees(
        self, shift: Shift, assigned_by_employee: dict[str, list[Shift]]
    ) -> tuple[list[Employee], dict[str, str]]:
        """Split the roster into those who may take ``shift`` and those who may not.

        Returns ``(eligible, blockers)`` where ``blockers`` maps employee id to
        the reason they were excluded -- the data behind an unfilled-shift
        explanation.
        """
        eligible: list[Employee] = []
        blockers: dict[str, str] = {}
        for employee in self.context.employees:
            result = self.check(employee, shift, assigned_by_employee.get(employee.id, []))
            if result.ok:
                eligible.append(employee)
            else:
                blockers[employee.id] = result.reason
        return eligible, blockers
