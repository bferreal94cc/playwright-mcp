"""Real-time exception handling: no-shows, call-outs and availability changes.

This is the capability the competitive research identified as the market gap.
When a shift breaks at 09:00, the manager needs options by 09:15, not a
rebuilt week.

The handler therefore re-solves only the damage: every untouched assignment is
pinned, and the vacated shift is re-filled from whoever is still legally
available. It returns *ranked options with reasons* rather than a single
answer, because the manager -- not the agent -- owns the call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .constraints import ConstraintEngine
from .domain import Assignment, Schedule, SchedulingContext, Shift
from .scoring import ObjectiveWeights, candidate_score, marginal_cost


class ExceptionType(str, Enum):
    NO_SHOW = "no_show"
    CALL_OUT = "call_out"
    AVAILABILITY_CHANGE = "availability_change"


@dataclass(frozen=True)
class ExceptionEvent:
    """Something broke on a published schedule."""

    type: ExceptionType
    shift_id: str
    employee_id: str
    reported_at: datetime
    note: str = ""


@dataclass
class ResolutionOption:
    """One way to cover the vacated shift, with its full cost of doing so."""

    employee_id: str
    employee_name: str
    cost_delta: float
    score: float
    rationale: str
    incurs_overtime: bool = False

    def describe(self) -> str:
        direction = "＋" if self.cost_delta >= 0 else "－"
        overtime = " (overtime)" if self.incurs_overtime else ""
        return (
            f"{self.employee_name}: {direction}${abs(self.cost_delta):.2f}{overtime} -- "
            f"{self.rationale}"
        )


@dataclass
class ExceptionResolution:
    """The handler's answer: ranked options, plus why anyone was excluded."""

    event: ExceptionEvent
    options: list[ResolutionOption] = field(default_factory=list)
    blockers: dict[str, str] = field(default_factory=dict)
    resolved_in_ms: float = 0.0

    @property
    def recommended(self) -> ResolutionOption | None:
        return self.options[0] if self.options else None

    @property
    def is_coverable(self) -> bool:
        return bool(self.options)

    def escalation_summary(self) -> str:
        """What to tell a manager when nobody can legally take the shift."""
        if self.is_coverable:
            return ""
        if not self.blockers:
            return "No one on the roster holds this role."
        counts: dict[str, int] = {}
        for reason in self.blockers.values():
            counts[reason] = counts.get(reason, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        detail = "; ".join(f"{reason} ({count})" for reason, count in ranked)
        return f"No legal coverage available. Blockers: {detail}"


@dataclass
class _Loads:
    """Per-employee load derived from a set of assignments."""

    shifts: dict[str, list[Shift]]
    hours: dict[str, float]
    undesirable: dict[str, int]


class ExceptionHandler:
    """Re-solves a single broken shift against a live schedule."""

    def __init__(
        self,
        context: SchedulingContext,
        weights: ObjectiveWeights | None = None,
        max_options: int = 3,
    ) -> None:
        self.context = context
        self.weights = weights or ObjectiveWeights.balanced()
        self.max_options = max_options
        self.engine = ConstraintEngine(context)

    def resolve(self, schedule: Schedule, event: ExceptionEvent) -> ExceptionResolution:
        """Rank replacements for the shift ``event`` broke.

        The schedule is not modified -- call :meth:`apply` once the manager (or
        an auto-approval policy) picks an option.
        """
        started = datetime.now()
        shift = self.context.shift_by_id(event.shift_id)
        if shift is None:
            raise ValueError(f"unknown shift '{event.shift_id}'")

        vacated = schedule.assignment_for_shift(event.shift_id)
        remaining = [a for a in schedule.assignments if a.shift_id != event.shift_id]
        loads = self._loads(remaining)

        # The person who called out is excluded regardless of what the
        # constraint engine thinks -- they have already said they cannot come.
        excluded = {event.employee_id}

        options: list[ResolutionOption] = []
        blockers: dict[str, str] = {}
        average_hours = self._average(list(loads.hours.values()))
        average_undesirable = self._average([float(v) for v in loads.undesirable.values()])
        baseline_cost = vacated.cost if vacated else 0.0

        for employee in self.context.employees:
            if employee.id in excluded:
                blockers[employee.id] = "reported unavailable for this shift"
                continue
            feasibility = self.engine.check(employee, shift, loads.shifts.get(employee.id, []))
            if not feasibility.ok:
                blockers[employee.id] = feasibility.reason
                continue

            hours_so_far = loads.hours.get(employee.id, 0.0)
            cost = marginal_cost(self.context, employee, shift, hours_so_far)
            score = candidate_score(
                self.context,
                employee,
                shift,
                hours_so_far,
                loads.undesirable.get(employee.id, 0),
                average_hours,
                average_undesirable,
                self.weights,
            )
            threshold = self.context.policy.overtime_threshold_hours
            options.append(
                ResolutionOption(
                    employee_id=employee.id,
                    employee_name=employee.name,
                    cost_delta=cost - baseline_cost,
                    score=score,
                    rationale=self._rationale(employee, shift, hours_so_far, average_hours),
                    incurs_overtime=hours_so_far + shift.hours > threshold,
                )
            )

        options.sort(key=lambda option: (-option.score, option.cost_delta, option.employee_id))
        elapsed = (datetime.now() - started).total_seconds() * 1000
        return ExceptionResolution(
            event=event,
            options=options[: self.max_options],
            blockers=blockers,
            resolved_in_ms=elapsed,
        )

    def apply(
        self, schedule: Schedule, event: ExceptionEvent, option: ResolutionOption
    ) -> Schedule:
        """Commit a chosen option, replacing the vacated assignment in place."""
        shift = self.context.shift_by_id(event.shift_id)
        if shift is None:
            raise ValueError(f"unknown shift '{event.shift_id}'")

        schedule.assignments = [a for a in schedule.assignments if a.shift_id != event.shift_id]
        loads = self._loads(schedule.assignments)
        cost = marginal_cost(
            self.context,
            self._employee(option.employee_id),
            shift,
            loads.hours.get(option.employee_id, 0.0),
        )
        schedule.assignments.append(
            Assignment(
                shift_id=shift.id,
                employee_id=option.employee_id,
                cost=cost,
                rationale=f"exception cover ({event.type.value}): {option.rationale}",
            )
        )
        schedule.unfilled = [u for u in schedule.unfilled if u.shift_id != shift.id]
        return schedule

    # -- helpers ----------------------------------------------------------

    def _loads(self, assignments: list[Assignment]) -> "_Loads":
        shifts: dict[str, list[Shift]] = {e.id: [] for e in self.context.employees}
        hours = {e.id: self.context.prior_hours.get(e.id, 0.0) for e in self.context.employees}
        undesirable = {
            e.id: self.context.prior_undesirable_counts.get(e.id, 0) for e in self.context.employees
        }
        for assignment in assignments:
            shift = self.context.shift_by_id(assignment.shift_id)
            if shift is None:
                continue
            shifts.setdefault(assignment.employee_id, []).append(shift)
            hours[assignment.employee_id] = hours.get(assignment.employee_id, 0.0) + shift.hours
            if shift.is_undesirable:
                undesirable[assignment.employee_id] = (
                    undesirable.get(assignment.employee_id, 0) + 1
                )
        return _Loads(shifts=shifts, hours=hours, undesirable=undesirable)

    def _employee(self, employee_id: str):
        employee = self.context.employee_by_id(employee_id)
        if employee is None:
            raise ValueError(f"unknown employee '{employee_id}'")
        return employee

    @staticmethod
    def _average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _rationale(self, employee, shift: Shift, hours_so_far: float, average: float) -> str:
        parts = []
        if employee.preferred_periods and employee.prefers(shift.period):
            parts.append(f"prefers {shift.period.value} shifts")
        if hours_so_far < average:
            parts.append(f"{average - hours_so_far:.1f}h below roster average")
        else:
            parts.append(f"already at {hours_so_far:.1f}h this week")
        if shift.required_certifications:
            parts.append("certifications current")
        return "; ".join(parts)
