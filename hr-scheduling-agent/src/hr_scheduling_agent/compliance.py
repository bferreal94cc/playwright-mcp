"""Independent compliance review of a finished schedule.

The optimizer already refuses to break a rule, so this auditor exists to catch
the cases the optimizer cannot: hand-edited schedules, exception covers applied
under time pressure, credentials that expire mid-week, and work that is legal
but distributed unfairly.

A note on fairness checking. Detecting bias requires knowing which groups to
compare, and this system deliberately does not infer protected characteristics
from names or any other proxy. Group labels are supplied explicitly by the
customer's HR team or not at all; with no labels the auditor still reports raw
distribution outliers, which is a workload-fairness signal rather than a
discrimination finding. The two are reported separately and never conflated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

from .constraints import ConstraintEngine
from .domain import Schedule, SchedulingContext, Shift


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    VIOLATION = "violation"


@dataclass(frozen=True)
class Violation:
    """A hard rule broken by a schedule as it currently stands."""

    shift_id: str
    employee_id: str
    rule: str
    severity: Severity = Severity.VIOLATION

    def describe(self) -> str:
        return f"[{self.severity.value}] {self.employee_id} on {self.shift_id}: {self.rule}"


@dataclass(frozen=True)
class ExpiringCredential:
    """A certification running out soon, or already lapsed."""

    employee_id: str
    certification: str
    expires_on: date
    days_remaining: int

    @property
    def severity(self) -> Severity:
        if self.days_remaining < 0:
            return Severity.VIOLATION
        return Severity.WARNING if self.days_remaining <= 30 else Severity.INFO

    def describe(self) -> str:
        if self.days_remaining < 0:
            return f"{self.employee_id}: {self.certification} expired {-self.days_remaining}d ago"
        return (
            f"{self.employee_id}: {self.certification} expires in "
            f"{self.days_remaining}d ({self.expires_on.isoformat()})"
        )


@dataclass(frozen=True)
class DistributionFinding:
    """An outlier in how work was shared out."""

    metric: str
    employee_id: str | None
    group: str | None
    observed: float
    reference: float
    detail: str

    def describe(self) -> str:
        return self.detail


@dataclass
class ComplianceReport:
    """Everything the admin console's compliance tab renders."""

    violations: list[Violation] = field(default_factory=list)
    expiring: list[ExpiringCredential] = field(default_factory=list)
    workload_outliers: list[DistributionFinding] = field(default_factory=list)
    group_disparities: list[DistributionFinding] = field(default_factory=list)
    uncovered_shifts: list[str] = field(default_factory=list)

    @property
    def is_compliant(self) -> bool:
        """True when nothing rises to the level of a violation."""
        lapsed = any(c.severity is Severity.VIOLATION for c in self.expiring)
        return not self.violations and not lapsed

    def summary(self) -> str:
        return (
            f"{len(self.violations)} violation(s), "
            f"{len(self.expiring)} credential alert(s), "
            f"{len(self.uncovered_shifts)} uncovered shift(s), "
            f"{len(self.group_disparities)} group disparity finding(s)"
        )


class ComplianceAuditor:
    """Re-checks a schedule that has already been built."""

    def __init__(
        self,
        context: SchedulingContext,
        credential_horizon_days: int = 30,
        workload_tolerance: float = 0.5,
        disparity_tolerance: float = 0.2,
    ) -> None:
        self.context = context
        self.engine = ConstraintEngine(context)
        self.credential_horizon_days = credential_horizon_days
        self.workload_tolerance = workload_tolerance
        self.disparity_tolerance = disparity_tolerance

    def audit(
        self,
        schedule: Schedule,
        as_of: date | None = None,
        groups: dict[str, str] | None = None,
    ) -> ComplianceReport:
        """Full review.

        ``groups`` optionally maps employee id to a customer-supplied group
        label (shift team, job family, or a protected class the customer is
        lawfully monitoring). Disparity findings are only produced when it is
        given.
        """
        as_of = as_of or date.today()
        report = ComplianceReport()
        self._check_assignments(schedule, report)
        self._check_credentials(schedule, report, as_of)
        self._check_workload(schedule, report)
        if groups:
            self._check_groups(schedule, report, groups)
        report.uncovered_shifts = [u.shift_id for u in schedule.unfilled]
        return report

    # -- individual reviews -----------------------------------------------

    def _check_assignments(self, schedule: Schedule, report: ComplianceReport) -> None:
        """Re-run every hard rule against the schedule as published."""
        shifts_by_employee: dict[str, list[Shift]] = {}
        for assignment in schedule.assignments:
            shift = self.context.shift_by_id(assignment.shift_id)
            if shift is not None:
                shifts_by_employee.setdefault(assignment.employee_id, []).append(shift)

        for assignment in schedule.assignments:
            employee = self.context.employee_by_id(assignment.employee_id)
            shift = self.context.shift_by_id(assignment.shift_id)
            if employee is None or shift is None:
                report.violations.append(
                    Violation(assignment.shift_id, assignment.employee_id, "unknown employee or shift")
                )
                continue
            # Check against the employee's *other* shifts, so the assignment
            # under review is not compared with itself.
            others = [s for s in shifts_by_employee[assignment.employee_id] if s.id != shift.id]
            result = self.engine.check(employee, shift, others)
            if not result.ok:
                report.violations.append(
                    Violation(shift.id, employee.id, result.reason, Severity.VIOLATION)
                )

    def _check_credentials(
        self, schedule: Schedule, report: ComplianceReport, as_of: date
    ) -> None:
        """Flag credentials that lapse while the schedule is still running."""
        horizon = as_of + timedelta(days=self.credential_horizon_days)
        working = {a.employee_id for a in schedule.assignments}
        for employee in self.context.employees:
            if employee.id not in working:
                continue
            for credential in employee.certifications:
                if credential.expires_on <= horizon:
                    report.expiring.append(
                        ExpiringCredential(
                            employee_id=employee.id,
                            certification=credential.name,
                            expires_on=credential.expires_on,
                            days_remaining=(credential.expires_on - as_of).days,
                        )
                    )
        report.expiring.sort(key=lambda c: (c.days_remaining, c.employee_id))

    def _check_workload(self, schedule: Schedule, report: ComplianceReport) -> None:
        """Flag anyone carrying far more or less than the roster average."""
        hours = schedule.hours_by_employee(self.context)
        scheduled = {e.id: hours.get(e.id, 0.0) for e in self.context.employees}
        if not scheduled:
            return
        average = sum(scheduled.values()) / len(scheduled)
        if average <= 0:
            return
        for employee_id, value in sorted(scheduled.items()):
            deviation = (value - average) / average
            if abs(deviation) < self.workload_tolerance:
                continue
            direction = "above" if deviation > 0 else "below"
            report.workload_outliers.append(
                DistributionFinding(
                    metric="weekly_hours",
                    employee_id=employee_id,
                    group=None,
                    observed=round(value, 2),
                    reference=round(average, 2),
                    detail=(
                        f"{employee_id} scheduled {value:.1f}h, "
                        f"{abs(deviation) * 100:.0f}% {direction} the {average:.1f}h average"
                    ),
                )
            )

        undesirable = schedule.undesirable_by_employee(self.context)
        counts = {e.id: float(undesirable.get(e.id, 0)) for e in self.context.employees}
        night_average = sum(counts.values()) / len(counts)
        if night_average <= 0:
            return
        for employee_id, value in sorted(counts.items()):
            if value <= night_average * (1 + self.workload_tolerance):
                continue
            report.workload_outliers.append(
                DistributionFinding(
                    metric="unsocial_shifts",
                    employee_id=employee_id,
                    group=None,
                    observed=value,
                    reference=round(night_average, 2),
                    detail=(
                        f"{employee_id} has {value:.0f} night/weekend shifts against a "
                        f"{night_average:.1f} average"
                    ),
                )
            )

    def _check_groups(
        self, schedule: Schedule, report: ComplianceReport, groups: dict[str, str]
    ) -> None:
        """Compare outcomes between customer-supplied groups."""
        hours = schedule.hours_by_employee(self.context)
        undesirable = schedule.undesirable_by_employee(self.context)

        for metric, values in (
            ("weekly_hours", {e.id: hours.get(e.id, 0.0) for e in self.context.employees}),
            (
                "unsocial_shifts",
                {e.id: float(undesirable.get(e.id, 0)) for e in self.context.employees},
            ),
        ):
            per_group: dict[str, list[float]] = {}
            for employee_id, value in values.items():
                label = groups.get(employee_id)
                if label is not None:
                    per_group.setdefault(label, []).append(value)
            means = {
                label: sum(vals) / len(vals) for label, vals in per_group.items() if vals
            }
            if len(means) < 2:
                continue
            overall = sum(means.values()) / len(means)
            if overall <= 0:
                continue
            for label, value in sorted(means.items()):
                deviation = (value - overall) / overall
                if abs(deviation) < self.disparity_tolerance:
                    continue
                direction = "more" if deviation > 0 else "less"
                report.group_disparities.append(
                    DistributionFinding(
                        metric=metric,
                        employee_id=None,
                        group=label,
                        observed=round(value, 2),
                        reference=round(overall, 2),
                        detail=(
                            f"group '{label}' averages {value:.1f} {metric.replace('_', ' ')}, "
                            f"{abs(deviation) * 100:.0f}% {direction} than the cross-group mean "
                            f"of {overall:.1f} -- review before publishing"
                        ),
                    )
                )
