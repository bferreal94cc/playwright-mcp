"""Every hard rule must veto, and must say why in words a manager can read."""

from __future__ import annotations

from datetime import date, time

from conftest import MONDAY, make_context, make_employee, make_shift
from hr_scheduling_agent.constraints import ConstraintEngine
from hr_scheduling_agent.domain import AvailabilityWindow, Certification, CompanyPolicy


def engine_for(employees, shifts, policy=None) -> ConstraintEngine:
    return ConstraintEngine(make_context(employees, shifts, policy))


class TestHardRules:
    def test_role_mismatch_blocks(self):
        employee = make_employee("e1", roles={"cook"})
        shift = make_shift("s1", role="server")
        result = engine_for([employee], [shift]).check(employee, shift, [])
        assert not result.ok
        assert "not qualified" in result.reason

    def test_missing_certification_blocks(self):
        employee = make_employee("e1")
        shift = make_shift("s1", certifications={"BLS"})
        result = engine_for([employee], [shift]).check(employee, shift, [])
        assert not result.ok
        assert "BLS" in result.reason

    def test_expired_certification_blocks(self):
        employee = make_employee(
            "e1", certifications=(Certification("BLS", date(2026, 8, 1)),)
        )
        shift = make_shift("s1", certifications={"BLS"})  # runs 2026-08-17
        result = engine_for([employee], [shift]).check(employee, shift, [])
        assert not result.ok
        assert "expired" in result.reason

    def test_shift_longer_than_policy_blocks(self):
        employee = make_employee("e1")
        shift = make_shift("s1", hours=14)
        policy = CompanyPolicy(max_shift_hours=12)
        result = engine_for([employee], [shift], policy).check(employee, shift, [])
        assert not result.ok
        assert "maximum shift length" in result.reason

    def test_outside_availability_blocks(self):
        employee = make_employee(
            "e1", availability=(AvailabilityWindow(0, time(14, 0), time(22, 0)),)
        )
        shift = make_shift("s1", start_hour=9)
        result = engine_for([employee], [shift]).check(employee, shift, [])
        assert not result.ok
        assert "availability" in result.reason

    def test_overlapping_assignment_blocks(self):
        employee = make_employee("e1")
        held = make_shift("held", start_hour=9, hours=8)
        wanted = make_shift("wanted", start_hour=12, hours=4)
        result = engine_for([employee], [held, wanted]).check(employee, wanted, [held])
        assert not result.ok
        assert "overlapping" in result.reason

    def test_insufficient_rest_blocks(self):
        employee = make_employee("e1")
        held = make_shift("held", day_offset=0, start_hour=14, hours=8)  # ends 22:00
        wanted = make_shift("wanted", day_offset=1, start_hour=4, hours=6)  # starts 04:00, 6h gap
        policy = CompanyPolicy(min_rest_hours_between_shifts=8)
        result = engine_for([employee], [held, wanted], policy).check(employee, wanted, [held])
        assert not result.ok
        assert "rest" in result.reason

    def test_adequate_rest_allows(self):
        employee = make_employee("e1")
        held = make_shift("held", day_offset=0, start_hour=9, hours=8)  # ends 17:00
        wanted = make_shift("wanted", day_offset=1, start_hour=9, hours=8)
        policy = CompanyPolicy(min_rest_hours_between_shifts=8)
        assert engine_for([employee], [held, wanted], policy).check(employee, wanted, [held]).ok

    def test_weekly_cap_blocks(self):
        employee = make_employee("e1")
        held = [make_shift(f"h{i}", day_offset=i, hours=8) for i in range(5)]  # 40h
        wanted = make_shift("wanted", day_offset=5, hours=8)
        policy = CompanyPolicy(max_hours_per_week=40)
        result = engine_for([employee], held + [wanted], policy).check(employee, wanted, held)
        assert not result.ok
        assert "weekly cap" in result.reason

    def test_employee_own_cap_is_respected(self):
        employee = make_employee("e1", max_hours=16)
        held = [make_shift(f"h{i}", day_offset=i, hours=8) for i in range(2)]
        wanted = make_shift("wanted", day_offset=3, hours=8)
        result = engine_for([employee], held + [wanted]).check(employee, wanted, held)
        assert not result.ok
        assert "16" in result.reason

    def test_consecutive_day_limit_blocks(self):
        employee = make_employee("e1")
        held = [make_shift(f"h{i}", day_offset=i, hours=4) for i in range(4)]
        wanted = make_shift("wanted", day_offset=4, hours=4)
        policy = CompanyPolicy(max_consecutive_days=4)
        result = engine_for([employee], held + [wanted], policy).check(employee, wanted, held)
        assert not result.ok
        assert "consecutive" in result.reason or "run past" in result.reason

    def test_non_consecutive_days_are_fine(self):
        employee = make_employee("e1")
        held = [make_shift("h0", day_offset=0, hours=4), make_shift("h1", day_offset=2, hours=4)]
        wanted = make_shift("wanted", day_offset=4, hours=4)
        policy = CompanyPolicy(max_consecutive_days=2)
        assert engine_for([employee], held + [wanted], policy).check(employee, wanted, held).ok

    def test_clean_assignment_allows(self):
        employee = make_employee("e1")
        shift = make_shift("s1")
        assert engine_for([employee], [shift]).check(employee, shift, []).ok


class TestEligibility:
    def test_splits_roster_and_explains_exclusions(self):
        good = make_employee("good", roles={"server"})
        wrong_role = make_employee("wrong", roles={"cook"})
        shift = make_shift("s1", role="server")

        eligible, blockers = ConstraintEngine(
            make_context([good, wrong_role], [shift])
        ).eligible_employees(shift, {})

        assert [e.id for e in eligible] == ["good"]
        assert "not qualified" in blockers["wrong"]

    def test_longest_run_counts_consecutive_days_only(self):
        run = ConstraintEngine._longest_run(
            {MONDAY, date(2026, 8, 18), date(2026, 8, 20)}
        )
        assert run == 2
