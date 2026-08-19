"""Domain model behaviour: time maths, availability, and cost policy."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytest

from conftest import MONDAY, make_employee, window
from hr_scheduling_agent.domain import (
    AvailabilityWindow,
    Certification,
    CompanyPolicy,
    ShiftPeriod,
    TimeWindow,
    week_of,
)


class TestTimeWindow:
    def test_rejects_backwards_window(self):
        start = datetime(2026, 8, 17, 12, 0)
        with pytest.raises(ValueError):
            TimeWindow(start=start, end=start - timedelta(hours=1))

    def test_hours(self):
        assert window(0, 9, 8).hours == 8.0

    def test_overlap_is_half_open(self):
        first = window(0, 9, 8)  # 09:00-17:00
        touching = window(0, 17, 4)  # 17:00-21:00
        assert not first.overlaps(touching), "back-to-back shifts must not count as overlapping"
        assert first.overlaps(window(0, 16, 4))

    def test_gap_hours_both_directions(self):
        morning = window(0, 9, 8)  # ends 17:00
        evening = window(0, 20, 2)  # starts 20:00
        assert morning.gap_hours_to(evening) == 3.0
        assert evening.gap_hours_to(morning) == 3.0

    def test_gap_is_negative_when_overlapping(self):
        assert window(0, 9, 8).gap_hours_to(window(0, 12, 4)) == -1.0


class TestAvailabilityWindow:
    def test_covers_shift_inside_window(self):
        availability = AvailabilityWindow(weekday=0, start=time(8, 0), end=time(18, 0))
        assert availability.covers(window(0, 9, 8))  # Monday 09:00-17:00

    def test_rejects_shift_starting_too_early(self):
        availability = AvailabilityWindow(weekday=0, start=time(10, 0), end=time(18, 0))
        assert not availability.covers(window(0, 9, 8))

    def test_rejects_shift_ending_too_late(self):
        availability = AvailabilityWindow(weekday=0, start=time(8, 0), end=time(16, 0))
        assert not availability.covers(window(0, 9, 8))

    def test_rejects_wrong_weekday(self):
        availability = AvailabilityWindow(weekday=2, start=time(0, 0), end=time(23, 59))
        assert not availability.covers(window(0, 9, 8))  # shift is a Monday

    def test_wrapping_window_covers_overnight_shift(self):
        overnight = AvailabilityWindow(weekday=0, start=time(19, 0), end=time(7, 0))
        assert overnight.covers(window(0, 19, 12))  # Mon 19:00 -> Tue 07:00

    def test_wrapping_window_rejects_overlong_shift(self):
        overnight = AvailabilityWindow(weekday=0, start=time(19, 0), end=time(7, 0))
        assert not overnight.covers(window(0, 19, 14))

    def test_non_wrapping_window_accepts_shift_ending_at_midnight(self):
        evening = AvailabilityWindow(weekday=0, start=time(16, 0), end=time(23, 59))
        assert evening.covers(window(0, 16, 8))  # 16:00 -> 00:00


class TestEmployee:
    def test_availability_respects_time_off(self):
        employee = make_employee("e1", unavailable=frozenset({MONDAY}))
        assert not employee.is_available_for(window(0, 9, 8))
        assert employee.is_available_for(window(1, 9, 8))

    def test_certification_validity_is_date_sensitive(self):
        employee = make_employee(
            "e1", certifications=(Certification("BLS", date(2026, 8, 20)),)
        )
        assert employee.certification_valid("BLS", date(2026, 8, 19))
        assert employee.certification_valid("BLS", date(2026, 8, 20)), "expiry day is inclusive"
        assert not employee.certification_valid("BLS", date(2026, 8, 21))
        assert not employee.certification_valid("ACLS", date(2026, 8, 19))


class TestShift:
    def test_period_buckets(self):
        assert ShiftPeriod.from_start(datetime(2026, 8, 17, 6)) is ShiftPeriod.MORNING
        assert ShiftPeriod.from_start(datetime(2026, 8, 17, 13)) is ShiftPeriod.AFTERNOON
        assert ShiftPeriod.from_start(datetime(2026, 8, 17, 21)) is ShiftPeriod.NIGHT

    def test_undesirable_covers_nights_and_weekends(self):
        from conftest import make_shift

        assert make_shift("night", day_offset=0, start_hour=20).is_undesirable
        assert make_shift("saturday", day_offset=5, start_hour=9).is_undesirable
        assert not make_shift("weekday-day", day_offset=1, start_hour=9).is_undesirable


class TestCompanyPolicy:
    def test_straight_time_below_threshold(self):
        policy = CompanyPolicy(overtime_threshold_hours=40, overtime_multiplier=1.5)
        assert policy.overtime_cost(base_hours=0, extra_hours=8, rate=20.0) == 160.0

    def test_splits_across_the_overtime_threshold(self):
        policy = CompanyPolicy(overtime_threshold_hours=40, overtime_multiplier=1.5)
        # 4h straight to reach 40, then 4h at 1.5x.
        assert policy.overtime_cost(base_hours=36, extra_hours=8, rate=20.0) == 4 * 20 + 4 * 30

    def test_fully_overtime_when_already_past_threshold(self):
        policy = CompanyPolicy(overtime_threshold_hours=40, overtime_multiplier=2.0)
        assert policy.overtime_cost(base_hours=44, extra_hours=2, rate=10.0) == 40.0


def test_week_of_returns_monday():
    assert week_of(date(2026, 8, 20)) == MONDAY
    assert week_of(MONDAY) == MONDAY
    assert week_of(date(2026, 8, 23)) == MONDAY  # Sunday still belongs to that week
