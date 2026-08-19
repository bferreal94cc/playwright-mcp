"""Integration bridges and urgency-based notification routing."""

from __future__ import annotations

from datetime import datetime

from conftest import make_context, make_employee, make_shift
from hr_scheduling_agent.domain import Assignment, Schedule
from hr_scheduling_agent.integrations import (
    InMemoryCalendar,
    InMemoryHRIS,
    NotificationEngine,
    Urgency,
)
from hr_scheduling_agent.optimizer import ScheduleOptimizer
from hr_scheduling_agent.providers import (
    ClaudeProvider,
    GoogleADKProvider,
    LLMResponse,
    StubProvider,
    provider_from_env,
)


class TestHRIS:
    def test_round_trips_the_roster(self):
        employees = [make_employee("e1"), make_employee("e2")]
        assert len(InMemoryHRIS(employees).fetch_employees()) == 2

    def test_push_writes_one_record_per_assignment(self):
        context = make_context(
            [make_employee("e1")], [make_shift("s1"), make_shift("s2", day_offset=1)]
        )
        schedule = ScheduleOptimizer(context).solve()
        hris = InMemoryHRIS(context.employees)
        written = hris.push_schedule(schedule, context)
        assert written == len(schedule.assignments)
        assert len(hris.published) == written

    def test_unknown_shifts_are_skipped(self):
        context = make_context([make_employee("e1")], [make_shift("s1")])
        schedule = Schedule(assignments=[Assignment("ghost-shift", "e1", 0.0)])
        assert InMemoryHRIS(context.employees).push_schedule(schedule, context) == 0


class TestCalendar:
    def test_publish_then_withdraw(self):
        calendar = InMemoryCalendar()
        employee = make_employee("e1")
        shift = make_shift("s1")
        event_id = calendar.publish(employee, shift)
        assert event_id in calendar.events
        assert calendar.withdraw(event_id)
        assert event_id not in calendar.events

    def test_withdrawing_an_unknown_event_is_false(self):
        assert not InMemoryCalendar().withdraw("nope")

    def test_event_ids_are_unique(self):
        calendar = InMemoryCalendar()
        employee = make_employee("e1")
        ids = {calendar.publish(employee, make_shift(f"s{i}")) for i in range(3)}
        assert len(ids) == 3


class TestNotificationRouting:
    def test_routine_goes_to_email_and_in_app(self):
        engine = NotificationEngine.in_memory()
        channels = engine.notify("e1", "subject", "body", Urgency.ROUTINE)
        assert set(channels) == {"email", "in_app"}

    def test_urgent_reaches_sms_and_push(self):
        engine = NotificationEngine.in_memory()
        channels = engine.notify("e1", "cover needed", "now", Urgency.URGENT)
        assert "sms" in channels and "push" in channels

    def test_important_adds_push_to_the_routine_channels(self):
        engine = NotificationEngine.in_memory()
        channels = engine.notify("e1", "subject", "body", Urgency.IMPORTANT)
        assert set(channels) == {"email", "in_app", "push"}

    def test_messages_land_in_the_channel_outbox(self):
        engine = NotificationEngine.in_memory()
        engine.notify("e1", "hello", "body", Urgency.ROUTINE)
        assert len(engine.channels["email"].outbox) == 1
        assert engine.channels["email"].outbox[0].recipient_id == "e1"
        assert engine.channels["sms"].outbox == []

    def test_missing_channel_is_skipped_without_raising(self):
        engine = NotificationEngine(channels={})
        assert engine.notify("e1", "s", "b", Urgency.URGENT) == []
        assert len(engine.sent) == 1


class TestBroadcast:
    def test_every_scheduled_person_is_reached_once(self):
        context = make_context(
            [make_employee("e1"), make_employee("e2")],
            [make_shift("s1"), make_shift("s2", day_offset=1)],
        )
        schedule = ScheduleOptimizer(context).solve()
        engine = NotificationEngine.in_memory()
        reached = engine.broadcast_schedule(schedule, context)
        recipients = {n.recipient_id for n in engine.sent}
        assert reached == len(recipients)
        assert recipients == {a.employee_id for a in schedule.assignments}

    def test_body_lists_the_shifts(self):
        context = make_context([make_employee("e1")], [make_shift("s1")])
        schedule = ScheduleOptimizer(context).solve()
        engine = NotificationEngine.in_memory()
        engine.broadcast_schedule(schedule, context)
        assert "loc-1" in engine.sent[0].body

    def test_nobody_scheduled_means_nobody_notified(self):
        context = make_context([make_employee("e1")], [make_shift("s1")])
        assert NotificationEngine.in_memory().broadcast_schedule(Schedule(), context) == 0


class TestProviders:
    def test_stub_matches_on_substring(self):
        provider = StubProvider({"weather": "sunny"}, default="fallback")
        assert provider.complete("what is the WEATHER").text == "sunny"
        assert provider.complete("anything else").text == "fallback"

    def test_stub_records_its_calls(self):
        provider = StubProvider()
        provider.complete("first")
        provider.complete("second")
        assert provider.calls == ["first", "second"]

    def test_json_parsing_handles_fenced_blocks(self):
        response = LLMResponse('```json\n{"intent":"post_job"}\n```', "stub", "s")
        assert response.as_json() == {"intent": "post_job"}

    def test_json_parsing_returns_none_for_prose(self):
        assert LLMResponse("not json", "stub", "s").as_json() is None

    def test_json_parsing_rejects_a_bare_list(self):
        assert LLMResponse("[1, 2]", "stub", "s").as_json() is None

    def test_env_selection_prefers_an_explicit_choice(self, monkeypatch):
        monkeypatch.setenv("HR_AGENT_LLM", "stub")
        monkeypatch.setenv("GOOGLE_API_KEY", "irrelevant")
        assert isinstance(provider_from_env(), StubProvider)

    def test_env_selection_uses_google_when_its_key_is_set(self, monkeypatch):
        monkeypatch.delenv("HR_AGENT_LLM", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        assert isinstance(provider_from_env(), GoogleADKProvider)

    def test_env_selection_uses_anthropic_when_its_key_is_set(self, monkeypatch):
        monkeypatch.delenv("HR_AGENT_LLM", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        assert isinstance(provider_from_env(), ClaudeProvider)

    def test_env_selection_falls_back_to_the_stub(self, monkeypatch):
        for name in ("HR_AGENT_LLM", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        assert isinstance(provider_from_env(), StubProvider)

    def test_cloud_providers_construct_without_their_sdk(self, monkeypatch):
        """Constructing must be free; only calling requires the optional extra."""
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        assert GoogleADKProvider().model
        assert ClaudeProvider().model
