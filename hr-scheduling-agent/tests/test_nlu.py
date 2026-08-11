"""Intent classification, entity extraction, and the LLM fallback path."""

from __future__ import annotations

from datetime import date

import pytest

from hr_scheduling_agent.nlu import (
    Intent,
    NLUPipeline,
    RuleBasedParser,
    ScriptedTranscriber,
    parser_for_context,
)
from hr_scheduling_agent.providers import StubProvider
from hr_scheduling_agent.scenarios import build_retail_context

MONDAY = date(2026, 8, 17)


@pytest.fixture
def parser() -> RuleBasedParser:
    return RuleBasedParser(
        known_roles={"server", "cook"},
        known_names={"Casey": "r03", "Devon": "r04"},
    )


class TestIntentClassification:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("I'm sick, calling out for tonight", Intent.REPORT_ABSENCE),
            ("I can't come in tomorrow", Intent.REPORT_ABSENCE),
            ("Can I swap my Friday shift?", Intent.SWAP_SHIFT),
            ("I need two days off next week", Intent.REQUEST_TIME_OFF),
            ("Requesting PTO for August", Intent.REQUEST_TIME_OFF),
            ("Build the schedule for next week", Intent.CREATE_SCHEDULE),
            ("Please generate the schedule", Intent.CREATE_SCHEDULE),
            ("We need to hire another cook", Intent.POST_JOB),
            ("Post a job for two servers", Intent.POST_JOB),
            ("When am I working?", Intent.QUERY_SCHEDULE),
            ("Who's on tonight?", Intent.QUERY_SCHEDULE),
        ],
    )
    def test_classifies_common_phrasings(self, parser, text, expected):
        assert parser.parse(text, reference=MONDAY).intent is expected

    def test_unrecognised_text_is_unknown_with_zero_confidence(self, parser):
        parsed = parser.parse("the quarterly figures look fine", reference=MONDAY)
        assert parsed.intent is Intent.UNKNOWN
        assert parsed.confidence == 0.0
        assert not parsed.is_confident

    def test_confident_matches_clear_the_threshold(self, parser):
        assert parser.parse("calling out today", reference=MONDAY).is_confident


class TestDateExtraction:
    def test_today_and_tomorrow(self, parser):
        parsed = parser.parse("calling out today", reference=MONDAY)
        assert parsed.entities["dates"] == ["2026-08-17"]
        parsed = parser.parse("calling out tomorrow", reference=MONDAY)
        assert parsed.entities["dates"] == ["2026-08-18"]

    def test_iso_dates(self, parser):
        parsed = parser.parse("time off on 2026-09-03", reference=MONDAY)
        assert "2026-09-03" in parsed.entities["dates"]

    def test_slash_dates(self, parser):
        parsed = parser.parse("time off on 9/3", reference=MONDAY)
        assert "2026-09-03" in parsed.entities["dates"]

    def test_month_name_dates(self, parser):
        parsed = parser.parse("time off on aug 20", reference=MONDAY)
        assert "2026-08-20" in parsed.entities["dates"]

    def test_bare_weekday_resolves_forward(self, parser):
        parsed = parser.parse("swap my friday shift", reference=MONDAY)
        assert parsed.entities["dates"] == ["2026-08-21"]

    def test_next_weekday_skips_to_the_following_week(self, parser):
        parsed = parser.parse("swap my shift next tuesday", reference=MONDAY)
        assert parsed.entities["dates"] == ["2026-08-25"]

    def test_next_week_sets_the_week_start(self, parser):
        parsed = parser.parse("build the schedule for next week", reference=MONDAY)
        assert parsed.entities["week_start"] == "2026-08-24"

    def test_invalid_dates_are_ignored(self, parser):
        parsed = parser.parse("time off on 13/45", reference=MONDAY)
        assert "dates" not in parsed.entities or "13" not in str(parsed.entities["dates"])


class TestOtherEntities:
    def test_times(self, parser):
        parsed = parser.parse("calling out for my 9am shift", reference=MONDAY)
        assert "09:00" in parsed.entities["times"]

    def test_twenty_four_hour_times(self, parser):
        parsed = parser.parse("swap my 14:30 shift", reference=MONDAY)
        assert "14:30" in parsed.entities["times"]

    def test_pm_times_convert(self, parser):
        parsed = parser.parse("swap my 7pm shift", reference=MONDAY)
        assert "19:00" in parsed.entities["times"]

    def test_role_detection(self, parser):
        assert parser.parse("hire a cook", reference=MONDAY).entities["role"] == "cook"

    def test_known_names_map_to_ids(self, parser):
        parsed = parser.parse("swap with Casey", reference=MONDAY)
        assert parsed.entities["employee_ids"] == ["r03"]

    def test_headcount_is_anchored_to_a_role(self, parser):
        parsed = parser.parse("we need to hire three more cooks", reference=MONDAY)
        assert parsed.entities["headcount"] == 3

    def test_days_are_not_mistaken_for_headcount(self, parser):
        """'two days off' counts days, not people."""
        parsed = parser.parse("I need two days off", reference=MONDAY)
        assert "headcount" not in parsed.entities

    def test_numeric_headcount(self, parser):
        parsed = parser.parse("post a job for 2 servers", reference=MONDAY)
        assert parsed.entities["headcount"] == 2


class TestPipeline:
    def test_confident_rules_skip_the_llm(self):
        provider = StubProvider(default='{"intent":"post_job","confidence":0.9}')
        pipeline = NLUPipeline(RuleBasedParser(), provider)
        parsed = pipeline.understand("I'm calling out today", reference=MONDAY)
        assert parsed.source == "rules"
        assert provider.calls == [], "the LLM must not be called when rules are sure"

    def test_falls_back_to_the_llm_when_unsure(self):
        provider = StubProvider(default='{"intent":"query_schedule","confidence":0.8}')
        pipeline = NLUPipeline(RuleBasedParser(), provider)
        parsed = pipeline.understand("what's the situation for the weekend", reference=MONDAY)
        assert parsed.intent is Intent.QUERY_SCHEDULE
        assert parsed.source == "llm:stub"
        assert provider.calls

    def test_works_with_no_provider_at_all(self):
        pipeline = NLUPipeline(RuleBasedParser())
        parsed = pipeline.understand("something unparseable", reference=MONDAY)
        assert parsed.intent is Intent.UNKNOWN

    def test_llm_failure_degrades_to_the_rule_result(self):
        class Broken(StubProvider):
            def complete(self, *args, **kwargs):
                raise RuntimeError("provider down")

        pipeline = NLUPipeline(RuleBasedParser(), Broken())
        parsed = pipeline.understand("unparseable text", reference=MONDAY)
        assert parsed.intent is Intent.UNKNOWN
        assert parsed.source == "rules"

    def test_malformed_llm_json_degrades(self):
        pipeline = NLUPipeline(RuleBasedParser(), StubProvider(default="not json at all"))
        parsed = pipeline.understand("unparseable text", reference=MONDAY)
        assert parsed.source == "rules"

    def test_unknown_llm_intent_degrades(self):
        provider = StubProvider(default='{"intent":"make_coffee","confidence":0.9}')
        pipeline = NLUPipeline(RuleBasedParser(), provider)
        parsed = pipeline.understand("unparseable text", reference=MONDAY)
        assert parsed.source == "rules"


class TestSpeechPath:
    def test_transcribes_then_classifies(self):
        pipeline = NLUPipeline(RuleBasedParser())
        transcriber = ScriptedTranscriber(["I can't come in today, I'm unwell"])
        parsed = pipeline.understand_speech(b"audio", transcriber, reference=MONDAY)
        assert parsed.intent is Intent.REPORT_ABSENCE
        assert parsed.entities["transcript"].startswith("I can't come in")
        assert transcriber.calls == 1


class TestContextPrimedParser:
    def test_picks_up_roles_and_names_from_a_scenario(self):
        parser = parser_for_context(build_retail_context())
        parsed = parser.parse("we need another cook", reference=MONDAY)
        assert parsed.entities["role"] == "cook"
        parsed = parser.parse("swap with Avery", reference=MONDAY)
        assert parsed.entities["employee_ids"] == ["r01"]
