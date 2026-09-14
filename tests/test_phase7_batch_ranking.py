"""
Unit and integration tests for Phase 7: Batch resume analysis, failure tolerance, and candidate ranking.
"""

from pathlib import Path
import random
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from bot import analyze_command
from formatter import format_candidate_ranking
from gemini_service import GeminiAPIError, GeminiInputError
from scoring_engine import rank_candidates
from session_manager import (
    add_resume,
    clear_session,
    get_all_sessions,
    set_job_description,
)


def setup_function():
    """Clear sessions before each test."""
    get_all_sessions().clear()


def test_single_resume_ranking():
    """1. Test ranking with 1 candidate."""
    candidates = [
        {"candidate_name": "Alice", "job_match_percentage": 85, "ats_score": 80, "original_filename": "alice.pdf"}
    ]
    ranked = rank_candidates(candidates)
    assert len(ranked) == 1
    assert ranked[0]["candidate_name"] == "Alice"


def test_two_resumes_ranking():
    """2. Test ranking with 2 candidates (higher Job Match % wins)."""
    candidates = [
        {"candidate_name": "Bob", "job_match_percentage": 70, "ats_score": 85, "original_filename": "bob.pdf"},
        {"candidate_name": "Alice", "job_match_percentage": 90, "ats_score": 75, "original_filename": "alice.pdf"},
    ]
    ranked = rank_candidates(candidates)
    assert len(ranked) == 2
    assert ranked[0]["candidate_name"] == "Alice"
    assert ranked[1]["candidate_name"] == "Bob"


def test_five_resumes_ranking():
    """3. Test ranking with 5 candidates."""
    candidates = [
        {"candidate_name": "C1", "job_match_percentage": 60, "ats_score": 60, "original_filename": "c1.pdf"},
        {"candidate_name": "C2", "job_match_percentage": 95, "ats_score": 90, "original_filename": "c2.pdf"},
        {"candidate_name": "C3", "job_match_percentage": 75, "ats_score": 70, "original_filename": "c3.pdf"},
        {"candidate_name": "C4", "job_match_percentage": 88, "ats_score": 85, "original_filename": "c4.pdf"},
        {"candidate_name": "C5", "job_match_percentage": 40, "ats_score": 50, "original_filename": "c5.pdf"},
    ]
    ranked = rank_candidates(candidates)
    expected_order = ["C2", "C4", "C3", "C1", "C5"]
    assert [c["candidate_name"] for c in ranked] == expected_order


def test_ten_resumes_ranking():
    """4. Test ranking with 10 candidates."""
    candidates = [
        {"candidate_name": f"Cand_{i}", "job_match_percentage": i * 10, "ats_score": i * 9, "original_filename": f"c{i}.pdf"}
        for i in range(1, 11)
    ]
    random.shuffle(candidates)
    ranked = rank_candidates(candidates)
    # Highest score (Cand_10 with 100%) should be first, lowest (Cand_1 with 10%) last
    assert ranked[0]["candidate_name"] == "Cand_10"
    assert ranked[-1]["candidate_name"] == "Cand_1"
    # Verify strict descending order by job match
    for i in range(len(ranked) - 1):
        assert ranked[i]["job_match_percentage"] >= ranked[i + 1]["job_match_percentage"]


def test_duplicate_resumes_ranking():
    """5. Test duplicate resumes with identical scores (deterministic tie-breaking)."""
    candidates = [
        {"candidate_name": "John Doe", "job_match_percentage": 80, "ats_score": 75, "original_filename": "resume_b.pdf"},
        {"candidate_name": "John Doe", "job_match_percentage": 80, "ats_score": 75, "original_filename": "resume_a.pdf"},
    ]
    ranked = rank_candidates(candidates)
    assert len(ranked) == 2
    # Filename breaks tie deterministically
    assert ranked[0]["original_filename"] == "resume_a.pdf"
    assert ranked[1]["original_filename"] == "resume_b.pdf"


def test_tie_scores_broken_by_ats():
    """6. Test tie in Job Match % broken by secondary ATS score."""
    candidates = [
        {"candidate_name": "Candidate Low ATS", "job_match_percentage": 85, "ats_score": 70, "original_filename": "low.pdf"},
        {"candidate_name": "Candidate High ATS", "job_match_percentage": 85, "ats_score": 92, "original_filename": "high.pdf"},
    ]
    ranked = rank_candidates(candidates)
    assert ranked[0]["candidate_name"] == "Candidate High ATS"
    assert ranked[1]["candidate_name"] == "Candidate Low ATS"


def test_very_different_candidate_profiles():
    """7. Test candidates with very different profiles (High Match vs High ATS)."""
    candidates = [
        # Profile 1: Excellent tech stack match, mediocre ATS formatting
        {"candidate_name": "Tech Star", "job_match_percentage": 92, "ats_score": 68, "original_filename": "tech.pdf"},
        # Profile 2: Beautiful resume structure/keywords, but missing key core skills
        {"candidate_name": "Format Master", "job_match_percentage": 70, "ats_score": 95, "original_filename": "format.pdf"},
    ]
    ranked = rank_candidates(candidates)
    # Job Match is primary factor
    assert ranked[0]["candidate_name"] == "Tech Star"
    assert ranked[1]["candidate_name"] == "Format Master"


def test_ranking_determinism():
    """8. Verify ranking is strictly deterministic across 50 repeated runs."""
    base_candidates = [
        {"candidate_name": "Alpha", "job_match_percentage": 80, "ats_score": 75, "original_filename": "a.pdf"},
        {"candidate_name": "Beta", "job_match_percentage": 90, "ats_score": 85, "original_filename": "b.pdf"},
        {"candidate_name": "Gamma", "job_match_percentage": 80, "ats_score": 85, "original_filename": "g.pdf"},
        {"candidate_name": "Delta", "job_match_percentage": 60, "ats_score": 60, "original_filename": "d.pdf"},
    ]
    reference = [c["candidate_name"] for c in rank_candidates(base_candidates)]

    for _ in range(50):
        shuffled = list(base_candidates)
        random.shuffle(shuffled)
        assert [c["candidate_name"] for c in rank_candidates(shuffled)] == reference


@pytest.mark.asyncio
async def test_batch_flow_with_one_failing_candidate(tmp_path):
    """
    9. Test batch flow where one candidate encounters an error (Gemini failure or invalid file),
    verifying that the other valid candidates are still successfully analyzed, ranked, and returned.
    """
    user_id = 7001
    set_job_description(user_id, "Python Engineer.")

    f1 = tmp_path / "resume_1.pdf"
    f1.write_bytes(b"%PDF-1.4 good")
    f2 = tmp_path / "resume_2.pdf"
    f2.write_bytes(b"%PDF-1.4 fail")
    f3 = tmp_path / "resume_3.pdf"
    f3.write_bytes(b"%PDF-1.4 good")

    add_resume(user_id, {"original_filename": "Alex.pdf", "local_path": str(f1), "file_id": "1", "file_size": 100})
    add_resume(user_id, {"original_filename": "Bad.pdf", "local_path": str(f2), "file_id": "2", "file_size": 100})
    add_resume(user_id, {"original_filename": "Charlie.pdf", "local_path": str(f3), "file_id": "3", "file_size": 100})

    def mock_analyze(resume_path, job_description):
        if "resume_2" in str(resume_path):
            raise GeminiAPIError("Simulated Gemini 500 Internal Error")
        if "resume_1" in str(resume_path):
            return {
                "candidate_name": "Alex",
                "ats_score": 75,
                "job_match_percentage": 80,
                "matching_skills": ["Python"],
                "missing_skills": [],
                "partial_skills": [],
                "skill_breakdown": [],
                "recommended_learning": [],
                "recommendation": "Good.",
            }
        return {
            "candidate_name": "Charlie",
            "ats_score": 90,
            "job_match_percentage": 95,
            "matching_skills": ["Python", "FastAPI"],
            "missing_skills": [],
            "partial_skills": [],
            "skill_breakdown": [],
            "recommended_learning": [],
            "recommendation": "Excellent.",
        }

    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    status_msg = MagicMock(edit_text=AsyncMock())
    replies = []

    async def mock_reply(*args, **kwargs):
        if "Analyzing" in args[0]:
            return status_msg
        replies.append(args[0])
        return MagicMock()

    update.message = MagicMock(reply_text=AsyncMock(side_effect=mock_reply))
    context = MagicMock()

    with patch("bot.analyze_resume_against_jd", side_effect=mock_analyze):
        await analyze_command(update, context)

    # Status message must indicate 2/3 evaluated
    status_msg.edit_text.assert_awaited_once()
    status_call = status_msg.edit_text.call_args[0][0]
    assert "2/3" in status_call

    # Must have ranking card
    ranking_card = replies[0]
    assert "CANDIDATE RANKING" in ranking_card
    assert "Charlie" in ranking_card
    assert "Alex" in ranking_card
    assert "Bad.pdf" in ranking_card  # Failed section
    assert "Failed Analyses" in ranking_card


def test_format_candidate_ranking_card_visuals():
    """10. Verify formatting of the ranking card with emojis and score lines."""
    ranked = [
        {"candidate_name": "Top Candidate", "job_match_percentage": 92, "ats_score": 88, "original_filename": "top.pdf"},
        {"candidate_name": "Second Candidate", "job_match_percentage": 85, "ats_score": 80, "original_filename": "second.pdf"},
    ]
    failed = [{"filename": "broken.pdf", "error": "Corrupted file"}]

    card = format_candidate_ranking(ranked, failed_resumes=failed)
    assert "🏆 <b>CANDIDATE RANKING</b>" in card
    assert "1️⃣ <b>Top Candidate</b>" in card
    assert "92%" in card
    assert "88%" in card
    assert "2️⃣ <b>Second Candidate</b>" in card
    assert "⚠️ <b>Failed Analyses:</b>" in card
    assert "broken.pdf" in card
