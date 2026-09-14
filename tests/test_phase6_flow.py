"""
Unit and integration tests for Phase 6: End-to-end candidate analysis and Telegram formatting.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from telegram.constants import ParseMode

from bot import analyze_command
from formatter import format_candidate_analysis, render_progress_bar
from gemini_service import GeminiAPIError
from session_manager import (
    add_resume,
    clear_session,
    get_all_sessions,
    set_job_description,
)


def setup_function():
    """Reset all sessions before each test."""
    get_all_sessions().clear()


def test_progress_bar_rendering():
    """Test progress bar visual output and bounds."""
    bar_0 = render_progress_bar(0)
    bar_50 = render_progress_bar(50)
    bar_100 = render_progress_bar(100)

    assert "[░░░░░░░░░░]" in bar_0
    assert "0%" in bar_0
    assert "[█████░░░░░]" in bar_50
    assert "50%" in bar_50
    assert "[██████████]" in bar_100
    assert "100%" in bar_100


def test_strong_candidate_formatting():
    """1. Test formatting for a strong candidate with high ATS and Job Match scores."""
    analysis = {
        "candidate_name": "Alice Chen",
        "ats_score": 92,
        "job_match_percentage": 88,
        "matching_skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "REST APIs"],
        "partial_skills": ["Kubernetes"],
        "missing_skills": [
            {"skill": "AWS", "importance": "preferred"},
        ],
        "skill_breakdown": [
            {"skill": "Python", "match_percentage": 100},
            {"skill": "FastAPI", "match_percentage": 100},
            {"skill": "PostgreSQL", "match_percentage": 90},
            {"skill": "AWS", "match_percentage": 20},
        ],
        "recommended_learning": [
            {
                "skill": "AWS",
                "reason": "Preferred cloud platform",
                "topics": ["AWS ECS", "S3 Storage", "IAM Roles"],
            }
        ],
        "recommendation": "Strongly recommended for technical interview. Excellent skill match.",
    }

    card = format_candidate_analysis(analysis, "Alice_Senior_Resume.pdf")

    assert "Alice Chen" in card
    assert "92/100" in card
    assert "88%" in card
    assert "Alice_Senior_Resume.pdf" in card
    assert "FastAPI" in card
    assert "🟢 <b>Preferred:</b>" in card
    assert "AWS ECS" in card
    assert "Strongly recommended" in card


def test_weak_candidate_formatting():
    """2. Test formatting for a weak candidate with low match scores and critical missing skills."""
    analysis = {
        "candidate_name": "Bob Miller",
        "ats_score": 28,
        "job_match_percentage": 18,
        "matching_skills": ["Git"],
        "partial_skills": [],
        "missing_skills": [
            {"skill": "Python", "importance": "critical"},
            {"skill": "FastAPI", "importance": "critical"},
            {"skill": "PostgreSQL", "importance": "important"},
        ],
        "skill_breakdown": [
            {"skill": "Python", "match_percentage": 0},
            {"skill": "FastAPI", "match_percentage": 0},
            {"skill": "Git", "match_percentage": 100},
        ],
        "recommended_learning": [
            {
                "skill": "Python",
                "reason": "Core programming language required for backend development.",
                "topics": ["Python basics", "Object-Oriented Programming", "Asyncio"],
            }
        ],
        "recommendation": "Not recommended. Lacks primary programming and framework requirements.",
    }

    card = format_candidate_analysis(analysis, "Bob_Resume.pdf")

    assert "Bob Miller" in card
    assert "28/100" in card
    assert "18%" in card
    assert "🔴 <b>Critical:</b>" in card
    assert "Python" in card
    assert "FastAPI" in card
    assert "Not recommended" in card


def test_candidate_with_categorized_missing_skills():
    """3. Test formatting with all three levels of missing skills: Critical, Important, and Preferred."""
    analysis = {
        "candidate_name": "Charlie Davis",
        "ats_score": 65,
        "job_match_percentage": 58,
        "matching_skills": ["Python", "Django"],
        "partial_skills": ["SQL"],
        "missing_skills": [
            {"skill": "FastAPI", "importance": "critical"},
            {"skill": "Docker", "importance": "important"},
            {"skill": "Terraform", "importance": "preferred"},
        ],
        "skill_breakdown": [
            {"skill": "Python", "match_percentage": 100},
            {"skill": "FastAPI", "match_percentage": 0},
        ],
        "recommended_learning": [
            {"skill": "FastAPI", "reason": "Required", "topics": ["Routing", "Pydantic"]}
        ],
        "recommendation": "Decent Python background, but requires upskilling in FastAPI and containerization.",
    }

    card = format_candidate_analysis(analysis, "Charlie_CV.pdf")

    assert "🔴 <b>Critical:</b>" in card
    assert "FastAPI" in card
    assert "🟡 <b>Important:</b>" in card
    assert "Docker" in card
    assert "🟢 <b>Preferred:</b>" in card
    assert "Terraform" in card


def test_unrelated_experience_formatting():
    """4. Test candidate with unrelated experience (e.g. Sales applying for Backend)."""
    analysis = {
        "candidate_name": "Diana Ross",
        "ats_score": 15,
        "job_match_percentage": 10,
        "matching_skills": [],
        "partial_skills": [],
        "missing_skills": [
            {"skill": "Software Engineering", "importance": "critical"},
            {"skill": "Python", "importance": "critical"},
        ],
        "skill_breakdown": [],
        "recommended_learning": [],
        "recommendation": "Candidate has a sales and marketing background with no software engineering experience.",
    }

    card = format_candidate_analysis(analysis, "Diana_Sales.pdf")

    assert "Diana Ross" in card
    assert "15/100" in card
    assert "10%" in card
    assert "<i>None explicitly matched</i>" in card
    assert "sales and marketing background" in card


def test_unusual_formatting_and_special_characters():
    """5. Test resume analysis containing HTML tags, ampersands, and quotes (verifies safe escaping)."""
    analysis = {
        "candidate_name": "Evan <Hacker> & Co.",
        "ats_score": 75,
        "job_match_percentage": 70,
        "matching_skills": ["C++", "C# & .NET", "HTML & CSS"],
        "partial_skills": ["<React/Vue>"],
        "missing_skills": [
            {"skill": "SQL & NoSQL <Databases>", "importance": "critical"},
        ],
        "skill_breakdown": [
            {"skill": "C# & .NET", "match_percentage": 85},
        ],
        "recommended_learning": [
            {
                "skill": "SQL & NoSQL",
                "reason": "Required for backend <data>",
                "topics": ["PostgreSQL & MongoDB", "JSON <BSON> types"],
            }
        ],
        "recommendation": "Solid developer. Knows C# & C++. Score: > 70%.",
    }

    card = format_candidate_analysis(analysis, "Evan_Resume<Special>.pdf")

    # Verify no raw HTML tags exist in user content
    assert "<Hacker>" not in card
    assert "&lt;Hacker&gt;" in card
    assert "&amp; Co." in card
    assert "&lt;React/Vue&gt;" in card
    assert "&lt;Databases&gt;" in card
    assert "Evan_Resume&lt;Special&gt;.pdf" in card


def test_resume_with_no_obvious_candidate_name():
    """6. Test fallback when resume has no candidate name or returns None / Unknown."""
    for empty_val in [None, "", "Unknown", "N/A", "none"]:
        analysis = {
            "candidate_name": empty_val,
            "ats_score": 60,
            "job_match_percentage": 55,
            "matching_skills": ["Python"],
            "missing_skills": [],
            "partial_skills": [],
            "skill_breakdown": [],
            "recommended_learning": [],
            "recommendation": "Evaluation complete.",
        }
        card = format_candidate_analysis(analysis, "John_Doe_2026.pdf")
        # Must fallback gracefully to include filename stem
        assert "Candidate (John_Doe_2026)" in card


@pytest.mark.asyncio
async def test_end_to_end_analyze_command_flow(tmp_path):
    """Test full /analyze Telegram flow with mocked Gemini service response."""
    user_id = 9001
    set_job_description(user_id, "Senior Backend Developer: Python, FastAPI, Docker.")

    # Create dummy local PDF file
    resume_file = tmp_path / "resume_1_Alex.pdf"
    resume_file.write_bytes(b"%PDF-1.4 mock content")

    add_resume(
        user_id=user_id,
        resume_meta={
            "original_filename": "Alex_Resume.pdf",
            "local_path": str(resume_file),
            "file_id": "file_9001",
            "file_size": 1024,
        },
    )

    mock_analysis = {
        "candidate_name": "Alex Mercer",
        "ats_score": 82,
        "job_match_percentage": 78,
        "matching_skills": ["Python", "FastAPI"],
        "missing_skills": [{"skill": "Docker", "importance": "critical"}],
        "partial_skills": [],
        "skill_breakdown": [{"skill": "Python", "match_percentage": 100}],
        "recommended_learning": [{"skill": "Docker", "reason": "Mandatory", "topics": ["Containers"]}],
        "recommendation": "Strong applicant.",
    }

    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    status_msg = MagicMock()
    status_msg.edit_text = AsyncMock()
    reply_msg = MagicMock()

    async def mock_reply_text(*args, **kwargs):
        if "Analyzing" in args[0]:
            return status_msg
        return reply_msg

    update.message = MagicMock(reply_text=AsyncMock(side_effect=mock_reply_text))
    context = MagicMock()

    with patch("bot.analyze_resume_against_jd", return_value=mock_analysis):
        await analyze_command(update, context)

    # Verify status message sent and then edited
    status_msg.edit_text.assert_awaited_once()
    edit_text_args, _ = status_msg.edit_text.call_args
    assert "Screening complete" in edit_text_args[0]

    # Verify candidate card sent
    update.message.reply_text.assert_awaited()
    # At least status notice + ranking card + candidate card
    assert update.message.reply_text.await_count >= 2


@pytest.mark.asyncio
async def test_analyze_command_handles_gemini_api_error(tmp_path):
    """Test that /analyze handles Gemini API failures gracefully without crashing."""
    user_id = 9002
    set_job_description(user_id, "DevOps Engineer.")

    resume_file = tmp_path / "resume_1.pdf"
    resume_file.write_bytes(b"%PDF-1.4 mock content")

    add_resume(
        user_id=user_id,
        resume_meta={
            "original_filename": "DevOps.pdf",
            "local_path": str(resume_file),
            "file_id": "file_9002",
            "file_size": 2048,
        },
    )

    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    status_msg = MagicMock(edit_text=AsyncMock())

    async def mock_reply(*args, **kwargs):
        if "Analyzing" in args[0]:
            return status_msg
        return MagicMock()

    update.message = MagicMock(reply_text=AsyncMock(side_effect=mock_reply))
    context = MagicMock()

    with patch("bot.analyze_resume_against_jd", side_effect=GeminiAPIError("Rate limit exceeded")):
        await analyze_command(update, context)

    # Verify error summary sent to user
    update.message.reply_text.assert_awaited()
    last_call = update.message.reply_text.call_args_list[-1]
    assert "Failed Analyses" in last_call[0][0]
    assert "DevOps.pdf" in last_call[0][0]
