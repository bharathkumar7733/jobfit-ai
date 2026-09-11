"""
Unit and integration tests for Phase 2: Job Description submission and session management.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest
from telegram.constants import ParseMode

from bot import (
    handle_document_before_jd,
    handle_job_description,
    help_command,
    reset_command,
    start_command,
)
from session_manager import (
    clear_session,
    get_all_sessions,
    get_job_description,
    get_session,
    set_job_description,
)


def setup_function():
    """Reset all sessions before every test."""
    get_all_sessions().clear()


def make_mock_update(user_id=123, text=None, is_document=False, first_name="TestUser"):
    """Creates a mock Telegram update with message and user info."""
    update = MagicMock()
    user = MagicMock()
    user.id = user_id
    user.first_name = first_name
    update.effective_user = user

    message = MagicMock()
    message.text = text
    message.reply_text = AsyncMock()

    if is_document:
        doc = MagicMock()
        doc.file_name = "resume.pdf"
        message.document = doc
    else:
        message.document = None

    update.message = message
    context = MagicMock()
    return update, context


@pytest.mark.asyncio
async def test_valid_job_description():
    """Test submitting a valid, well-formed Job Description."""
    user_id = 1001
    valid_jd = (
        "Senior Backend Engineer\n"
        "Requirements:\n"
        "- 5+ years experience in Python and FastAPI\n"
        "- Experience with PostgreSQL and Redis\n"
        "- Knowledge of Docker and CI/CD pipelines"
    )
    update, context = make_mock_update(user_id=user_id, text=valid_jd)

    await handle_job_description(update, context)

    # Verify session state and stored JD
    session = get_session(user_id)
    assert session["job_description"] == valid_jd
    assert session["state"] == "waiting_for_resumes"
    assert get_job_description(user_id) == valid_jd

    # Verify response message
    update.message.reply_text.assert_awaited_once()
    reply, kwargs = update.message.reply_text.call_args
    assert "Job Description received" in reply[0]
    assert kwargs.get("parse_mode") == ParseMode.HTML


@pytest.mark.asyncio
async def test_empty_or_whitespace_job_description():
    """Test that empty or whitespace-only JD input is rejected."""
    user_id = 1002
    update, context = make_mock_update(user_id=user_id, text="    \n\t   ")

    await handle_job_description(update, context)

    # Session must not store empty JD
    session = get_session(user_id)
    assert session["job_description"] is None

    update.message.reply_text.assert_awaited_once()
    reply, _ = update.message.reply_text.call_args
    assert "cannot be empty" in reply[0]


@pytest.mark.asyncio
async def test_very_short_job_description():
    """Test that excessively short JD input (< 30 chars) is rejected with guidance."""
    user_id = 1003
    update, context = make_mock_update(user_id=user_id, text="Need Python dev")

    await handle_job_description(update, context)

    session = get_session(user_id)
    assert session["job_description"] is None

    update.message.reply_text.assert_awaited_once()
    reply, _ = update.message.reply_text.call_args
    assert "too short" in reply[0]


@pytest.mark.asyncio
async def test_excessively_long_job_description():
    """Test that input exceeding 10,000 characters is rejected gracefully."""
    user_id = 1004
    huge_jd = "Python Developer " * 700  # > 11,000 chars
    update, context = make_mock_update(user_id=user_id, text=huge_jd)

    await handle_job_description(update, context)

    session = get_session(user_id)
    assert session["job_description"] is None

    update.message.reply_text.assert_awaited_once()
    reply, _ = update.message.reply_text.call_args
    assert "too long" in reply[0]


@pytest.mark.asyncio
async def test_job_description_with_special_characters_and_unicode():
    """Test that JDs with HTML tags, emojis, and special symbols are stored accurately."""
    user_id = 1005
    special_jd = (
        "🚀 Fullstack Developer <React/Node.js & Python>\n"
        "Must know: C++ & C# & SQL; Salary: $120k-$150k; Experience: 3+ yrs.\n"
        "Special chars: <html>, &amp;, \", ', %, @, #"
    )
    update, context = make_mock_update(user_id=user_id, text=special_jd)

    await handle_job_description(update, context)

    session = get_session(user_id)
    # The raw text should be preserved verbatim in the session
    assert session["job_description"] == special_jd
    assert "<html>" in session["job_description"]
    assert "🚀" in session["job_description"]


@pytest.mark.asyncio
async def test_multiple_users_isolation():
    """Verify that User A and User B maintain completely isolated Job Descriptions."""
    user_a = 2001
    user_b = 2002

    jd_a = "Frontend Engineer: React, TypeScript, Tailwind CSS, Next.js, Redux."
    jd_b = "DevOps Engineer: Kubernetes, Terraform, AWS, Docker, CI/CD, Helm."

    update_a, context_a = make_mock_update(user_id=user_a, text=jd_a)
    update_b, context_b = make_mock_update(user_id=user_b, text=jd_b)

    await handle_job_description(update_a, context_a)
    await handle_job_description(update_b, context_b)

    # Ensure neither user's JD leaked into the other
    assert get_job_description(user_a) == jd_a
    assert get_job_description(user_b) == jd_b
    assert get_job_description(user_a) != get_job_description(user_b)


@pytest.mark.asyncio
async def test_reset_after_jd_submission():
    """Test that /reset removes the stored JD and sets session back to idle."""
    user_id = 3001
    jd = "Data Scientist: Python, Pandas, PyTorch, Scikit-learn, Machine Learning."
    update_jd, context_jd = make_mock_update(user_id=user_id, text=jd)
    await handle_job_description(update_jd, context_jd)

    assert get_job_description(user_id) == jd

    # Now execute /reset
    update_reset, context_reset = make_mock_update(user_id=user_id)
    await reset_command(update_reset, context_reset)

    # Verify JD is completely removed
    assert get_job_description(user_id) is None
    assert get_session(user_id)["resumes"] == []
    assert get_session(user_id)["state"] == "idle"


@pytest.mark.asyncio
async def test_sending_document_before_jd():
    """Test that uploading a document before submitting a JD prompts for the JD first."""
    user_id = 4001
    update, context = make_mock_update(user_id=user_id, is_document=True)

    await handle_document_before_jd(update, context)

    update.message.reply_text.assert_awaited_once()
    reply, _ = update.message.reply_text.call_args
    assert "send the Job Description first" in reply[0]


@pytest.mark.asyncio
async def test_commands_while_waiting_for_jd():
    """Test that sending /help or /start does not get swallowed or saved as the JD."""
    user_id = 5001

    # First call /start
    update_start, context_start = make_mock_update(user_id=user_id)
    await start_command(update_start, context_start)
    assert get_session(user_id)["state"] == "waiting_for_jd"
    assert get_job_description(user_id) is None

    # Then call /help
    update_help, context_help = make_mock_update(user_id=user_id)
    await help_command(update_help, context_help)
    assert get_job_description(user_id) is None


@pytest.mark.asyncio
async def test_prevent_accidental_jd_overwrite():
    """Test that once a JD is accepted, subsequent plain text alerts user instead of silently overwriting."""
    user_id = 6001
    jd_initial = "Backend Developer with Python, Django, REST APIs, and PostgreSQL."
    update1, context1 = make_mock_update(user_id=user_id, text=jd_initial)
    await handle_job_description(update1, context1)

    assert get_job_description(user_id) == jd_initial

    # Attempt to send another text message
    update2, context2 = make_mock_update(user_id=user_id, text="Accidental second text message")
    await handle_job_description(update2, context2)

    # Initial JD must remain unchanged
    assert get_job_description(user_id) == jd_initial
    reply, _ = update2.message.reply_text.call_args
    assert "already set" in reply[0]
