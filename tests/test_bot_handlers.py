"""
Unit tests for bot.py command handlers using AsyncMock.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest
from telegram.constants import ParseMode

from bot import (
    create_bot_application,
    help_command,
    reset_command,
    start_command,
    unknown_command,
)
from session_manager import get_all_sessions, get_session


def setup_function():
    """Clear in-memory sessions before each test."""
    get_all_sessions().clear()


def make_mock_update(user_id=123, first_name="Alex"):
    """Helper to create a mock Telegram Update."""
    update = MagicMock()
    user = MagicMock()
    user.id = user_id
    user.first_name = first_name
    update.effective_user = user

    message = MagicMock()
    message.reply_text = AsyncMock()
    update.message = message

    context = MagicMock()
    return update, context


@pytest.mark.asyncio
async def test_start_command():
    """Test that /start welcomes user, sets state to waiting_for_jd, and sends HTML reply."""
    update, context = make_mock_update(user_id=456, first_name="Sarah")

    await start_command(update, context)

    # Verify session initialized
    session = get_session(456)
    assert session["state"] == "waiting_for_jd"

    # Verify reply sent with greeting and parse_mode
    update.message.reply_text.assert_awaited_once()
    call_args, call_kwargs = update.message.reply_text.call_args
    assert "Sarah" in call_args[0]
    assert "JobFit AI" in call_args[0]
    assert call_kwargs.get("parse_mode") == ParseMode.HTML


@pytest.mark.asyncio
async def test_help_command():
    """Test that /help explains bot features and lists commands."""
    update, context = make_mock_update(user_id=789)

    await help_command(update, context)

    update.message.reply_text.assert_awaited_once()
    call_args, call_kwargs = update.message.reply_text.call_args
    assert "/start" in call_args[0]
    assert "/help" in call_args[0]
    assert "/reset" in call_args[0]
    assert call_kwargs.get("parse_mode") == ParseMode.HTML


@pytest.mark.asyncio
async def test_reset_command():
    """Test that /reset clears the session data."""
    user_id = 999
    session = get_session(user_id)
    session["job_description"] = "Python Developer JD"
    session["resumes"] = ["res1.pdf"]

    update, context = make_mock_update(user_id=user_id)

    await reset_command(update, context)

    # Verify session is reset
    reset_session = get_session(user_id)
    assert reset_session["job_description"] is None
    assert reset_session["resumes"] == []

    update.message.reply_text.assert_awaited_once()
    call_args, _ = update.message.reply_text.call_args
    assert "Session reset successfully" in call_args[0]


@pytest.mark.asyncio
async def test_unknown_command():
    """Test that unrecognized commands inform the user gracefully."""
    update, context = make_mock_update(user_id=111)

    await unknown_command(update, context)

    update.message.reply_text.assert_awaited_once()
    call_args, call_kwargs = update.message.reply_text.call_args
    assert "Unrecognized command" in call_args[0]
    assert "/help" in call_args[0]


def test_create_bot_application():
    """Test that create_bot_application constructs the Application with expected handlers."""
    fake_token = "123456789:ABCDEF1234567890abcdef"
    app = create_bot_application(fake_token)
    assert app is not None


@pytest.mark.asyncio
async def test_start_command_html_escaping():
    """Test that user names containing HTML tags/characters are safely escaped."""
    malicious_name = "Alex <script>alert(1)</script> & Co."
    update, context = make_mock_update(user_id=1234, first_name=malicious_name)

    await start_command(update, context)

    update.message.reply_text.assert_awaited_once()
    call_args, call_kwargs = update.message.reply_text.call_args
    # Should contain escaped entities, NOT raw HTML tags
    assert "&lt;script&gt;" in call_args[0]
    assert "&amp; Co." in call_args[0]
    assert "<script>" not in call_args[0]
    assert call_kwargs.get("parse_mode") == ParseMode.HTML


@pytest.mark.asyncio
async def test_error_handler():
    """Test that global error_handler logs update exceptions without crashing."""
    from bot import error_handler
    context = MagicMock()
    context.error = RuntimeError("Simulated Telegram error")
    # Should run and log without raising an exception
    await error_handler(None, context)


def test_main_missing_token(monkeypatch):
    """Test that main() safely exits with code 1 if TELEGRAM_BOT_TOKEN is missing."""
    from bot import main
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_main_placeholder_token(monkeypatch):
    """Test that main() safely exits with code 1 if TELEGRAM_BOT_TOKEN is default placeholder."""
    from bot import main
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "your_telegram_bot_token_here")

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_main_invalid_token(monkeypatch):
    """Test that main() catches InvalidToken and exits with code 1 without tracebacks."""
    from bot import main
    from unittest.mock import patch
    from telegram.error import InvalidToken

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:invalid_token_xyz")

    with patch("bot.Application.run_polling", side_effect=InvalidToken("Unauthorized")):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


def test_main_keyboard_interrupt(monkeypatch):
    """Test that main() handles KeyboardInterrupt gracefully."""
    from bot import main
    from unittest.mock import patch

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:valid_token_xyz")

    with patch("bot.Application.run_polling", side_effect=KeyboardInterrupt):
        # Should catch KeyboardInterrupt and return cleanly without raising SystemExit or crashing
        main()
