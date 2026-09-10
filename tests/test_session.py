"""
Tests for session_manager.py
"""

import pytest
from session_manager import clear_session, get_session, get_all_sessions


def setup_function():
    """Clear all sessions before each test."""
    get_all_sessions().clear()


def test_get_session_creates_default():
    """Verify get_session initializes a clean default session."""
    user_id = 12345
    session = get_session(user_id)

    assert session["job_description"] is None
    assert session["resumes"] == []
    assert session["state"] == "idle"


def test_session_state_mutation():
    """Verify session data persists across get_session calls for the same user."""
    user_id = 12345
    session = get_session(user_id)
    session["job_description"] = "Senior Python Developer"
    session["resumes"].append("resume1.pdf")
    session["state"] = "waiting_for_resumes"

    retrieved = get_session(user_id)
    assert retrieved["job_description"] == "Senior Python Developer"
    assert len(retrieved["resumes"]) == 1
    assert retrieved["state"] == "waiting_for_resumes"


def test_clear_session():
    """Verify clear_session resets user data to default."""
    user_id = 99999
    session = get_session(user_id)
    session["job_description"] = "DevOps Engineer"
    session["resumes"] = ["res1.pdf", "res2.pdf"]

    clear_session(user_id)

    reset_session = get_session(user_id)
    assert reset_session["job_description"] is None
    assert reset_session["resumes"] == []
    assert reset_session["state"] == "idle"


def test_user_session_isolation():
    """Verify that multiple users have completely isolated sessions."""
    user_a = 101
    user_b = 202

    session_a = get_session(user_a)
    session_b = get_session(user_b)

    session_a["job_description"] = "JD for User A"
    session_b["job_description"] = "JD for User B"

    assert get_session(user_a)["job_description"] == "JD for User A"
    assert get_session(user_b)["job_description"] == "JD for User B"


def test_clear_session_in_place():
    """Verify that clear_session mutates the existing dict reference in-place."""
    user_id = 777
    original_ref = get_session(user_id)
    original_ref["job_description"] = "Some JD"

    clear_session(user_id)

    # original_ref must reflect the cleared state
    assert original_ref["job_description"] is None
    assert original_ref["resumes"] == []
