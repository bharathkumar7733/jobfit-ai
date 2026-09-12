"""
Unit and integration tests for Phase 3: Resume upload, validation, storage, and cleanup.
"""

import os
from pathlib import Path
import shutil
from unittest.mock import AsyncMock, MagicMock
import pytest
from telegram.constants import ParseMode

from bot import (
    MAX_RESUME_SIZE_BYTES,
    analyze_command,
    handle_resume_document,
    reset_command,
)
from session_manager import (
    clear_session,
    get_all_sessions,
    get_resumes,
    get_session,
    get_user_temp_dir,
    set_job_description,
)


def setup_function():
    """Clear in-memory sessions and temp files before each test."""
    get_all_sessions().clear()
    temp_root = Path(__file__).resolve().parent.parent / "temp" / "users"
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)


def teardown_function():
    """Clean up temp files after each test."""
    temp_root = Path(__file__).resolve().parent.parent / "temp" / "users"
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)


def make_mock_document_update(
    user_id=101,
    file_name="resume.pdf",
    mime_type="application/pdf",
    file_size=1024,
    file_id="file_123",
    valid_pdf_content=True,
):
    """Helper to create a mock Telegram Update with document and file downloader."""
    update = MagicMock()
    user = MagicMock()
    user.id = user_id
    user.first_name = "Alice"
    update.effective_user = user

    document = MagicMock()
    document.file_name = file_name
    document.mime_type = mime_type
    document.file_size = file_size
    document.file_id = file_id

    message = MagicMock()
    message.document = document
    message.reply_text = AsyncMock()
    update.message = message

    context = MagicMock()
    bot = MagicMock()

    async def mock_download_to_drive(custom_path):
        os.makedirs(os.path.dirname(custom_path), exist_ok=True)
        with open(custom_path, "wb") as f:
            if valid_pdf_content:
                f.write(b"%PDF-1.4 mock binary content for resume")
            else:
                f.write(b"NOT_A_VALID_PDF_HEADER")

    mock_file = MagicMock()
    mock_file.download_to_drive = AsyncMock(side_effect=mock_download_to_drive)
    bot.get_file = AsyncMock(return_value=mock_file)
    context.bot = bot

    return update, context


@pytest.mark.asyncio
async def test_upload_single_pdf():
    """Test uploading one valid PDF resume after setting a Job Description."""
    user_id = 101
    set_job_description(user_id, "Senior Python Developer with FastAPI skills.")

    update, context = make_mock_document_update(
        user_id=user_id,
        file_name="Alice_Resume.pdf",
        file_id="tg_alice_1",
    )

    await handle_resume_document(update, context)

    resumes = get_resumes(user_id)
    assert len(resumes) == 1
    assert resumes[0]["original_filename"] == "Alice_Resume.pdf"
    assert resumes[0]["file_id"] == "tg_alice_1"
    assert Path(resumes[0]["local_path"]).exists()

    update.message.reply_text.assert_awaited_once()
    reply, _ = update.message.reply_text.call_args
    assert "Resume 1 received" in reply[0]
    assert "Alice_Resume.pdf" in reply[0]


@pytest.mark.asyncio
async def test_upload_multiple_pdfs_preserves_order():
    """Test uploading multiple PDFs and verifying sequential ordering and counts."""
    user_id = 102
    set_job_description(user_id, "Fullstack Engineer with React & Node.")

    filenames = ["Resume_A.pdf", "Resume_B.pdf", "Resume_C.pdf"]
    for i, fname in enumerate(filenames, start=1):
        update, context = make_mock_document_update(
            user_id=user_id,
            file_name=fname,
            file_id=f"file_{i}",
        )
        await handle_resume_document(update, context)

    resumes = get_resumes(user_id)
    assert len(resumes) == 3
    assert [r["original_filename"] for r in resumes] == filenames
    for r in resumes:
        assert Path(r["local_path"]).exists()


@pytest.mark.asyncio
async def test_upload_five_pdfs():
    """Test uploading five PDF resumes."""
    user_id = 103
    set_job_description(user_id, "Data Engineer with Spark and Airflow.")

    for i in range(1, 6):
        update, context = make_mock_document_update(
            user_id=user_id,
            file_name=f"Candidate_{i}.pdf",
            file_id=f"file_id_{i}",
        )
        await handle_resume_document(update, context)

    resumes = get_resumes(user_id)
    assert len(resumes) == 5
    for i, r in enumerate(resumes, start=1):
        assert r["original_filename"] == f"Candidate_{i}.pdf"
        assert Path(r["local_path"]).exists()


@pytest.mark.asyncio
async def test_reject_non_pdf_files():
    """Test rejecting unsupported files (e.g., .rtf, .txt, .png)."""
    user_id = 104
    set_job_description(user_id, "DevOps Engineer with Kubernetes.")

    bad_files = [
        ("resume.rtf", "application/rtf"),
        ("notes.txt", "text/plain"),
        ("picture.png", "image/png"),
    ]

    for fname, mime in bad_files:
        update, context = make_mock_document_update(
            user_id=user_id,
            file_name=fname,
            mime_type=mime,
        )
        await handle_resume_document(update, context)
        reply, _ = update.message.reply_text.call_args
        assert "Unsupported file format" in reply[0]

    assert len(get_resumes(user_id)) == 0


@pytest.mark.asyncio
async def test_reject_corrupted_pdf_file():
    """Test that a file with non-PDF header bytes is rejected and removed from disk."""
    user_id = 105
    set_job_description(user_id, "QA Automation Engineer.")

    update, context = make_mock_document_update(
        user_id=user_id,
        file_name="corrupt.pdf",
        valid_pdf_content=False,
    )

    await handle_resume_document(update, context)

    assert len(get_resumes(user_id)) == 0
    reply, _ = update.message.reply_text.call_args
    assert "Corrupted or invalid PDF" in reply[0]


@pytest.mark.asyncio
async def test_reject_oversized_file():
    """Test that files exceeding MAX_RESUME_SIZE_BYTES (15 MB) are rejected."""
    user_id = 106
    set_job_description(user_id, "Cloud Architect.")

    update, context = make_mock_document_update(
        user_id=user_id,
        file_name="large_portfolio.pdf",
        file_size=MAX_RESUME_SIZE_BYTES + 1024,
    )

    await handle_resume_document(update, context)

    assert len(get_resumes(user_id)) == 0
    reply, _ = update.message.reply_text.call_args
    assert "File is too large" in reply[0]


@pytest.mark.asyncio
async def test_reject_empty_zero_byte_file():
    """Test that 0-byte files are rejected."""
    user_id = 107
    set_job_description(user_id, "Software Tester.")

    update, context = make_mock_document_update(
        user_id=user_id,
        file_name="empty.pdf",
        file_size=0,
    )

    await handle_resume_document(update, context)

    assert len(get_resumes(user_id)) == 0
    reply, _ = update.message.reply_text.call_args
    assert "File is empty" in reply[0]


@pytest.mark.asyncio
async def test_duplicate_filenames_handled_safely():
    """Test that uploading multiple resumes with the same filename does not overwrite local files."""
    user_id = 108
    set_job_description(user_id, "Backend Developer.")

    update1, context1 = make_mock_document_update(
        user_id=user_id,
        file_name="resume.pdf",
        file_id="file_1",
    )
    await handle_resume_document(update1, context1)

    update2, context2 = make_mock_document_update(
        user_id=user_id,
        file_name="resume.pdf",
        file_id="file_2",
    )
    await handle_resume_document(update2, context2)

    resumes = get_resumes(user_id)
    assert len(resumes) == 2
    # Local paths must be distinct
    path1 = Path(resumes[0]["local_path"])
    path2 = Path(resumes[1]["local_path"])
    assert path1 != path2
    assert path1.exists()
    assert path2.exists()


@pytest.mark.asyncio
async def test_upload_before_jd_is_rejected():
    """Test that attempting to upload a resume without submitting a JD first is rejected."""
    user_id = 109
    # No JD set

    update, context = make_mock_document_update(user_id=user_id, file_name="early_resume.pdf")
    await handle_resume_document(update, context)

    assert len(get_resumes(user_id)) == 0
    reply, _ = update.message.reply_text.call_args
    assert "Please send the Job Description first" in reply[0]


@pytest.mark.asyncio
async def test_reset_deletes_downloaded_files():
    """Test that /reset removes downloaded PDF files from disk."""
    user_id = 110
    set_job_description(user_id, "Site Reliability Engineer.")

    update_doc, context_doc = make_mock_document_update(user_id=user_id, file_name="sre.pdf")
    await handle_resume_document(update_doc, context_doc)

    resumes = get_resumes(user_id)
    assert len(resumes) == 1
    file_path = Path(resumes[0]["local_path"])
    assert file_path.exists()

    # Call /reset
    update_reset = MagicMock()
    update_reset.effective_user = MagicMock(id=user_id)
    update_reset.message = MagicMock(reply_text=AsyncMock())
    context_reset = MagicMock()

    await reset_command(update_reset, context_reset)

    # Verify session is empty AND file is wiped from disk
    assert len(get_resumes(user_id)) == 0
    assert not file_path.exists()


@pytest.mark.asyncio
async def test_multi_user_isolation_for_resumes():
    """Verify that User A cannot see or access User B's files and resumes."""
    user_a = 201
    user_b = 202

    set_job_description(user_a, "Job Description for User A.")
    set_job_description(user_b, "Job Description for User B.")

    update_a, context_a = make_mock_document_update(user_id=user_a, file_name="A_resume.pdf")
    update_b, context_b = make_mock_document_update(user_id=user_b, file_name="B_resume.pdf")

    await handle_resume_document(update_a, context_a)
    await handle_resume_document(update_b, context_b)

    resumes_a = get_resumes(user_a)
    resumes_b = get_resumes(user_b)

    assert len(resumes_a) == 1
    assert len(resumes_b) == 1
    assert resumes_a[0]["original_filename"] == "A_resume.pdf"
    assert resumes_b[0]["original_filename"] == "B_resume.pdf"
    assert resumes_a[0]["local_path"] != resumes_b[0]["local_path"]
    assert str(user_a) in resumes_a[0]["local_path"]
    assert str(user_b) in resumes_b[0]["local_path"]


@pytest.mark.asyncio
async def test_analyze_command():
    """Test /analyze command response before and after uploading resumes."""
    user_id = 301

    # Case 1: No JD set
    update, context = make_mock_document_update(user_id=user_id)
    await analyze_command(update, context)
    reply, _ = update.message.reply_text.call_args
    assert "Job Description missing" in reply[0]

    # Case 2: JD set, but 0 resumes
    set_job_description(user_id, "Backend Python Engineer with FastAPI.")
    update.message.reply_text.reset_mock()
    await analyze_command(update, context)
    reply, _ = update.message.reply_text.call_args
    assert "No resumes uploaded" in reply[0]

    # Case 3: Resumes uploaded
    await handle_resume_document(update, context)
    update.message.reply_text.reset_mock()

    mock_analysis = {
        "candidate_name": "Test Candidate",
        "ats_score": 80,
        "job_match_percentage": 75,
        "matching_skills": ["Python"],
        "missing_skills": [],
        "partial_skills": [],
        "skill_breakdown": [],
        "recommended_learning": [],
        "recommendation": "Good candidate.",
    }

    from unittest.mock import patch
    with patch("bot.analyze_resume_against_jd", return_value=mock_analysis):
        await analyze_command(update, context)

    reply, _ = update.message.reply_text.call_args
    assert "CANDIDATE ANALYSIS" in reply[0]
    assert "Test Candidate" in reply[0]
