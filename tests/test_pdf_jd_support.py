"""
Tests for PDF Job Description uploads and text extraction.
"""

import os
from pathlib import Path
import shutil
from unittest.mock import AsyncMock, MagicMock
import pytest
from pypdf import PdfWriter

from bot import (
    handle_document_before_jd,
    handle_resume_document,
)
from gemini_service import (
    GeminiInputError,
    extract_text_from_pdf,
)
from session_manager import (
    clear_session,
    get_all_sessions,
    get_job_description,
    get_resumes,
    get_session,
)


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean sessions and temp files before each test."""
    get_all_sessions().clear()
    temp_root = Path(__file__).resolve().parent.parent / "temp" / "users"
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    yield
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)


def create_sample_pdf(file_path: Path, text: str) -> Path:
    """Helper to generate a real, readable PDF with text."""
    # Create a basic PDF with text using PyPDF or copy sample_resume.pdf
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    # PyPDF add_blank_page creates a blank page; for text, we can use an existing sample or write minimal PDF stream
    # Minimal standard PDF with text stream
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"5 0 obj<</Length " + str(len(text) + 40).encode("latin1") + b">>\n"
        b"stream\n"
        b"BT /F1 12 Tf 50 700 Td (" + text.encode("latin1") + b") Tj ET\n"
        b"endstream\n"
        b"endobj\n"
        b"xref\n"
        b"0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000052 00000 n \n"
        b"0000000101 00000 n \n"
        b"0000000212 00000 n \n"
        b"0000000277 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n"
        b"370\n"
        b"%%EOF\n"
    )
    file_path.write_bytes(pdf_content)
    return file_path


def make_mock_pdf_update(
    user_id: int,
    file_path: Path,
    file_name: str | None = None,
    mime_type: str = "application/pdf",
):
    """Creates a mock Telegram update representing a PDF document upload."""
    update = MagicMock()
    user = MagicMock()
    user.id = user_id
    user.first_name = "David"
    update.effective_user = user

    document = MagicMock()
    document.file_name = file_name or file_path.name
    document.mime_type = mime_type
    document.file_size = file_path.stat().st_size if file_path.exists() else 1024
    document.file_id = f"tg_pdf_jd_{user_id}"

    message = MagicMock()
    message.document = document
    message.reply_text = AsyncMock()
    update.message = message

    context = MagicMock()
    bot = MagicMock()

    async def mock_download_to_drive(custom_path):
        os.makedirs(os.path.dirname(custom_path), exist_ok=True)
        shutil.copyfile(file_path, custom_path)

    mock_file = MagicMock()
    mock_file.download_to_drive = AsyncMock(side_effect=mock_download_to_drive)
    bot.get_file = AsyncMock(return_value=mock_file)
    context.bot = bot

    return update, context


# ==========================================
# 1. Tests for extract_text_from_pdf
# ==========================================

def test_extract_text_from_sample_pdf():
    """Test extracting text from sample_resume.pdf in workspace."""
    sample_path = Path(__file__).resolve().parent.parent / "sample_resume.pdf"
    text = extract_text_from_pdf(sample_path)
    assert "John Doe" in text
    assert "Python Developer" in text


def test_extract_text_from_generated_pdf(tmp_path):
    """Test extracting text from custom generated PDF."""
    pdf_path = tmp_path / "custom.pdf"
    content = "Senior DevOps Engineer with Terraform Kubernetes and CI CD experience"
    create_sample_pdf(pdf_path, content)

    extracted = extract_text_from_pdf(pdf_path)
    assert "Senior DevOps Engineer" in extracted
    assert "Kubernetes" in extracted


def test_extract_text_from_nonexistent_pdf(tmp_path):
    """Test extracting text from non-existent PDF raises GeminiInputError."""
    with pytest.raises(GeminiInputError, match="does not exist"):
        extract_text_from_pdf(tmp_path / "not_found.pdf")


# ==========================================
# 2. Tests for Job Description via PDF
# ==========================================

@pytest.mark.asyncio
async def test_submit_job_description_via_pdf(tmp_path):
    """Test uploading a valid PDF Job Description file sets the JD and advances state."""
    user_id = 801
    jd_pdf = tmp_path / "Job_Description_Backend.pdf"
    content = "Senior Backend Engineer position requiring 5 plus years of Python FastAPI and PostgreSQL."
    create_sample_pdf(jd_pdf, content)

    update, context = make_mock_pdf_update(user_id=user_id, file_path=jd_pdf)
    await handle_resume_document(update, context)

    session = get_session(user_id)
    assert session["job_description"] is not None
    assert "Senior Backend Engineer" in session["job_description"]
    assert session["state"] == "waiting_for_resumes"

    update.message.reply_text.assert_awaited_once()
    reply, _ = update.message.reply_text.call_args
    assert "Job Description received" in reply[0]
    assert "Job_Description_Backend.pdf" in reply[0]


@pytest.mark.asyncio
async def test_submit_short_job_description_via_pdf(tmp_path):
    """Test uploading a PDF JD with fewer than 30 characters is rejected."""
    user_id = 802
    jd_pdf = tmp_path / "Short_JD.pdf"
    create_sample_pdf(jd_pdf, "Python dev")

    update, context = make_mock_pdf_update(user_id=user_id, file_path=jd_pdf)
    await handle_resume_document(update, context)

    session = get_session(user_id)
    assert session["job_description"] is None
    reply, _ = update.message.reply_text.call_args
    assert "too short" in reply[0]


@pytest.mark.asyncio
async def test_corrupted_pdf_job_description(tmp_path):
    """Test that a non-PDF file named .pdf is rejected when uploaded as JD."""
    user_id = 803
    corrupt_pdf = tmp_path / "bad_jd.pdf"
    corrupt_pdf.write_bytes(b"NOT_A_VALID_PDF_HEADER_1234567890123456789012345678901234567890")

    update, context = make_mock_pdf_update(user_id=user_id, file_path=corrupt_pdf)
    await handle_resume_document(update, context)

    session = get_session(user_id)
    assert session["job_description"] is None
    reply, _ = update.message.reply_text.call_args
    assert "Corrupted or invalid PDF" in reply[0]
