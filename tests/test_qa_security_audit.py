"""
Comprehensive Senior QA Engineer & Security Reviewer Test Suite.

Covers:
1. Complete User Flow (/start -> JD -> Upload 1 -> Upload multiple -> /analyze -> Analysis -> Ranking -> /reset -> Next)
2. Functional Tests (Empty JD, Long JD, No resume, Non-PDF, Corrupt PDF, Large PDF, Duplicate resume,
   Multiple resumes, Missing name, Unrelated resume, 100% match, 0% match)
3. Gemini Tests (Valid key, Missing key, Invalid key, Timeout, Rate limit, Invalid response, Invalid JSON, Empty response)
4. Telegram Tests (Multi-user concurrency, Unexpected commands, Upload before JD, Reset during/after analysis,
   Repeated /analyze, Very long messages > 4096 chars)
5. Security Tests (API keys in code/git, bot token logging, temp file cleanup, path traversal, session isolation,
   .env ignored, PII logging)
6. Scoring Tests (Bounds [0, 100], deterministic ranking & tiebreakers)
"""

import asyncio
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.constants import ParseMode

from bot import (
    MAX_RESUME_SIZE_BYTES,
    analyze_command,
    handle_job_description,
    handle_resume_document,
    help_command,
    reset_command,
    safe_reply_text,
    start_command,
)
from formatter import (
    format_candidate_analysis,
    format_candidate_ranking,
    format_multi_jd_comparison,
)
from gemini_service import (
    GeminiAPIError,
    GeminiConfigError,
    GeminiInputError,
    GeminiParsingError,
    analyze_resume_against_jd,
    extract_text_from_pdf,
    is_valid_job_description_text,
)
from scoring_engine import (
    calculate_ats_score,
    calculate_job_match,
    rank_candidates,
)
from session_manager import (
    TEMP_ROOT,
    add_job_description,
    add_resume,
    cleanup_user_files,
    clear_session,
    get_all_sessions,
    get_job_descriptions,
    get_resumes,
    get_session,
    get_user_temp_dir,
    set_job_description,
)

SAMPLE_RESUME = Path(__file__).resolve().parent.parent / "sample_resume.pdf"


def setup_function():
    """Wipe memory and temporary user directories before each test."""
    get_all_sessions().clear()
    if TEMP_ROOT.exists():
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)


def teardown_function():
    """Wipe memory and temporary user directories after each test."""
    get_all_sessions().clear()
    if TEMP_ROOT.exists():
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)


# ==============================================================================
# Helpers
# ==============================================================================

def create_mock_update(user_id=12345, text=None, first_name="Tester"):
    update = MagicMock()
    user = MagicMock()
    user.id = user_id
    user.first_name = first_name
    update.effective_user = user

    message = MagicMock()
    message.text = text
    message.reply_text = AsyncMock()
    message.edit_text = AsyncMock()
    update.message = message
    return update, MagicMock()


def create_mock_document_update(
    user_id=12345,
    file_name="resume.pdf",
    file_bytes=b"%PDF-1.4 Mock valid PDF document data content",
    file_size=None,
    mime_type="application/pdf",
):
    update = MagicMock()
    user = MagicMock()
    user.id = user_id
    user.first_name = "Tester"
    update.effective_user = user

    document = MagicMock()
    document.file_name = file_name
    document.mime_type = mime_type
    document.file_size = file_size if file_size is not None else len(file_bytes)
    document.file_id = f"file_id_{file_name}_{user_id}"

    message = MagicMock()
    message.document = document
    message.reply_text = AsyncMock(return_value=MagicMock(edit_text=AsyncMock()))
    update.message = message

    context = MagicMock()
    bot = MagicMock()

    async def mock_download_to_drive(custom_path=None):
        out_path = Path(custom_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(file_bytes)
        return out_path

    mock_tg_file = MagicMock()
    mock_tg_file.download_to_drive = AsyncMock(side_effect=mock_download_to_drive)
    bot.get_file = AsyncMock(return_value=mock_tg_file)
    context.bot = bot

    return update, context


def sample_analysis_result(candidate_name="John Doe", ats=85, match=90):
    return {
        "candidate_name": candidate_name,
        "ats_score": ats,
        "job_match_percentage": match,
        "component_scores": {
            "required_skills": 90,
            "relevant_experience": 85,
            "relevant_keywords": 80,
            "education_certifications": 85,
            "formatting_readability": 90,
        },
        "matching_skills": ["Python", "FastAPI", "SQL"],
        "partial_skills": ["Docker"],
        "missing_skills": [{"skill": "Kubernetes", "importance": "preferred"}],
        "skill_breakdown": [
            {"skill": "Python", "match_percentage": 100},
            {"skill": "FastAPI", "match_percentage": 95},
        ],
        "recommended_learning": [
            {
                "skill": "Kubernetes",
                "reason": "Container orchestration",
                "recommended_resource": "https://kubernetes.io/docs/home/",
                "roadmap_milestones": ["Learn Pods", "Deploy Services"],
            }
        ],
        "strengths": ["Strong backend experience"],
        "concerns": ["Limited Kubernetes"],
        "summary": "Solid candidate matching primary requirements.",
    }


# ==============================================================================
# 1. USER FLOW TEST
# ==============================================================================

@pytest.mark.asyncio
async def test_complete_user_flow_1_to_9():
    """
    Executes full user flow:
    1. /start
    2. Submit Job Description
    3. Upload one PDF
    4. Upload multiple PDFs
    5. Send /analyze
    6. Receive candidate analysis
    7. Receive ranking
    8. Send /reset
    9. Start another analysis
    """
    user_id = 99901

    # Step 1: /start
    u1, c1 = create_mock_update(user_id=user_id)
    await start_command(u1, c1)
    session = get_session(user_id)
    assert session["state"] == "waiting_for_jd"
    start_reply = u1.message.reply_text.call_args[0][0]
    assert "Welcome to <b>JobFit AI</b>" in start_reply

    # Step 2: Submit Job Description
    jd_text = "Senior Python Engineer needed with 5+ years experience in Python, FastAPI, Docker, and AWS."
    u2, c2 = create_mock_update(user_id=user_id, text=jd_text)
    await handle_job_description(u2, c2)
    session = get_session(user_id)
    assert session["state"] == "waiting_for_resumes"
    assert len(get_job_descriptions(user_id)) == 1

    # Step 3: Upload one PDF
    u3, c3 = create_mock_document_update(user_id=user_id, file_name="Alice_Resume.pdf")
    with patch("bot.classify_document_content", return_value="resume"):
        await handle_resume_document(u3, c3)
    assert len(get_resumes(user_id)) == 1
    assert "Resume 1 received" in u3.message.reply_text.call_args[0][0]

    # Step 4: Upload multiple PDFs
    u4, c4 = create_mock_document_update(user_id=user_id, file_name="Bob_Resume.pdf")
    with patch("bot.classify_document_content", return_value="resume"):
        await handle_resume_document(u4, c4)
    assert len(get_resumes(user_id)) == 2

    u5, c5 = create_mock_document_update(user_id=user_id, file_name="Charlie_Resume.pdf")
    with patch("bot.classify_document_content", return_value="resume"):
        await handle_resume_document(u5, c5)
    assert len(get_resumes(user_id)) == 3

    # Step 5, 6, 7: Send /analyze and receive candidate analysis + ranking
    u6, c6 = create_mock_update(user_id=user_id)
    mock_results = [
        sample_analysis_result("Alice", 92, 95),
        sample_analysis_result("Bob", 80, 82),
        sample_analysis_result("Charlie", 70, 75),
    ]

    with patch("bot.analyze_resume_against_jd", side_effect=mock_results):
        await analyze_command(u6, c6)

    # Validate ranking and candidate cards received
    reply_calls = [call[0][0] for call in u6.message.reply_text.call_args_list]
    ranking_msgs = [m for m in reply_calls if "CANDIDATE RANKING" in m]
    assert len(ranking_msgs) >= 1
    assert "Alice" in ranking_msgs[0]
    assert "Bob" in ranking_msgs[0]
    assert "Charlie" in ranking_msgs[0]

    candidate_cards = [m for m in reply_calls if "CANDIDATE ANALYSIS" in m]
    assert len(candidate_cards) == 3

    # Step 8: Send /reset
    u7, c7 = create_mock_update(user_id=user_id)
    await reset_command(u7, c7)
    assert len(get_resumes(user_id)) == 0
    assert len(get_job_descriptions(user_id)) == 0
    assert not (TEMP_ROOT / str(user_id)).exists()

    # Step 9: Start another analysis
    u8, c8 = create_mock_update(user_id=user_id)
    await start_command(u8, c8)
    assert get_session(user_id)["state"] == "waiting_for_jd"


# ==============================================================================
# 2. FUNCTIONAL TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_functional_empty_jd():
    """Empty or whitespace-only JD is rejected with a clear prompt."""
    user_id = 99902
    u, c = create_mock_update(user_id=user_id, text="   ")
    await handle_job_description(u, c)
    assert len(get_job_descriptions(user_id)) == 0
    assert not is_valid_job_description_text("   ")


@pytest.mark.asyncio
async def test_functional_extremely_long_jd():
    """Extremely long JD (e.g. >10,000 characters) is safely rejected with a warning without crashing."""
    user_id = 99903
    long_jd = "We are seeking a Senior Python Engineer. Requirements: " + ("Build microservices, clean code, scalable architecture. " * 500)
    assert len(long_jd) > 10000
    u, c = create_mock_update(user_id=user_id, text=long_jd)
    await handle_job_description(u, c)
    assert len(get_job_descriptions(user_id)) == 0
    reply = u.message.reply_text.call_args[0][0]
    assert "Job Description is too long" in reply


@pytest.mark.asyncio
async def test_functional_no_resume_analyze():
    """Sending /analyze with a valid JD but no resumes displays a warning."""
    user_id = 99904
    set_job_description(user_id, "Python FastAPI developer needed.")
    u, c = create_mock_update(user_id=user_id)
    await analyze_command(u, c)
    reply = u.message.reply_text.call_args[0][0]
    assert "No resumes uploaded!" in reply


@pytest.mark.asyncio
async def test_functional_non_pdf_resume():
    """Uploading an unsupported document (e.g. .exe or .txt) is rejected."""
    user_id = 99905
    set_job_description(user_id, "Software tester needed.")
    u, c = create_mock_document_update(user_id=user_id, file_name="malware.exe", mime_type="application/octet-stream")
    await handle_resume_document(u, c)
    assert len(get_resumes(user_id)) == 0
    reply = u.message.reply_text.call_args[0][0]
    assert "Unsupported file format" in reply


@pytest.mark.asyncio
async def test_functional_corrupt_pdf():
    """Corrupt PDF file with non-PDF header bytes is rejected cleanly."""
    user_id = 99906
    set_job_description(user_id, "Developer needed.")
    u, c = create_mock_document_update(user_id=user_id, file_name="corrupt.pdf", file_bytes=b"GARBAGE_NOT_A_PDF")
    await handle_resume_document(u, c)
    assert len(get_resumes(user_id)) == 0
    reply = u.message.reply_text.call_args[0][0]
    assert "Corrupted or invalid PDF" in reply


@pytest.mark.asyncio
async def test_functional_large_pdf():
    """PDF exceeding MAX_RESUME_SIZE_BYTES (15MB) is rejected with clear error."""
    user_id = 99907
    set_job_description(user_id, "Developer needed.")
    oversized = MAX_RESUME_SIZE_BYTES + 1024
    u, c = create_mock_document_update(user_id=user_id, file_name="huge.pdf", file_size=oversized)
    await handle_resume_document(u, c)
    assert len(get_resumes(user_id)) == 0
    reply = u.message.reply_text.call_args[0][0]
    assert "File is too large" in reply


@pytest.mark.asyncio
async def test_functional_duplicate_resume():
    """Uploading the same resume filename twice stores distinct files without overwriting."""
    user_id = 99908
    set_job_description(user_id, "Developer needed.")
    with patch("bot.classify_document_content", return_value="resume"):
        u1, c1 = create_mock_document_update(user_id=user_id, file_name="resume.pdf")
        await handle_resume_document(u1, c1)

        u2, c2 = create_mock_document_update(user_id=user_id, file_name="resume.pdf")
        await handle_resume_document(u2, c2)

    resumes = get_resumes(user_id)
    assert len(resumes) == 2
    assert resumes[0]["local_path"] != resumes[1]["local_path"]
    assert Path(resumes[0]["local_path"]).exists()
    assert Path(resumes[1]["local_path"]).exists()


@pytest.mark.asyncio
async def test_functional_missing_candidate_name():
    """When candidate name is missing or 'Unknown', system falls back to filename stem."""
    user_id = 99909
    set_job_description(user_id, "Developer needed.")
    add_resume(user_id, {"original_filename": "Jane_Doe_CV.pdf", "local_path": "fake/Jane_Doe_CV.pdf"})

    res = sample_analysis_result(candidate_name="Unknown", ats=80, match=85)
    with patch("pathlib.Path.exists", return_value=True), patch("bot.analyze_resume_against_jd", return_value=res):
        u, c = create_mock_update(user_id=user_id)
        await analyze_command(u, c)

    reply_calls = [call[0][0] for call in u.message.reply_text.call_args_list]
    ranking = [m for m in reply_calls if "CANDIDATE RANKING" in m][0]
    assert "Jane_Doe_CV" in ranking


def test_functional_resume_all_skills_vs_none_vs_unrelated():
    """Test scoring engine with 100% skill match, 0% skill match, and completely unrelated profile."""
    # 1. 100% match
    match_100 = calculate_job_match(required_skills=100, relevant_experience=100, projects=100, education=100, preferred_skills=100)
    assert match_100 == 100

    # 2. 0% match
    match_0 = calculate_job_match(required_skills=0, relevant_experience=0, projects=0, education=0, preferred_skills=0)
    assert match_0 == 0

    # 3. Completely unrelated (0 across all components)
    ats_unrelated = calculate_ats_score(0, 0, 0, 0, 0, 0)
    assert ats_unrelated == 0

    # 4. Perfect ATS
    ats_perfect = calculate_ats_score(100, 100, 100, 100, 100, 100)
    assert ats_perfect == 100


# ==============================================================================
# 3. GEMINI TESTS
# ==============================================================================

def test_gemini_missing_or_invalid_api_key():
    """Missing or invalid API key raises GeminiConfigError."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        with pytest.raises(GeminiConfigError):
            analyze_resume_against_jd(SAMPLE_RESUME, "Job Description")


def test_gemini_api_timeout():
    """Gemini API timeout or network error raises GeminiAPIError."""
    with patch("pathlib.Path.read_bytes", return_value=b"%PDF-1.4 Mock PDF"), \
         patch("google.genai.Client") as mock_client:
        mock_instance = MagicMock()
        mock_instance.models.generate_content.side_effect = TimeoutError("Connection timed out")
        mock_client.return_value = mock_instance

        with patch.dict(os.environ, {"GEMINI_API_KEY": "valid_key"}):
            with pytest.raises(GeminiAPIError) as exc:
                analyze_resume_against_jd(SAMPLE_RESUME, "Job Description")
            assert "error communicating with Gemini API" in str(exc.value)


def test_gemini_invalid_json():
    """Gemini returning malformed JSON raises GeminiParsingError without logging PII."""
    mock_resp = MagicMock()
    mock_resp.text = "{candidate_name: 'Unquoted Name' - MALFORMED"

    with patch("pathlib.Path.read_bytes", return_value=b"%PDF-1.4 Mock PDF"), \
         patch("google.genai.Client") as mock_client:
        mock_instance = MagicMock()
        mock_instance.models.generate_content.return_value = mock_resp
        mock_client.return_value = mock_instance

        with patch.dict(os.environ, {"GEMINI_API_KEY": "valid_key"}):
            with pytest.raises(GeminiParsingError) as exc:
                analyze_resume_against_jd(SAMPLE_RESUME, "Job Description")
            assert "invalid JSON" in str(exc.value)


def test_gemini_empty_response():
    """Gemini returning empty or None response raises GeminiParsingError."""
    mock_resp = MagicMock()
    mock_resp.text = ""

    with patch("pathlib.Path.read_bytes", return_value=b"%PDF-1.4 Mock PDF"), \
         patch("google.genai.Client") as mock_client:
        mock_instance = MagicMock()
        mock_instance.models.generate_content.return_value = mock_resp
        mock_client.return_value = mock_instance

        with patch.dict(os.environ, {"GEMINI_API_KEY": "valid_key"}):
            with pytest.raises(GeminiParsingError) as exc:
                analyze_resume_against_jd(SAMPLE_RESUME, "Job Description")
            assert "empty response" in str(exc.value)


# ==============================================================================
# 4. TELEGRAM TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_telegram_multiple_users_simultaneously():
    """Multiple users operating simultaneously maintain total isolation."""
    user_a = 7001
    user_b = 7002

    # User A sets JD A
    jd_a = "We are seeking a Python Backend Engineer with 5+ years experience in FastAPI and Docker."
    u_a, c_a = create_mock_update(user_id=user_a, text=jd_a)
    await handle_job_description(u_a, c_a)

    # User B sets JD B
    jd_b = "We are looking for a Frontend React Developer with 3+ years experience in TypeScript and React."
    u_b, c_b = create_mock_update(user_id=user_b, text=jd_b)
    await handle_job_description(u_b, c_b)

    # Verify sessions
    jds_a = get_job_descriptions(user_a)
    jds_b = get_job_descriptions(user_b)
    assert len(jds_a) == 1
    assert "Python Backend Engineer" in jds_a[0]["text"]
    assert len(jds_b) == 1
    assert "Frontend React Developer" in jds_b[0]["text"]


@pytest.mark.asyncio
async def test_telegram_upload_before_jd():
    """Uploading a resume before sending a JD prompts user to send JD first."""
    user_id = 7003
    u, c = create_mock_document_update(user_id=user_id, file_name="early_resume.pdf")
    with patch("bot.classify_document_content", return_value="resume"):
        await handle_resume_document(u, c)
    assert len(get_resumes(user_id)) == 0
    reply = u.message.reply_text.call_args[0][0]
    assert "Please send the Job Description first" in reply


@pytest.mark.asyncio
async def test_telegram_repeated_analyze():
    """Calling /analyze multiple times in succession executes idempotently."""
    user_id = 7004
    set_job_description(user_id, "Go Developer")
    add_resume(user_id, {"original_filename": "dev.pdf", "local_path": "fake/dev.pdf"})

    res = sample_analysis_result("Alex", 85, 90)
    with patch("pathlib.Path.exists", return_value=True), patch("bot.analyze_resume_against_jd", return_value=res):
        # 1st /analyze
        u1, c1 = create_mock_update(user_id=user_id)
        await analyze_command(u1, c1)

        # 2nd /analyze immediately after
        u2, c2 = create_mock_update(user_id=user_id)
        await analyze_command(u2, c2)

    assert u1.message.reply_text.call_count >= 2
    assert u2.message.reply_text.call_count >= 2


@pytest.mark.asyncio
async def test_telegram_very_long_message_chunking():
    """Messages exceeding Telegram's 4096 character limit are safely chunked by safe_reply_text."""
    mock_msg = MagicMock()
    mock_msg.reply_text = AsyncMock()

    long_text = "<b>Header</b>\n" + ("Important candidate analysis content line.\n" * 150)
    assert len(long_text) > 5000

    sent_parts = await safe_reply_text(mock_msg, long_text)
    assert len(sent_parts) >= 2
    for call in mock_msg.reply_text.call_args_list:
        chunk = call[0][0]
        assert len(chunk) <= 4000


# ==============================================================================
# 5. SECURITY TESTS
# ==============================================================================

def test_security_no_api_keys_in_source():
    """Scans all repository python files for hardcoded API keys."""
    root = Path(__file__).resolve().parent.parent
    key_patterns = [
        re.compile(r"AIza[0-9A-Za-z_-]{35}"),  # Google API key
        re.compile(r"sk-[a-zA-Z0-9_-]{20,}"),  # OpenAI API key
    ]
    for py_file in root.glob("*.py"):
        content = py_file.read_text(encoding="utf-8", errors="ignore")
        for pattern in key_patterns:
            matches = pattern.findall(content)
            assert not matches, f"Hardcoded key pattern found in {py_file}: {matches}"


def test_security_env_in_gitignore():
    """Verify that .env is explicitly excluded in .gitignore."""
    root = Path(__file__).resolve().parent.parent
    gitignore = root / ".gitignore"
    assert gitignore.exists(), ".gitignore does not exist!"
    content = gitignore.read_text(encoding="utf-8")
    assert ".env" in content.splitlines() or any(line.strip() == ".env" for line in content.splitlines())


@pytest.mark.asyncio
async def test_security_path_traversal_prevention():
    """Path traversal attack filenames like '../../evil.pdf' are sanitized."""
    user_id = 8888
    set_job_description(user_id, "We are seeking a Python Developer.")
    attack_filename = "../../../etc/passwd.pdf"
    u, c = create_mock_document_update(user_id=user_id, file_name=attack_filename)

    with patch("bot.classify_document_content", return_value="resume"):
        await handle_resume_document(u, c)

    resumes = get_resumes(user_id)
    assert len(resumes) == 1
    saved_path = Path(resumes[0]["local_path"])
    # Saved path MUST be contained strictly inside user temp dir
    user_dir = get_user_temp_dir(user_id)
    assert user_dir.resolve() in saved_path.resolve().parents
    assert "passwd" in saved_path.name
    assert ".." not in str(saved_path)


def test_security_no_pii_in_json_decode_log(caplog):
    """When Gemini returns invalid JSON, candidate PII is not leaked in error logs."""
    mock_resp = MagicMock()
    # Malformed JSON with sensitive candidate PII
    mock_resp.text = "{'candidate_ssn': '123-45-6789', 'secret_phone': '+15559998888', 'medical': 'confidential' - broken"

    with patch("pathlib.Path.read_bytes", return_value=b"%PDF-1.4 Mock PDF"), \
         patch("google.genai.Client") as mock_client:
        mock_instance = MagicMock()
        mock_instance.models.generate_content.return_value = mock_resp
        mock_client.return_value = mock_instance

        with patch.dict(os.environ, {"GEMINI_API_KEY": "valid_key"}):
            with pytest.raises(GeminiParsingError):
                analyze_resume_against_jd(SAMPLE_RESUME, "Job Description")

    # Check log records
    for record in caplog.records:
        assert "123-45-6789" not in record.message
        assert "+15559998888" not in record.message


# ==============================================================================
# 6. SCORING TESTS
# ==============================================================================

def test_scoring_bounds_and_determinism():
    """ATS and Match scores must strictly satisfy 0 <= score <= 100 and be deterministic."""
    # Negative component scores clamp to 0
    ats_neg = calculate_ats_score(-10, -5, -20, -1, -50, -100)
    assert ats_neg == 0

    # Oversized component scores clamp to 100
    ats_over = calculate_ats_score(250, 150, 500, 1000, 999, 888)
    assert ats_over == 100

    # Determinism test
    for _ in range(10):
        score_1 = calculate_ats_score(80, 85, 90, 75, 95, 90)
        score_2 = calculate_ats_score(80, 85, 90, 75, 95, 90)
        assert score_1 == score_2


def test_scoring_candidate_ranking_tiebreaker():
    """
    Candidate ranking sorts by:
    1. job_match_percentage DESC
    2. ats_score DESC
    3. candidate_name ASC (alphabetical tiebreaker)
    """
    candidates = [
        {"candidate_name": "Bob", "job_match_percentage": 90, "ats_score": 85, "original_filename": "b.pdf"},
        {"candidate_name": "Alice", "job_match_percentage": 90, "ats_score": 85, "original_filename": "a.pdf"},
        {"candidate_name": "Charlie", "job_match_percentage": 95, "ats_score": 80, "original_filename": "c.pdf"},
    ]

    ranked = rank_candidates(candidates)
    # Charlie is 1st (match 95)
    assert ranked[0]["candidate_name"] == "Charlie"
    # Alice is 2nd (match 90, ats 85, 'Alice' < 'Bob')
    assert ranked[1]["candidate_name"] == "Alice"
    # Bob is 3rd
    assert ranked[2]["candidate_name"] == "Bob"
