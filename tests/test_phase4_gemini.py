"""
Unit and integration tests for Phase 4: Gemini resume and JD analysis service.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from gemini_service import (
    GeminiAPIError,
    GeminiConfigError,
    GeminiInputError,
    GeminiParsingError,
    ResumeAnalysisOutput,
    analyze_resume_against_jd,
)

SAMPLE_RESUME_PATH = Path(__file__).resolve().parent.parent / "sample_resume.pdf"
SAMPLE_JD = (
    "Backend Python Engineer\n"
    "Requirements:\n"
    "- Python, FastAPI\n"
    "- PostgreSQL, Docker\n"
    "- REST APIs\n"
    "Education: BS in Computer Science"
)

VALID_MOCK_GEMINI_RESPONSE = json.dumps({
    "candidate_name": "Jane Smith",
    "ats_score": 85,
    "job_match_percentage": 82,
    "matching_skills": ["Python", "FastAPI", "REST APIs"],
    "missing_skills": [
        {"skill": "Docker", "importance": "important"},
        {"skill": "PostgreSQL", "importance": "critical"},
    ],
    "partial_skills": ["SQL"],
    "skill_breakdown": [
        {"skill": "Python", "match_percentage": 100},
        {"skill": "FastAPI", "match_percentage": 100},
        {"skill": "Docker", "match_percentage": 0},
        {"skill": "PostgreSQL", "match_percentage": 0},
    ],
    "recommended_learning": [
        {
            "skill": "PostgreSQL",
            "reason": "Critical database requirement",
            "topics": ["PostgreSQL queries", "Indexing", "Transactions"],
        },
        {
            "skill": "Docker",
            "reason": "Containerization required for microservices",
            "topics": ["Dockerfiles", "Docker Compose"],
        },
    ],
    "recommendation": "Strong candidate with solid Python & FastAPI foundations.",
})


@pytest.fixture(autouse=True)
def ensure_sample_pdf():
    """Ensure a minimal valid sample PDF resume exists for tests."""
    if not SAMPLE_RESUME_PATH.exists():
        pdf_content = (
            b"%PDF-1.4\n"
            b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
            b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >> endobj\n"
            b"xref\n0 4\n0000000000 65535 f\n"
            b"trailer << /Size 4 /Root 1 0 R >>\nstartxref\n100\n%%EOF"
        )
        SAMPLE_RESUME_PATH.write_bytes(pdf_content)


def test_missing_api_key(monkeypatch):
    """Test that missing or placeholder API key raises GeminiConfigError."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(GeminiConfigError, match="GEMINI_API_KEY is missing"):
        analyze_resume_against_jd(SAMPLE_RESUME_PATH, SAMPLE_JD, api_key="")

    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    with pytest.raises(GeminiConfigError, match="GEMINI_API_KEY is missing"):
        analyze_resume_against_jd(SAMPLE_RESUME_PATH, SAMPLE_JD, api_key="your_gemini_api_key_here")


def test_empty_job_description():
    """Test that empty or whitespace-only JD raises GeminiInputError."""
    with pytest.raises(GeminiInputError, match="Job Description cannot be empty"):
        analyze_resume_against_jd(SAMPLE_RESUME_PATH, "   \n   ", api_key="test_key")


def test_invalid_resume_missing_file():
    """Test that non-existent resume file raises GeminiInputError."""
    non_existent = Path("non_existent_resume_9999.pdf")
    with pytest.raises(GeminiInputError, match="does not exist"):
        analyze_resume_against_jd(non_existent, SAMPLE_JD, api_key="test_key")


def test_invalid_resume_empty_file(tmp_path):
    """Test that 0-byte file raises GeminiInputError."""
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    with pytest.raises(GeminiInputError, match="Resume file is empty"):
        analyze_resume_against_jd(empty_file, SAMPLE_JD, api_key="test_key")


def test_invalid_resume_corrupt_header(tmp_path):
    """Test that file without %PDF magic header raises GeminiInputError."""
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"NOT_A_VALID_PDF_DOCUMENT")

    with pytest.raises(GeminiInputError, match="missing %PDF header"):
        analyze_resume_against_jd(corrupt_file, SAMPLE_JD, api_key="test_key")


@patch("google.genai.Client")
def test_successful_gemini_request(mock_client_cls):
    """Test successful Gemini request and schema verification."""
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = VALID_MOCK_GEMINI_RESPONSE
    mock_instance.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_instance

    result = analyze_resume_against_jd(SAMPLE_RESUME_PATH, SAMPLE_JD, api_key="test_key")

    # Verify return dictionary structure and values
    assert result["candidate_name"] == "Jane Smith"
    assert result["ats_score"] == 85
    assert result["job_match_percentage"] == 82
    assert "Python" in result["matching_skills"]
    assert len(result["missing_skills"]) == 2
    assert len(result["skill_breakdown"]) == 4
    assert len(result["recommended_learning"]) == 2
    assert "Strong candidate" in result["recommendation"]


@patch("google.genai.Client")
def test_invalid_api_key(mock_client_cls):
    """Test handling of invalid API key error."""
    mock_instance = MagicMock()
    mock_instance.models.generate_content.side_effect = Exception("API_KEY_INVALID: 401 Unauthorized")
    mock_client_cls.return_value = mock_instance

    with pytest.raises(GeminiConfigError, match="Invalid GEMINI_API_KEY"):
        analyze_resume_against_jd(SAMPLE_RESUME_PATH, SAMPLE_JD, api_key="invalid_key")


@patch("google.genai.Client")
def test_invalid_gemini_json_response(mock_client_cls):
    """Test handling of invalid / non-JSON output from Gemini."""
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is plain text and not valid JSON"
    mock_instance.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_instance

    with pytest.raises(GeminiParsingError, match="Gemini returned invalid JSON"):
        analyze_resume_against_jd(SAMPLE_RESUME_PATH, SAMPLE_JD, api_key="test_key")


@patch("google.genai.Client")
def test_empty_gemini_response(mock_client_cls):
    """Test handling of empty text response from Gemini."""
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = ""
    mock_instance.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_instance

    with pytest.raises(GeminiParsingError, match="empty response"):
        analyze_resume_against_jd(SAMPLE_RESUME_PATH, SAMPLE_JD, api_key="test_key")


@patch("google.genai.Client")
def test_schema_mismatch_missing_keys(mock_client_cls):
    """Test handling when response JSON does not comply with required schema."""
    mock_instance = MagicMock()
    mock_response = MagicMock()
    # Missing candidate_name and scores
    mock_response.text = json.dumps({"incomplete": "data"})
    mock_instance.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_instance

    with pytest.raises(GeminiParsingError, match="Response schema validation failed"):
        analyze_resume_against_jd(SAMPLE_RESUME_PATH, SAMPLE_JD, api_key="test_key")


@patch("google.genai.Client")
def test_api_failure_rate_limit_or_server_error(mock_client_cls):
    """Test handling when Gemini encounters a server error or rate limit."""
    from google.genai.errors import APIError

    mock_instance = MagicMock()
    api_error = APIError(429, {"error": {"message": "ResourceExhausted: Quota exceeded"}})
    mock_instance.models.generate_content.side_effect = api_error
    mock_client_cls.return_value = mock_instance

    with pytest.raises(GeminiAPIError, match="Gemini API call failed"):
        analyze_resume_against_jd(SAMPLE_RESUME_PATH, SAMPLE_JD, api_key="test_key")
