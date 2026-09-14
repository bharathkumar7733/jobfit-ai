"""
Tests for Smart Intake, Document Classification, and Learning Roadmap with Links.

Covers:
1. Rejection of non-JD text messages with explicit "Invalid message" responses.
2. Validation of genuine Job Descriptions.
3. Classification of uploaded documents into JDs vs Resumes.
4. Handling multiple JDs (2 or 3) with 1 Resume and evaluating across roles.
5. Formatting and rendering of Multi-JD comparison cards.
6. Formatting of Phased Learning Roadmaps and official documentation/learning links.
"""

from pathlib import Path
import shutil
from unittest.mock import AsyncMock, MagicMock, patch
import docx
from pypdf import PdfWriter
import pytest
from telegram.constants import ParseMode

from bot import (
    analyze_command,
    handle_jd_selection_callback,
    handle_job_description,
    handle_resume_document,
    handle_unsupported_media,
    use_jd_command,
)
from formatter import (
    format_candidate_analysis,
    format_multi_jd_comparison,
)
from gemini_service import (
    classify_document_content,
    is_valid_job_description_text,
)
from session_manager import (
    add_job_description,
    add_resume,
    clear_session,
    get_active_jds_for_analysis,
    get_all_sessions,
    get_job_descriptions,
    get_resumes,
    get_selected_jd_index,
    get_session,
    set_selected_jd,
)


@pytest.fixture(autouse=True)
def clean_sessions():
    """Reset session state before and after each test."""
    get_all_sessions().clear()
    temp_root = Path(__file__).resolve().parent.parent / "temp" / "users"
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    yield
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)


def make_mock_text_update(user_id: int, text: str):
    """Helper to mock a text update."""
    update = MagicMock()
    user = MagicMock(id=user_id, first_name="Tester")
    update.effective_user = user
    message = MagicMock(text=text, reply_text=AsyncMock())
    update.message = message
    context = MagicMock()
    return update, context


def create_pdf(file_path: Path, text: str) -> Path:
    """Helper to create a valid minimal PDF."""
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


def create_docx(file_path: Path, paragraphs: list[str]) -> Path:
    """Helper to create a valid minimal DOCX."""
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    doc.save(str(file_path))
    return file_path


def make_mock_doc_update(user_id: int, file_path: Path, file_name: str | None = None):
    """Helper to mock a document update."""
    update = MagicMock()
    user = MagicMock(id=user_id, first_name="Alice")
    update.effective_user = user

    doc_meta = MagicMock()
    doc_meta.file_name = file_name or file_path.name
    doc_meta.file_size = file_path.stat().st_size
    doc_meta.file_id = f"file_{user_id}_{doc_meta.file_name}"
    doc_meta.mime_type = "application/pdf" if file_path.suffix.lower() == ".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    message = MagicMock(document=doc_meta, reply_text=AsyncMock())
    update.message = message
    context = MagicMock()

    async def mock_download_to_drive(custom_path):
        shutil.copyfile(file_path, custom_path)

    tg_file = MagicMock()
    tg_file.download_to_drive = AsyncMock(side_effect=mock_download_to_drive)
    context.bot = MagicMock(get_file=AsyncMock(return_value=tg_file))

    return update, context


# ================================================================
# 1. Text Message Validation Tests
# ================================================================

def test_is_valid_job_description_text_helper():
    """Verify is_valid_job_description_text distinguishes real JDs from non-JDs."""
    # Casual messages should be rejected
    assert not is_valid_job_description_text("hi")
    assert not is_valid_job_description_text("hello bot, can you help me?")
    assert not is_valid_job_description_text("what is your name and what can you do for me?")
    assert not is_valid_job_description_text("I love programming in Python and reading tech news.")

    # Resume pasted as text should be rejected
    resume_text = (
        "Alice Johnson\n"
        "Email: alice@example.com | Phone: +1 555-123-4567\n"
        "Education: B.Tech in Computer Science, GPA 3.9\n"
        "Work Experience: Software Engineer at Acme Corp (2021-2024)\n"
        "Skills: Python, Django, PostgreSQL, Docker"
    )
    assert not is_valid_job_description_text(resume_text)

    # Legitimate JDs should pass
    valid_jd = (
        "Senior Backend Engineer\n"
        "We are looking for a Python developer with 4+ years of experience.\n"
        "Requirements:\n"
        "- Strong expertise in Python, FastAPI, and SQL\n"
        "- Experience building and scaling cloud microservices"
    )
    assert is_valid_job_description_text(valid_jd)


@pytest.mark.asyncio
async def test_handle_job_description_rejects_casual_text():
    """Test that casual chat messages are rejected with 'Invalid message'."""
    user_id = 2001
    update, context = make_mock_text_update(user_id, "hello bot, what can you do for me today?")

    await handle_job_description(update, context)

    # Verify no JD was set
    assert len(get_job_descriptions(user_id)) == 0
    update.message.reply_text.assert_awaited_once()
    reply_text, _ = update.message.reply_text.call_args
    assert "Invalid message" in reply_text[0]
    assert "Your message is invalid" in reply_text[0]


@pytest.mark.asyncio
async def test_handle_job_description_accepts_real_jd():
    """Test that a valid job description text is accepted."""
    user_id = 2002
    valid_jd = (
        "Senior Python Engineer Opening\n"
        "Responsibilities:\n"
        "- Build scalable microservices and APIs\n"
        "Requirements:\n"
        "- 5+ years experience in Python and PostgreSQL"
    )
    update, context = make_mock_text_update(user_id, valid_jd)

    await handle_job_description(update, context)

    jds = get_job_descriptions(user_id)
    assert len(jds) == 1
    assert "Senior Python Engineer" in jds[0]["text"]
    reply_text, _ = update.message.reply_text.call_args
    assert "Job Description received" in reply_text[0]


# ================================================================
# 2. Smart Document Intake: 2 or 3 JDs and 1 Resume
# ================================================================

def test_classify_document_content_accuracy():
    """Verify document classifier distinguishes JDs from Resumes."""
    jd_content = (
        "Job Title: Senior Backend Developer\n"
        "About the role: We are looking for an experienced developer.\n"
        "Key Responsibilities:\n"
        "- Architect distributed systems\n"
        "Requirements:\n"
        "- 5+ years Python and Docker\n"
        "Benefits and Perks: Health insurance, 401k"
    )
    assert classify_document_content(jd_content, "Backend_Engineer.pdf") == "job_description"

    resume_content = (
        "John Doe\n"
        "john.doe@example.com | +1 555-987-6543 | linkedin.com/in/johndoe\n"
        "Education: Bachelor of Science in Software Engineering\n"
        "Work Experience: Software Developer at Tech Solutions (2020-2023)\n"
        "Technical Skills: Python, Django, AWS, Git"
    )
    assert classify_document_content(resume_content, "John_Doe.pdf") == "resume"


@pytest.mark.asyncio
async def test_upload_two_jds_and_one_resume(tmp_path):
    """Test user uploading 2 JDs and 1 resume: bot divides what is JD and what is resume."""
    user_id = 2003

    # JD 1 (PDF)
    jd1_file = tmp_path / "Backend_JD.pdf"
    create_pdf(
        jd1_file,
        "Job Description: Backend Engineer\n"
        "We are looking for a Python developer.\n"
        "Requirements: 3+ years experience in Python and Docker."
    )
    up1, ctx1 = make_mock_doc_update(user_id, jd1_file)
    await handle_resume_document(up1, ctx1)

    # JD 2 (DOCX)
    jd2_file = tmp_path / "Frontend_JD.docx"
    create_docx(
        jd2_file,
        [
            "Job Description: Frontend Engineer",
            "We are seeking a React specialist.",
            "Requirements: 3+ years experience with React, TypeScript, and CSS.",
            "Responsibilities: Build engaging user interfaces."
        ]
    )
    up2, ctx2 = make_mock_doc_update(user_id, jd2_file)
    await handle_resume_document(up2, ctx2)

    # Resume (PDF)
    resume_file = tmp_path / "Alice_Resume.pdf"
    create_pdf(
        resume_file,
        "Alice Johnson\n"
        "alice@example.com | 123-456-7890\n"
        "Education: B.Tech Computer Science\n"
        "Work Experience: Full Stack Engineer at Alpha Corp\n"
        "Projects: Built React frontend and Python backend microservices"
    )
    up3, ctx3 = make_mock_doc_update(user_id, resume_file)
    await handle_resume_document(up3, ctx3)

    # Verify session state: exactly 2 JDs and 1 Resume
    jds = get_job_descriptions(user_id)
    resumes = get_resumes(user_id)

    assert len(jds) == 2
    assert "Backend_JD.pdf" in jds[0]["filename"]
    assert "Frontend_JD.docx" in jds[1]["filename"]

    assert len(resumes) == 1
    assert resumes[0]["original_filename"] == "Alice_Resume.pdf"


# ================================================================
# 3. Multi-JD Analysis Flow & Comparison Card
# ================================================================

@pytest.mark.asyncio
async def test_analyze_with_multiple_jds_and_one_resume(tmp_path):
    """Test /analyze evaluates 1 resume across multiple JDs and sends comparison card."""
    user_id = 2004

    add_job_description(
        user_id,
        "Senior Backend Engineer. Requirements: Python, FastAPI, Docker.",
        "Backend_Role.pdf"
    )
    add_job_description(
        user_id,
        "Frontend Engineer. Requirements: React, TypeScript, Tailwind.",
        "Frontend_Role.docx"
    )

    resume_path = tmp_path / "candidate_alice.pdf"
    create_pdf(resume_path, "Alice Johnson - Python & Docker Developer")

    add_resume(
        user_id,
        {
            "original_filename": "Alice_Resume.pdf",
            "local_path": str(resume_path),
            "file_id": "tg_alice_1",
            "file_size": 1024,
            "file_type": "pdf",
        }
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

    # Mock analyze_resume_against_jd to return different scores for the two roles
    def mock_analyze(resume_path, job_description):
        if "Backend" in job_description:
            return {
                "candidate_name": "Alice Johnson",
                "ats_score": 90,
                "job_match_percentage": 92,
                "matching_skills": ["Python", "Docker"],
                "missing_skills": [{"skill": "FastAPI", "importance": "important"}],
                "partial_skills": [],
                "skill_breakdown": [{"skill": "Python", "match_percentage": 100}],
                "recommended_learning": [
                    {
                        "skill": "FastAPI",
                        "reason": "Backend API framework",
                        "topics": ["Pydantic v2", "Dependency Injection"],
                        "resource_url": "https://fastapi.tiangolo.com",
                    }
                ],
                "learning_roadmap": [
                    {
                        "step_number": 1,
                        "phase": "Phase 1: API Foundations",
                        "skill_focus": "FastAPI",
                        "duration": "1-2 Weeks",
                        "key_topics": ["Routing", "Request Validation"],
                        "learning_resource_url": "https://fastapi.tiangolo.com/tutorial/",
                    }
                ],
                "recommendation": "Strong backend fit.",
            }
        else:
            return {
                "candidate_name": "Alice Johnson",
                "ats_score": 45,
                "job_match_percentage": 40,
                "matching_skills": [],
                "missing_skills": [{"skill": "React", "importance": "critical"}],
                "partial_skills": [],
                "skill_breakdown": [{"skill": "React", "match_percentage": 0}],
                "recommended_learning": [],
                "learning_roadmap": [],
                "recommendation": "Weak frontend fit.",
            }

    with patch("bot.analyze_resume_against_jd", side_effect=mock_analyze):
        await analyze_command(update, context)

    # Verify status message was edited upon completion
    status_msg.edit_text.assert_awaited_once()
    edit_args, _ = status_msg.edit_text.call_args
    assert "Screening complete" in edit_args[0]

    # Verify replies: multi-job comparison card + individual role cards
    all_replies = [call[0][0] for call in update.message.reply_text.call_args_list]
    joined_replies = "\n".join(all_replies)

    assert "JOB FIT COMPARISON FOR CANDIDATE" in joined_replies
    assert "Backend_Role.pdf" in joined_replies
    assert "Best Fit!" in joined_replies
    assert "Frontend_Role.docx" in joined_replies
    assert "Target Role:" in joined_replies
    assert "https://fastapi.tiangolo.com" in joined_replies


# ================================================================
# 4. Learning Roadmap and Authoritative Links Formatter Tests
# ================================================================

def test_learning_roadmap_and_links_formatting():
    """Test format_candidate_analysis renders clickable links and structured roadmap."""
    analysis = {
        "candidate_name": "Bharath Kumar",
        "ats_score": 85,
        "job_match_percentage": 82,
        "matching_skills": ["Python", "FastAPI", "SQL"],
        "partial_skills": ["Docker"],
        "missing_skills": [
            {"skill": "Kubernetes", "importance": "critical"},
            {"skill": "Redis", "importance": "important"},
        ],
        "skill_breakdown": [
            {"skill": "Python", "match_percentage": 100},
            {"skill": "Kubernetes", "match_percentage": 20},
        ],
        "recommended_learning": [
            {
                "skill": "Kubernetes",
                "reason": "Container orchestration for production",
                "topics": ["Pods, Services, Deployments", "ConfigMaps & Secrets"],
                "resource_url": "https://kubernetes.io/docs/home/",
            }
        ],
        "learning_roadmap": [
            {
                "step_number": 1,
                "phase": "Phase 1: Foundations",
                "skill_focus": "Docker & Containerization",
                "duration": "1 Week",
                "key_topics": ["Dockerfiles", "Multi-stage builds"],
                "learning_resource_url": "https://docs.docker.com",
            },
            {
                "step_number": 2,
                "phase": "Phase 2: Core Orchestration",
                "skill_focus": "Kubernetes Architecture",
                "duration": "2-3 Weeks",
                "key_topics": ["Cluster setup", "Deployments & Ingress"],
                "learning_resource_url": "https://kubernetes.io/docs/tutorials/",
            },
        ],
        "recommendation": "Great technical foundation. Upskilling in container orchestration will make candidate interview-ready.",
    }

    card = format_candidate_analysis(
        analysis=analysis,
        resume_filename="Bharath_Resume.docx",
        jd_title="Cloud Backend Lead.docx",
    )

    # Assertions
    assert "Bharath Kumar" in card
    assert "Cloud Backend Lead.docx" in card
    assert "Target Role:" in card
    assert "85/100" in card
    assert "82%" in card

    # Authoritative links assertions
    assert 'href="https://kubernetes.io/docs/home/"' in card
    assert "Official Documentation / Learning Link" in card

    # Roadmap assertions
    assert "🗺️ <b>Learning Roadmap:</b>" in card
    assert "Phase 1: Foundations" in card
    assert "Docker &amp; Containerization" in card
    assert 'href="https://docs.docker.com"' in card
    assert "Phase 2: Core Orchestration" in card
    assert 'href="https://kubernetes.io/docs/tutorials/"' in card


def test_format_multi_jd_comparison():
    """Test format_multi_jd_comparison renders accurate ranking badges."""
    evals = [
        {"jd_title": "AI Backend Engineer", "job_match_percentage": 94, "ats_score": 90},
        {"jd_title": "Full Stack Engineer", "job_match_percentage": 76, "ats_score": 80},
        {"jd_title": "Mobile Developer", "job_match_percentage": 42, "ats_score": 50},
    ]
    card = format_multi_jd_comparison(
        candidate_name="Alice Johnson",
        resume_filename="Alice_Resume.pdf",
        jd_evaluations=evals,
    )

    assert "JOB FIT COMPARISON FOR CANDIDATE" in card
    assert "Alice Johnson" in card
    assert "Alice_Resume.pdf" in card
    assert "AI Backend Engineer" in card
    assert "Best Fit!" in card
    assert "94%" in card
    assert "Full Stack Engineer" in card
    assert "76%" in card
    assert "Mobile Developer" in card
    assert "42%" in card


@pytest.mark.asyncio
async def test_hi_or_invalid_text_in_jd_step_prompts_for_jd():
    """Verify sending 'hi' or invalid text during JD step prompts specifically for Job Description."""
    user_id = 9001
    update, context = make_mock_text_update(user_id, "hi")

    await handle_job_description(update, context)

    update.message.reply_text.assert_awaited_once()
    reply_text, _ = update.message.reply_text.call_args
    assert "Invalid message" in reply_text[0]
    assert "Your message is invalid" in reply_text[0]
    assert "Job Description" in reply_text[0]


@pytest.mark.asyncio
async def test_hi_or_invalid_text_in_resume_step_prompts_for_resumes():
    """Verify sending 'hi' or invalid text during Resume step prompts specifically for Resumes."""
    user_id = 9002
    # Set JD first so the session enters waiting_for_resumes
    add_job_description(user_id, "Senior Backend Engineer with Python, FastAPI, and Postgres.", "JD.pdf")

    update, context = make_mock_text_update(user_id, "hi")
    await handle_job_description(update, context)

    update.message.reply_text.assert_awaited_once()
    reply_text, _ = update.message.reply_text.call_args
    assert "Invalid message" in reply_text[0]
    assert "Your message is invalid" in reply_text[0]
    assert "already set" in reply_text[0]
    assert "resumes" in reply_text[0].lower()


@pytest.mark.asyncio
async def test_unsupported_media_in_jd_step_and_resume_step():
    """Verify unsupported media messages (photos, stickers, audio) prompt correctly based on step."""
    # 1. JD step
    user_id = 9003
    update_jd, context_jd = make_mock_text_update(user_id, "")
    await handle_unsupported_media(update_jd, context_jd)
    reply_jd, _ = update_jd.message.reply_text.call_args
    assert "Invalid message" in reply_jd[0]
    assert "Job Description" in reply_jd[0]

    # 2. Resume step
    add_job_description(user_id, "Senior Backend Engineer with Python, FastAPI, and Postgres.", "JD.pdf")
    update_res, context_res = make_mock_text_update(user_id, "")
    await handle_unsupported_media(update_res, context_res)
    reply_res, _ = update_res.message.reply_text.call_args
    assert "Invalid message" in reply_res[0]
    assert "resumes" in reply_res[0].lower()


@pytest.mark.asyncio
async def test_direct_resume_upload_before_jd_asks_for_jd_because_resume_uploaded(tmp_path):
    """Verify that uploading a resume directly before JD prompts to give JD because user uploaded a resume."""
    user_id = 9010
    resume_file = tmp_path / "Candidate_Resume.pdf"
    create_pdf(resume_file, "Alice Johnson - Software Engineer - Python and AWS experience")

    update, context = make_mock_doc_update(user_id, resume_file)
    await handle_resume_document(update, context)

    update.message.reply_text.assert_awaited_once()
    reply_text, _ = update.message.reply_text.call_args
    assert "send the Job Description first" in reply_text[0]
    assert "You uploaded a resume" in reply_text[0]
    assert "give me the" in reply_text[0]
    assert "because you uploaded a resume" in reply_text[0]
    assert len(get_resumes(user_id)) == 0


@pytest.mark.asyncio
async def test_multiple_jds_upload_identifies_and_prompts_with_buttons(tmp_path):
    """Verify that uploading multiple JD PDFs prompts the user to select which JD to use."""
    user_id = 9011
    # 1. First JD
    jd1_file = tmp_path / "Backend_JD.pdf"
    create_pdf(jd1_file, "Backend Engineer Opening. Requirements: 5+ years Python, PostgreSQL, Docker.")
    update1, context1 = make_mock_doc_update(user_id, jd1_file)
    await handle_resume_document(update1, context1)

    reply1, _ = update1.message.reply_text.call_args
    assert "Job Description received" in reply1[0]
    assert len(get_job_descriptions(user_id)) == 1

    # 2. Second JD
    jd2_file = tmp_path / "DevOps_JD.pdf"
    create_pdf(jd2_file, "DevOps Engineer Role. Requirements: Kubernetes, Terraform, AWS CI/CD pipelines.")
    update2, context2 = make_mock_doc_update(user_id, jd2_file)
    await handle_resume_document(update2, context2)

    assert len(get_job_descriptions(user_id)) == 2
    reply2, kwargs2 = update2.message.reply_text.call_args
    assert "Multiple Job Descriptions Detected (2)" in reply2[0]
    assert "Backend_JD.pdf" in reply2[0]
    assert "DevOps_JD.pdf" in reply2[0]
    assert "Which Job Description should I use for evaluation?" in reply2[0]
    assert "reply_markup" in kwargs2
    # Verify buttons
    markup = kwargs2["reply_markup"]
    assert len(markup.inline_keyboard) == 3  # JD1, JD2, Compare All


@pytest.mark.asyncio
async def test_select_jd_via_text_reply():
    """Verify selecting which JD to use by replying with '1', '2', or 'all'."""
    user_id = 9012
    add_job_description(user_id, "Python Developer JD", "Python_Role.pdf")
    add_job_description(user_id, "Java Developer JD", "Java_Role.pdf")

    # User replies with "1"
    update1, context1 = make_mock_text_update(user_id, "1")
    await handle_job_description(update1, context1)
    reply1, _ = update1.message.reply_text.call_args
    assert "Selected Job Description:" in reply1[0]
    assert "Python_Role.pdf" in reply1[0]
    assert get_selected_jd_index(user_id) == 0

    # User replies with "2"
    update2, context2 = make_mock_text_update(user_id, "2")
    await handle_job_description(update2, context2)
    reply2, _ = update2.message.reply_text.call_args
    assert "Selected Job Description:" in reply2[0]
    assert "Java_Role.pdf" in reply2[0]
    assert get_selected_jd_index(user_id) == 1

    # User replies with "all"
    update_all, context_all = make_mock_text_update(user_id, "all")
    await handle_job_description(update_all, context_all)
    reply_all, _ = update_all.message.reply_text.call_args
    assert "Compare Across All Job Descriptions" in reply_all[0]
    assert get_selected_jd_index(user_id) == "all"


@pytest.mark.asyncio
async def test_select_jd_via_callback_query():
    """Verify interactive inline button click selects the JD."""
    user_id = 9013
    add_job_description(user_id, "Frontend JD", "Frontend.pdf")
    add_job_description(user_id, "Backend JD", "Backend.pdf")

    # Mock callback query for select_jd_1 (Backend)
    update = MagicMock()
    query = MagicMock()
    query.data = "select_jd_1"
    query.from_user = MagicMock(id=user_id)
    query.answer = AsyncMock()
    query.message = MagicMock(reply_text=AsyncMock())
    update.callback_query = query
    context = MagicMock()

    await handle_jd_selection_callback(update, context)

    query.answer.assert_awaited_once()
    assert get_selected_jd_index(user_id) == 1
    reply, _ = query.message.reply_text.call_args
    assert "Selected Job Description:" in reply[0]
    assert "Backend.pdf" in reply[0]


@pytest.mark.asyncio
async def test_use_jd_command():
    """Verify /use_jd command lists JDs and allows setting choice."""
    user_id = 9014
    add_job_description(user_id, "Role A JD", "Role_A.pdf")
    add_job_description(user_id, "Role B JD", "Role_B.pdf")

    # /use_jd without args lists available JDs
    update1, context1 = make_mock_text_update(user_id, "/use_jd")
    context1.args = []
    await use_jd_command(update1, context1)
    reply1, kwargs1 = update1.message.reply_text.call_args
    assert "Available Job Descriptions (2)" in reply1[0]
    assert "Role_A.pdf" in reply1[0]
    assert "Role_B.pdf" in reply1[0]
    assert "reply_markup" in kwargs1

    # /use_jd 1 sets Role_A
    update2, context2 = make_mock_text_update(user_id, "/use_jd 1")
    context2.args = ["1"]
    await use_jd_command(update2, context2)
    assert get_selected_jd_index(user_id) == 0
    reply2, _ = update2.message.reply_text.call_args
    assert "Role_A.pdf" in reply2[0]


@pytest.mark.asyncio
async def test_analyze_with_selected_jd_evaluates_single_jd(tmp_path):
    """Verify that when 1 JD is selected among multiple JDs, /analyze evaluates only against that JD."""
    user_id = 9015
    add_job_description(user_id, "Backend Python JD", "Backend.pdf")
    add_job_description(user_id, "Frontend React JD", "Frontend.pdf")

    # Select Backend JD (index 0)
    set_selected_jd(user_id, 0)
    assert len(get_active_jds_for_analysis(user_id)) == 1

    resume_file = tmp_path / "Alice_Resume.pdf"
    create_pdf(resume_file, "Alice Johnson - Python expert")
    add_resume(user_id, {
        "original_filename": "Alice_Resume.pdf",
        "local_path": str(resume_file),
        "file_id": "file_123",
        "file_size": 1024,
        "file_type": "pdf",
    })

    update, context = make_mock_text_update(user_id, "/analyze")
    status_msg = MagicMock(edit_text=AsyncMock())
    update.message.reply_text = AsyncMock(return_value=status_msg)

    evaluated_jds = []

    def mock_analyze(resume_path, job_description):
        evaluated_jds.append(job_description)
        return {
            "candidate_name": "Alice Johnson",
            "ats_score": 90,
            "job_match_percentage": 92,
            "matching_skills": ["Python"],
            "missing_skills": [],
            "partial_skills": [],
            "skill_breakdown": [],
            "recommended_learning": [],
            "learning_roadmap": [],
            "recommendation": "Strong fit for Backend.",
        }

    with patch("bot.analyze_resume_against_jd", side_effect=mock_analyze):
        await analyze_command(update, context)

    # Must only evaluate once against the selected JD!
    assert len(evaluated_jds) == 1
    assert evaluated_jds[0] == "Backend Python JD"


@pytest.mark.asyncio
async def test_upload_two_jd_docx_files_both_considered_jds_and_prompts_for_selection(tmp_path):
    """Verify that uploading JD_01...docx and JD_02...docx treats both as JDs and asks to choose JD 1 or JD 2."""
    user_id = 9020
    # 1. Create and upload JD 1
    jd1_file = tmp_path / "JD_01_Senior_FullStack_Developer.docx"
    create_docx(jd1_file, [
        "Senior Full Stack Developer",
        "Company: TechCorp",
        "Position Summary: We are looking for an experienced Full Stack Developer.",
        "Required Skills: React, Node.js, Python, PostgreSQL",
        "Key Responsibilities: Design and build scalable web apps.",
    ])
    update1, context1 = make_mock_doc_update(user_id, jd1_file)
    await handle_resume_document(update1, context1)

    assert len(get_job_descriptions(user_id)) == 1
    reply1, _ = update1.message.reply_text.call_args
    assert "Job Description received" in reply1[0]

    # 2. Create and upload JD 2
    jd2_file = tmp_path / "JD_02_Data_Scientist.docx"
    create_docx(jd2_file, [
        "Data Scientist",
        "Company: DataInsights Inc",
        "Position Summary: We are seeking a Data Scientist to develop ML models.",
        "Required Skills: Python, Machine Learning, SQL, Pandas",
        "Key Responsibilities: Develop and train ML models, communicate insights.",
    ])
    update2, context2 = make_mock_doc_update(user_id, jd2_file)
    await handle_resume_document(update2, context2)

    # Must have 2 JDs and 0 resumes!
    assert len(get_job_descriptions(user_id)) == 2
    assert len(get_resumes(user_id)) == 0

    reply2, kwargs2 = update2.message.reply_text.call_args
    assert "Multiple Job Descriptions Detected (2)" in reply2[0]
    assert "JD_01_Senior_FullStack_Developer.docx" in reply2[0]
    assert "JD_02_Data_Scientist.docx" in reply2[0]
    assert "Which Job Description should I use for evaluation?" in reply2[0]
    assert "reply_markup" in kwargs2

    # Verify inline buttons for JD 1, JD 2, and Compare All
    markup = kwargs2["reply_markup"]
    assert len(markup.inline_keyboard) == 3
    assert markup.inline_keyboard[0][0].callback_data == "select_jd_0"
    assert markup.inline_keyboard[1][0].callback_data == "select_jd_1"
    assert markup.inline_keyboard[2][0].callback_data == "select_jd_all"



