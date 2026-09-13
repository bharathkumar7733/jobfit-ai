"""
JobFit AI - Telegram Bot (Phase 1)

This module initializes the Telegram Bot, handles basic commands (/start, /help, /reset),
and manages per-user session state.
"""

import html
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import InvalidToken, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from formatter import (
    ORDINAL_EMOJIS,
    format_candidate_analysis,
    format_candidate_ranking,
    format_multi_jd_comparison,
)
from gemini_service import (
    GeminiAPIError,
    GeminiConfigError,
    GeminiInputError,
    GeminiServiceError,
    analyze_resume_against_jd,
    classify_document_content,
    extract_text_from_docx,
    extract_text_from_pdf,
    is_valid_job_description_text,
)
from scoring_engine import rank_candidates
from session_manager import (
    add_job_description,
    add_resume,
    clear_session,
    get_active_jds_for_analysis,
    get_job_description,
    get_job_descriptions,
    get_resumes,
    get_selected_jd,
    get_selected_jd_index,
    get_session,
    get_user_temp_dir,
    set_job_description,
    set_selected_jd,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Explicitly anchor .env to the directory containing this script
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command."""
    if not update.effective_user or not update.message:
        return

    user_id = update.effective_user.id
    user_name = html.escape(update.effective_user.first_name or "there")

    # Initialize / refresh user session
    clear_session(user_id)
    session = get_session(user_id)
    session["state"] = "waiting_for_jd"

    welcome_text = (
        f"👋 Hello, <b>{user_name}</b>! Welcome to <b>JobFit AI</b>.\n\n"
        "I am your AI-powered assistant for screening candidate resumes against a Job Description.\n\n"
        "<b>Here is how it works:</b>\n"
        "1. Send me a <b>Job Description</b> (paste text, or upload a <b>.pdf</b> or <b>.docx</b> file).\n"
        "2. Upload one or more <b>PDF or DOCX Resumes</b>.\n"
        "3. Send <code>/analyze</code> to get ATS scores, skill matches, and candidate rankings.\n\n"
        "👉 To begin, please send me the <b>Job Description</b> (text, .pdf, or .docx document)."
    )

    await update.message.reply_text(welcome_text, parse_mode=ParseMode.HTML)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /help command."""
    if not update.message:
        return

    help_text = (
        "🤖 <b>JobFit AI - Help Guide</b>\n\n"
        "JobFit AI evaluates candidates against a job description using predefined criteria:\n"
        "• <b>ATS Score (0-100):</b> Format, experience, keywords, and education.\n"
        "• <b>Job Match %:</b> Technical and role-specific skill alignment.\n"
        "• <b>Skill Analysis:</b> Matching, missing, and partial skills.\n"
        "• <b>Candidate Ranking:</b> Ranks all uploaded candidates.\n\n"
        "<b>Supported Formats:</b>\n"
        "• <b>Job Description:</b> Plain text message, <b>.pdf</b>, or <b>.docx</b> document.\n"
        "• <b>Resumes:</b> <b>.pdf</b> or <b>.docx</b> documents.\n\n"
        "<b>Available Commands:</b>\n"
        "• /start - Start a new screening session\n"
        "• /analyze - Begin candidate resume evaluation\n"
        "• /help - Display this help guide\n"
        "• /reset - Clear your current session data (JD & uploaded resumes)\n\n"
        "Need a fresh start? Send /reset anytime."
    )

    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /reset command."""
    if not update.effective_user or not update.message:
        return

    user_id = update.effective_user.id
    clear_session(user_id)

    reset_text = (
        "🔄 <b>Session reset successfully!</b>\n\n"
        "Your Job Description and uploaded resumes have been cleared.\n"
        "Send /start whenever you are ready to begin a new analysis."
    )

    await update.message.reply_text(reset_text, parse_mode=ParseMode.HTML)


def try_parse_jd_selection(text: str, total_jds: int) -> Optional[Any]:
    """
    Checks if text is an attempt to select a JD from multiple JDs.
    Returns 0-based integer index, 'all', or None.
    """
    cleaned = text.strip().lower()
    if cleaned in ("all", "both", "all jds", "compare all", "/use_jd all"):
        return "all"
    match = re.match(r"^(?:(?:/use_jd|use\s*jd|jd|use|option)\s*)?(\d+)$", cleaned)
    if match:
        num = int(match.group(1))
        if 1 <= num <= total_jds:
            return num - 1
    return None


async def handle_job_description(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming text messages containing the Job Description or selection."""
    if not update.effective_user or not update.message or update.message.text is None:
        return

    user_id = update.effective_user.id
    session = get_session(user_id)
    raw_text = update.message.text.strip()
    all_jds = get_job_descriptions(user_id)
    total_jds = len(all_jds)

    # 1. If multiple JDs exist, check if user is selecting which JD to use (e.g. "1", "2", "all")
    if total_jds > 1:
        selection = try_parse_jd_selection(raw_text, total_jds)
        if selection is not None:
            set_selected_jd(user_id, selection)
            if selection == "all":
                confirm_msg = (
                    "✅ <b>Active Mode: Compare Across All Job Descriptions</b>\n\n"
                    f"The candidate(s) will be evaluated against all {total_jds} Job Descriptions.\n\n"
                    "👉 Upload candidate <b>resumes</b> (or send <code>/analyze</code> if resumes are already uploaded)."
                )
            else:
                sel_name = all_jds[selection].get("filename", f"Job Description #{selection+1}")
                confirm_msg = (
                    f"✅ <b>Selected Job Description:</b> <code>{html.escape(sel_name)}</code>\n\n"
                    f"The candidate(s) will be evaluated specifically for <b>{html.escape(sel_name)}</b>.\n\n"
                    "👉 Upload candidate <b>resumes</b> (or send <code>/analyze</code> if resumes are already uploaded)."
                )
            await update.message.reply_text(confirm_msg, parse_mode=ParseMode.HTML)
            return

    # If the user already uploaded a JD and is in waiting_for_resumes state
    if session.get("state") == "waiting_for_resumes":
        already_set_msg = (
            "⚠️ <b>Invalid message.</b>\n\n"
            "Your message is invalid. The Job Description is already set!\n\n"
            "👉 Please upload candidate <b>resumes</b> as <b>.pdf</b> or <b>.docx</b> files (or send <code>/analyze</code> when you are finished).\n"
            "If you want to submit a different Job Description, send /reset first."
        )
        if total_jds > 1:
            already_set_msg += (
                f"\n\n💡 <i>Tip: You have {total_jds} Job Descriptions loaded. "
                "Reply with <code>1</code>, <code>2</code>, or <code>all</code> to choose which JD to use.</i>"
            )
        await update.message.reply_text(already_set_msg, parse_mode=ParseMode.HTML)
        return

    # Check for empty or whitespace-only input
    if not raw_text:
        empty_msg = (
            "⚠️ <b>Invalid message.</b>\n\n"
            "Your message is invalid. The Job Description cannot be empty.\n\n"
            "👉 Please provide the <b>Job Description</b> (paste the job requirements, or upload a <b>.pdf</b> or <b>.docx</b> file)."
        )
        await update.message.reply_text(empty_msg, parse_mode=ParseMode.HTML)
        return

    # Check minimum length (at least 30 characters for a realistic JD)
    if len(raw_text) < 30:
        short_msg = (
            "⚠️ <b>Invalid message.</b>\n\n"
            "Your message is invalid. The Job Description is too short.\n\n"
            "👉 Please provide a valid <b>Job Description</b> (at least 30 characters) including required skills and responsibilities, or upload a <b>.pdf</b> or <b>.docx</b> file."
        )
        await update.message.reply_text(short_msg, parse_mode=ParseMode.HTML)
        return

    # Check maximum length to prevent excessive payloads
    if len(raw_text) > 10000:
        long_msg = (
            "⚠️ <b>Invalid message.</b>\n\n"
            "Your message is invalid. The Job Description is too long.\n\n"
            "👉 Please provide a concise <b>Job Description</b> (under 10,000 characters) focusing on key requirements."
        )
        await update.message.reply_text(long_msg, parse_mode=ParseMode.HTML)
        return

    # Validate whether the text is a genuine Job Description
    if not is_valid_job_description_text(raw_text):
        lower_t = raw_text.lower()
        if any(w in lower_t for w in ("resume", "curriculum vitae", "education:", "gpa", "work experience:")):
            invalid_msg = (
                "⚠️ <b>Please send the Job Description first.</b>\n\n"
                "You pasted a resume, but no Job Description is set yet!\n\n"
                "👉 Please give me the <b>Job Description</b> first (paste job requirements, or upload a <b>.pdf</b> or <b>.docx</b> file) "
                "because you uploaded a resume, so I know what requirements to evaluate it against."
            )
        else:
            invalid_msg = (
                "⚠️ <b>Invalid message.</b>\n\n"
                "Your message is invalid. It does not appear to be a Job Description.\n\n"
                "👉 Please provide a valid <b>Job Description</b> with role title, responsibilities, and required skills (or upload a <b>.pdf</b> or <b>.docx</b> file)."
            )
        await update.message.reply_text(invalid_msg, parse_mode=ParseMode.HTML)
        return

    # Store first JD and update session state
    set_job_description(user_id, raw_text, "Text Job Description")

    confirmation_text = (
        "✅ <b>Job Description received!</b>\n\n"
        "Now upload your candidate <b>PDF or DOCX resumes</b>.\n"
        "When you are finished uploading, send <code>/analyze</code> to begin the evaluation."
    )
    await update.message.reply_text(confirmation_text, parse_mode=ParseMode.HTML)


# Maximum allowed file size for resume uploads (15 MB)
MAX_RESUME_SIZE_BYTES = 15 * 1024 * 1024


async def handle_resume_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles document uploads (PDF/DOCX resumes or Job Descriptions)."""
    if not update.effective_user or not update.message or not update.message.document:
        return

    user_id = update.effective_user.id
    session = get_session(user_id)
    document = update.message.document
    original_filename = document.file_name or "document"
    ext = Path(original_filename).suffix.lower()

    # Guard 1: File extension check
    if ext not in (".pdf", ".docx"):
        if session.get("state") == "waiting_for_resumes":
            warning_text = (
                "⚠️ <b>Unsupported file format.</b>\n\n"
                "Your message is invalid. "
                f"Received: <code>{html.escape(original_filename)}</code>\n\n"
                "👉 Please upload candidate <b>resumes</b> as <b>.pdf</b> or <b>.docx</b> files."
            )
        else:
            warning_text = (
                "⚠️ <b>Unsupported file format.</b>\n\n"
                "Your message is invalid. "
                f"Received: <code>{html.escape(original_filename)}</code>\n\n"
                "👉 Please provide a <b>Job Description</b> (paste text, or upload a <b>.pdf</b> or <b>.docx</b> file)."
            )
        await update.message.reply_text(warning_text, parse_mode=ParseMode.HTML)
        return

    # Guard 2: MIME type check (when string is provided by Telegram)
    if isinstance(document.mime_type, str) and document.mime_type:
        mime_lower = document.mime_type.lower()
        if ext == ".pdf" and mime_lower not in ("application/pdf", "application/octet-stream"):
            mime_msg = (
                "⚠️ <b>Invalid file type.</b>\n\n"
                f"The file <code>{html.escape(original_filename)}</code> is not recognized as a valid PDF document."
            )
            await update.message.reply_text(mime_msg, parse_mode=ParseMode.HTML)
            return
        elif ext == ".docx":
            allowed_docx_mimes = (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/zip",
                "application/x-zip-compressed",
                "application/octet-stream",
                "application/msword",
            )
            if mime_lower not in allowed_docx_mimes:
                mime_msg = (
                    "⚠️ <b>Invalid file type.</b>\n\n"
                    f"The file <code>{html.escape(original_filename)}</code> is not recognized as a valid Word document."
                )
                await update.message.reply_text(mime_msg, parse_mode=ParseMode.HTML)
                return

    # Guard 3: File size check
    if isinstance(document.file_size, (int, float)):
        if document.file_size == 0:
            await update.message.reply_text(
                "⚠️ <b>File is empty.</b>\n\nPlease upload a valid, non-empty document.",
                parse_mode=ParseMode.HTML,
            )
            return
        if document.file_size > MAX_RESUME_SIZE_BYTES:
            size_mb = MAX_RESUME_SIZE_BYTES // (1024 * 1024)
            await update.message.reply_text(
                f"⚠️ <b>File is too large.</b>\n\n"
                f"Maximum allowed file size is {size_mb} MB. Please upload a smaller document.",
                parse_mode=ParseMode.HTML,
            )
            return

    # Fast-path check: If no JD is set yet, and filename clearly indicates a resume
    # (e.g. "early_resume.pdf", "Alice_Resume.pdf"), reject early as required by existing tests
    existing_jds = get_job_descriptions(user_id)
    clean_lower = original_filename.lower()
    is_resume_filename = (
        bool(re.search(r"(^|[_\-\s])(resume|cv|curriculum|biodata|profile|candidate)([_\-\s\d\.]|$)", clean_lower))
        and not bool(re.search(r"(^|[_\-\s])(jd|job[_\-\s]*desc|job|desc|requirement|position|role|opening|hiring)", clean_lower))
    )
    is_jd_filename = (
        bool(re.search(r"(^|[_\-\s])(jd|job[_\-\s]*desc|job[_\-\s]*description|job|requirement|position|role|opening|hiring|vacancy)([_\-\s\d\.]|$)", clean_lower))
        or bool(re.search(r"(^|[_\-\s])jd\d*([_\-\s\.]|$)", clean_lower))
    )
    if not existing_jds and is_resume_filename:
        warning_text = (
            "⚠️ <b>Please send the Job Description first.</b>\n\n"
            f"You uploaded a resume (<code>{html.escape(original_filename)}</code>), but no Job Description is set yet!\n\n"
            "👉 Please give me the <b>Job Description</b> first (paste text, or upload a <b>.pdf</b> or <b>.docx</b> file) "
            "because you uploaded a resume, so I know what requirements to evaluate it against."
        )
        await update.message.reply_text(warning_text, parse_mode=ParseMode.HTML)
        return

    # Prepare local destination
    clean_base = re.sub(r"[^a-zA-Z0-9_.-]", "_", Path(original_filename).name)
    existing_resumes = get_resumes(user_id)
    seq_num = len(existing_resumes) + 1
    unique_filename = f"doc_{seq_num}_{clean_base}"
    dest_path = get_user_temp_dir(user_id) / unique_filename

    # Download file from Telegram
    try:
        tg_file = await context.bot.get_file(document.file_id)
        await tg_file.download_to_drive(custom_path=dest_path)
    except Exception as exc:
        logger.error(f"Failed downloading document from Telegram: {exc}")
        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        await update.message.reply_text(
            "⚠️ <b>Download failed.</b>\n\n"
            "Unable to download your document. Please try again.",
            parse_mode=ParseMode.HTML,
        )
        return

    # Validate header bytes
    try:
        with open(dest_path, "rb") as f:
            header = f.read(4)

        if ext == ".pdf":
            if not header.startswith(b"%PDF"):
                dest_path.unlink(missing_ok=True)
                await update.message.reply_text(
                    f"⚠️ <b>Corrupted or invalid PDF.</b>\n\n"
                    f"The file <code>{html.escape(original_filename)}</code> is not a valid PDF document.",
                    parse_mode=ParseMode.HTML,
                )
                return
        elif ext == ".docx":
            if not header.startswith(b"PK\x03\x04"):
                dest_path.unlink(missing_ok=True)
                await update.message.reply_text(
                    f"⚠️ <b>Corrupted or invalid DOCX.</b>\n\n"
                    f"The file <code>{html.escape(original_filename)}</code> is not a valid Word document.",
                    parse_mode=ParseMode.HTML,
                )
                return
    except Exception as read_err:
        logger.error(f"Error reading file header: {read_err}")
        dest_path.unlink(missing_ok=True)
        await update.message.reply_text(
            "⚠️ <b>File error.</b>\n\nCould not read the uploaded file. Please try again.",
            parse_mode=ParseMode.HTML,
        )
        return

    # Determine if document is a Resume or Job Description
    extracted_text = ""
    try:
        if ext == ".pdf":
            extracted_text = extract_text_from_pdf(dest_path)
        elif ext == ".docx":
            extracted_text = extract_text_from_docx(dest_path)
    except Exception as extract_err:
        logger.warning(f"Could not extract text during intake: {extract_err}")
        extracted_text = ""

    if is_jd_filename and not is_resume_filename:
        doc_type = "job_description"
    elif is_resume_filename and not is_jd_filename:
        doc_type = "resume"
    elif existing_jds and not extracted_text:
        doc_type = "resume"
    else:
        doc_type = classify_document_content(extracted_text, original_filename)

    if doc_type == "job_description":
        dest_path.unlink(missing_ok=True)

        if len(extracted_text) < 30:
            await update.message.reply_text(
                "⚠️ <b>Job Description in document is too short.</b>\n\n"
                "Please provide a more detailed Job Description (at least 30 characters) including required skills and responsibilities.",
                parse_mode=ParseMode.HTML,
            )
            return

        if len(extracted_text) > 10000:
            await update.message.reply_text(
                "⚠️ <b>Job Description in document is too long.</b>\n\n"
                "Please provide a concise description (under 10,000 characters) focusing on key requirements.",
                parse_mode=ParseMode.HTML,
            )
            return

        is_first_jd = len(existing_jds) == 0
        add_job_description(user_id, extracted_text, original_filename)

        if is_first_jd:
            confirmation_text = (
                f"✅ <b>Job Description received!</b> (from <code>{html.escape(original_filename)}</code>)\n\n"
                "Now upload your candidate <b>PDF or DOCX resumes</b>.\n"
                "When you are finished uploading, send <code>/analyze</code> to begin the evaluation."
            )
            await update.message.reply_text(confirmation_text, parse_mode=ParseMode.HTML)
            return
        else:
            all_jds = get_job_descriptions(user_id)
            total_jds = len(all_jds)
            total_res = len(get_resumes(user_id))

            jd_list_lines = []
            keyboard_buttons = []
            for i, j in enumerate(all_jds):
                fname = j.get("filename", f"Job Description #{i+1}")
                emoji = ORDINAL_EMOJIS[i] if i < len(ORDINAL_EMOJIS) else f"{i+1}️⃣"
                jd_list_lines.append(f"{emoji} <b>{html.escape(fname)}</b>")
                keyboard_buttons.append([
                    InlineKeyboardButton(
                        text=f"{emoji} Use {fname[:25]}",
                        callback_data=f"select_jd_{i}",
                    )
                ])

            keyboard_buttons.append([
                InlineKeyboardButton(
                    text="🌟 Compare Across All JDs",
                    callback_data="select_jd_all",
                )
            ])
            reply_markup = InlineKeyboardMarkup(keyboard_buttons)

            confirmation_text = (
                f"📑 <b>Multiple Job Descriptions Detected ({total_jds})!</b>\n\n"
                f"Received additional JD: <code>{html.escape(original_filename)}</code>\n\n"
                "I have identified multiple Job Descriptions in your session:\n"
                + "\n".join(jd_list_lines)
                + "\n\n"
                "👉 <b>Which Job Description should I use for evaluation?</b>\n"
                "Click a button below, or reply with the number (e.g. <code>1</code>, <code>2</code>, or <code>all</code>).\n\n"
                f"📊 Current Resumes uploaded: <b>{total_res}</b>"
            )
            await update.message.reply_text(
                confirmation_text,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup,
            )
            return

    # Document is a candidate RESUME
    if not existing_jds:
        dest_path.unlink(missing_ok=True)
        warning_text = (
            "⚠️ <b>Please send the Job Description first.</b>\n\n"
            f"You uploaded a resume (<code>{html.escape(original_filename)}</code>), but no Job Description is set yet!\n\n"
            "👉 Please give me the <b>Job Description</b> first (paste text, or upload a <b>.pdf</b> or <b>.docx</b> file) "
            "because you uploaded a resume, so I know what requirements to evaluate it against."
        )
        await update.message.reply_text(warning_text, parse_mode=ParseMode.HTML)
        return

    actual_size = dest_path.stat().st_size
    resume_meta = {
        "original_filename": original_filename,
        "local_path": str(dest_path),
        "file_id": document.file_id,
        "file_size": actual_size,
        "file_type": "docx" if ext == ".docx" else "pdf",
    }
    add_resume(user_id, resume_meta)

    total_count = len(get_resumes(user_id))
    size_kb = max(1, actual_size // 1024)

    confirmation_msg = (
        f"✅ <b>Resume {total_count} received:</b> <code>{html.escape(original_filename)}</code> ({size_kb} KB)\n\n"
        f"Total candidate resumes ready: <b>{total_count}</b>\n\n"
        f"Upload another PDF or DOCX resume, or send <code>/analyze</code> when you are finished."
    )
    await update.message.reply_text(confirmation_msg, parse_mode=ParseMode.HTML)


# Alias for backward-compatibility with Phase 2 test imports
handle_document_before_jd = handle_resume_document


async def safe_reply_text(
    message: Any,
    text: str,
    parse_mode: Any = ParseMode.HTML,
    **kwargs: Any,
) -> List[Any]:
    """
    Safely sends a message via Telegram, splitting into chunks if text exceeds 4000 characters.
    Handles HTML tags or plain text without truncation errors.
    """
    MAX_CHUNK = 4000
    if len(text) <= MAX_CHUNK:
        try:
            return [await message.reply_text(text, parse_mode=parse_mode, **kwargs)]
        except TelegramError:
            return [await message.reply_text(text, parse_mode=None, **kwargs)]

    lines = text.splitlines(keepends=True)
    chunks = []
    curr = ""
    for line in lines:
        if len(curr) + len(line) > MAX_CHUNK:
            if curr:
                chunks.append(curr)
                curr = ""
            if len(line) > MAX_CHUNK:
                for i in range(0, len(line), MAX_CHUNK):
                    chunks.append(line[i : i + MAX_CHUNK])
            else:
                curr = line
        else:
            curr += line
    if curr:
        chunks.append(curr)

    sent = []
    for chunk in chunks:
        try:
            sent_msg = await message.reply_text(chunk, parse_mode=parse_mode, **kwargs)
        except TelegramError:
            sent_msg = await message.reply_text(chunk, parse_mode=None, **kwargs)
        sent.append(sent_msg)
    return sent


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /analyze command, executing Gemini analysis and rendering formatted cards."""
    if not update.effective_user or not update.message:
        return

    user_id = update.effective_user.id
    jds = get_job_descriptions(user_id)
    resumes = get_resumes(user_id)

    if not jds:
        await update.message.reply_text(
            "⚠️ <b>Job Description missing!</b>\n\n"
            "Please send the Job Description first before running <code>/analyze</code>.",
            parse_mode=ParseMode.HTML,
        )
        return

    if not resumes:
        await update.message.reply_text(
            "⚠️ <b>No resumes uploaded!</b>\n\n"
            "Please upload at least one candidate PDF or DOCX resume before sending <code>/analyze</code>.",
            parse_mode=ParseMode.HTML,
        )
        return

    active_jds = get_active_jds_for_analysis(user_id)

    # Scenario 1: Multi-JD mode (2+ active Job Descriptions, 1 candidate Resume)
    if len(active_jds) > 1 and len(resumes) == 1:
        candidate_resume = resumes[0]
        resume_filename = candidate_resume.get("original_filename", "resume.pdf")
        local_path = Path(candidate_resume.get("local_path", ""))

        if not local_path.exists():
            await update.message.reply_text(
                f"⚠️ <b>File not found:</b> <code>{html.escape(resume_filename)}</code> is no longer available on the server. Please upload it again.",
                parse_mode=ParseMode.HTML,
            )
            return

        status_msg = await update.message.reply_text(
            f"🔍 <b>Analyzing candidate against {len(active_jds)} Job Descriptions...</b>\n"
            "⏳ <i>Evaluating candidate fit across roles...</i>",
            parse_mode=ParseMode.HTML,
        )

        successful_evals = []
        failed_evals = []

        for idx, jd_item in enumerate(active_jds, start=1):
            jd_name = jd_item.get("filename", f"Job Description #{idx}")
            jd_text = jd_item.get("text", "")

            try:
                eval_result = analyze_resume_against_jd(
                    resume_path=local_path,
                    job_description=jd_text,
                )
                eval_result["jd_title"] = jd_name
                eval_result["original_filename"] = resume_filename
                successful_evals.append(eval_result)
            except GeminiConfigError as cfg_err:
                logger.error(f"Gemini config error: {cfg_err}")
                await update.message.reply_text(
                    "⚠️ <b>Configuration Error:</b> Gemini API key is missing or invalid. Please check the bot settings.",
                    parse_mode=ParseMode.HTML,
                )
                return
            except GeminiInputError as inp_err:
                logger.error(f"Input error for {jd_name}: {inp_err}")
                failed_evals.append({"filename": jd_name, "error": str(inp_err)})
            except GeminiAPIError as api_err:
                logger.error(f"API error for {jd_name}: {api_err}")
                failed_evals.append({"filename": jd_name, "error": "Gemini API service error"})
            except Exception as exc:
                logger.error(f"Unexpected error for {jd_name}: {exc}")
                failed_evals.append({"filename": jd_name, "error": "Unexpected processing error"})

        try:
            if successful_evals:
                await status_msg.edit_text(
                    f"✅ <b>Screening complete!</b> Evaluated candidate against {len(successful_evals)}/{len(active_jds)} Job Descriptions.",
                    parse_mode=ParseMode.HTML,
                )
            else:
                await status_msg.edit_text(
                    "❌ <b>Analysis failed for all Job Descriptions.</b> Please see details below.",
                    parse_mode=ParseMode.HTML,
                )
        except Exception:
            pass

        if successful_evals:
            ranked_evals = sorted(
                successful_evals,
                key=lambda x: (
                    max(0, min(100, int(x.get("job_match_percentage", 0)))),
                    max(0, min(100, int(x.get("ats_score", 0)))),
                ),
                reverse=True,
            )
            raw_cand_name = ranked_evals[0].get("candidate_name") or ""
            if not raw_cand_name or raw_cand_name.lower() in ("unknown", "n/a", "none", "candidate", "not specified"):
                cand_name = Path(resume_filename).stem
            else:
                cand_name = raw_cand_name

            # 1. Multi-JD comparison card
            comparison_card = format_multi_jd_comparison(
                candidate_name=cand_name,
                resume_filename=resume_filename,
                jd_evaluations=ranked_evals,
            )
            await safe_reply_text(update.message, comparison_card)

            # 2. Detailed analysis card for each JD in ranked order
            for ev in ranked_evals:
                card_html = format_candidate_analysis(
                    analysis=ev,
                    resume_filename=resume_filename,
                    jd_title=ev.get("jd_title"),
                )
                await safe_reply_text(update.message, card_html)

        elif failed_evals:
            fail_summary = format_candidate_ranking([], failed_resumes=failed_evals)
            await safe_reply_text(update.message, fail_summary)

        return

    # Scenario 2: Standard screening with single active JD across 1 or more candidate resumes
    target_jd_item = active_jds[0] if active_jds else jds[0]
    primary_jd = target_jd_item["text"]
    primary_jd_title = target_jd_item.get("filename")
    num_resumes = len(resumes)
    status_msg = await update.message.reply_text(
        f"🔍 <b>Analyzing {num_resumes} candidate resume{'s' if num_resumes > 1 else ''}...</b>\n"
        "⏳ <i>Evaluating candidates against the Job Description...</i>",
        parse_mode=ParseMode.HTML,
    )

    successful_candidates = []
    failed_candidates = []

    for idx, resume in enumerate(resumes, start=1):
        resume_filename = resume.get("original_filename", f"resume_{idx}")
        local_path = Path(resume.get("local_path", ""))

        if not local_path.exists():
            failed_candidates.append({
                "filename": resume_filename,
                "error": "Local file not found on server.",
            })
            continue

        try:
            analysis_result = analyze_resume_against_jd(
                resume_path=local_path,
                job_description=primary_jd,
            )
            analysis_result["original_filename"] = resume_filename
            successful_candidates.append(analysis_result)

        except GeminiConfigError as cfg_err:
            logger.error(f"Gemini config error: {cfg_err}")
            await update.message.reply_text(
                "⚠️ <b>Configuration Error:</b> Gemini API key is missing or invalid. Please check the bot settings.",
                parse_mode=ParseMode.HTML,
            )
            return
        except GeminiInputError as inp_err:
            logger.error(f"Gemini input error for {resume_filename}: {inp_err}")
            failed_candidates.append({
                "filename": resume_filename,
                "error": str(inp_err),
            })
        except GeminiAPIError as api_err:
            logger.error(f"Gemini API error for {resume_filename}: {api_err}")
            failed_candidates.append({
                "filename": resume_filename,
                "error": "Gemini API service error",
            })
        except Exception as exc:
            logger.error(f"Unexpected error analyzing {resume_filename}: {exc}")
            failed_candidates.append({
                "filename": resume_filename,
                "error": "Unexpected processing error",
            })

    # Update initial status message
    try:
        if successful_candidates:
            await status_msg.edit_text(
                f"✅ <b>Screening complete!</b> Evaluated {len(successful_candidates)}/{num_resumes} resume{'s' if num_resumes > 1 else ''}.",
                parse_mode=ParseMode.HTML,
            )
        else:
            await status_msg.edit_text(
                "❌ <b>Analysis failed for all uploaded resumes.</b> Please see details below.",
                parse_mode=ParseMode.HTML,
            )
    except Exception:
        pass

    # If any candidates were successfully analyzed, rank and display results
    if successful_candidates:
        ranked_candidates = rank_candidates(successful_candidates)

        ranking_card = format_candidate_ranking(
            ranked_candidates=ranked_candidates,
            failed_resumes=failed_candidates,
        )
        await safe_reply_text(update.message, ranking_card)

        for cand in ranked_candidates:
            card_html = format_candidate_analysis(
                analysis=cand,
                resume_filename=cand.get("original_filename", "resume.pdf"),
                jd_title=primary_jd_title,
            )
            await safe_reply_text(update.message, card_html)
    elif failed_candidates:
        fail_summary = format_candidate_ranking([], failed_resumes=failed_candidates)
        await safe_reply_text(update.message, fail_summary)


async def handle_jd_selection_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles interactive button clicks when selecting a Job Description."""
    query = update.callback_query
    if not query or not query.data or not query.data.startswith("select_jd_"):
        return

    await query.answer()
    user_id = query.from_user.id
    raw_choice = query.data.replace("select_jd_", "")
    all_jds = get_job_descriptions(user_id)
    total_jds = len(all_jds)

    if raw_choice == "all":
        set_selected_jd(user_id, "all")
        confirm_text = (
            "✅ <b>Active Mode: Compare Across All Job Descriptions</b>\n\n"
            f"The candidate(s) will be evaluated against all {total_jds} Job Descriptions.\n\n"
            "👉 Upload candidate <b>resumes</b> (or send <code>/analyze</code> if resumes are already uploaded)."
        )
    else:
        try:
            idx = int(raw_choice)
            if 0 <= idx < total_jds:
                set_selected_jd(user_id, idx)
                sel_name = all_jds[idx].get("filename", f"Job Description #{idx+1}")
                confirm_text = (
                    f"✅ <b>Selected Job Description:</b> <code>{html.escape(sel_name)}</code>\n\n"
                    f"The candidate(s) will be evaluated specifically for <b>{html.escape(sel_name)}</b>.\n\n"
                    "👉 Upload candidate <b>resumes</b> (or send <code>/analyze</code> if resumes are already uploaded)."
                )
            else:
                confirm_text = "⚠️ Invalid selection index. Please choose a valid Job Description."
        except ValueError:
            confirm_text = "⚠️ Invalid selection. Please choose a valid Job Description."

    if query.message:
        await query.message.reply_text(confirm_text, parse_mode=ParseMode.HTML)


async def use_jd_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command /use_jd allowing users to switch active JD."""
    if not update.effective_user or not update.message:
        return

    user_id = update.effective_user.id
    all_jds = get_job_descriptions(user_id)
    total_jds = len(all_jds)

    if not all_jds:
        await update.message.reply_text(
            "⚠️ <b>No Job Descriptions found.</b> Please upload a Job Description first.",
            parse_mode=ParseMode.HTML,
        )
        return

    args = context.args or []
    if not args:
        jd_list_lines = []
        keyboard_buttons = []
        for i, j in enumerate(all_jds):
            fname = j.get("filename", f"Job Description #{i+1}")
            emoji = ORDINAL_EMOJIS[i] if i < len(ORDINAL_EMOJIS) else f"{i+1}️⃣"
            jd_list_lines.append(f"{emoji} <b>{html.escape(fname)}</b>")
            keyboard_buttons.append([
                InlineKeyboardButton(
                    text=f"{emoji} Use {fname[:25]}",
                    callback_data=f"select_jd_{i}",
                )
            ])
        keyboard_buttons.append([
            InlineKeyboardButton(
                text="🌟 Compare Across All JDs",
                callback_data="select_jd_all",
            )
        ])
        reply_markup = InlineKeyboardMarkup(keyboard_buttons)
        msg = (
            f"📑 <b>Available Job Descriptions ({total_jds}):</b>\n\n"
            + "\n".join(jd_list_lines)
            + "\n\n👉 <b>Which Job Description should I use?</b>\n"
            "Click an option below or send <code>/use_jd &lt;number&gt;</code> or <code>/use_jd all</code>."
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
        return

    choice = args[0].strip().lower()
    selection = try_parse_jd_selection(choice, total_jds)
    if selection is None:
        await update.message.reply_text(
            f"⚠️ Invalid choice: <code>{html.escape(choice)}</code>. Please enter a number (1 to {total_jds}) or <code>all</code>.",
            parse_mode=ParseMode.HTML,
        )
        return

    set_selected_jd(user_id, selection)
    if selection == "all":
        confirm_text = (
            "✅ <b>Active Mode: Compare Across All Job Descriptions</b>\n\n"
            f"The candidate(s) will be evaluated against all {total_jds} Job Descriptions.\n\n"
            "👉 Upload candidate <b>resumes</b> (or send <code>/analyze</code> if resumes are already uploaded)."
        )
    else:
        sel_name = all_jds[selection].get("filename", f"Job Description #{selection+1}")
        confirm_text = (
            f"✅ <b>Selected Job Description:</b> <code>{html.escape(sel_name)}</code>\n\n"
            f"The candidate(s) will be evaluated specifically for <b>{html.escape(sel_name)}</b>.\n\n"
            "👉 Upload candidate <b>resumes</b> (or send <code>/analyze</code> if resumes are already uploaded)."
        )
    await update.message.reply_text(confirm_text, parse_mode=ParseMode.HTML)


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles unrecognized commands gracefully."""
    if not update.message:
        return

    unknown_text = (
        "❓ <b>Unrecognized command.</b>\n\n"
        "Use /help to see the list of available commands."
    )

    await update.message.reply_text(unknown_text, parse_mode=ParseMode.HTML)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log uncaught exceptions caused by Telegram updates."""
    logger.error("Exception occurred while handling an update:", exc_info=context.error)


async def handle_unsupported_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles non-text, non-document messages (photos, stickers, voice notes, audio, video, etc.)."""
    if not update.effective_user or not update.message:
        return

    user_id = update.effective_user.id
    session = get_session(user_id)

    if session.get("state") == "waiting_for_resumes":
        msg = (
            "⚠️ <b>Invalid message.</b>\n\n"
            "Your message is invalid. The Job Description is already set!\n\n"
            "👉 Please upload candidate <b>resumes</b> as <b>.pdf</b> or <b>.docx</b> files (or send <code>/analyze</code> when you are finished)."
        )
    else:
        msg = (
            "⚠️ <b>Invalid message.</b>\n\n"
            "Your message is invalid.\n\n"
            "👉 Please provide a valid <b>Job Description</b> by pasting text or uploading a <b>.pdf</b> or <b>.docx</b> file."
        )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


def create_bot_application(token: str) -> Application:
    """Builds and returns the configured Telegram Application with resilient network timeouts."""
    app = (
        ApplicationBuilder()
        .token(token)
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .pool_timeout(30.0)
        .build()
    )

    # Register command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("use_jd", use_jd_command))

    # Callback query handler for interactive JD selection buttons
    app.add_handler(CallbackQueryHandler(handle_jd_selection_callback, pattern=r"^select_jd_"))

    # Document handler for resume uploads
    app.add_handler(MessageHandler(filters.Document.ALL, handle_resume_document))

    # Plain text messages are treated as Job Description input
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_job_description))

    # Fallback handler for unknown commands
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))

    # Fallback handler for unsupported media (photos, voice notes, stickers, audio, video)
    app.add_handler(MessageHandler(~filters.COMMAND & ~filters.Document.ALL & ~filters.TEXT, handle_unsupported_media))

    # Register global error handler
    app.add_error_handler(error_handler)

    return app


def main() -> None:
    """Entrypoint to validate config and start polling."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token or token.strip() == "your_telegram_bot_token_here":
        logger.error(
            "CRITICAL: TELEGRAM_BOT_TOKEN is missing or not set in .env!\n"
            "Please open .env and provide a valid bot token from @BotFather."
        )
        sys.exit(1)

    logger.info("Initializing JobFit AI Telegram Bot...")
    app = create_bot_application(token)

    logger.info("Bot started successfully. Polling for updates...")
    try:
        app.run_polling(bootstrap_retries=10, timeout=30)
    except InvalidToken:
        logger.error(
            "CRITICAL: TELEGRAM_BOT_TOKEN was rejected by the Telegram API (Invalid Token).\n"
            "Please verify that the token in .env is correct and active."
        )
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt). Exiting cleanly...")
    except TelegramError as err:
        logger.error(f"Telegram API error occurred: {err}")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Unexpected error occurred: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
