# JobFit AI

[![Deployment Status](https://img.shields.io/badge/Deployment-Live%20on%20Render%20(24%2F7)-brightgreen?style=for-the-badge&logo=render&logoColor=white)](https://t.me/hashira_Job_fit_bot)
[![Telegram Bot](https://img.shields.io/badge/Telegram-@hashira__Job__fit__bot-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/hashira_Job_fit_bot)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-3.6%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com/)

> ### 🟢 Bot is Deployed & Live 24/7 in the Cloud!
> **JobFit AI is currently deployed and running live on Render.**  
> Anyone can test it immediately on Telegram without installing or running any code locally:  
> 👉 **[Click here to chat with @hashira_Job_fit_bot](https://t.me/hashira_Job_fit_bot)**  
> *(Simply tap the link, press `/start`, and upload a job description & resume to test it live!)*

**JobFit AI** is an intelligent, production-ready Telegram screening bot that automates candidate resume evaluation against Job Descriptions (JDs). Powered by Google Gemini (`gemini-3.6-flash`) and a deterministic mathematical scoring engine, JobFit AI extracts candidate qualifications, computes ATS (Applicant Tracking System) and Job Match scores, identifies matching and missing skills, generates personalized learning roadmaps with documentation links, and ranks multiple candidates for hiring teams.

---

## Table of Contents

- [What the Project Does](#what-the-project-does)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Environment Variables](#environment-variables)
- [How to Create a Telegram Bot](#how-to-create-a-telegram-bot)
- [How to Configure Gemini API](#how-to-configure-gemini-api)
- [How to Run the Bot](#how-to-run-the-bot)
- [User Workflow](#user-workflow)
- [Example Output](#example-output)
- [Scoring Methodology](#scoring-methodology)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## What the Project Does

Recruiters and hiring managers often spend hours manually screening resumes against lengthy job descriptions. JobFit AI streamlines this initial screening funnel into Telegram:

1. **Intake**: Accept Job Descriptions as plain text or uploaded documents (`.pdf`, `.docx`).
2. **Document Classification**: Distinguish whether an uploaded file is a Job Description or a candidate Resume using heuristic classifiers.
3. **Multi-Candidate Screening**: Collect one or more candidate resumes in `.pdf` or `.docx` format.
4. **AI Analysis**: Extract skills, analyze professional experience, and highlight skill gaps using Google Gemini.
5. **Deterministic Scoring**: Calculate transparent, bounded ($0-100$) ATS Scores and Job Match percentages using a normalized weighted formula.
6. **Candidate Ranking**: Sort candidates in descending order of fit with deterministic tiebreaking and medal icons.
7. **Actionable Recommendations**: Provide missing skill roadmaps and authoritative documentation links (e.g., official docs, roadmap.sh) for candidate upskilling.

---

## Features

- **Document Flexibility**: Accepts Job Descriptions and Resumes as plain text, PDF (`.pdf`), and Microsoft Word (`.docx`).
- **Smart Content Classification**: Automatically identifies whether an uploaded document is a Job Description or a candidate Resume based on filename and content markers.
- **Multi-JD Support**: Allows loading multiple Job Descriptions and selecting an active JD via interactive inline buttons or text commands (`/use_jd`).
- **Multi-Candidate Batch Screening**: Upload multiple resumes in a single session; the bot processes each candidate and compiles a comparative leaderboard.
- **Deterministic Scoring Engine**: Decouples score calculation from LLM hallucinations. Component scores are normalized and clamped strictly to $[0, 100]$.
- **Visual Progress & Formatting**: HTML-formatted Telegram cards featuring ASCII progress bars (`[██████░░░░] 60%`), section dividers, and categorized skill breakdowns.
- **Upskilling Roadmaps**: Missing skills are flagged by priority (`critical`, `important`, `preferred`) with milestone roadmaps and direct educational links.
- **Enterprise-Grade Security**:
  - Absolute session isolation across concurrent Telegram users.
  - Path traversal protections preventing directory escape attacks.
  - Sanitized logging ensuring candidate PII is never written to server logs.
  - Automatic purging of temporary files on session reset (`/reset`) and restart (`/start`).
  - Telegram 4,096-character message chunking with HTML tag safety.

---

## Architecture

The system follows a modular, decoupled architecture:

```
                      +-------------------+
                      |   Telegram User   |
                      +---------+---------+
                                | (HTTPS Polling)
                                v
                      +-------------------+
                      |      bot.py       | <---> session_manager.py
                      | (Telegram Handler)|     (Per-User Isolated State)
                      +----+---------+----+
                           |         |
         Intake / Parsing  |         | Card Formatting
                           v         v
                +-----------------------+      +-------------------+
                |   gemini_service.py   | ---> |   formatter.py    |
                | (pypdf, python-docx,  |      | (HTML Cards &     |
                |  google-genai Client) |      |  Progress Bars)   |
                +-----------+-----------+      +-------------------+
                            |
                            v Component Scores
                +-----------------------+
                |   scoring_engine.py   |
                | (Deterministic Math & |
                |  Candidate Ranking)   |
                +-----------------------+
```

1. **`bot.py`**: Telegram dispatch layer handling updates, commands, interactive callbacks, and safe message chunking.
2. **`session_manager.py`**: In-memory state store and disk space manager (`temp/users/{user_id}/`).
3. **`gemini_service.py`**: Document parsers (`pypdf`, `python-docx`), document classification, and structured schema extraction via Google GenAI SDK.
4. **`scoring_engine.py`**: Pure, deterministic Python math engine implementing weighted scoring formulas and ranking logic.
5. **`formatter.py`**: Presentation layer formatting Telegram HTML messages and visual progress meters.

---

## Tech Stack

- **Language**: Python 3.10+ (Tested on Python 3.12)
- **Telegram Bot Framework**: `python-telegram-bot` (v21+)
- **LLM Engine**: `google-genai` (Official Google GenAI SDK, using `gemini-3.6-flash`)
- **Schema Validation**: `pydantic` (v2+)
- **Document Parsing**: `pypdf` (v4+), `python-docx` (v1.1+)
- **Configuration**: `python-dotenv`
- **Testing & Quality Assurance**: `pytest`, `pytest-asyncio`

---

## Project Structure

```
hashira_1stround/
├── .env.example              # Template environment configuration
├── .gitignore                # Git exclusions (secrets, venv, caches, temp files)
├── pytest.ini                # Pytest configuration and asyncio modes
├── requirements.txt          # Production dependencies
├── bot.py                    # Main Telegram bot application & update handlers
├── gemini_service.py         # Google Gemini integration, parsing & classification
├── scoring_engine.py         # Deterministic ATS & Match scoring algorithms
├── formatter.py              # Telegram HTML message formatting & progress bars
├── session_manager.py        # In-memory user state & temp file management
└── tests/                    # Comprehensive automated test suite
    ├── test_bot_handlers.py
    ├── test_docx_support.py
    ├── test_pdf_jd_support.py
    ├── test_phase2_jd.py
    ├── test_phase3_resumes.py
    ├── test_phase4_gemini.py
    ├── test_phase5_scoring.py
    ├── test_phase6_flow.py
    ├── test_phase7_batch_ranking.py
    ├── test_qa_security_audit.py
    ├── test_session.py
    └── test_smart_intake_and_roadmap.py
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.10 or higher installed.
- Git installed.
- A Telegram account.
- A Google Gemini API Key.

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/jobfit-ai.git
cd jobfit-ai
```

### 3. Create and Activate Virtual Environment

On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Environment Variables

Copy the sample environment file to `.env`:

On Linux / macOS:
```bash
cp .env.example .env
```

On Windows:
```powershell
Copy-Item .env.example .env
```

Open `.env` in your text editor and configure your secrets:

```env
# Telegram Bot Configuration
# Obtain your token from @BotFather on Telegram
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz

# Google Gemini API Configuration
# Obtain your key from https://aistudio.google.com/
GEMINI_API_KEY=AIzaSyYourGeminiApiKeyHere
```

> **Security Note:** Never commit your `.env` file to version control. It is strictly excluded in `.gitignore`.

---

## How to Create a Telegram Bot

1. Open Telegram and search for `@BotFather`.
2. Send `/start` and then `/newbot`.
3. Choose a display name (e.g., `My JobFit AI Bot`).
4. Choose a unique username ending in `bot` (e.g., `my_jobfit_screener_bot`).
5. BotFather will reply with an API token (e.g., `8556715439:AAG...`).
6. Copy this token and paste it as `TELEGRAM_BOT_TOKEN` in your `.env` file.

---

## How to Configure Gemini API

1. Visit [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google account.
3. Click **Get API key** and create a new key.
4. Copy the API key.
5. Paste it as `GEMINI_API_KEY` in your `.env` file.

---

## How to Run the Bot

Once `.env` is configured, start the bot:

```bash
python bot.py
```

Console output confirming successful startup:
```
2026-09-10 18:25:27 - __main__ - INFO - Initializing JobFit AI Telegram Bot...
2026-09-10 18:25:28 - __main__ - INFO - Bot started successfully. Polling for updates...
2026-09-10 18:25:29 - telegram.ext.Application - INFO - Application started
```

To run the automated test suite:
```bash
pytest
```

---

## User Workflow

```
Recruiter / User                          JobFit AI Bot
       |                                         |
       | ---------- 1. /start -----------------> | Resets session, displays welcome instructions
       |                                         |
       | ---- 2. Send JD (Text, PDF, DOCX) ----> | Validates format, stores JD requirements
       |                                         |
       | -- 3. Upload Resumes (PDF, DOCX) -----> | Classifies & indexes candidate documents
       |                                         |
       | ---------- 4. /analyze ---------------> | Analyzes candidates, computes scores,
       |                                         | returns rankings & candidate breakdown cards
       |                                         |
       | ---------- 5. /reset -----------------> | Clears session data & purges disk temp files
```

### Available Bot Commands

- `/start`: Starts a new screening session and clears previous data.
- `/help`: Displays the usage guide, scoring rubrics, and supported file types.
- `/analyze`: Triggers batch evaluation of uploaded resumes against the active JD.
- `/use_jd`: Switch active Job Description when multiple JDs are uploaded.
- `/reset`: Cleans up all session data and removes downloaded files from disk.

---

## Example Output

### 1. Candidate Ranking Card
```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏆 CANDIDATE RANKING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1️⃣ Alice Johnson (Alice_Johnson_CV.pdf)
   💼 Job Match: 94% | 🎯 ATS: 92%

2️⃣ Bob Chen (Bob_Chen_Resume.docx)
   💼 Job Match: 78% | 🎯 ATS: 81%

3️⃣ Charlie Davis (Charlie_Davis.pdf)
   💼 Job Match: 45% | 🎯 ATS: 52%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 2. Detailed Candidate Card
```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👤 CANDIDATE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 Resume: Alice_Johnson_CV.pdf
👤 Candidate: Alice Johnson

🎯 ATS SCORE: 92/100
[█████████░] 92%

💼 JOB MATCH: 94%
[█████████░] 94%

✅ Matching Skills:
• Python, FastAPI, Docker, PostgreSQL, REST APIs

⚠️ Partial Skills:
• Kubernetes (Familiarity with pods, needs deployment experience)

❌ Missing Skills:
• AWS (preferred)

🗺️ Recommended Learning Roadmap:
• AWS: Cloud infrastructure and serverless services
  🔗 Resource: https://docs.aws.amazon.com
  Milestones:
  - Complete AWS Cloud Practitioner essentials
  - Deploy Dockerized FastAPI to AWS ECS / Fargate

💪 Strengths:
• 5+ years building scalable distributed backend systems.
• Proven expertise with relational databases and query optimization.

⚠️ Areas of Concern:
• Limited hands-on cloud deployment experience.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Scoring Methodology

JobFit AI uses a **hybrid evaluation model**: Gemini extracts structured qualification criteria, while Python's deterministic math engine computes final scores. This guarantees that two resumes with identical qualifications receive identical scores without LLM temperature drift.

### 1. ATS Score Rubric (0 – 100)

| Category | Weight | Evaluation Criteria |
| :--- | :---: | :--- |
| **Required Skills** | **40%** | Match against core, mandatory technical requirements. |
| **Relevant Experience** | **20%** | Years of domain experience and seniority alignment. |
| **Relevant Keywords** | **15%** | Industry-standard terminology, libraries, and tools. |
| **Resume Structure** | **10%** | Clear headings, reverse-chronological order, readability. |
| **Projects & Impact** | **10%** | Measurable accomplishments, business outcomes, repositories. |
| **Education & Certifications** | **5%** | Relevant degree or accredited certifications. |

### 2. Job Match Percentage (0 – 100)

| Category | Weight | Evaluation Criteria |
| :--- | :---: | :--- |
| **Required Skills** | **50%** | Direct coverage of mandatory role competencies. |
| **Relevant Experience** | **25%** | Direct role-specific industry experience. |
| **Projects & Portfolio** | **15%** | Practical application aligned with role duties. |
| **Preferred Skills** | **5%** | Nice-to-have or bonus qualifications. |
| **Education** | **5%** | Educational alignment with job prerequisites. |

### 3. Ranking & Tiebreakers

Candidates are ranked using a 4-tier deterministic comparator:
1. `Job Match Percentage` (Descending)
2. `ATS Score` (Descending)
3. `Candidate Name` (Ascending / Alphabetical)
4. `Filename` (Ascending / Alphabetical)

---

## Limitations

- **File Format Scope**: Supports `.pdf`, `.docx`, and plain text. Legacy binary `.doc` files, images of scanned resumes (OCR), or spreadsheets (`.xlsx`) are not supported.
- **Single Model Quota**: Operates on Google Gemini API quotas. High-concurrency screening across dozens of resumes is subject to your API tier rate limits.
- **In-Memory Sessions**: User sessions are maintained in process memory. If the bot server restarts, active sessions reset, requiring users to resend `/start`.
- **Stateless Cloud Storage**: Does not persist resume files to external databases (e.g., S3/PostgreSQL); files are strictly temporary and quarantined in local sandbox directories.

---

## Future Improvements

- [ ] **Persistent Database Backend**: Support PostgreSQL or Redis to persist historical screening results across server restarts.
- [ ] **Async Batch Concurrency**: Evaluate multiple resumes concurrently using `asyncio.gather` while respecting Gemini rate limits.
- [ ] **Export to CSV / PDF**: Allow recruiters to download candidate comparison leaderboards as a CSV spreadsheet or summary PDF report.
- [ ] **OCR Support**: Support image-based scanned resumes using OCR parsing libraries.
- [ ] **Webhook Mode**: Provide production deployment configuration via Telegram Webhooks in addition to long-polling.
