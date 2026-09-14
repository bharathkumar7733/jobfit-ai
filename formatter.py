"""
Telegram Message Formatter for JobFit AI.

Transforms structured candidate analysis JSON into clean, readable,
safe HTML formatted messages for Telegram.
"""

import html
from pathlib import Path
from typing import Any, Dict, List, Optional


def render_progress_bar(percentage: int, total_blocks: int = 10) -> str:
    """Renders a text-based progress bar, e.g. [██████░░░░] 60%."""
    pct = max(0, min(100, int(percentage)))
    filled_blocks = int(round((pct / 100.0) * total_blocks))
    filled_blocks = max(0, min(total_blocks, filled_blocks))
    empty_blocks = total_blocks - filled_blocks
    bar = "█" * filled_blocks + "░" * empty_blocks
    return f"<code>[{bar}]</code> {pct}%"


def format_candidate_analysis(
    analysis: Dict[str, Any],
    resume_filename: str = "resume.pdf",
    jd_title: Optional[str] = None,
) -> str:
    """
    Formats a candidate's structured analysis into a comprehensive Telegram HTML card.

    Includes:
    - Candidate name (with fallback)
    - ATS Score (0-100)
    - Job Match Percentage (0-100)
    - Matching Skills
    - Missing Skills (categorized into Critical, Important, Preferred)
    - Partially Matching Skills
    - Skill Breakdown with visual progress bars
    - Recommended Learning with topics
    - Actionable Recommendation
    """
    # 1. Candidate Name
    raw_name = (analysis.get("candidate_name") or "").strip()
    if not raw_name or raw_name.lower() in ("unknown", "n/a", "none", "candidate", "not specified"):
        clean_name = f"Candidate ({Path(resume_filename).stem})"
    else:
        clean_name = raw_name
    candidate_name = html.escape(clean_name)
    clean_filename = html.escape(resume_filename)

    # 2. Scores
    ats_score = max(0, min(100, int(analysis.get("ats_score", 0))))
    job_match = max(0, min(100, int(analysis.get("job_match_percentage", 0))))

    # 3. Matching Skills
    matching_skills = analysis.get("matching_skills") or []
    if matching_skills:
        matching_skills_str = "\n".join(
            f"• {html.escape(str(s))}" for s in matching_skills
        )
    else:
        matching_skills_str = "<i>None explicitly matched</i>"

    # 4. Partial Skills
    partial_skills = analysis.get("partial_skills") or []
    if partial_skills:
        partial_skills_str = "\n".join(
            f"• {html.escape(str(s))}" for s in partial_skills
        )
    else:
        partial_skills_str = "<i>None</i>"

    # 5. Missing Skills (Grouped by Importance: Critical, Important, Preferred)
    raw_missing = analysis.get("missing_skills") or []
    critical_skills: List[str] = []
    important_skills: List[str] = []
    preferred_skills: List[str] = []

    for item in raw_missing:
        if isinstance(item, dict):
            skill_name = str(item.get("skill") or "").strip()
            importance = str(item.get("importance") or "important").lower()
        else:
            skill_name = str(item).strip()
            importance = "important"

        if not skill_name:
            continue

        if "critical" in importance:
            critical_skills.append(skill_name)
        elif "preferred" in importance or "nice" in importance:
            preferred_skills.append(skill_name)
        else:
            important_skills.append(skill_name)

    missing_sections = []
    if critical_skills:
        crit_str = "\n".join(f"  • {html.escape(s)}" for s in critical_skills)
        missing_sections.append(f"🔴 <b>Critical:</b>\n{crit_str}")
    if important_skills:
        imp_str = "\n".join(f"  • {html.escape(s)}" for s in important_skills)
        missing_sections.append(f"🟡 <b>Important:</b>\n{imp_str}")
    if preferred_skills:
        pref_str = "\n".join(f"  • {html.escape(s)}" for s in preferred_skills)
        missing_sections.append(f"🟢 <b>Preferred:</b>\n{pref_str}")

    if missing_sections:
        missing_skills_str = "\n".join(missing_sections)
    else:
        missing_skills_str = "<i>No critical or important skills missing!</i>"

    # 6. Skill Breakdown
    skill_breakdown = analysis.get("skill_breakdown") or []
    if skill_breakdown:
        breakdown_lines = []
        for item in skill_breakdown:
            if isinstance(item, dict):
                s_name = html.escape(str(item.get("skill") or ""))
                s_pct = max(0, min(100, int(item.get("match_percentage", 0))))
                bar = render_progress_bar(s_pct)
                breakdown_lines.append(f"• <b>{s_name}:</b> {bar}")
            else:
                breakdown_lines.append(f"• {html.escape(str(item))}")
        skill_breakdown_str = "\n".join(breakdown_lines)
    else:
        skill_breakdown_str = "<i>Not available</i>"

    # 7. Recommended Learning
    recommended_learning = analysis.get("recommended_learning") or []
    if recommended_learning:
        learning_lines = []
        for i, item in enumerate(recommended_learning, start=1):
            if isinstance(item, dict):
                skill = html.escape(str(item.get("skill") or "Topic"))
                reason = item.get("reason")
                reason_line = f"   <i>{html.escape(reason)}</i>\n" if reason else ""
                topics = item.get("topics") or []
                if topics:
                    topics_str = "\n".join(f"   ▫️ {html.escape(str(t))}" for t in topics)
                else:
                    topics_str = "   ▫️ Practical hands-on tutorials"
                
                url = item.get("resource_url")
                url_line = ""
                if url and isinstance(url, str) and url.startswith("http"):
                    clean_url = html.escape(url.strip())
                    url_line = f"\n   🔗 <a href=\"{clean_url}\">Official Documentation / Learning Link</a>"

                learning_lines.append(f"<b>{i}. {skill}</b>\n{reason_line}{topics_str}{url_line}")
            else:
                learning_lines.append(f"<b>{i}. {html.escape(str(item))}</b>")
        recommended_learning_str = "\n\n".join(learning_lines)
    else:
        recommended_learning_str = "<i>No additional courses required.</i>"

    # 8. Learning Roadmap
    learning_roadmap = analysis.get("learning_roadmap") or []
    if learning_roadmap:
        roadmap_lines = []
        for step in learning_roadmap:
            if isinstance(step, dict):
                p_name = html.escape(str(step.get("phase") or f"Phase {step.get('step_number', 1)}"))
                focus = html.escape(str(step.get("skill_focus") or ""))
                dur = html.escape(str(step.get("duration") or ""))
                dur_str = f" ({dur})" if dur else ""
                focus_str = f" — <b>{focus}</b>" if focus else ""
                header = f"• <b>{p_name}</b>{dur_str}{focus_str}"
                
                step_topics = step.get("key_topics") or []
                t_str = ""
                if step_topics:
                    t_str = "\n" + "\n".join(f"  ▫️ {html.escape(str(t))}" for t in step_topics)
                
                url = step.get("learning_resource_url")
                url_str = ""
                if url and isinstance(url, str) and url.startswith("http"):
                    clean_url = html.escape(url.strip())
                    url_str = f"\n  🔗 <a href=\"{clean_url}\">Official Guide / Documentation</a>"
                
                roadmap_lines.append(f"{header}{t_str}{url_str}")
            else:
                roadmap_lines.append(f"• {html.escape(str(step))}")
        roadmap_str = "\n\n".join(roadmap_lines)
    else:
        roadmap_str = None

    # 9. Recommendation
    raw_rec = analysis.get("recommendation") or "Candidate review completed."
    recommendation = html.escape(raw_rec.strip())

    roadmap_block = f"\n\n🗺️ <b>Learning Roadmap:</b>\n{roadmap_str}" if roadmap_str else ""

    target_job_line = f"🎯 <b>Target Role:</b> <code>{html.escape(jd_title)}</code>\n" if jd_title else ""

    # Build the full formatted card
    card = (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "👤 <b>CANDIDATE ANALYSIS</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📄 <b>Resume:</b> <code>{clean_filename}</code>\n"
        f"{target_job_line}"
        f"👤 <b>Candidate:</b> <b>{candidate_name}</b>\n\n"
        f"🎯 <b>ATS SCORE:</b> <b>{ats_score}/100</b>\n"
        f"💼 <b>JOB MATCH:</b> <b>{job_match}%</b>\n\n"
        f"✅ <b>Matching Skills:</b>\n{matching_skills_str}\n\n"
        f"⚠️ <b>Partially Matching Skills:</b>\n{partial_skills_str}\n\n"
        f"❌ <b>Missing Skills:</b>\n{missing_skills_str}\n\n"
        f"📊 <b>Skill Breakdown:</b>\n{skill_breakdown_str}\n\n"
        f"📚 <b>Recommended Learning:</b>\n{recommended_learning_str}"
        f"{roadmap_block}\n\n"
        f"⭐ <b>Recommendation:</b>\n{recommendation}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return card


ORDINAL_EMOJIS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]


def format_candidate_ranking(
    ranked_candidates: List[Dict[str, Any]],
    failed_resumes: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Renders the summary 🏆 CANDIDATE RANKING card for multiple candidates.
    """
    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🏆 <b>CANDIDATE RANKING</b>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not ranked_candidates:
        lines.append("<i>No candidates could be successfully evaluated.</i>")
    else:
        for idx, cand in enumerate(ranked_candidates, start=1):
            emoji = ORDINAL_EMOJIS[idx - 1] if idx <= len(ORDINAL_EMOJIS) else f"<b>#{idx}</b>"
            raw_name = (cand.get("candidate_name") or "").strip()
            filename = cand.get("original_filename") or f"resume_{idx}.pdf"
            if not raw_name or raw_name.lower() in ("unknown", "n/a", "none", "candidate", "not specified"):
                clean_name = f"Candidate ({Path(filename).stem})"
            else:
                clean_name = raw_name

            job_match = max(0, min(100, int(cand.get("job_match_percentage", 0))))
            ats_score = max(0, min(100, int(cand.get("ats_score", 0))))

            lines.append(
                f"{emoji} <b>{html.escape(clean_name)}</b> (<code>{html.escape(filename)}</code>)\n"
                f"   💼 <b>Job Match:</b> {job_match}% | 🎯 <b>ATS:</b> {ats_score}%\n"
            )

    if failed_resumes:
        lines.append("⚠️ <b>Failed Analyses:</b>")
        for f in failed_resumes:
            fname = html.escape(f.get("filename", "unknown.pdf"))
            err = html.escape(f.get("error", "Processing error"))
            lines.append(f"• <code>{fname}</code>: <i>{err}</i>")
        lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


def format_multi_jd_comparison(
    candidate_name: str,
    resume_filename: str,
    jd_evaluations: List[Dict[str, Any]],
) -> str:
    """
    Renders the comparison card when evaluating 1 candidate against 2+ Job Descriptions.

    jd_evaluations is a list of dicts, sorted by best match first.
    Each item contains:
      - 'jd_title': e.g. "Senior Backend Developer" or "Backend_JD.pdf"
      - 'job_match_percentage': int
      - 'ats_score': int
    """
    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🎯 <b>JOB FIT COMPARISON FOR CANDIDATE</b>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"👤 <b>Candidate:</b> <b>{html.escape(candidate_name)}</b> (<code>{html.escape(resume_filename)}</code>)\n",
    ]

    for idx, eval_item in enumerate(jd_evaluations, start=1):
        emoji = ORDINAL_EMOJIS[idx - 1] if idx <= len(ORDINAL_EMOJIS) else f"<b>#{idx}</b>"
        title = html.escape(str(eval_item.get("jd_title", f"Role #{idx}")))
        match_pct = max(0, min(100, int(eval_item.get("job_match_percentage", 0))))
        ats_score = max(0, min(100, int(eval_item.get("ats_score", 0))))

        badge = " 🌟 <b>Best Fit!</b>" if idx == 1 else ""
        lines.append(
            f"{emoji} <b>{title}</b>{badge}\n"
            f"   💼 <b>Job Match:</b> {match_pct}% | 🎯 <b>ATS:</b> {ats_score}%\n"
        )

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)

