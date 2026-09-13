"""
Scoring Engine for JobFit AI.

Calculates deterministic ATS Score and Job Match Percentage using predefined weighted formulas.
Normalizes component scores (0-100), handles missing/empty categories gracefully,
and prevents arbitrary or hallucinated AI scores.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

# =======================================================
# Scoring Weights Configuration
# =======================================================
ATS_WEIGHTS: Dict[str, float] = {
    "required_skills": 0.40,
    "relevant_experience": 0.20,
    "relevant_keywords": 0.15,
    "projects": 0.10,
    "education": 0.05,
    "resume_structure": 0.10,
}

JOB_MATCH_WEIGHTS: Dict[str, float] = {
    "required_skills": 0.50,
    "relevant_experience": 0.25,
    "projects": 0.15,
    "education": 0.05,
    "preferred_skills": 0.05,
}


class CandidateEvaluationComponents(BaseModel):
    """Component scores (0-100) extracted from resume and JD analysis."""
    required_skills: Optional[float] = Field(default=0.0, description="0-100 match on mandatory skills")
    relevant_experience: Optional[float] = Field(default=0.0, description="0-100 relevance of past work experience")
    relevant_keywords: Optional[float] = Field(default=0.0, description="0-100 presence of relevant industry keywords")
    projects: Optional[float] = Field(default=0.0, description="0-100 alignment of portfolio/projects")
    education: Optional[float] = Field(default=0.0, description="0-100 alignment of educational background")
    resume_structure: Optional[float] = Field(default=0.0, description="0-100 formatting, readability, sectioning")
    preferred_skills: Optional[float] = Field(default=None, description="0-100 match on preferred/bonus skills (None if not specified)")


def clamp_score(value: Optional[float]) -> Optional[float]:
    """Clamps a component score strictly to the [0.0, 100.0] interval."""
    if value is None:
        return None
    try:
        val = float(value)
        return max(0.0, min(100.0, val))
    except (TypeError, ValueError):
        return 0.0


def calculate_weighted_score(
    components: Dict[str, Optional[float]],
    weights: Dict[str, float],
) -> int:
    """
    Computes a deterministic, normalized weighted score between 0 and 100.

    If a category is None (i.e. not applicable or missing from the JD),
    the remaining active categories are proportionally normalized so the
    total effective weight always equals 1.0 (100%).

    If all categories are None or total active weight is 0, returns 0.
    """
    total_active_weight = 0.0
    weighted_sum = 0.0

    for category, weight in weights.items():
        raw_val = components.get(category)
        if raw_val is not None:
            clamped_val = clamp_score(raw_val)
            if clamped_val is not None:
                weighted_sum += clamped_val * weight
                total_active_weight += weight

    # Zero requirements or no applicable categories
    if total_active_weight <= 0.0:
        return 0

    # Normalize by active weight to account for non-applicable / missing categories
    normalized_score = weighted_sum / total_active_weight

    # Ensure bounds and round cleanly to nearest integer
    final_score = round(max(0.0, min(100.0, normalized_score)))
    return int(final_score)


def calculate_ats_score(
    required_skills: Optional[float] = 0.0,
    relevant_experience: Optional[float] = 0.0,
    relevant_keywords: Optional[float] = 0.0,
    projects: Optional[float] = 0.0,
    education: Optional[float] = 0.0,
    resume_structure: Optional[float] = 0.0,
) -> int:
    """
    Calculates ATS Score (0 - 100) using predefined rubric:
    - Required Skills: 40%
    - Relevant Experience: 20%
    - Keywords: 15%
    - Projects: 10%
    - Education: 5%
    - Resume Structure: 10%
    """
    components = {
        "required_skills": required_skills,
        "relevant_experience": relevant_experience,
        "relevant_keywords": relevant_keywords,
        "projects": projects,
        "education": education,
        "resume_structure": resume_structure,
    }
    return calculate_weighted_score(components, ATS_WEIGHTS)


def calculate_job_match(
    required_skills: Optional[float] = 0.0,
    relevant_experience: Optional[float] = 0.0,
    projects: Optional[float] = 0.0,
    education: Optional[float] = 0.0,
    preferred_skills: Optional[float] = None,
) -> int:
    """
    Calculates Job Match Percentage (0 - 100) using predefined rubric:
    - Required Skills: 50%
    - Relevant Experience: 25%
    - Projects: 15%
    - Education: 5%
    - Preferred Skills: 5%
    """
    components = {
        "required_skills": required_skills,
        "relevant_experience": relevant_experience,
        "projects": projects,
        "education": education,
        "preferred_skills": preferred_skills,
    }
    return calculate_weighted_score(components, JOB_MATCH_WEIGHTS)


def compute_candidate_scores(
    components: CandidateEvaluationComponents,
) -> Dict[str, int]:
    """
    Calculates both ATS Score and Job Match Percentage from candidate evaluation components.
    """
    ats = calculate_ats_score(
        required_skills=components.required_skills,
        relevant_experience=components.relevant_experience,
        relevant_keywords=components.relevant_keywords,
        projects=components.projects,
        education=components.education,
        resume_structure=components.resume_structure,
    )
    job_match = calculate_job_match(
        required_skills=components.required_skills,
        relevant_experience=components.relevant_experience,
        projects=components.projects,
        education=components.education,
        preferred_skills=components.preferred_skills,
    )
    return {
        "ats_score": ats,
        "job_match_percentage": job_match,
    }


def rank_candidates(candidates: list) -> list:
    """
    Ranks candidate analysis results deterministically.

    Ranking Order:
    1. Job Match Percentage (descending, primary)
    2. ATS Score (descending, secondary tiebreaker)
    3. Candidate Name / Filename (ascending, tertiary tiebreaker for strict determinism)
    """
    return sorted(
        candidates,
        key=lambda c: (
            -int(c.get("job_match_percentage", 0)),
            -int(c.get("ats_score", 0)),
            str(c.get("candidate_name") or ""),
            str(c.get("original_filename") or ""),
        ),
    )
