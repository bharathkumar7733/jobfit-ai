"""
Unit tests for Phase 5: Scoring Engine.

Verifies deterministic calculation of ATS Score and Job Match Percentage
using predefined weights, handling missing categories, empty requirements,
and boundary cases.
"""

import pytest
from scoring_engine import (
    ATS_WEIGHTS,
    JOB_MATCH_WEIGHTS,
    CandidateEvaluationComponents,
    calculate_ats_score,
    calculate_job_match,
    clamp_score,
    compute_candidate_scores,
)


def test_perfect_candidate():
    """1. Perfect candidate with 100% in all categories."""
    ats = calculate_ats_score(
        required_skills=100,
        relevant_experience=100,
        relevant_keywords=100,
        projects=100,
        education=100,
        resume_structure=100,
    )
    job_match = calculate_job_match(
        required_skills=100,
        relevant_experience=100,
        projects=100,
        education=100,
        preferred_skills=100,
    )

    assert ats == 100
    assert job_match == 100
    assert 0 <= ats <= 100
    assert 0 <= job_match <= 100


def test_completely_unsuitable_candidate():
    """2. Candidate with 0% across all categories."""
    ats = calculate_ats_score(
        required_skills=0,
        relevant_experience=0,
        relevant_keywords=0,
        projects=0,
        education=0,
        resume_structure=0,
    )
    job_match = calculate_job_match(
        required_skills=0,
        relevant_experience=0,
        projects=0,
        education=0,
        preferred_skills=0,
    )

    assert ats == 0
    assert job_match == 0
    assert 0 <= ats <= 100
    assert 0 <= job_match <= 100


def test_half_required_skills():
    """
    3. Candidate with exactly 50% on required skills, and 100% on other categories.
    ATS calculation:
      50 * 0.40 = 20
      100 * 0.20 = 20
      100 * 0.15 = 15
      100 * 0.10 = 10
      100 * 0.05 = 5
      100 * 0.10 = 10
      Total ATS = 80
    Job Match calculation:
      50 * 0.50 = 25
      100 * 0.25 = 25
      100 * 0.15 = 15
      100 * 0.05 = 5
      100 * 0.05 = 5
      Total Job Match = 75
    """
    ats = calculate_ats_score(
        required_skills=50,
        relevant_experience=100,
        relevant_keywords=100,
        projects=100,
        education=100,
        resume_structure=100,
    )
    job_match = calculate_job_match(
        required_skills=50,
        relevant_experience=100,
        projects=100,
        education=100,
        preferred_skills=100,
    )

    assert ats == 80
    assert job_match == 75
    assert 0 <= ats <= 100
    assert 0 <= job_match <= 100


def test_excellent_experience_missing_skills():
    """
    4. Candidate with 100% experience but 0% required skills and other categories.
    ATS: 100 * 0.20 = 20
    Job Match: 100 * 0.25 = 25
    """
    ats = calculate_ats_score(
        required_skills=0,
        relevant_experience=100,
        relevant_keywords=0,
        projects=0,
        education=0,
        resume_structure=0,
    )
    job_match = calculate_job_match(
        required_skills=0,
        relevant_experience=100,
        projects=0,
        education=0,
        preferred_skills=0,
    )

    assert ats == 20
    assert job_match == 25
    assert 0 <= ats <= 100
    assert 0 <= job_match <= 100


def test_skills_no_experience():
    """
    5. Candidate with 100% required skills but 0% experience and other categories.
    ATS: 100 * 0.40 = 40
    Job Match: 100 * 0.50 = 50
    """
    ats = calculate_ats_score(
        required_skills=100,
        relevant_experience=0,
        relevant_keywords=0,
        projects=0,
        education=0,
        resume_structure=0,
    )
    job_match = calculate_job_match(
        required_skills=100,
        relevant_experience=0,
        projects=0,
        education=0,
        preferred_skills=0,
    )

    assert ats == 40
    assert job_match == 50
    assert 0 <= ats <= 100
    assert 0 <= job_match <= 100


def test_missing_education_information():
    """
    6. Candidate evaluation where education is None (not specified in JD).
    Remaining active weights should normalize to 100%.
    ATS: active weight = 0.95.
    If all other categories are 100, normalized score = (95 / 0.95) = 100.
    """
    ats = calculate_ats_score(
        required_skills=100,
        relevant_experience=100,
        relevant_keywords=100,
        projects=100,
        education=None,
        resume_structure=100,
    )
    job_match = calculate_job_match(
        required_skills=100,
        relevant_experience=100,
        projects=100,
        education=None,
        preferred_skills=100,
    )

    assert ats == 100
    assert job_match == 100


def test_missing_preferred_skills():
    """
    7. Candidate evaluation where preferred_skills is None (no bonus skills in JD).
    In Job Match: active weight = 0.50 + 0.25 + 0.15 + 0.05 = 0.95.
    If candidate has 80 in all required categories, normalized result should be 80.
    """
    job_match = calculate_job_match(
        required_skills=80,
        relevant_experience=80,
        projects=80,
        education=80,
        preferred_skills=None,
    )

    assert job_match == 80
    assert 0 <= job_match <= 100


def test_empty_requirements():
    """
    8. Case where all component criteria are None (e.g. Empty requirements).
    Should gracefully return 0 rather than dividing by zero or crashing.
    """
    ats = calculate_ats_score(
        required_skills=None,
        relevant_experience=None,
        relevant_keywords=None,
        projects=None,
        education=None,
        resume_structure=None,
    )
    job_match = calculate_job_match(
        required_skills=None,
        relevant_experience=None,
        projects=None,
        education=None,
        preferred_skills=None,
    )

    assert ats == 0
    assert job_match == 0


def test_score_boundary_cases():
    """
    9. Boundary cases: negative values, values > 100, floating point precision, rounding.
    Scores must strictly clamp: 0 <= score <= 100.
    """
    # Negative component scores should clamp to 0
    assert clamp_score(-15.5) == 0.0
    assert clamp_score(125.0) == 100.0

    ats_underflow = calculate_ats_score(
        required_skills=-50,
        relevant_experience=-10,
        relevant_keywords=0,
        projects=0,
        education=0,
        resume_structure=0,
    )
    assert ats_underflow == 0

    ats_overflow = calculate_ats_score(
        required_skills=200,
        relevant_experience=150,
        relevant_keywords=110,
        projects=105,
        education=100,
        resume_structure=100,
    )
    assert ats_overflow == 100

    job_match_overflow = calculate_job_match(
        required_skills=120,
        relevant_experience=110,
        projects=100,
        education=100,
        preferred_skills=100,
    )
    assert job_match_overflow == 100

    # Rounding precision test
    # e.g., 82.4 should round to 82, 82.6 should round to 83
    # 82.4 on required skills (40%), 0 on rest: 82.4 * 0.40 = 32.96 -> rounds to 33
    ats_round = calculate_ats_score(required_skills=82.4)
    assert ats_round == 33
    assert 0 <= ats_round <= 100


def test_compute_candidate_scores_helper():
    """Test compute_candidate_scores with CandidateEvaluationComponents model."""
    components = CandidateEvaluationComponents(
        required_skills=80,
        relevant_experience=70,
        relevant_keywords=90,
        projects=60,
        education=100,
        resume_structure=85,
        preferred_skills=50,
    )

    result = compute_candidate_scores(components)
    assert "ats_score" in result
    assert "job_match_percentage" in result
    assert 0 <= result["ats_score"] <= 100
    assert 0 <= result["job_match_percentage"] <= 100
