import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

# In-memory dictionary: user_id (int) -> session data (dict)
_sessions: Dict[int, Dict[str, Any]] = {}

# User temp files directory root
TEMP_ROOT = Path(__file__).resolve().parent / "temp" / "users"


def get_user_temp_dir(user_id: int) -> Path:
    """Returns the dedicated temporary directory for a specific user."""
    user_dir = TEMP_ROOT / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def cleanup_user_files(user_id: int) -> None:
    """Removes all downloaded temporary files and folders for a given user."""
    user_dir = TEMP_ROOT / str(user_id)
    if user_dir.exists():
        try:
            shutil.rmtree(user_dir)
        except OSError:
            pass


def get_default_session() -> Dict[str, Any]:
    """Returns a clean, initial session dictionary."""
    return {
        "job_description": None,
        "job_descriptions": [],
        "resumes": [],
        "state": "idle",
        "selected_jd_index": None,
    }


def get_session(user_id: int) -> Dict[str, Any]:
    """
    Retrieve the session for a given user.
    If no session exists, creates and returns a default session.
    """
    if user_id not in _sessions:
        _sessions[user_id] = get_default_session()
    return _sessions[user_id]


def clear_session(user_id: int) -> None:
    """Clears/resets the session data and deletes temporary files for a user."""
    cleanup_user_files(user_id)
    if user_id in _sessions:
        _sessions[user_id].clear()
        _sessions[user_id].update(get_default_session())
    else:
        _sessions[user_id] = get_default_session()


def get_all_sessions() -> Dict[int, Dict[str, Any]]:
    """Returns all active sessions (useful for debugging/testing)."""
    return _sessions


def set_job_description(user_id: int, jd_text: str, filename: Optional[str] = None) -> None:
    """Sets the job description for a given user and advances state to waiting_for_resumes."""
    session = get_session(user_id)
    session["job_description"] = jd_text
    session["state"] = "waiting_for_resumes"
    if "job_descriptions" not in session:
        session["job_descriptions"] = []
    fname = filename or "Text Job Description"
    if not any(j.get("filename") == fname and j.get("text") == jd_text for j in session["job_descriptions"]):
        session["job_descriptions"].append({
            "filename": fname,
            "text": jd_text,
        })


def add_job_description(user_id: int, jd_text: str, filename: Optional[str] = None) -> None:
    """Adds a Job Description to the list of JDs and updates the active JD."""
    set_job_description(user_id, jd_text, filename)


def get_job_descriptions(user_id: int) -> List[Dict[str, Any]]:
    """Retrieves all stored job descriptions for a user."""
    return get_session(user_id).get("job_descriptions", [])


def get_job_description(user_id: int) -> Optional[str]:
    """Retrieves the stored job description for a given user, if any."""
    session = get_session(user_id)
    idx = session.get("selected_jd_index")
    jds = session.get("job_descriptions", [])
    if isinstance(idx, int) and 0 <= idx < len(jds):
        return jds[idx]["text"]
    return session.get("job_description")


def set_selected_jd(user_id: int, selection: Any) -> bool:
    """
    Sets the active JD selection for the user.
    selection can be:
      - an int index (0, 1, 2...)
      - 'all' for evaluating across all JDs
    Returns True if valid and set, False otherwise.
    """
    session = get_session(user_id)
    jds = session.get("job_descriptions", [])
    if isinstance(selection, str) and selection.strip().lower() == "all":
        session["selected_jd_index"] = "all"
        return True
    try:
        idx = int(selection)
        if 0 <= idx < len(jds):
            session["selected_jd_index"] = idx
            session["job_description"] = jds[idx]["text"]
            return True
    except (ValueError, TypeError):
        pass
    return False


def get_selected_jd_index(user_id: int) -> Any:
    """Returns current selected_jd_index (int, 'all', or None)."""
    return get_session(user_id).get("selected_jd_index")


def get_selected_jd(user_id: int) -> Optional[Dict[str, Any]]:
    """Returns the selected JD dict, or the primary JD dict if single JD or default."""
    session = get_session(user_id)
    jds = session.get("job_descriptions", [])
    if not jds:
        return None
    idx = session.get("selected_jd_index")
    if isinstance(idx, int) and 0 <= idx < len(jds):
        return jds[idx]
    return jds[0]


def get_active_jds_for_analysis(user_id: int) -> List[Dict[str, Any]]:
    """
    Returns the list of JDs to evaluate against.
    If 'all' is selected, returns all stored JDs.
    If an index is selected, returns [jds[idx]].
    If only 1 JD exists, returns [jds[0]].
    If multiple JDs exist and nothing selected yet, returns all stored JDs.
    """
    session = get_session(user_id)
    jds = session.get("job_descriptions", [])
    if not jds:
        return []
    idx = session.get("selected_jd_index")
    if idx == "all":
        return jds
    if isinstance(idx, int) and 0 <= idx < len(jds):
        return [jds[idx]]
    return jds


def add_resume(user_id: int, resume_meta: Dict[str, Any]) -> None:
    """Adds a validated resume metadata dictionary to the user's session."""
    session = get_session(user_id)
    session["resumes"].append(resume_meta)


def get_resumes(user_id: int) -> List[Dict[str, Any]]:
    """Retrieves the list of uploaded resume dictionaries for a given user."""
    return get_session(user_id).get("resumes", [])
