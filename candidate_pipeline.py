import pandas as pd

# Map for past company tier: 1 = top-tier, 3 = lower-tier
TIER_MAP = {1: 1.0, 2: 0.8, 3: 0.6}


def load_data(path: str) -> pd.DataFrame:
    """Load the CSV file from disk."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the applicant dataset."""
    df = df.copy()

    # Expected numeric columns (change names if your CSV is different)
    numeric_cols = [
        "Age",
        "Years of Experience",
        "Test Score",
        "Interview Score",
        "Past Company Tier",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Categorical handling
    if "Degree" in df.columns:
        df["Degree"] = df["Degree"].fillna("Unknown")

    if "Location Preference" in df.columns:
        df["Location Preference"] = df["Location Preference"].fillna("Not Specified")

    if "Technical Skills" in df.columns:
        df["Technical Skills"] = df["Technical Skills"].fillna("")
    else:
        df["Technical Skills"] = ""

    # Standardize skills to a list
    df["skills_list"] = (
        df["Technical Skills"]
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.split(",")
        .apply(lambda skills: [s.strip() for s in skills if s.strip()])
    )

    return df


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized components and final score for each candidate."""
    df = df.copy()

    # Normalized numeric features
    df["test_norm"] = df["Test Score"] / 100.0
    df["interview_norm"] = df["Interview Score"] / 10.0
    df["exp_norm"] = df["Years of Experience"].clip(upper=10) / 10.0

    df["tier_norm"] = df["Past Company Tier"].map(TIER_MAP)
    df["tier_norm"] = df["tier_norm"].fillna(0.6)  # default if missing

    # Skill-based bonus
    def skill_bonus(skills):
        skills = set(skills)
        bonus = 0.0
        if "python" in skills:
            bonus += 0.03
        if "sql" in skills:
            bonus += 0.03
        if "excel" in skills or "power bi" in skills or "tableau" in skills:
            bonus += 0.02
        return bonus

    df["skill_bonus"] = df["skills_list"].apply(skill_bonus)

    # Default weights (these can be changed later by sliders in the app)
    w_test = 0.4
    w_interview = 0.3
    w_exp = 0.2
    w_tier = 0.1

    df["final_score"] = (
        w_test * df["test_norm"]
        + w_interview * df["interview_norm"]
        + w_exp * df["exp_norm"]
        + w_tier * df["tier_norm"]
        + df["skill_bonus"]
    )

    return df


def recompute_scores_with_weights(
    df: pd.DataFrame,
    w_test: float,
    w_interview: float,
    w_exp: float,
    w_tier: float,
) -> pd.DataFrame:
    """Recalculate final_score with new weights (weights already normalized)."""
    df = df.copy()
    df["final_score"] = (
        w_test * df["test_norm"]
        + w_interview * df["interview_norm"]
        + w_exp * df["exp_norm"]
        + w_tier * df["tier_norm"]
        + df["skill_bonus"]
    )
    return df


def get_top_candidates(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return top N candidates sorted by final_score."""
    return df.sort_values("final_score", ascending=False).head(n)
