import pandas as pd
from typing import Dict, List, Any

def load_data(base_path: str = "dataset") -> Dict[str, pd.DataFrame]:
    return {
        "claims": pd.read_csv(f"{base_path}/claims.csv"),
        "history": pd.read_csv(f"{base_path}/user_history.csv"),
        "evidence": pd.read_csv(f"{base_path}/evidence_requirements.csv")
    }

def get_user_history(history_df: pd.DataFrame, user_id: str) -> Dict[str, str]:
    user_data = history_df[history_df["user_id"] == user_id]
    if user_data.empty:
        return {"history_summary": "No previous history.", "history_flags": "none"}
    row = user_data.iloc[0]
    return {
        "history_summary": row["history_summary"],
        "history_flags": row["history_flags"]
    }

def get_evidence_requirements(evidence_df: pd.DataFrame, claim_object: str) -> str:
    relevant = evidence_df[(evidence_df["claim_object"] == claim_object) | (evidence_df["claim_object"] == "all")]
    reqs = []
    for _, row in relevant.iterrows():
        reqs.append(f"- Applies to: {row['applies_to']} -> Minimum Evidence: {row['minimum_image_evidence']}")
    return "\n".join(reqs)

def prepare_claim_context(row: pd.Series, history_df: pd.DataFrame, evidence_df: pd.DataFrame) -> Dict[str, Any]:
    user_id = row["user_id"]
    claim_object = row["claim_object"]
    
    history_info = get_user_history(history_df, user_id)
    evidence_reqs = get_evidence_requirements(evidence_df, claim_object)
    
    image_paths = row["image_paths"].split(";") if pd.notna(row["image_paths"]) else []
    
    return {
        "user_id": user_id,
        "image_paths": image_paths,
        "user_claim": row["user_claim"],
        "claim_object": claim_object,
        "history_summary": history_info["history_summary"],
        "history_flags": history_info["history_flags"],
        "evidence_requirements": evidence_reqs
    }
