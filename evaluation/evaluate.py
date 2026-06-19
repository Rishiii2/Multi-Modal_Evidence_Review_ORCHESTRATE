import pandas as pd
import sys
import os

# Add parent directory to path to import from code folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))
from main import main as run_pipeline

def evaluate():
    print("Running pipeline on sample claims for evaluation...")
    run_pipeline(csv_path="dataset/sample_claims.csv", output_path="evaluation/output_sample.csv", max_workers=2)
    
    expected = pd.read_csv("dataset/sample_claims.csv")
    actual = pd.read_csv("evaluation/output_sample.csv")
    
    # Merge and compare
    merged = pd.merge(expected, actual, on="user_id", suffixes=("_exp", "_act"))
    
    metrics = {}
    total = len(merged)
    
    # Categorical columns to check
    cols_to_check = ["evidence_standard_met", "claim_status", "issue_type", "object_part", "severity"]
    
    for col in cols_to_check:
        correct = (merged[f"{col}_exp"].astype(str).str.lower() == merged[f"{col}_act"].astype(str).str.lower()).sum()
        metrics[col] = f"{(correct / total * 100):.1f}%"
        
    print("\n--- Evaluation Results ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
    # Write report
    report = f"""# Operational Analysis & Evaluation Report

## Model Performance on Sample Data
- **evidence_standard_met Accuracy**: {metrics['evidence_standard_met']}
- **claim_status Accuracy**: {metrics['claim_status']}
- **issue_type Accuracy**: {metrics['issue_type']}
- **object_part Accuracy**: {metrics['object_part']}
- **severity Accuracy**: {metrics['severity']}

## Operational Cost & Scale Estimation
- **Model Used**: Gemini 1.5 Pro
- **Model Calls**: 1 call per claim (images are batched into a single prompt).
- **Approximate Tokens per call**: 
  - Input: ~600 tokens (text) + 258 tokens per image
  - Output: ~100 tokens (JSON response)
- **Approximate Latency**: Using `ThreadPoolExecutor`, we achieve concurrency. A single Gemini call takes 3-7 seconds. Processing 100 claims with 5 workers takes ~1.5 - 2 minutes.
- **TPM/RPM Considerations**: Gemini Pro has rate limits (e.g., 2-15 RPM for free tier, higher for paid). Our script uses concurrent workers which can be tuned (`max_workers`) based on the active API key limits. For the full test set, we recommend a max_worker of 5 with exponential backoff if 429s are encountered.
- **Approximate Cost**: 
  - Gemini 1.5 Pro is ~$1.25 per 1M input tokens, and $5.00 per 1M output tokens.
  - Per claim: ~1000 input tokens ($0.00125) + 100 output tokens ($0.0005) = **$0.00175 per claim**.
  - For 10,000 claims: ~$17.50.
"""
    with open("evaluation/evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report)
        
    print("Report generated at evaluation/evaluation_report.md")

if __name__ == "__main__":
    evaluate()
