# Operational Analysis & Evaluation Report

## Model Performance on Sample Data
- **evidence_standard_met Accuracy**: 10.0%
- **claim_status Accuracy**: 10.0%
- **issue_type Accuracy**: 15.0%
- **object_part Accuracy**: 5.0%
- **severity Accuracy**: 10.0%

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
