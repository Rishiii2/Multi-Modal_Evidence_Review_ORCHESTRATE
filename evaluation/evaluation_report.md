# Operational Analysis & Evaluation Report

## Architecture Overview
This system uses a **V5 Multi-Agent Orchestrator** to analyze multi-modal evidence for insurance claims. To prevent hallucination and improve grounding, it relies on a Two-Stage (Map-Reduce) pipeline with a Critic Auditor Loop.

### Operational Metrics

- **Model Calls per Claim**:
  - **VisionPerceptionAgent**: 1 call per image. If `issue_visible` is false, it uses a **Zoom Fallback** triggering 1 additional call to analyze 4 cropped quadrants.
  - **SynthesisAgent**: 1 call per claim to merge findings.
  - **CriticAgent**: 1 call per claim to audit the Synthesis output.
  - *Average*: ~4-5 model calls per claim depending on the number of images and whether zoom is triggered.

- **Token Usage (Approximate)**:
  - Each image uses ~258 tokens (base resizing to 1024x1024). 
  - Input tokens per claim: ~1,500 - 3,000 tokens.
  - Output tokens per claim: ~400 tokens.

- **Cost Estimation**:
  - Assuming Gemini 1.5 Pro:
  - At $3.50 / 1M Input Tokens and $10.50 / 1M Output Tokens.
  - **Cost per Claim**: ~$0.012.
  - **Full Dataset (100 rows)**: ~$1.20.

- **Latency**:
  - Vision calls run sequentially per claim, but claims run concurrently via `ThreadPoolExecutor` (max_workers=5).
  - Single claim latency: ~15-25 seconds.
  - Total batch latency: ~5-10 minutes for 100 rows, heavily dependent on API rate limits.

### Throttling & Reliability
- **Tenacity Retries**: All LLM calls are wrapped in an `Exponential Backoff` (min=15s, max=120s) to gracefully handle TPM/RPM rate limits without dropping the process.
- **Fail-Safe Fallback**: If an unrecoverable exception occurs, `main.py` explicitly appends a `not_enough_information` fallback row to guarantee the output CSV matches the input CSV length perfectly, avoiding disqualification.
- **Image Caching**: A SHA-256 caching layer bypasses LLM calls if an image has already been analyzed.

### Pre-Screening Efficiency
Instead of spending tokens on invalid images, the pipeline uses OpenCV (Laplacian variance and mean pixels) and PIL (EXIF scanning) to programmatically reject blurry, heavily glared, or manipulated (e.g., Photoshop/screenshots) images in 0.05 seconds. This drastically reduces cost and prevents the LLM from hallucinating on garbage inputs.
