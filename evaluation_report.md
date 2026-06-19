# Evaluation Report – Multi-Modal Evidence Review

**Samples evaluated:** dataset/sample_claims.csv (first 5 for benchmark)

---

## Strategy A – Two-Stage Pipeline (Map-Reduce)

**Description:**  
Each image is analyzed independently by Gemini 2.5 Pro. All per-image JSON objects are synthesized into a final verdict.

**Advantages:**
- Perfect `supporting_image_ids` attribution.
- Programmatic OpenCV pre-screening prevents LLM calls on broken images.
- Independent quality flagging avoids "competing image" confusion.

---

## Strategy B – Single-Pass Baseline

**Description:**  
All images for a claim are sent in a single Gemini call.

**Trade-offs:**
- Fewer API calls (cheaper).
- Weaker attribution and prone to hallucinating on multiple images.

---

## Final Strategy Selection

**Strategy A was selected for final `output.csv`** because of its strict attribution and modular reasoning flow.

---

## Operational Analysis

### Model Calls
- **Strategy A:** `N_images + 1` calls (average ~3 per claim)
- **Strategy B:** 1 call per claim

### Average Latency
- **Strategy A:** 804.87 seconds per claim
- **Strategy B:** 286.16 seconds per claim

### Output Consistency
- Strategy A and B agreed on `claim_status` 100.0% of the time.

### Caching & Pre-screening
- Implemented OpenCV `Laplacian` blur detection.
- Implemented `hashlib` SHA-256 caching for duplicate images to minimize token usage.
