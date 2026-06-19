# V5 SOTA Multi-Agent Architecture

This project is a highly sophisticated, State-of-the-Art (SOTA) Multi-Agent AI system designed for extreme accuracy in Multi-Modal Evidence Review. It goes beyond simple API calls and mimics the cognitive loops of expert human adjusters.

## The Agent Orchestration Loop

Instead of a monolithic script, `agent.py` orchestrates 4 independent AI personas:

### 1. VisionPerceptionAgent (with Progressive Spatial Cropping)
The Vision Agent first performs a deterministic OpenCV scan (Laplacian variance for blur, mean pixels for glare). If the image is valid, it sends it to the Vision-Language Model. 
**[SOTA Feature] Zoom Fallback**: If the wide-shot image is deemed "no damage visible", the agent uses `PIL` to crop the image into 4 high-resolution quadrants. It then re-prompts the VLM to analyze the zoomed crops to catch micro-scratches that were lost to VLM token compression.

### 2. FraudAnalysisAgent
A deterministic logic engine that calculates a `fraud_score` based on rejected claims history. If the score is >0.5, it strictly enforces manual review flags.

### 3. SynthesisAgent
Gathers the output from the VisionPerceptionAgent and the FraudAnalysisAgent to generate an initial `ClaimVerdict`. It uses Few-Shot prompting and explicit checklists to guarantee structure.

### 4. CriticAgent (Reflect & Critique)
**[SOTA Feature] LLM-as-a-Judge**: Before the verdict is finalized, it is passed to the `CriticAgent`. This auditor model checks the `SynthesisAgent`'s logic for hallucinations or contradictions against the Evidence Requirements. If it catches an error, it overrides the verdict with a `corrected_verdict`.

## Operational Efficiency
- **SHA-256 Caching**: Identical raw image bytes bypass the LLM and are pulled instantly from memory.
- **Structured Logging**: `run_log.jsonl` provides continuous observability for latency, confidence, and verdicts.
- **Evaluation Benchmark**: `evaluation/main.py` explicitly proves the superiority of this Multi-Agent architecture against a single-pass baseline.