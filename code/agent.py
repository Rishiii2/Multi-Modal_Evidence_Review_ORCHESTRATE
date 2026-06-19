import os
import time
import json
import logging
import hashlib
import cv2
import numpy as np
import PIL.Image
from PIL import ImageEnhance
import google.generativeai as genai
from typing import Dict, Any, List
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from schema import SingleImageAnalysis, ClaimVerdict, RiskFlag, CriticVerdict

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logging.warning("GEMINI_API_KEY is not set.")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-pro")

IMAGE_CACHE = {}

# ==========================================
# FRAUD ANALYSIS AGENT
# ==========================================
class FraudAnalysisAgent:
    @staticmethod
    def calculate_fraud_score(history_summary: Any) -> float:
        try:
            if isinstance(history_summary, str):
                history_dict = json.loads(history_summary)
            else:
                history_dict = history_summary
            
            # Base variables
            prev = int(history_dict.get('previous_claims', history_dict.get('past_claim_count', 0)))
            rej = int(history_dict.get('rejected_claims', history_dict.get('rejected_claim', 0)))
            last_90 = int(history_dict.get('last_90_days_claim_count', 0))
            flags_str = str(history_dict.get('history_flags', history_dict.get('flags', ''))).lower()
            
            if prev == 0: 
                return 0.1
                
            # Base ratio
            score = min(1.0, rej / prev)
            
            # Velocity penalty
            if last_90 >= 3:
                score += 0.3
                
            # Direct flag penalty
            if "fraud" in flags_str or "manipulation" in flags_str:
                score = max(score, 0.95)
            elif "suspicious" in flags_str:
                score = max(score, 0.7)
                
            return min(1.0, score)
        except Exception as e:
            logging.error(f"Fraud scoring error: {e}")
            return 0.2

# ==========================================
# VISION PERCEPTION AGENT
# ==========================================
class VisionPerceptionAgent:
    STAGE_1_SYSTEM = """
    You are a forensic insurance claims investigator analyzing a SINGLE image.
    Your task is to carefully inspect this specific image to identify any damage to the object claimed.
    Use 'internal_reasoning' to describe the lighting, angles, and any visible damage.
    Do not assume damage from the user's claim if you cannot visually see it.

    User Claim Conversation:
    {user_claim}

    Claim Object: {claim_object}
    Image ID: {image_id}

    Evaluate this image and provide the exact structured output.
    """
    
    ZOOM_SYSTEM = """
    You are analyzing 4 high-resolution zoomed quadrants of the original image (Top-Left, Top-Right, Bottom-Left, Bottom-Right).
    Look extremely closely for micro-scratches, small dents, or hairline cracks that were missed in the wide shot.
    
    User Claim Conversation:
    {user_claim}
    
    Claim Object: {claim_object}
    Image ID: {image_id}
    
    Evaluate these zoomed crops combined and provide the exact structured output.
    """

    @staticmethod
    def pre_screen_image(img_path: str) -> Dict[str, Any]:
        result = {"valid": True, "flags": []}
        
        # 1. Image Format / Manipulation Check
        try:
            with PIL.Image.open(img_path) as pimg:
                ext = img_path.lower().split('.')[-1]
                actual_format = pimg.format.lower() if pimg.format else ""
                
                # If a .jpg is actually a PNG/WEBP, it's often a screenshot or downloaded edit
                if ext in ["jpg", "jpeg"] and actual_format not in ["jpeg", "jpg"]:
                    result["flags"].append("non_original_image")
                    result["flags"].append("possible_manipulation")
                    
                # Check EXIF for Photoshop or editing software
                exif = pimg.getexif()
                if exif:
                    from PIL import ExifTags
                    for k, v in exif.items():
                        if k in ExifTags.TAGS and ExifTags.TAGS[k] == 'Software':
                            if any(sw in str(v).lower() for sw in ['photoshop', 'gimp', 'lightroom']):
                                result["flags"].append("possible_manipulation")
        except Exception:
            pass

        # 2. OpenCV Quality Checks
        img = cv2.imread(img_path)
        if img is None:
            return {"valid": False, "reason": "damage_not_visible", "flags": result["flags"]}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance < 10.0:
            return {"valid": False, "reason": "blurry_image", "flags": result["flags"]}
            
        mean_brightness = np.mean(gray)
        if mean_brightness < 10 or mean_brightness > 245:
            return {"valid": False, "reason": "low_light_or_glare", "flags": result["flags"]}
            
        return result

    @staticmethod
    def enhance_image(img: PIL.Image.Image) -> PIL.Image.Image:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        sharpness = ImageEnhance.Sharpness(img)
        img = sharpness.enhance(1.5)
        return img
        
    @staticmethod
    def get_quadrants(img: PIL.Image.Image) -> List[PIL.Image.Image]:
        width, height = img.size
        w2, h2 = width // 2, height // 2
        return [
            img.crop((0, 0, w2, h2)),
            img.crop((w2, 0, width, h2)),
            img.crop((0, h2, w2, height)),
            img.crop((w2, h2, width, height))
        ]

    @staticmethod
    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=15, max=120))
    def _call_gemini(contents):
        return model.generate_content(
            contents,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=SingleImageAnalysis,
                temperature=0.0
            )
        )

    def analyze(self, img_path: str, context: Dict[str, Any]) -> dict:
        full_path = img_path if os.path.exists(img_path) else os.path.join("dataset", img_path)
        image_id = os.path.basename(img_path).split('.')[0]
        
        with open(full_path, "rb") as f:
            img_bytes = f.read()
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        
        if img_hash in IMAGE_CACHE:
            return IMAGE_CACHE[img_hash]
            
        screen_res = self.pre_screen_image(full_path)
        if not screen_res["valid"]:
            fast_analysis = {
                "image_id": image_id,
                "internal_reasoning": f"Programmatic prescreen failed: {screen_res['reason']}",
                "is_usable": False,
                "issue_visible": False,
                "issue_type": "unknown",
                "object_part": "unknown",
                "damage_description": "Image rejected by pre-screen.",
                "quality_flags": [screen_res["reason"]] + screen_res.get("flags", []),
                "confidence": 1.0,
                "severity_estimate": "unknown"
            }
            IMAGE_CACHE[img_hash] = fast_analysis
            return fast_analysis
            
        img = PIL.Image.open(full_path).convert("RGB")
        img = self.enhance_image(img)
        img.thumbnail((1024, 1024))
        
        prompt = self.STAGE_1_SYSTEM.format(
            user_claim=context["user_claim"],
            claim_object=context["claim_object"],
            image_id=image_id
        )
        
        response = self._call_gemini([prompt, img])
        analysis_data = json.loads(response.text)
        
        # Inject programmatic EXIF flags into LLM output
        existing_flags = analysis_data.get("quality_flags", [])
        for f in screen_res.get("flags", []):
            if f not in existing_flags:
                existing_flags.append(f)
        analysis_data["quality_flags"] = existing_flags
        
        # Progressive Spatial Cropping (Zoom Agent)
        if not analysis_data.get("issue_visible", False):
            quadrants = self.get_quadrants(img)
            zoom_prompt = self.ZOOM_SYSTEM.format(
                user_claim=context["user_claim"],
                claim_object=context["claim_object"],
                image_id=image_id
            )
            try:
                time.sleep(1)
                zoom_response = self._call_gemini([zoom_prompt] + quadrants)
                zoom_data = json.loads(zoom_response.text)
                if zoom_data.get("issue_visible", False):
                    analysis_data = zoom_data
                    analysis_data["internal_reasoning"] += " [DAMAGE FOUND VIA ZOOM AGENT]"
            except Exception as e:
                logging.error(f"Zoom agent failed: {e}")
                
        IMAGE_CACHE[img_hash] = analysis_data
        return analysis_data

# ==========================================
# SYNTHESIS AGENT
# ==========================================
class SynthesisAgent:
    STAGE_2_SYSTEM = """
    You are the lead forensic insurance claims investigator.
    You have received independent analyses of individual images for a claim.
    You must synthesize these findings to make a final ClaimVerdict.
    
    CRITICAL INSTRUCTION: You MUST strictly cross-reference the visible damage against the "Evidence Requirements Checklist". 
    If the provided images do not meet the minimum requirements, you MUST set `evidence_standard_met` to false.

    User Claim Conversation:
    {user_claim}

    Claim Object: {claim_object}

    Fraud Risk Score (0.0 to 1.0): {fraud_score} (If > 0.5, strictly enforce manual_review_required)

    Evidence Requirements Checklist:
    {evidence_requirements}

    Individual Image Analyses (JSON):
    {image_analyses}

    Synthesize these findings into a forensic, objective JSON verdict. Ensure you extract the exact image IDs that support your decision.
    """

    @staticmethod
    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=15, max=120))
    def synthesize(context: Dict[str, Any], fraud_score: float, analyses: List[Dict[str, Any]]) -> dict:
        prompt = SynthesisAgent.STAGE_2_SYSTEM.format(
            user_claim=context["user_claim"],
            claim_object=context["claim_object"],
            fraud_score=fraud_score,
            evidence_requirements=context["evidence_requirements"],
            image_analyses=json.dumps(analyses, indent=2)
        )
        response = model.generate_content(
            [prompt],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ClaimVerdict,
                temperature=0.0
            )
        )
        return json.loads(response.text)

# ==========================================
# CRITIC AGENT (Reflect & Critique Loop)
# ==========================================
class CriticAgent:
    CRITIC_SYSTEM = """
    You are the Senior AI Auditor. Review the SynthesisAgent's Initial Verdict.
    You must think step-by-step (Chain-of-Thought) internally, then provide your output.
    
    CRITICAL AUDIT TASKS:
    1. Check Evidence Rules: Does the verdict say `evidence_standard_met=true` but the minimum evidence requirements (e.g. 2 angles) were not provided? If so, this is a hallucination. Override it.
    2. Check Fraud Rules: If the Fraud Score > 0.5, `risk_flags` MUST contain `user_history_risk` and `manual_review_required`.
    3. Grounding: Are the `supporting_image_ids` actually the images that showed damage?
    
    User Claim: {user_claim}
    Fraud Score: {fraud_score}
    Evidence Requirements: {evidence_requirements}
    Initial Verdict: {initial_verdict}
    
    If it is mathematically and logically perfect, set `approved: true`.
    If you catch an error, set `approved: false` and return a `corrected_verdict`.
    """

    @staticmethod
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=15, max=60))
    def critique(context: Dict[str, Any], fraud_score: float, initial_verdict: dict) -> dict:
        prompt = CriticAgent.CRITIC_SYSTEM.format(
            user_claim=context["user_claim"],
            fraud_score=fraud_score,
            evidence_requirements=context["evidence_requirements"],
            initial_verdict=json.dumps(initial_verdict, indent=2)
        )
        response = model.generate_content(
            [prompt],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=CriticVerdict,
                temperature=0.0
            )
        )
        result = json.loads(response.text)
        if result.get("approved"):
            return initial_verdict
        else:
            return result.get("corrected_verdict", initial_verdict)

# ==========================================
# ORCHESTRATOR
# ==========================================
def log_claim(user_id: str, n_images: int, total_ms: float, verdict: str):
    log_file = "run_log.jsonl"
    record = {
        "user_id": user_id,
        "n_images": n_images,
        "total_ms": round(total_ms, 2),
        "verdict": verdict
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")

def process_claim(context: Dict[str, Any]) -> dict:
    t0 = time.time()
    
    fraud_score = FraudAnalysisAgent.calculate_fraud_score(context.get("history_summary", {}))
    
    vision_agent = VisionPerceptionAgent()
    analyses = []
    for img_path in context["image_paths"]:
        try:
            time.sleep(0.5)
            analysis = vision_agent.analyze(img_path, context)
            analyses.append(analysis)
        except Exception as e:
            logging.error(f"Vision Agent error on {img_path}: {e}")

    try:
        # Synthesis Stage
        initial_verdict = SynthesisAgent.synthesize(context, fraud_score, analyses)
        
        # Override risk flags from strict inputs
        hist_flag = context["history_flags"]
        if hist_flag != "none" and hist_flag not in initial_verdict["risk_flags"]:
            initial_verdict["risk_flags"].append(hist_flag)
            
        if fraud_score > 0.5 and "user_history_risk" not in initial_verdict["risk_flags"]:
            initial_verdict["risk_flags"].append("user_history_risk")
            initial_verdict["risk_flags"].append("manual_review_required")

        # Critique Stage
        final_verdict = CriticAgent.critique(context, fraud_score, initial_verdict)
        
        # Enforce overrides again just in case the critic dropped them
        if hist_flag != "none" and hist_flag not in final_verdict["risk_flags"]:
            final_verdict["risk_flags"].append(hist_flag)
        if fraud_score > 0.5 and "user_history_risk" not in final_verdict["risk_flags"]:
            final_verdict["risk_flags"].append("user_history_risk")
            final_verdict["risk_flags"].append("manual_review_required")
            
        # Enforce exact Pydantic schema structure
        validated_verdict = ClaimVerdict.model_validate(final_verdict).model_dump()
            
        t1 = time.time()
        log_claim(
            user_id=context["user_id"],
            n_images=len(context["image_paths"]),
            total_ms=(t1 - t0) * 1000,
            verdict=validated_verdict.get("claim_status", "unknown")
        )
            
        return validated_verdict
    except Exception as e:
        logging.error(f"Agent Orchestration error for user {context['user_id']}: {e}")
        return {
            "evidence_standard_met": False,
            "evidence_standard_met_reason": str(e)[:200],
            "risk_flags": ["none"],
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": "API Error or failure during orchestration.",
            "supporting_image_ids": ["none"],
            "valid_image": False,
            "severity": "unknown"
        }

FALLBACK_SYSTEM = """
You are a forensic insurance claims investigator.
Analyze all provided images at once.
Make a final ClaimVerdict.
User Claim Conversation:
{user_claim}
Claim Object: {claim_object}
History Summary: {history_summary}
Evidence Requirements: {evidence_requirements}
"""

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=15, max=120))
def call_gemini_fallback(contents):
    return model.generate_content(
        contents,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=ClaimVerdict,
            temperature=0.0
        )
    )

def process_claim_single_pass(context: Dict[str, Any]) -> dict:
    t0 = time.time()
    prompt = FALLBACK_SYSTEM.format(
        user_claim=context["user_claim"],
        claim_object=context["claim_object"],
        history_summary=context["history_summary"],
        evidence_requirements=context["evidence_requirements"]
    )
    contents = [prompt]
    for img_path in context["image_paths"]:
        try:
            full_path = img_path if os.path.exists(img_path) else os.path.join("dataset", img_path)
            img = PIL.Image.open(full_path).convert("RGB")
            img.thumbnail((1024, 1024))
            contents.append(img)
            contents.append(f"Image ID: {os.path.basename(img_path).split('.')[0]}")
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
    try:
        response = call_gemini_fallback(contents)
        return json.loads(response.text)
    except Exception as e:
        return {
            "evidence_standard_met": False,
            "evidence_standard_met_reason": str(e)[:200],
            "risk_flags": ["none"],
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": "API Error",
            "supporting_image_ids": ["none"],
            "valid_image": False,
            "severity": "unknown"
        }
