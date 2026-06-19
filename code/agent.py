import os
import time
import json
import logging
import hashlib
import cv2
import numpy as np
import PIL.Image
from PIL import ImageEnhance
from typing import Dict, Any, List
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from schema import SingleImageAnalysis, ClaimVerdict, RiskFlag, CriticVerdict

from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logging.warning("GEMINI_API_KEY is not set.")
client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-2.5-flash-lite"

IMAGE_CACHE = {}

@retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=1, min=1, max=1))
def call_gemini(contents, response_schema=None):
    config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json"
    )
    if response_schema:
        config.response_schema = response_schema
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=config
    )
    return response

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
            
            prev = int(history_dict.get('previous_claims', history_dict.get('past_claim_count', 0)))
            rej = int(history_dict.get('rejected_claims', history_dict.get('rejected_claim', 0)))
            last_90 = int(history_dict.get('last_90_days_claim_count', 0))
            flags_str = str(history_dict.get('history_flags', history_dict.get('flags', ''))).lower()
            
            if prev == 0: return 0.1
            score = min(1.0, rej / prev)
            if last_90 >= 3: score += 0.3
            if "fraud" in flags_str or "manipulation" in flags_str: score = max(score, 0.95)
            elif "suspicious" in flags_str: score = max(score, 0.7)
            return min(1.0, score)
        except Exception as e:
            logging.error(f"Fraud scoring error: {e}")
            return 0.2

# ==========================================
# VISION PERCEPTION AGENT
# ==========================================
class VisionPerceptionAgent:
    STAGE_1_SYSTEM = '''
    You are a forensic insurance claims investigator analyzing a SINGLE image.
    Your task is to carefully inspect this specific image to identify any damage to the object claimed.
    Use 'internal_reasoning' to describe the lighting, angles, and any visible damage.
    Do not assume damage from the user's claim if you cannot visually see it.

    User Claim Conversation:
    {user_claim}

    Claim Object: {claim_object}
    Image ID: {image_id}

    Evaluate this image and provide the exact structured output.
    '''
    
    ZOOM_SYSTEM = '''
    You are analyzing 4 high-resolution zoomed quadrants of the original image.
    Look extremely closely for micro-scratches, small dents, or hairline cracks.
    
    User Claim Conversation:
    {user_claim}
    
    Claim Object: {claim_object}
    Image ID: {image_id}
    
    Evaluate these zoomed crops combined and provide the exact structured output.
    '''

    @staticmethod
    def pre_screen_image(img_path: str) -> Dict[str, Any]:
        result = {"valid": True, "flags": []}
        try:
            with PIL.Image.open(img_path) as pimg:
                ext = img_path.lower().split('.')[-1]
                actual_format = pimg.format.lower() if pimg.format else ""
                if ext in ["jpg", "jpeg"] and actual_format not in ["jpeg", "jpg"]:
                    result["flags"].extend(["non_original_image", "possible_manipulation"])
                exif = pimg.getexif()
                if exif:
                    from PIL import ExifTags
                    for k, v in exif.items():
                        if k in ExifTags.TAGS and ExifTags.TAGS[k] == 'Software':
                            if any(sw in str(v).lower() for sw in ['photoshop', 'gimp', 'lightroom']):
                                result["flags"].append("possible_manipulation")
        except Exception:
            pass
        img = cv2.imread(img_path)
        if img is None: return {"valid": False, "reason": "damage_not_visible", "flags": result["flags"]}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 10.0: return {"valid": False, "reason": "blurry_image", "flags": result["flags"]}
        mean_brightness = np.mean(gray)
        if mean_brightness < 10 or mean_brightness > 245: return {"valid": False, "reason": "low_light_or_glare", "flags": result["flags"]}
        return result

    @staticmethod
    def enhance_image(img: PIL.Image.Image) -> PIL.Image.Image:
        img = ImageEnhance.Contrast(img).enhance(1.2)
        return ImageEnhance.Sharpness(img).enhance(1.5)
        
    @staticmethod
    def get_quadrants(img: PIL.Image.Image) -> List[PIL.Image.Image]:
        w2, h2 = img.size[0] // 2, img.size[1] // 2
        return [img.crop((0,0,w2,h2)), img.crop((w2,0,img.size[0],h2)), img.crop((0,h2,w2,img.size[1])), img.crop((w2,h2,img.size[0],img.size[1]))]

    def analyze(self, img_path: str, context: Dict[str, Any]) -> dict:
        full_path = img_path if os.path.exists(img_path) else os.path.join("dataset", img_path)
        image_id = os.path.basename(img_path).split('.')[0]
        
        with open(full_path, "rb") as f:
            img_hash = hashlib.sha256(f.read()).hexdigest()
        if img_hash in IMAGE_CACHE: return IMAGE_CACHE[img_hash]
            
        screen_res = self.pre_screen_image(full_path)
        if not screen_res["valid"]:
            res = {"image_id": image_id, "internal_reasoning": f"Programmatic prescreen failed: {screen_res['reason']}", "is_usable": False, "issue_visible": False, "issue_type": "unknown", "object_part": "unknown", "damage_description": "Image rejected by pre-screen.", "quality_flags": [screen_res["reason"]] + screen_res.get("flags", []), "confidence": 1.0, "severity_estimate": "unknown"}
            IMAGE_CACHE[img_hash] = res
            return res
            
        img = PIL.Image.open(full_path).convert("RGB")
        img = self.enhance_image(img)
        img.thumbnail((1024, 1024))
        
        prompt = self.STAGE_1_SYSTEM.format(user_claim=context["user_claim"], claim_object=context["claim_object"], image_id=image_id)
        
        response = call_gemini([prompt, img], SingleImageAnalysis)
        analysis_data = json.loads(response.text.strip().removeprefix("```json").removesuffix("```").strip())
        
        existing_flags = analysis_data.get("quality_flags", [])
        for f in screen_res.get("flags", []):
            if f not in existing_flags: existing_flags.append(f)
        analysis_data["quality_flags"] = existing_flags
        
        if not analysis_data.get("issue_visible", False):
            try:
                zoom_prompt = self.ZOOM_SYSTEM.format(user_claim=context["user_claim"], claim_object=context["claim_object"], image_id=image_id)
                zoom_response = call_gemini([zoom_prompt] + self.get_quadrants(img), SingleImageAnalysis)
                zoom_data = json.loads(zoom_response.text.strip().removeprefix("```json").removesuffix("```").strip())
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
    STAGE_2_SYSTEM = '''
    You are the lead forensic insurance claims investigator.
    You must strictly cross-reference the visible damage against the "Evidence Requirements Checklist". 
    If the provided images do not meet the minimum requirements, you MUST set evidence_standard_met to false.

    User Claim Conversation:
    {user_claim}
    Claim Object: {claim_object}
    Fraud Risk Score: {fraud_score}
    Evidence Requirements Checklist:
    {evidence_requirements}

    Individual Image Analyses (JSON):
    {image_analyses}
    '''

    @staticmethod
    def synthesize(context: Dict[str, Any], fraud_score: float, analyses: List[Dict[str, Any]]) -> dict:
        prompt = SynthesisAgent.STAGE_2_SYSTEM.format(
            user_claim=context["user_claim"], claim_object=context["claim_object"],
            fraud_score=fraud_score, evidence_requirements=context["evidence_requirements"],
            image_analyses=json.dumps(analyses, indent=2)
        )
        response = call_gemini([prompt], ClaimVerdict)
        return json.loads(response.text.strip().removeprefix("```json").removesuffix("```").strip())

# ==========================================
# CRITIC AGENT
# ==========================================
class CriticAgent:
    CRITIC_SYSTEM = '''
    You are the Senior AI Auditor. Review the SynthesisAgent's Initial Verdict.
    Check Evidence Rules and Fraud Rules. If Fraud Score > 0.5, risk_flags MUST contain user_history_risk and manual_review_required.
    
    User Claim: {user_claim}
    Fraud Score: {fraud_score}
    Evidence Requirements: {evidence_requirements}
    Initial Verdict: {initial_verdict}
    '''

    @staticmethod
    def critique(context: Dict[str, Any], fraud_score: float, initial_verdict: dict) -> dict:
        prompt = CriticAgent.CRITIC_SYSTEM.format(
            user_claim=context["user_claim"], fraud_score=fraud_score,
            evidence_requirements=context["evidence_requirements"],
            initial_verdict=json.dumps(initial_verdict, indent=2)
        )
        response = call_gemini([prompt], CriticVerdict)
        result = json.loads(response.text.strip().removeprefix("```json").removesuffix("```").strip())
        return initial_verdict if result.get("approved") else result.get("corrected_verdict", initial_verdict)

# ==========================================
# ORCHESTRATOR
# ==========================================
def log_claim(user_id: str, n_images: int, total_ms: float, verdict: str):
    with open("run_log.jsonl", "a") as f:
        f.write(json.dumps({"user_id": user_id, "n_images": n_images, "total_ms": round(total_ms, 2), "verdict": verdict}) + "\n")

def process_claim(context: Dict[str, Any]) -> dict:
    t0 = time.time()
    fraud_score = FraudAnalysisAgent.calculate_fraud_score(context.get("history_summary", {}))
    vision_agent = VisionPerceptionAgent()
    analyses = []
    for img_path in context["image_paths"]:
        try:
            analyses.append(vision_agent.analyze(img_path, context))
        except Exception as e:
            logging.error(f"Vision Agent error on {img_path}: {e}")

    try:
        initial_verdict = SynthesisAgent.synthesize(context, fraud_score, analyses)
        hist_flag = context["history_flags"]
        if hist_flag != "none" and hist_flag not in initial_verdict["risk_flags"]: initial_verdict["risk_flags"].append(hist_flag)
        if fraud_score > 0.5 and "user_history_risk" not in initial_verdict["risk_flags"]:
            initial_verdict["risk_flags"].extend(["user_history_risk", "manual_review_required"])

        final_verdict = CriticAgent.critique(context, fraud_score, initial_verdict)
        if hist_flag != "none" and hist_flag not in final_verdict["risk_flags"]: final_verdict["risk_flags"].append(hist_flag)
        if fraud_score > 0.5 and "user_history_risk" not in final_verdict["risk_flags"]:
            final_verdict["risk_flags"].extend(["user_history_risk", "manual_review_required"])
            
        validated_verdict = ClaimVerdict.model_validate(final_verdict).model_dump()
        log_claim(context["user_id"], len(context["image_paths"]), (time.time() - t0) * 1000, validated_verdict.get("claim_status", "unknown"))
        return validated_verdict
    except Exception as e:
        logging.error(f"Agent Orchestration error for user {context['user_id']}: {e}")
        return {"evidence_standard_met": False, "evidence_standard_met_reason": str(e)[:200], "risk_flags": ["none"], "issue_type": "unknown", "object_part": "unknown", "claim_status": "not_enough_information", "claim_status_justification": "API Error", "supporting_image_ids": ["none"], "valid_image": False, "severity": "unknown"}

FALLBACK_SYSTEM = '''
You are a forensic insurance claims investigator. Analyze all provided images at once. Make a final ClaimVerdict.
User Claim: {user_claim}
Object: {claim_object}
History: {history_summary}
Requirements: {evidence_requirements}
'''

def process_claim_single_pass(context: Dict[str, Any]) -> dict:
    prompt = FALLBACK_SYSTEM.format(user_claim=context["user_claim"], claim_object=context["claim_object"], history_summary=context["history_summary"], evidence_requirements=context["evidence_requirements"])
    contents = [prompt]
    for img_path in context["image_paths"]:
        try:
            full_path = img_path if os.path.exists(img_path) else os.path.join("dataset", img_path)
            img = PIL.Image.open(full_path).convert("RGB")
            img.thumbnail((1024, 1024))
            contents.append(img)
            contents.append(f"Image ID: {os.path.basename(img_path).split('.')[0]}")
        except Exception:
            pass
    try:
        response = call_gemini(contents, ClaimVerdict)
        return json.loads(response.text.strip().removeprefix("```json").removesuffix("```").strip())
    except Exception as e:
        return {"evidence_standard_met": False, "evidence_standard_met_reason": str(e)[:200], "risk_flags": ["none"], "issue_type": "unknown", "object_part": "unknown", "claim_status": "not_enough_information", "claim_status_justification": "API Error", "supporting_image_ids": ["none"], "valid_image": False, "severity": "unknown"}
