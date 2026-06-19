import os
import time
import json
import logging
import PIL.Image
from PIL import ImageEnhance
import google.generativeai as genai
from typing import Dict, Any, List
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from schema import SingleImageAnalysis, ClaimVerdict, RiskFlag

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logging.warning("GEMINI_API_KEY is not set. Please ensure it's provided in your .env or environment variables.")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-pro")

def enhance_image(img: PIL.Image.Image) -> PIL.Image.Image:
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    sharpness = ImageEnhance.Sharpness(img)
    img = sharpness.enhance(1.5)
    return img

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

STAGE_2_SYSTEM = """
You are the lead forensic insurance claims investigator.
You have received independent analyses of individual images (Stage 1) for a claim.
You must synthesize these findings, along with the user's historical risk and evidence requirements, to make a final ClaimVerdict.

User Claim Conversation:
{user_claim}

Claim Object: {claim_object}

User History Summary:
{history_summary}

Evidence Requirements:
{evidence_requirements}

Individual Image Analyses (JSON):
{image_analyses}

Synthesize these findings and provide the exact structured output.
Make sure to add any user history risk flags.
"""

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=15, max=120))
def call_gemini_stage1(contents):
    return model.generate_content(
        contents,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=SingleImageAnalysis,
            temperature=0.0
        )
    )

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=15, max=120))
def call_gemini_stage2(contents):
    return model.generate_content(
        contents,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=ClaimVerdict,
            temperature=0.0
        )
    )

def process_claim(context: Dict[str, Any]) -> dict:
    analyses = []
    for img_path in context["image_paths"]:
        try:
            full_path = img_path if os.path.exists(img_path) else os.path.join("dataset", img_path)
            img = PIL.Image.open(full_path).convert("RGB")
            img = enhance_image(img)
            img.thumbnail((1024, 1024))
            
            image_id = os.path.basename(img_path).split('.')[0]
            
            prompt = STAGE_1_SYSTEM.format(
                user_claim=context["user_claim"],
                claim_object=context["claim_object"],
                image_id=image_id
            )
            
            response = call_gemini_stage1([prompt, img])
            analyses.append(json.loads(response.text))
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"Error processing image {img_path}: {e}")
            
    prompt = STAGE_2_SYSTEM.format(
        user_claim=context["user_claim"],
        claim_object=context["claim_object"],
        history_summary=context["history_summary"],
        evidence_requirements=context["evidence_requirements"],
        image_analyses=json.dumps(analyses, indent=2)
    )
    
    try:
        response = call_gemini_stage2([prompt])
        result = json.loads(response.text)
        
        hist_flag = context["history_flags"]
        if hist_flag != "none" and hist_flag not in result["risk_flags"]:
            result["risk_flags"].append(hist_flag)
            
        return result
    except Exception as e:
        logging.error(f"Gemini API error during synthesis for user {context['user_id']}: {e}")
        return {
            "evidence_standard_met": False,
            "evidence_standard_met_reason": str(e)[:200],
            "risk_flags": ["none"],
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": "API Error or failure during synthesis.",
            "supporting_image_ids": ["none"],
            "valid_image": False,
            "severity": "unknown"
        }
