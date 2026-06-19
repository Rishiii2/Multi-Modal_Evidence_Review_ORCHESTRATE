import os
import time
import logging
import PIL.Image
from PIL import ImageEnhance
import google.generativeai as genai
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from schema import AgentDecision, RiskFlag

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logging.warning("GEMINI_API_KEY is not set. Please ensure it's provided in your .env or environment variables.")
genai.configure(api_key=api_key)

# We use gemini-2.5-pro since it supports multimodal large context and strict Pydantic JSON outputs.
model = genai.GenerativeModel("gemini-2.5-pro")

SYSTEM_PROMPT = """
You are a forensic insurance claims investigator specializing in damage evaluation.
You must review the user's claim conversation, their historical risk, and the provided images.
Follow the "Evidence Requirements" strictly. 
Images are the primary source of truth. Do not let user history override clear visual evidence.

First, use the `internal_reasoning` field to carefully describe each image. 
Note the lighting, angles, visible parts, and any damage. Determine if the evidence matches the claim.
Then, fill out the rest of the strict JSON structure.

User Claim Conversation:
{user_claim}

Claim Object: {claim_object}

User History Summary:
{history_summary}

Evidence Requirements:
{evidence_requirements}

Evaluate the evidence and provide the exact structured output requested.
"""

def enhance_image(img: PIL.Image.Image) -> PIL.Image.Image:
    # Slightly enhance contrast and sharpness to make damage (scratches/dents) more visible to VLM
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    sharpness = ImageEnhance.Sharpness(img)
    img = sharpness.enhance(1.5)
    return img

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=15, max=120))
def call_gemini(contents):
    response = model.generate_content(
        contents,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=AgentDecision,
            temperature=0.0
        )
    )
    return response

def process_claim(context: Dict[str, Any]) -> dict:
    prompt = SYSTEM_PROMPT.format(
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
            img = enhance_image(img)
            img.thumbnail((1024, 1024))
            contents.append(img)
            contents.append(f"Image ID: {os.path.basename(img_path).split('.')[0]}")
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")

    try:
        response = call_gemini(contents)
        import json
        result = json.loads(response.text)
        
        hist_flag = context["history_flags"]
        if hist_flag != "none" and hist_flag not in result["risk_flags"]:
            result["risk_flags"].append(hist_flag)
            
        return result
    except Exception as e:
        logging.error(f"Gemini API error for user {context['user_id']}: {e}")
        return {
            "evidence_standard_met": False,
            "evidence_standard_met_reason": str(e)[:200],
            "risk_flags": ["none"],
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": "API Error or failure to process images.",
            "supporting_image_ids": ["none"],
            "valid_image": False,
            "severity": "unknown"
        }
