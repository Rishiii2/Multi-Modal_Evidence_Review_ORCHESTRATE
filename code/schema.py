from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFORMATION = "not_enough_information"

class IssueType(str, Enum):
    DENT = "dent"
    SCRATCH = "scratch"
    CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"
    BROKEN_PART = "broken_part"
    MISSING_PART = "missing_part"
    TORN_PACKAGING = "torn_packaging"
    CRUSHED_PACKAGING = "crushed_packaging"
    WATER_DAMAGE = "water_damage"
    STAIN = "stain"
    NONE = "none"
    UNKNOWN = "unknown"

class ObjectPart(str, Enum):
    FRONT_BUMPER = "front_bumper"
    REAR_BUMPER = "rear_bumper"
    DOOR = "door"
    HOOD = "hood"
    WINDSHIELD = "windshield"
    SIDE_MIRROR = "side_mirror"
    HEADLIGHT = "headlight"
    TAILLIGHT = "taillight"
    FENDER = "fender"
    QUARTER_PANEL = "quarter_panel"
    BODY = "body"
    SCREEN = "screen"
    KEYBOARD = "keyboard"
    TRACKPAD = "trackpad"
    HINGE = "hinge"
    LID = "lid"
    CORNER = "corner"
    PORT = "port"
    BASE = "base"
    BOX = "box"
    PACKAGE_CORNER = "package_corner"
    PACKAGE_SIDE = "package_side"
    SEAL = "seal"
    LABEL = "label"
    CONTENTS = "contents"
    ITEM = "item"
    UNKNOWN = "unknown"

class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"

class RiskFlag(str, Enum):
    NONE = "none"
    BLURRY_IMAGE = "blurry_image"
    CROPPED_OR_OBSTRUCTED = "cropped_or_obstructed"
    LOW_LIGHT_OR_GLARE = "low_light_or_glare"
    WRONG_ANGLE = "wrong_angle"
    WRONG_OBJECT = "wrong_object"
    WRONG_OBJECT_PART = "wrong_object_part"
    DAMAGE_NOT_VISIBLE = "damage_not_visible"
    CLAIM_MISMATCH = "claim_mismatch"
    POSSIBLE_MANIPULATION = "possible_manipulation"
    NON_ORIGINAL_IMAGE = "non_original_image"
    TEXT_INSTRUCTION_PRESENT = "text_instruction_present"
    USER_HISTORY_RISK = "user_history_risk"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"

class SingleImageAnalysis(BaseModel):
    image_id: str = Field(description="The file name or ID of the evaluated image")
    internal_reasoning: str = Field(description="Visual description of the damage, lighting, and object in THIS specific image.")
    is_usable: bool = Field(description="True if the image is clear enough to evaluate.")
    issue_visible: bool = Field(description="True if damage is visible in this image.")
    issue_type: IssueType = Field(description="The visible issue type in this image.")
    object_part: ObjectPart = Field(description="The relevant object part in this image.")
    damage_description: str = Field(description="Brief description of the damage.")
    quality_flags: List[RiskFlag] = Field(description="Any risk flags specific to this image (e.g. blurry_image, wrong_angle).")
    confidence: float = Field(description="Confidence from 0.0 to 1.0.")
    severity_estimate: Severity = Field(description="Estimated severity based ONLY on this image.")

class ClaimVerdict(BaseModel):
    internal_reasoning: str = Field(description="Final reasoning synthesis taking all single image analyses into account.")
    evidence_standard_met: bool = Field(description="true if the image set is sufficient to evaluate the claim")
    evidence_standard_met_reason: str = Field(description="short reason for the evidence decision")
    risk_flags: List[RiskFlag] = Field(description="risk flags found in the claim or history")
    issue_type: IssueType = Field(description="visible issue type")
    object_part: ObjectPart = Field(description="relevant object part")
    claim_status: ClaimStatus = Field(description="final decision on the claim")
    claim_status_justification: str = Field(description="concise image-grounded explanation")
    supporting_image_ids: List[str] = Field(description="image IDs supporting the decision, use ['none'] if none")
    valid_image: bool = Field(description="true if the image set is usable for automated review")
    severity: Severity = Field(description="estimated damage severity")

class CriticVerdict(BaseModel):
    approved: bool = Field(description="True if the initial verdict is completely logically sound and contains no contradictions.")
    critique_reasoning: str = Field(description="The critic's internal reasoning and logic checking.")
    corrected_verdict: ClaimVerdict = Field(description="The corrected verdict if approved is False. If approved is True, return the original verdict.")
