from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=150)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class AnalystCreateUserRequest(RegisterRequest):
    role: str = Field(default='user', pattern='^(user|analyst)$')


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserOut(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    role: str
    is_active: bool

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = 'bearer'
    user: UserOut


class PredictionRequest(BaseModel):
    brand: Optional[str] = 'nolimit_srilanka'
    platform: str = 'facebook'
    age_range: str = '25-34'
    gender: str = 'Female'
    occupation: str = 'Employed'
    income: str = '25,000-50,000'
    platform_usage_count: int = Field(default=2, ge=1, le=10)
    ad_exposure_ordinal: int = Field(default=4, ge=1, le=5)
    influencer_trust: int = Field(default=4, ge=1, le=5)
    influencer_recommend: int = Field(default=4, ge=1, le=5)
    ad_appeal: int = Field(default=4, ge=1, le=5)
    brand_engagement_trust: int = Field(default=4, ge=1, le=5)
    social_media_presence: int = Field(default=4, ge=1, le=5)
    social_proof_confidence: int = Field(default=4, ge=1, le=5)
    past_purchase_flag: int = Field(default=3, ge=1, le=5)
    overall_influence: int = Field(default=4, ge=1, le=5)


class TrainingRequest(BaseModel):
    brand: Optional[str] = None
    generate_pdf_report: bool = False


class MessageResponse(BaseModel):
    message: str
    details: Optional[Dict[str, Any]] = None


class DashboardResponse(BaseModel):
    payload: Dict[str, Any]
