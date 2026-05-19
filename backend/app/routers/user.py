from fastapi import APIRouter, Depends, Query

from ..dependencies import get_current_user
from ..models import User
from ..schemas import PredictionRequest
from ..services.analytics import get_analytics_service

router = APIRouter(prefix='/api/user', tags=['user'])

# GET endpoint used to load user dashboard data
@router.get('/dashboard')
def dashboard(brand: str | None = Query(default=None), _: User = Depends(get_current_user)):
    return get_analytics_service().user_dashboard(brand)

# Purchase Prediction Endpoint
@router.post('/predict')
def predict(payload: PredictionRequest, _: User = Depends(get_current_user)):
    return get_analytics_service().predict_purchase_intent(payload.model_dump())
