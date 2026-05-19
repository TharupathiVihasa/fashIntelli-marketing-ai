from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies import require_analyst
from ..models import User
from ..schemas import TrainingRequest, UserOut
from ..services.analytics import get_analytics_service

router = APIRouter(prefix='/api/analyst', tags=['analyst'])

# Analyst Dashboard Endpoint
@router.get('/dashboard')
def dashboard(_: User = Depends(require_analyst)):
    return get_analytics_service().analyst_dashboard()

# Model Training Endpoint
@router.post('/train')
def train(payload: TrainingRequest, _: User = Depends(require_analyst)):
    return get_analytics_service().train_models(payload.brand, generate_pdf_report=payload.generate_pdf_report)

# Dataset Upload Endpoint
@router.post('/upload-datasets')
async def upload_datasets(
    survey_file: UploadFile | None = File(default=None),
    social_file: UploadFile | None = File(default=None),
    train_immediately: bool = Form(default=False),
    brand: str | None = Form(default=None),
    generate_pdf_report: bool = Form(default=False),
    _: User = Depends(require_analyst),
):
    try:
        survey_bytes = await survey_file.read() if survey_file else None
        social_bytes = await social_file.read() if social_file else None
        return get_analytics_service().upload_datasets(
            survey_bytes=survey_bytes,
            survey_filename=survey_file.filename if survey_file else None,
            social_bytes=social_bytes,
            social_filename=social_file.filename if social_file else None,
            train_immediately=train_immediately,
            brand=brand,
            generate_pdf_report=generate_pdf_report,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

# GET endpoint to retrieve all registered users

@router.get('/users', response_model=list[UserOut])
def users(db: Session = Depends(get_db), _: User = Depends(require_analyst)):
    return db.query(User).order_by(User.created_at.desc()).all()
