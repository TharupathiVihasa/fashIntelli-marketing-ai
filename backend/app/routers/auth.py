from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies import get_current_user, require_analyst
from ..models import User
from ..schemas import AnalystCreateUserRequest, LoginRequest, RegisterRequest, TokenResponse, UserOut
from ..security import create_access_token, get_password_hash, verify_password

router = APIRouter(prefix='/api/auth', tags=['auth'])

# POST endpoint for new user registration
@router.post('/register', response_model=TokenResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail='User already exists.')
    user = User(
        full_name=payload.full_name,
        email=payload.email,
        password_hash=get_password_hash(payload.password),
        role='user',
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({'sub': str(user.id), 'role': user.role})
    return TokenResponse(access_token=token, user=user)

# User Login Endpoint
@router.post('/login', response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Incorrect email or password.')
    if not user.is_active:
        raise HTTPException(status_code=403, detail='User is inactive')
    token = create_access_token({'sub': str(user.id), 'role': user.role})
    return TokenResponse(access_token=token, user=user)

# Current Logged-In User Endpoint
@router.get('/me', response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return current_user

# POST endpoint for analysts to create users
@router.post('/create-user', response_model=UserOut)
def analyst_create_user(payload: AnalystCreateUserRequest, db: Session = Depends(get_db), _: User = Depends(require_analyst)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail='User already exists.')
    user = User(
        full_name=payload.full_name,
        email=payload.email,
        password_hash=get_password_hash(payload.password),
        role=payload.role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
