from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .database import get_db
from .models import User
from .security import decode_token

security = HTTPBearer(auto_error=False)

# authentication and authorization
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')
    token = credentials.credentials
    try:
        payload = decode_token(token)
        user_id = int(payload.get('sub'))
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid or expired token')

    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='User not found or inactive')
    return user


def require_analyst(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != 'analyst':
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Analyst access required')
    return current_user
