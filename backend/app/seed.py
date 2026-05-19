from sqlalchemy.orm import Session

from .models import User
from .security import get_password_hash

# Automatically create default analyst and user accounts
DEFAULT_USERS = [
    {
        'full_name': 'Lead Analyst',
        'email': 'analyst@example.com',
        'password': 'Analyst123!',
        'role': 'analyst',
    },
    {
        'full_name': 'Fashion User',
        'email': 'user@example.com',
        'password': 'User12345!',
        'role': 'user',
    },
]


def seed_default_users(db: Session) -> None:
    for entry in DEFAULT_USERS:
        existing = db.query(User).filter(User.email == entry['email']).first()
        if existing:
            continue
        db.add(User(
            full_name=entry['full_name'],
            email=entry['email'],
            password_hash=get_password_hash(entry['password']),
            role=entry['role'],
            is_active=True,
        ))
    db.commit()
