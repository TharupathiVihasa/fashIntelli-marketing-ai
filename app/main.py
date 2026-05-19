from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .database import Base, SessionLocal, engine
from .routers import analyst, auth, public, user
from .seed import seed_default_users

# Enable CORS so React frontend can communicate with FastAPI backend
app = FastAPI(title='FashIntelli API', version='1.1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

BASE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BASE_DIR.parent
Base.metadata.create_all(bind=engine)
with SessionLocal() as db:
    seed_default_users(db)

static_dir = BASE_DIR / 'app' / 'static'
static_dir.mkdir(parents=True, exist_ok=True)
app.mount('/static', StaticFiles(directory=static_dir), name='static')

figures_dir = BASE_DIR / 'outputs' / 'figures'
figures_dir.mkdir(parents=True, exist_ok=True)
app.mount('/figures', StaticFiles(directory=figures_dir), name='figures')

app.include_router(auth.router)
app.include_router(public.router)
app.include_router(user.router)
app.include_router(analyst.router)


@app.get('/api/health')
def health():
    return {'status': 'ok', 'service': 'FashIntelli API'}


@app.get('/api')
def api_root():
    return {
        'service': 'FashIntelli API',
        'status': 'ok',
        'docs': '/docs',
        'health': '/api/health',
        'landing': '/api/public/landing',
        'note': 'Open the root URL / for the web interface, or /docs for interactive API docs.',
    }


frontend_dist = ROOT_DIR / 'frontend' / 'dist'
if frontend_dist.exists():
    app.mount('/', StaticFiles(directory=frontend_dist, html=True), name='frontend')
else:
    @app.get('/')
    def root_fallback():
        return JSONResponse({
            'message': 'Frontend build not found. Start the backend for API access, or build/run the frontend separately.',
            'api': '/api',
            'docs': '/docs',
        })
