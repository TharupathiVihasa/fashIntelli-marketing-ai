# FashIntelli
Author - K.D.Tharupathi Vihasa
IIT ID - 20212041
UOW ID - w1871490

**Explainable AI Decision Support System for Fashion Marketing**



## Role separation

### Analyst
Analysts can access:
- full analytics
- model training
- feature importance
- model performance charts
- user creation and registry

### User
Users can access:
- sign up and sign in
- brand selection
- platform recommendation
- campaign brief
- prediction studio
- PDF summary download
- dataset download




## Frontend and backend run steps

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```
- GUI: `http://localhost:8000`
- API info: `http://localhost:8000/api`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/health`
### Frontend source development (optional)
The project also includes the React frontend source under `frontend/src/`.
For normal use, you do **not** need to run the frontend separately because the backend already serves the completed GUI from `frontend/dist`.

If you want to work on the frontend source manually:
```bash
cd frontend
npm install
npm run dev
```

## Important if you used an older broken version before

Delete this file once before running the new pack:

```text
backend/storage/fashintelli.db
```

That ensures the default accounts and database state are recreated cleanly.

## Default accounts

- Analyst: `analyst@example.com` / `Analyst123!`
- User: `user@example.com` / `User12345!`


## Main folders

- `backend/` → FastAPI backend
- `frontend/` → frontend source and completed GUI bundle
- `backend/artifacts/` → trained artifacts
- `backend/outputs/` → generated outputs and figures

## Notes on training

Brand-specific **engagement** retraining is supported when the selected brand has enough social rows.
Brand-specific **purchase-intention** retraining still depends on the survey dataset having a reliable `brand` field, so the analyst interface keeps this as a transparent data-readiness note.



- Analysts can upload new survey and social datasets from the frontend when they follow the same template format.
- Analysts can optionally trigger retraining immediately after upload.
- User-facing results automatically reflect the latest trained backend outputs.
- Dataset and PDF downloads are fixed and now return working files.
