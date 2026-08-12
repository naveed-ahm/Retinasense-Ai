# RetinaSense AI

End-to-end AI-powered retinal analysis system. Upload fundus photographs, get multi-class diagnosis via deep learning (EfficientNetB3), and generate clinical reports as PDFs.

**Stack:** FastAPI · SQLAlchemy async · PostgreSQL / SQLite · React 18 (Vite) · Recharts · Docker · TensorFlow 2.21

## Architecture

```
retinasense/
├── backend/           FastAPI server (Python 3.13)
│   ├── app/
│   │   ├── ai/        Model training, inference, Grad-CAM
│   │   ├── dataset/   Dataset pipeline (EyePACS, Glaucoma, AMD)
│   │   ├── models/    SQLAlchemy ORM (User, Patient, Scan, Prediction, Report)
│   │   ├── routers/   10 API routers (auth, patients, scans, reports, …)
│   │   ├── schemas/   Pydantic request/response models
│   │   ├── services/  Business logic (auth, AI, reports, analytics, PDF)
│   │   └── utils/     JWT, password hashing, rate limiting
│   └── tests/         pytest suite (55 tests)
├── frontend/          React SPA (Vite, TypeScript, Tailwind)
│   ├── src/
│   │   ├── app/       App shell + page components
│   │   └── services/  API client + React hooks
│   └── __tests__/     vitest suite (9 tests)
└── docker-compose.yml PostgreSQL 16 + backend + frontend (nginx)
```

## Quick Start

### Prerequisites
- Python 3.13+
- Node.js 22+ and pnpm 11+ (pnpm 11 requires Node ≥ 22.13)
- PostgreSQL 16 (optional — SQLite used by default in dev)

### Backend

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Server starts at `http://127.0.0.1:8000`. Tables are created automatically on first start. Interactive API docs at `http://127.0.0.1:8000/docs`.

### Frontend

```bash
cd frontend
pnpm install
pnpm run dev
```

Opens at `http://localhost:5173`. The SPA calls the backend directly at `http://127.0.0.1:8000/api` (CORS is configured for the dev origin); start the backend first.

### Creating the first user

There is no public sign-up — users are provisioned by an operator. The dev database (`backend/retinasense.db`, gitignored) already contains a demo user:

- **`dr.rajan@apollo.org` / `demo1234`**

For a fresh database (e.g. Docker/PostgreSQL), create a user with the app's own helpers:

```bash
cd backend
python -c "
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.models.user import User
from app.utils.security import hash_password
engine = create_engine('postgresql://retinasense:retinasense_secret@localhost:5432/retinasense')
with Session(engine) as s:
    s.add(User(email='admin@retinasense.org', password_hash=hash_password('ChangeMe!123'),
               first_name='Admin', last_name='User', role='Admin'))
    s.commit()
print('user created')
"
```

## Docker

```bash
docker compose up -d --build
```

Starts 3 services:

| Service | Container | Port |
|---------|-----------|------|
| PostgreSQL 16 | `retinasense-postgres` | 5432 |
| Backend (FastAPI) | `retinasense-backend` | 8000 |
| Frontend (nginx) | `retinasense-frontend` | 3000 |

The nginx container serves the built SPA and proxies `/api/*` to the backend. To run real inference, mount a trained model file at `app/ai/models` (the `model_data` volume). Without a model the API runs in **mock mode** (`is_mock: true`) so the UI stays fully usable.

### Production configuration

Copy `.env.example` to `.env` and adjust values — every variable is optional and falls back to the defaults in `docker-compose.yml`:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `POSTGRES_PASSWORD` | `retinasense_secret` | Postgres password (also used in the backend `DATABASE_URL`) |
| `SECRET_KEY` | `retinasense-secret-key-change-in-production` | JWT signing key — generate with `python -c "import secrets; print(secrets.token_hex(32))"` |
| `CORS_ORIGINS` | dev origins (`localhost:5173`, `localhost:3000`, `frontend:80`) | JSON list of allowed browser origins |
| `DEBUG` | `false` | Debug mode |
| `BACKEND_PORT` | `8000` | Published backend port |
| `FRONTEND_PORT` | `3000` | Published frontend port |

> **Security note:** in production you must set a strong `SECRET_KEY` and `POSTGRES_PASSWORD`. The default values trigger a backend startup warning. Terminate TLS at a reverse proxy in front of `FRONTEND_PORT` (the bundled nginx serves plain HTTP).

## API Endpoints

All endpoints below `/api/*` require `Authorization: Bearer <JWT>` unless marked otherwise.

### Auth
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/auth/login` | — | Login, returns JWT (rate-limited: 5/min/IP) |
| POST | `/api/auth/forgot-password` | — | Request password reset |
| POST | `/api/auth/request-access` | — | Submit access request |

### Dashboard & Analytics
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/dashboard/stats` | JWT | KPIs, charts, activity |
| GET | `/api/analytics/summary` | JWT | Aggregate analytics |
| GET | `/api/analysis/latest` | JWT | Latest scan analysis |

### Patients
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/patients` | JWT | List/search patients (paginated) |
| POST | `/api/patients` | JWT | Create patient |
| GET | `/api/patients/{id}` | JWT | Patient detail + scan history |
| GET | `/api/patients/{id}/report/pdf` | JWT | Download patient PDF report |
| DELETE | `/api/patients/{id}` | JWT | Delete patient |

### Scans
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/scans` | JWT | Upload fundus image + analyze |
| GET | `/api/scans` | JWT | List scans (search/filter/sort, paginated) |
| GET | `/api/scans/{id}/status` | JWT | Poll scan progress |
| GET | `/api/scans/{id}/analysis` | JWT | AI analysis results |
| GET | `/api/scans/{id}/report` | JWT | Clinical report data |
| GET | `/api/scans/{id}/report/pdf` | JWT | Download per-scan PDF report |
| DELETE | `/api/scans/{id}/report` | JWT | Delete a scan's report |
| DELETE | `/api/scans/{id}` | JWT | Delete a scan (and its report/prediction) |

### Reports
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/reports/latest` | JWT | Latest report data |
| GET | `/api/reports/latest/pdf` | JWT | Download latest report PDF |

### Settings
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET/PUT | `/api/settings/profile` | JWT | User profile (email uniqueness enforced) |
| PUT | `/api/settings/password` | JWT | Change password (min 8 chars, must differ) |
| GET/PUT | `/api/settings/notifications` | JWT | Notification preferences |
| PUT | `/api/settings/theme` | JWT | Theme preference |

### Images & Health
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/images/{name}` | JWT | Serve uploaded/processed images |
| GET | `/api/health` | — | Liveness + `model_loaded` flag |

## Security

- JWT access tokens (with `iat`/`iss` claims), bcrypt password hashing.
- Login rate limiting (5 attempts/min per IP) and account-disable checks.
- Security headers (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, etc.) applied by middleware.
- Upload validation: content-type sniffing, size cap (10 MB), extension allowlist.
- Responses compressed with GZip for large JSON payloads.

## Testing

```bash
# Backend (pytest) — 55 tests
cd backend && python -m pytest tests/ -q --asyncio-mode=auto

# Frontend (vitest) — 9 tests
cd frontend && pnpm run test
```

## AI Model

- **Architecture:** EfficientNetB3 (transfer learning, 12.9M params)
- **Classes:** Normal, DR, Glaucoma, AMD, Hypertensive Retinopathy, Other
- **Outputs:** Per-class probabilities, Grad-CAM heatmaps, clinical findings
- **Training:** `python -m app.ai.train`
- The trained model is a frozen artifact; retraining is out of scope for app changes.

## Dataset Pipeline

```
datasets/
├── eye_pacs/       Diabetic Retinopathy (5 grades)
├── glaucoma/       Glaucoma detection
├── amd/            Age-related Macular Degeneration
├── hypertensive/   Hypertensive Retinopathy
└── healthy/        Normal retinas
```

Run: `python -m app.dataset.pipeline`

## Project Status

| Phase | Status |
|-------|--------|
| Dataset pipeline | Done |
| AI model (EfficientNetB3) | Done |
| Database (SQLAlchemy async) | Done |
| Patient history & dashboard analytics | Done |
| Clinical PDF reports (per-scan, per-patient) | Done |
| Security hardening (rate limits, headers, validation) | Done |
| Docker (3 services) | Done |
| Frontend API integration | Done |
| Tests (55 backend + 9 frontend passing) | Done |
| Documentation | Done |
