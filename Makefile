.PHONY: install backend frontend dev

# One-time setup: backend venv + deps, frontend node_modules
install:
	cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
	cd frontend && npm install

# Backend only — http://localhost:8787 (loads backend/.env if present)
backend:
	cd backend && if [ -f .env ]; then set -a; . ./.env; set +a; fi; exec .venv/bin/uvicorn app.main:app --reload --port 8787

# Frontend only — http://localhost:5173
frontend:
	cd frontend && npm run dev

# Both together; Ctrl-C stops both
dev:
	$(MAKE) -j2 backend frontend
