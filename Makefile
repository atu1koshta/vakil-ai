.PHONY: install backend frontend dev eval

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

# Retrieval eval — writes backend/evals/results/<TAG>.json (needs Ollama)
# Usage: make eval TAG=dense-baseline [PROFILE=nomic-default]
# TAG defaults to profile name; PROFILE defaults to active profile
eval:
	cd backend && if [ -f .env ]; then set -a; . ./.env; set +a; fi; .venv/bin/python -m app.evals.run_eval $(TAG) $(if $(PROFILE),--profile $(PROFILE))
