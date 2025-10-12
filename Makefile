.PHONY: help build up down logs clean download-models test

help:
	@echo "AI-Chatbot Commands:"
	@echo ""
	@echo "  make build              - Build all Docker images"
	@echo "  make up                 - Start all services"
	@echo "  make down               - Stop all services"
	@echo "  make logs               - Show logs (all services)"
	@echo "  make logs-ingestion     - Show ingestion logs"
	@echo "  make download-models    - Download Docling models (one-time)"
	@echo "  make test-ingestion     - Test ingestion API"
	@echo "  make clean              - Clean volumes and cache"
	@echo "  make restart            - Restart all services"
	@echo ""
	@echo "First-time setup:"
	@echo "  1. make build"
	@echo "  2. make download-models"
	@echo "  3. make up"

build:
	@echo "Building Docker images..."
	docker-compose build

up:
	@echo "Starting services..."
	docker-compose up -d
	@echo ""
	@echo "Services started:"
	@echo "  - Ingestion API: http://localhost:6000"
	@echo "  - Online RAG: http://localhost:7000"
	@echo "  - LLM: http://localhost:8000"
	@echo "  - Embeddings: http://localhost:8080"
	@echo "  - Qdrant: http://localhost:6333"

down:
	@echo "Stopping services..."
	docker-compose down

logs:
	docker-compose logs -f

logs-ingestion:
	docker-compose logs -f ingestion

logs-llm:
	docker-compose logs -f llm

logs-embeddings:
	docker-compose logs -f embeddings

restart:
	@echo "Restarting services..."
	docker-compose restart

# Download Docling models (run once with internet)
download-models:
	@echo "Downloading ALL Docling models..."
	@echo "This will download:"
	@echo "  - Layout detection model (~200MB)"
	@echo "  - TableFormer model (~150MB)"
	@echo "  - Formula detection (~50MB)"
	@echo "  - Code detection (~30MB)"
	@echo "Total: ~500-600MB (one-time only)"
	docker-compose run --rm ingestion python download_models_full.py
	@echo "✓ All models downloaded to docling_models volume"
	@echo "  Services can now run COMPLETELY offline"

download-models-with-vlm:
	@echo "Downloading ALL Docling models + VLM..."
	@echo "WARNING: VLM adds ~3-5GB and requires GPU"
	docker-compose run --rm ingestion python download_models_full.py --with-vlm
	@echo "✓ All models including VLM downloaded"

# Test ingestion API
test-ingestion:
	@echo "Testing ingestion API..."
	curl -F "file=@tests/pdf/Knoleges.pdf" http://localhost:6000/ingest | jq

test-pdf:
	curl -F "file=@tests/pdf/Knoleges.pdf" http://localhost:6000/ingest | jq

test-docx:
	curl -F "file=@tests/docx/Darckstore.docx" http://localhost:6000/ingest | jq

test-pptx:
	curl -F "file=@tests/pptx/Platon presentation.pptx" http://localhost:6000/ingest | jq

# Clean everything
clean:
	@echo "Cleaning volumes and cache..."
	docker-compose down -v
	@echo "✓ Volumes removed (models will need to be re-downloaded)"

clean-cache:
	@echo "Cleaning build cache..."
	docker system prune -f

# Health checks
health:
	@echo "Checking service health..."
	@curl -s http://localhost:6000/health || echo "Ingestion: DOWN"
	@curl -s http://localhost:7000/health || echo "RAG: DOWN"
	@curl -s http://localhost:8000/health || echo "LLM: DOWN"
	@curl -s http://localhost:6333/health || echo "Qdrant: DOWN"

# Complete first-time setup
setup: build download-models up
	@echo ""
	@echo "✓ Setup complete!"
	@echo ""
	@echo "Services are running:"
	@echo "  - Ingestion: http://localhost:6000"
	@echo "  - RAG: http://localhost:7000"
	@echo ""
	@echo "Test with:"
	@echo "  make test-ingestion"

# Development commands
dev-ingestion:
	cd services/ingestion && uvicorn api:app --reload --host 0.0.0.0 --port 6000

dev-rag:
	cd services/online-rag && uvicorn api.gateway:app --reload --host 0.0.0.0 --port 7000

# Show status
status:
	@echo "Service Status:"
	@docker-compose ps
