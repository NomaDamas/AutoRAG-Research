.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running ty"
	@uv run ty check
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@uv run deptry .

.PHONY: test docker-up docker-down docker-wait clean-docker

# PostgreSQL 컨테이너 시작
docker-up:
	@echo "🐘 Starting PostgreSQL containers..."
	@mkdir -p postgresql/pgdata
	@cd postgresql && docker compose --env-file .env up -d

# PostgreSQL 준비 대기
docker-wait:
	@echo "⏳ Waiting for PostgreSQL to be ready..."
	@until docker compose -f postgresql/docker-compose.yml --env-file postgresql/.env exec -T db pg_isready -U postgres > /dev/null 2>&1; do \
		sleep 1; \
	done
	@echo "✅ PostgreSQL is ready!"

# PostgreSQL 컨테이너 중지 및 삭제
docker-down:
	@echo "🛑 Stopping PostgreSQL containers..."
	@cd postgresql && docker compose --env-file .env down

# 완전 정리 (볼륨 포함)
clean-docker:
	@echo "🧹 Cleaning up PostgreSQL containers and volumes..."
	@cd postgresql && docker compose --env-file .env down -v
	@echo "🗑️  Removing pgdata directory..."
	@rm -rf postgresql/pgdata

# 테스트 실행 (PostgreSQL 자동 관리)
test: docker-up docker-wait ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml -m "not gpu"; \
	TEST_EXIT_CODE=$$?; \
	$(MAKE) clean-docker; \
	exit $$TEST_EXIT_CODE

# 테스트만 실행 (컨테이너는 유지)
test-only: ## Run tests without managing Docker containers
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml -m "not gpu and not data"

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: publish
publish: ## Publish a release to PyPI.
	@echo "🚀 Publishing."
	@uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
