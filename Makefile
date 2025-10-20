help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

init: ## Init env
	uv python pin 3.12.9
	uv venv .venv
	uv sync

init-dev:  ## Init dev env
	uv python pin 3.12.9
	uv venv .venv
	uv sync --all-extras --dev
	rm -f .git/hooks/pre-commit && rm -f .git/hooks/pre-commit.legacy
	pre-commit install

format:  ## Run formatting
	black .
	isort . --skip-gitignore --profile black
