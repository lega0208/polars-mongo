.DEFAULT_GOAL := help
UV := uv

.PHONY: help
help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "%-12s %s\n", $$1, $$2}'

.PHONY: install
install: ## Create the venv and install dev dependencies
	$(UV) sync

.PHONY: dev
dev: install ## Build the Rust extension in debug mode into the venv
	$(UV) run maturin develop --uv

.PHONY: release
release: install ## Build the Rust extension with optimizations
	$(UV) run maturin develop --uv --release

.PHONY: wheel
wheel: install ## Build a distributable wheel into target/wheels
	$(UV) run maturin build --release

.PHONY: test
test: dev ## Run Rust and Python tests
	cargo test
	$(UV) run pytest

.PHONY: lint
lint: ## Lint and type-check Rust and Python
	cargo fmt --check
	cargo clippy --all-targets -- -D warnings
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run ty check

.PHONY: fmt
fmt: ## Format Rust and Python
	cargo fmt
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

.PHONY: clean
clean: ## Remove build artifacts
	cargo clean
	rm -rf .venv .pytest_cache .ruff_cache
	find python -name '*.so' -delete
