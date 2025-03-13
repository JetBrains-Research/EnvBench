format:
	ruff format --config pyproject.toml
	ruff check --fix --config pyproject.toml
lint:
	ruff format --check --config pyproject.toml
	ruff check --config pyproject.toml
	mypy --config-file pyproject.toml .
	pyright