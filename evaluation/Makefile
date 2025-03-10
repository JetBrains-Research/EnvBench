format:
	poetry run ruff format --config pyproject.toml
	poetry run ruff check --fix --config pyproject.toml
lint:
	poetry run ruff format --check --config pyproject.toml
	poetry run ruff check --config pyproject.toml
	poetry run mypy --config-file pyproject.toml .
	poetry run pyright
test:
	poetry run pytest