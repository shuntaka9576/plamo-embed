format:
	uv run ruff format

lint:
	uv run ruff check

all: format lint

exec:
	rm -f blog_search.db
	uv run blog_search_demo.py

.PHONY: format lint all exec