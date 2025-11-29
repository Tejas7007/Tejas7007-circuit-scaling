# Convenience targets
.PHONY: fmt test

fmt:
	black src scripts tests
	isort src scripts tests

test:
	pytest -q
