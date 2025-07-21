clean:
	rm -rf public
	rm -rf .hugo_build.lock
	rm -rf .ruff_cache

bibtex:
	@echo "Generating BibTeX citations for all posts:"
	@echo "=========================================="
	@for post in content/posts/*.md; do \
		if [ -f "$$post" ]; then \
			echo ""; \
			echo "% File: $$post"; \
			python scripts/generate_bibtex_citation.py "$$post"; \
		fi; \
	done

pre_commit:
	pre-commit run --all-files
