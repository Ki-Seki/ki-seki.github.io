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
