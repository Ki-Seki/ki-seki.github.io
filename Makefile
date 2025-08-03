clean:
	rm -rf public
	rm -rf .hugo_build.lock

pre_commit:
	pre-commit run --all-files
