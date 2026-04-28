# Copilot Instructions

## Project Overview

This is **The Kiseki Log**, a personal Hugo static site using the
[PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme (a git
submodule). It is deployed via GitHub Pages.

## Repo Structure

- `content/posts/` — long-form blog posts (Hugo page bundles)
- `content/moments/` — short reflections with `location` and `mood` front
  matter
- `content/cv.md` — curriculum vitae page
- `archetypes/` — Hugo content archetypes/templates
- `assets/` — custom CSS (`assets/css/extended/`) and JS (`assets/js/`)
- `layouts/` — Hugo layout overrides and custom partials
- `static/` — static assets and feature pages (`static/features/`)
- `config.yml` — Hugo site configuration

## Local Development

```bash
# Serve the site locally (requires Hugo extended)
hugo server

# Run all pre-commit checks
make pre_commit
```

## Markdown Rules

All Markdown files are linted with
[markdownlint-cli](https://github.com/igorshubovych/markdownlint-cli) using
the rules in `.markdownlint.yaml`. Pre-commit hooks also enforce
trailing-whitespace removal and a newline at end of file.

**Disabled rules** (anything else from the default ruleset is enabled):

| Rule | Description |
|------|-------------|
| MD013 | Line length |
| MD033 | Inline HTML |
| MD034 | Bare URL |
| MD041 | First line heading |
| MD046 | Code block style |
| MD060 | Fenced code block language |

**When writing or editing any `.md` file:**

- No trailing whitespace on any line.
- File must end with exactly one newline.
- Headings must be surrounded by blank lines (MD022).
- Lists must be surrounded by blank lines (MD032).
- Fenced code blocks must be surrounded by blank lines (MD031).
- Heading levels must increment by one level at a time (MD001).
- No duplicate headings within a document (MD024).
- No hard tabs; use spaces (MD010).
- No multiple consecutive blank lines (MD012).

Run `make pre_commit` to verify all files pass before committing.

## Content Conventions

### Posts

Created with `hugo new posts/<slug>/index.md`. Front matter includes `title`,
`date`, `author`, `summary`, `tags`, and `math`.

### Moments

Created with `hugo new moments/<slug>/index.md`. Front matter includes
`location` and `mood` in addition to the standard fields.

### Shortcodes

Custom shortcodes available: `media`, `admonition`, `bibtex`. The `media`
shortcode accepts images, YouTube, Vimeo, and plain video URLs.
