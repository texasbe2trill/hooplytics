# Contributing to Hooplytics

Thanks for your interest in improving Hooplytics! This guide covers everything you need to get a working dev environment and submit a useful change.

## Ways to contribute

- **Bug reports** — open an issue using the bug template; include reproduction steps, expected vs. actual behavior, and your Python version.
- **Feature requests** — open an issue using the feature template and describe the analytics workflow you're trying to support.
- **Pull requests** — small, focused changes are easiest to review. If your change is large, please open an issue first to align on direction.
- **Discussions** — questions, ideas, and "show & tell" go in [GitHub Discussions](https://github.com/texasbe2trill/hooplytics/discussions).

## Development setup

Hooplytics targets **Python 3.11+**.

```bash
git clone https://github.com/texasbe2trill/hooplytics.git
cd hooplytics
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

The `all` extra installs the `web`, `notebook`, and `test` optional dependencies (Streamlit, OpenAI/Anthropic SDKs, JupyterLab, pytest). If you only plan to work on a specific surface, install just the extras you need:

```bash
pip install -e ".[test]"           # CLI + tests only
pip install -e ".[web,test]"       # Streamlit dashboard + tests
pip install -e ".[notebook,test]"  # Jupyter notebook + tests
```

### Environment variables

Copy `.env.example` to `.env` and fill in any keys you need:

```bash
cp .env.example .env
```

- `ODDS_API_KEY` — free tier (500 req/month) at [the-odds-api.com](https://the-odds-api.com/). Only required if you want live line context.
- `HOOPLYTICS_PRETRAINED_BUNDLE` — optional path to a pre-trained `.joblib` bundle. Without it, the app auto-detects the newest file under `.hooplytics_cache/models/`.

The Streamlit AI scout features additionally read `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` if you want to enable them. They're optional — the app runs fine without either.

## Running the app

```bash
hooplytics --help          # Typer CLI
hooplytics-web             # Streamlit dashboard
jupyter lab hooplytics.ipynb
```

## Running tests

```bash
pytest tests/
```

CI runs the same command on every push and PR (see `.github/workflows/ci.yml`). Please make sure tests pass locally before opening a PR.

## Pull request checklist

- [ ] The change is focused — one feature or fix per PR.
- [ ] Tests pass locally (`pytest tests/`).
- [ ] New behavior is covered by a test where it's reasonably testable.
- [ ] No secrets, API keys, or `.env` contents in the diff.
- [ ] Documentation (README or docstrings) is updated if user-facing behavior changes.
- [ ] Commit messages are descriptive — we loosely follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).

Open the PR against `main`. A maintainer will review and may request changes.

## Code style

- Follow the patterns in surrounding code; we don't currently enforce a formatter, but consistency matters.
- Prefer clear names over comments. Comment the *why*, not the *what*.
- Keep functions small and pure where possible — the modeling code in `hooplytics/` leans heavily on this.

## Reporting security issues

Please **don't** open a public issue for security vulnerabilities. Email the maintainer directly via the contact link on the [GitHub profile](https://github.com/texasbe2trill).

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
