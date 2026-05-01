# Changelog

All notable changes to Hooplytics are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `CONTRIBUTING.md` with dev setup, env variables, and PR checklist.
- GitHub Actions CI workflow (`.github/workflows/ci.yml`) running pytest on Python 3.11 and 3.12.
- Issue templates for bug reports and feature requests.
- Star CTA in README for users who find the project useful.

## [0.1.2] - 2026-05-01

### Added
- Anthropic (Claude) AI provider support alongside OpenAI for the in-app scout.
- Player performance PDF reports with redesigned conviction map, recent-form context, and AI scout picks section.
- Live scores panel and market-feature expansion in the Streamlit dashboard.
- Market-anchored calibration for model predictions.
- Historical odds backfilling for enhanced player outcomes analysis.
- Matchup grounding enriched with team information and today's slate.
- Session management for API keys and prebuilt model bundles.
- User-facing reasons surfaced when AI is unavailable in reports.

### Changed
- Player name display and performance overview layout adapted for clarity.
- AI scout report layout and context presentation refined.
- Player history table improved with headers and empty-state messaging.
- Edge confidence quadrant and signal index formatting enhanced.

### Fixed
- Improved error handling for player data loading via `NBADataUnavailable` exceptions, with friendlier UI messages on empty results.

## [0.1.0] - Initial public release

### Added
- RACE model with pregame-safe rolling features.
- Streamlit dashboard with eight purpose-built pages.
- Typer CLI with rich-rendered tables and `--json` output.
- Jupyter notebook walkthrough.
- The Odds API integration for live line context.
- PDF report generation.
- Multiple prebuilt RACE bundles (`race_fast`, `race_playoffs`, etc.).

[Unreleased]: https://github.com/texasbe2trill/hooplytics/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/texasbe2trill/hooplytics/releases/tag/v0.1.2
