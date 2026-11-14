# AI-Based Visual Risk and Compliance Intelligence System

## Overview

A Flask-based personal decision-support platform that combines financial activity, academic progress, fitness habits, goals, forecasting, simulations, and AI guidance in one dashboard.

## Features

- Personal dashboard with financial, study, fitness, and goal summaries.
- Income and expense tracking with savings metrics and forecasts.
- Study logs, productivity tracking, and GPA estimation.
- Fitness activity tracking with duration, calories, and activity summaries.
- What-if simulations for saving, investing, spending, studying, and exercising.
- Gemini assistant for finance, study, fitness, and goal-related questions.
- Domain guardrails and offline fallback for unavailable Gemini responses.
- Authentication with user and administrator roles.
- Administrator dashboard with platform metrics and searchable user management.
- Administrator profile editing, including optional password changes.

## Milestones

Each milestone is a self-contained Flask application with its own templates.

| Milestone | Scope |
|---|---|
| 1 | Application foundation and authentication |
| 2 | Dashboard, finance, study, and fitness workflows |
| 3 | AI assistant and future simulations |
| 4 | Administrator workspace and profile management |

Milestone 4 is the current development version.

## Main flows

1. Users register and sign in to access their dashboard.
2. Users record and manage financial, study, fitness, and goal data.
3. The dashboard calculates summaries, trends, forecasts, and progress.
4. Simulations project the effect of possible decisions and store their results.
5. The assistant responds to supported questions through Gemini or a safe offline fallback.
6. Administrators monitor platform activity and manage user accounts.

## Technology

Flask, Flask-Login, Flask-SQLAlchemy, SQLite, Jinja, Bootstrap, Chart.js, NumPy, scikit-learn, and Google Gemini.

## Configuration

The root `.env` file may contain:

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Flask session security |
| `GEMINI_API_KEY` | Enables Gemini responses |
| `GEMINI_MODEL` | Selects the Gemini model; defaults to `gemini-flash-lite-latest` |

Secrets, virtual environments, caches, and SQLite databases are excluded from Git.

## Demo accounts

| Account | Username | Password |
|---|---|---|
| Administrator | `admin` | `admin123` |
| Demo users | `demo.user`, `maya.sharma`, `arjun.nair`, `riya.patel` | `demo12345` |
