# AI-Based Visual Risk and Compliance Intelligence System

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?logo=sqlalchemy&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-7952B3?logo=bootstrap&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?logo=chart.js&logoColor=white)

## Overview

The AI-Based Visual Risk and Compliance Intelligence System is a personal decision-support web application. It brings financial activity, academic progress, fitness habits, goals, forecasts, and what-if analysis into one authenticated dashboard so users can understand current patterns and make better-informed decisions.

## Core capabilities

- **Personal dashboard:** Summarizes financial, academic, fitness, and goal metrics with visual trends.
- **Financial intelligence:** Records income and expenses, calculates savings indicators, and produces forecasts.
- **Study intelligence:** Tracks subjects, study hours, productivity, and GPA estimates.
- **Fitness intelligence:** Logs activities, duration, calories, streaks, and activity breakdowns.
- **Future simulation:** Compares scenarios such as saving, investing, spending, studying more, or exercising more, then stores the projected outcome and risk level.
- **AI assistant:** Provides practical guidance about finances, studies, fitness, and personal goals through optional Google Gemini integration, with an offline fallback when the provider is unavailable.
- **Authentication and roles:** Supports registration, login, protected user pages, and administrator access.
- **Administrator workspace:** Provides platform metrics, user search, user detail views, role management, mock accounts, and profile editing.

## Project structure

The project is organized into milestone folders. Each milestone contains a self-contained Flask application and its Jinja templates:

```text
Milestone 1/   Initial application foundation
Milestone 2/   Dashboard, finance, study, and fitness workflows
Milestone 3/   Expanded assistant and future-simulation experience
Milestone 4/   Administrator workspace and profile management
```

## Technology

The application uses Flask for web routing, Flask-Login for authentication, Flask-SQLAlchemy with SQLite for persistence, Jinja and Bootstrap for the interface, Chart.js for visualizations, NumPy and scikit-learn for analytical calculations, and Google Gemini for assistant responses when configured.

## Application flows

### User onboarding

1. A visitor opens the public landing page.
2. A new user registers with a username, email, and password.
3. The application validates the account details, hashes the password, and stores the user.
4. The user logs in and is redirected to the personal dashboard.

### Personal tracking

1. The user records transactions, study sessions, fitness activities, and goals.
2. Each record is associated with the authenticated user.
3. Records can be reviewed, edited, or deleted from their respective pages.
4. The dashboard aggregates the records into metrics, charts, forecasts, and progress indicators.

### Simulation flow

1. The user opens Future Simulation and selects a scenario.
2. The user enters an amount for saving, investing, spending, studying, or exercising.
3. The server validates the scenario and amount.
4. The system calculates the current value, projected value, impact, recommendation, and risk level.
5. The result is stored in the user's simulation history and shown on the page.

### AI assistant flow

1. The user submits a question about finances, studies, fitness, or goals.
2. The assistant sends the request asynchronously so the page remains open.
3. When Gemini is configured, the question is sent to the selected Gemini model.
4. The response is stored with the user's conversation history and displayed in the chat.
5. If Gemini is unavailable, the application returns a safe offline response instead of failing.

### Administrator flow

1. The administrator signs in with the administrator account and opens the Admin menu.
2. The admin dashboard displays platform-wide users, transactions, study activity, fitness activity, and goals.
3. Manage Users provides searchable user accounts with detail and edit actions.
4. The administrator profile shows account information and platform counts, and supports editing account details or changing the password.
5. Administrator-only checks protect dashboard, user management, and profile routes.

### Startup and demo flow

When the application starts, it creates the SQLite tables if needed, creates the default administrator account when absent, and seeds repeat-safe demo users with transaction, study, fitness, and goal records. This allows user and administrator pages to display meaningful data immediately.

## Configuration

The application reads these values from the root `.env` file:

- `SECRET_KEY` for Flask sessions.
- `GEMINI_API_KEY` to enable Gemini responses.
- `GEMINI_MODEL` to select the Gemini model; defaults to `gemini-flash-lite-latest`.

The `.env` file, virtual environments, caches, and SQLite databases remain local and are excluded by `.gitignore`.

## Demo accounts

| Account | Username | Password |
|---|---|---|
| Administrator | `admin` | `admin123` |
| Demo users | `demo.user`, `maya.sharma`, `arjun.nair`, `riya.patel` | `demo12345` |

## Input handling

User-owned records can be edited or deleted only by their owner. Numeric inputs are validated at the route boundary, optional fitness calories default to zero, and invalid fitness submissions return a validation message instead of an application error page.
