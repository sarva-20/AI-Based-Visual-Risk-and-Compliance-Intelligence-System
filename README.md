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

## Project structure

The project is organized into milestone folders. Each milestone contains a self-contained Flask application and its Jinja templates:

```text
Milestone 1/   Initial application foundation
Milestone 2/   Dashboard, finance, study, and fitness workflows
Milestone 3/   Expanded assistant and future-simulation experience
```

## Technology

The application uses Flask for web routing, Flask-Login for authentication, Flask-SQLAlchemy with SQLite for persistence, Jinja and Bootstrap for the interface, Chart.js for visualizations, NumPy and scikit-learn for analytical calculations, and Google Gemini for assistant responses when configured.
