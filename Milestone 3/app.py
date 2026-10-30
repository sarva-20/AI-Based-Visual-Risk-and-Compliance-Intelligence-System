from collections import defaultdict
from datetime import datetime, timedelta
import math
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-key-change-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///digital_twin.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"


@app.context_processor
def utility_processor():
    return {"datetime": datetime}


try:
    from google import genai
    from google.genai import errors as genai_errors
except ImportError:
    genai = None
    genai_errors = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
if genai is not None and GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer, default=25)
    occupation = db.Column(db.String(100), default="Student")
    monthly_income = db.Column(db.Float, default=50000)
    monthly_expenses = db.Column(db.Float, default=35000)
    savings = db.Column(db.Float, default=100000)
    study_hours_per_week = db.Column(db.Float, default=20)
    fitness_hours_per_week = db.Column(db.Float, default=5)
    sleep_hours_per_day = db.Column(db.Float, default=7)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    transactions = db.relationship("Transaction", backref="user", lazy=True)
    study_logs = db.relationship("StudyLog", backref="user", lazy=True)
    fitness_logs = db.relationship("FitnessLog", backref="user", lazy=True)
    goals = db.relationship("Goal", backref="user", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    category = db.Column(db.String(50))
    amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(10))


class StudyLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    hours = db.Column(db.Float, nullable=False)
    subject = db.Column(db.String(100))
    productivity_score = db.Column(db.Integer)


class FitnessLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    activity = db.Column(db.String(50))
    duration_min = db.Column(db.Integer, nullable=False)
    calories_burned = db.Column(db.Integer, default=0)


class Goal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    description = db.Column(db.String(200))
    target_date = db.Column(db.Date)
    achieved = db.Column(db.Boolean, default=False)


class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    question = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship("User", backref="chat_messages")


class SimulationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    scenario = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    result_desc = db.Column(db.String(200))
    result_impact = db.Column(db.String(200))
    result_risk = db.Column(db.String(20))
    result_rec = db.Column(db.String(200))
    projected = db.Column(db.Float)
    current = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship("User", backref="simulations")


def generate_forecast(data, periods=6):
    """Return a linear forecast with small random variation."""
    if len(data) < 2:
        return [data[-1] if data else 0] * periods
    x_values = np.arange(len(data)).reshape(-1, 1)
    model = LinearRegression().fit(x_values, np.asarray(data))
    future = np.arange(len(data), len(data) + periods).reshape(-1, 1)
    forecast = model.predict(future)
    spread = np.std(data) * 0.1 if np.std(data) > 0 else 1
    return (forecast + np.random.normal(0, spread, periods)).tolist()


def calculate_financial_metrics(user):
    """Calculate income, expenses, net savings, and savings rate."""
    income = sum(t.amount for t in user.transactions if t.type == "income")
    expenses = sum(t.amount for t in user.transactions if t.type == "expense")
    net = income - expenses
    return {"income": income, "expenses": expenses, "net": net, "rate": net / income * 100 if income else 0}


def predict_gpa(user):
    """Estimate GPA from logged study time and productivity."""
    logs = user.study_logs
    if not logs:
        return {"gpa": 3.0, "efficiency": 70, "recommendation": "Log study sessions to get predictions."}
    total_hours = sum(log.hours for log in logs)
    average_productivity = sum(log.productivity_score or 0 for log in logs) / len(logs)
    gpa = min(4.0, max(1.0, 2.0 + total_hours / 100 + average_productivity / 100))
    recommendation = "Increase study hours." if total_hours < 20 else "Improve focus and quality." if average_productivity < 70 else "Keep up the good work!"
    return {"gpa": round(gpa, 2), "efficiency": round(average_productivity, 1), "recommendation": recommendation}


def simulate_scenario(user, action, amount):
    """Return the projected result for a supported life scenario."""
    scenarios = {
        "save": (f"Save {amount:,} monthly", f"Your savings grow to {user.savings + amount * 12:,.0f} in 1 year.", "Low", "Great for long-term security"),
        "invest": (f"Invest {amount:,} monthly", f"With 10% returns, investment grows to {amount * 12 * 1.1:,.0f} in 1 year.", "Medium", "Balanced approach for growth"),
        "spend": (f"Spend {amount:,} monthly extra", f"Your savings reduce to {user.savings - amount * 12:,.0f} in 1 year.", "High", "Consider reducing discretionary spending"),
        "study_more": (f"Study {amount} more hours per week", f"Your GPA could increase by {amount * 0.1:.2f} points.", "Low", "Consistent effort yields results"),
        "exercise": (f"Exercise {amount} minutes daily", f"You could burn {amount * 30 * 30:,.0f} calories monthly.", "Low", "Great for overall health"),
    }
    description, impact, risk, recommendation = scenarios.get(action, scenarios["save"])
    return {"desc": description, "impact": impact, "risk": risk, "rec": recommendation}


def create_default_admin():
    """Create the default admin account when one does not exist."""
    admin = User.query.filter_by(username="admin").first()
    if admin is None:
        admin = User(username="admin", email="admin@digitaltwin.ai", is_admin=True)
        admin.set_password("admin123")
        db.session.add(admin)
        db.session.commit()
    return admin


def seed_demo_data(user):
    """Add representative study and fitness records once for the demo user."""
    today = datetime.now().date()
    if not StudyLog.query.filter_by(user_id=user.id, subject="Python").first():
        study_sessions = [
            (13, 1.5, "Python", 78),
            (11, 2.0, "Machine Learning", 84),
            (9, 1.0, "Database Systems", 72),
            (7, 2.5, "Python", 88),
            (5, 1.5, "Machine Learning", 81),
            (3, 2.0, "Database Systems", 90),
            (1, 2.5, "Python", 92),
        ]
        db.session.add_all(
            [
                StudyLog(
                    user_id=user.id,
                    date=today - timedelta(days=days_ago),
                    hours=hours,
                    subject=subject,
                    productivity_score=productivity,
                )
                for days_ago, hours, subject, productivity in study_sessions
            ]
        )
    if not FitnessLog.query.filter_by(user_id=user.id, activity="Walking").first():
        fitness_sessions = [
            (13, "Walking", 35, 155),
            (11, "Strength training", 45, 280),
            (9, "Cycling", 40, 320),
            (7, "Running", 30, 295),
            (5, "Yoga", 25, 90),
            (3, "Strength training", 50, 315),
            (1, "Running", 35, 340),
        ]
        db.session.add_all(
            [
                FitnessLog(
                    user_id=user.id,
                    date=today - timedelta(days=days_ago),
                    activity=activity,
                    duration_min=duration,
                    calories_burned=calories,
                )
                for days_ago, activity, duration, calories in fitness_sessions
            ]
        )
    db.session.commit()


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


@app.route("/")
def index():
    return render_template("index.html")


def _simple_page(title, message):
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{title}</title></head><body><main><h1>{title}</h1><p>{message}</p><p><a href='{url_for('index')}'>Return home</a></p></main></body></html>"


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/terms")
def terms():
    return render_template("terms.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        if not username or not email or not password:
            flash("All fields are required.")
        elif password != request.form.get("confirm_password", ""):
            flash("Passwords do not match.")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.")
        elif User.query.filter_by(username=username).first():
            flash("Username already exists.")
        elif User.query.filter_by(email=email).first():
            flash("Email already registered.")
        else:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash("Registration successful! Please log in.")
            return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form.get("username", "").strip()).first()
        if user and user.check_password(request.form.get("password", "")):
            login_user(user)
            endpoint = "admin_dashboard" if user.is_admin else "user_dashboard"
            return redirect(url_for(endpoint))
        flash("Invalid username or password.")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        return redirect(url_for("user_dashboard"))
    return render_template("user/dashboard.html", user=current_user, metrics=calculate_financial_metrics(current_user), gpa=predict_gpa(current_user), total_fitness=sum(log.duration_min for log in current_user.fitness_logs), months=[], income_data=[], expense_data=[], study_days=[], study_hours=[], fitness_days=[], fitness_minutes=[], goal_progress=[], savings_forecast=[], gpa_forecast=[], projected_fitness=0, total_study_hours=0, study_streak=0, fitness_streak=0, subjects=[], subject_hours=[], activities=[], activity_minutes=[], life_score=0)


@app.route("/admin/users")
@login_required
def admin_users():
    if not current_user.is_admin:
        return redirect(url_for("user_dashboard"))
    return _simple_page("Manage Users", "User management is available to administrators.")


@app.route("/user/simulation")
@login_required
def user_simulation():
    current_gpa = predict_gpa(current_user)["gpa"]
    current_fitness = sum(log.duration_min for log in current_user.fitness_logs)
    result = None
    result_id = request.args.get("result", type=int)
    if result_id:
        result = SimulationHistory.query.filter_by(id=result_id, user_id=current_user.id).first()
    current_values = {
        "savings": current_user.savings,
        "gpa": current_gpa,
        "fitness": current_fitness,
    }
    return render_template(
        "user/simulation.html",
        user=current_user,
        current_values=current_values,
        result=result,
        history=SimulationHistory.query.filter_by(user_id=current_user.id).order_by(SimulationHistory.timestamp.desc()).limit(10).all(),
    )


@app.route("/user/simulation/run", methods=["POST"])
@login_required
def run_simulation():
    scenario = request.form.get("scenario", "save")
    allowed_scenarios = {"save", "invest", "spend", "study_more", "exercise"}
    if scenario not in allowed_scenarios:
        flash("Please choose a valid scenario.", "warning")
        return redirect(url_for("user_simulation"))

    try:
        amount = float(request.form.get("amount", "0"))
    except ValueError:
        amount = -1
    if not math.isfinite(amount) or amount < 0:
        flash("Enter a non-negative amount for the simulation.", "warning")
        return redirect(url_for("user_simulation"))

    current_gpa = predict_gpa(current_user)["gpa"]
    current_fitness = sum(log.duration_min for log in current_user.fitness_logs)
    result = simulate_scenario(current_user, scenario, amount)
    if scenario in {"save", "invest", "spend"}:
        current = current_user.savings
        projected = {
            "save": current + amount * 12,
            "invest": current + amount * 12 * 1.1,
            "spend": current - amount * 12,
        }[scenario]
    elif scenario == "study_more":
        current = current_gpa
        projected = min(4.0, current + amount * 0.1)
    else:
        current = current_fitness
        projected = current + amount * 30

    history_entry = SimulationHistory(
        user_id=current_user.id,
        scenario=scenario,
        amount=amount,
        result_desc=result["desc"],
        result_impact=result["impact"],
        result_risk=result["risk"],
        result_rec=result["rec"],
        current=current,
        projected=projected,
    )
    db.session.add(history_entry)
    db.session.commit()
    return redirect(url_for("user_simulation", result=history_entry.id))


@app.route("/user/assistant", methods=["GET", "POST"])
@login_required
def user_assistant():
    if request.method == "POST":
        question = request.form.get("query", "").strip()
        if not question:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"status": "error", "message": "Please enter a question."}), 400
            flash("Please enter a question.", "warning")
            return redirect(url_for("user_assistant"))

        response = (
            "I can help you think through finances, study planning, fitness, and life goals. "
            "Try asking for a specific recommendation based on one of those areas."
        )
        if gemini_client is not None:
            prompt = (
                "You are a concise personal decision-support assistant. "
                "Give practical, safe advice about finances, study, fitness, or goals. "
                f"User question: {question}"
            )
            try:
                generated = gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
                response = getattr(generated, "text", response)
            except (AttributeError, ValueError, TypeError):
                flash("The AI assistant could not generate a response. Showing the offline response instead.", "warning")
            except Exception:
                flash("The AI assistant could not generate a response. Showing the offline response instead.", "warning")

        message = ChatMessage(user_id=current_user.id, question=question, response=response)
        db.session.add(message)
        db.session.commit()

        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({
                "status": "success",
                "question": message.question,
                "response": message.response,
                "message_id": message.id,
                "timestamp": message.timestamp.isoformat(),
            })

        return redirect(url_for("user_assistant"))

    messages = ChatMessage.query.filter_by(user_id=current_user.id).order_by(ChatMessage.timestamp.desc()).all()
    if request.args.get("new"):
        messages = []
    elif request.args.get("load"):
        selected = ChatMessage.query.filter_by(id=request.args["load"], user_id=current_user.id).first()
        if selected is not None:
            messages = [selected]
    return render_template("user/assistance.html", user=current_user, all_messages=messages)


@app.route("/settings")
@login_required
def settings():
    return _simple_page("Settings", "Account settings are ready for configuration.")


@app.route("/user/dashboard")
@login_required
def user_dashboard():
    metrics = calculate_financial_metrics(current_user)
    gpa_info = predict_gpa(current_user)
    total_fitness = sum(log.duration_min for log in current_user.fitness_logs)
    income_by_month = defaultdict(float)
    expense_by_month = defaultdict(float)
    for transaction in current_user.transactions:
        month = transaction.date.strftime("%Y-%m")
        (income_by_month if transaction.type == "income" else expense_by_month)[month] += transaction.amount
    months = sorted(set(income_by_month) | set(expense_by_month))[-6:]
    study_by_day = defaultdict(float)
    fitness_by_day = defaultdict(float)
    subjects = defaultdict(float)
    activities = defaultdict(float)
    for log in current_user.study_logs:
        study_by_day[log.date.strftime("%Y-%m-%d")] += log.hours
        if log.subject:
            subjects[log.subject] += log.hours
    for log in current_user.fitness_logs:
        fitness_by_day[log.date.strftime("%Y-%m-%d")] += log.duration_min
        if log.activity:
            activities[log.activity] += log.duration_min
    study_days = sorted(study_by_day)[-14:]
    fitness_days = sorted(fitness_by_day)[-14:]
    goal_progress = []
    for goal in current_user.goals:
        if not goal.achieved and goal.target_date:
            remaining = (goal.target_date - datetime.now().date()).days
            progress = 100 if remaining < 0 else min(100, max(0, 100 - remaining / 30 * 100))
            goal_progress.append({"description": goal.description, "progress": round(progress, 1)})
    return render_template(
        "user/dashboard.html", user=current_user, metrics=metrics, gpa=gpa_info, total_fitness=total_fitness,
        months=months, income_data=[income_by_month[m] for m in months], expense_data=[expense_by_month[m] for m in months],
        study_days=study_days, study_hours=[study_by_day[d] for d in study_days], fitness_days=fitness_days,
        fitness_minutes=[fitness_by_day[d] for d in fitness_days], goal_progress=goal_progress,
        savings_forecast=[current_user.savings + i * 5000 for i in range(6)],
        gpa_forecast=[round(gpa_info["gpa"] + i * 0.05, 2) for i in range(3)], projected_fitness=total_fitness * 2,
        total_study_hours=sum(log.hours for log in current_user.study_logs), study_streak=0, fitness_streak=0,
        subjects=list(subjects), subject_hours=list(subjects.values()), activities=list(activities),
        activity_minutes=list(activities.values()), life_score=round(metrics["rate"] / 30 * 20 + gpa_info["gpa"] / 4 * 30),
    )


@app.route("/user/financial", methods=["GET", "POST"])
@login_required
def user_financial():
    if request.method == "POST":
        db.session.add(Transaction(user_id=current_user.id, date=datetime.strptime(request.form["date"], "%Y-%m-%d").date(), category=request.form.get("category", "Other"), amount=float(request.form["amount"]), type=request.form["type"]))
        db.session.commit()
        flash("Transaction added.")
        return redirect(url_for("user_financial"))
    transactions = current_user.transactions
    category_totals = defaultdict(float)
    for transaction in transactions:
        if transaction.type == "expense":
            category_totals[transaction.category] += transaction.amount
    expenses = [t.amount for t in transactions if t.type == "expense"]
    return render_template("user/financial.html", user=current_user, transactions=transactions[-10:][::-1], forecast=generate_forecast(expenses), metrics=calculate_financial_metrics(current_user), categories=list(category_totals), category_amounts=list(category_totals.values()))


def _owned_transaction(transaction_id):
    return Transaction.query.filter_by(id=transaction_id, user_id=current_user.id).first()


@app.route("/edit_transaction", methods=["POST"])
@login_required
def edit_transaction():
    transaction = _owned_transaction(request.form.get("id"))
    if transaction is None:
        flash("Transaction not found or you do not have permission.", "danger")
        return redirect(url_for("user_financial"))
    try:
        transaction.date = datetime.strptime(request.form["date"], "%Y-%m-%d").date()
        transaction.category = request.form["category"]
        transaction.amount = float(request.form["amount"])
        transaction.type = request.form["type"]
        db.session.commit()
        flash("Transaction updated successfully!", "success")
    except (KeyError, ValueError) as error:
        db.session.rollback()
        flash(f"Error updating transaction: {error}", "danger")
    return redirect(url_for("user_financial"))


@app.route("/delete_transaction/<int:id>", methods=["POST"])
@login_required
def delete_transaction(id):
    transaction = _owned_transaction(id)
    if transaction is None:
        flash("Transaction not found or you do not have permission.", "danger")
        return redirect(url_for("user_financial"))
    db.session.delete(transaction)
    db.session.commit()
    flash("Transaction deleted.", "info")
    return redirect(url_for("user_financial"))


def _parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d").date()


@app.route("/user/study", methods=["GET", "POST"])
@login_required
def user_study():
    if request.method == "POST":
        db.session.add(StudyLog(user_id=current_user.id, date=_parse_date(request.form["date"]), hours=float(request.form["hours"]), subject=request.form.get("subject"), productivity_score=int(request.form["productivity"])))
        db.session.commit()
        flash("Study session logged.", "success")
        return redirect(url_for("user_study"))
    logs = current_user.study_logs
    by_day = defaultdict(float)
    subjects = defaultdict(float)
    for log in logs:
        by_day[log.date.strftime("%Y-%m-%d")] += log.hours
        if log.subject:
            subjects[log.subject] += log.hours
    dates = sorted(by_day)[-14:]
    productivity_by_day = defaultdict(list)
    for log in logs:
        productivity_by_day[log.date.strftime("%Y-%m-%d")].append(log.productivity_score or 0)
    productivity_dates = sorted(productivity_by_day)[-14:]
    productivity_average = [
        sum(productivity_by_day[date]) / len(productivity_by_day[date])
        for date in productivity_dates
    ]
    return render_template(
        "user/study.html",
        user=current_user,
        logs=logs[-10:][::-1],
        gpa=predict_gpa(current_user),
        study_days=dates,
        study_hours=[by_day[d] for d in dates],
        subjects=list(subjects),
        subject_hours=list(subjects.values()),
        prod_dates=productivity_dates,
        prod_avg=productivity_average,
        total_hours=sum(log.hours for log in logs),
        avg_productivity=sum(log.productivity_score or 0 for log in logs) / len(logs) if logs else 0,
        study_streak=0,
    )


@app.route("/edit_study_log", methods=["POST"])
@login_required
def edit_study_log():
    log = StudyLog.query.filter_by(id=request.form.get("id"), user_id=current_user.id).first()
    if log is None:
        flash("Log not found or you do not have permission.", "danger")
        return redirect(url_for("user_study"))
    log.date = _parse_date(request.form["date"])
    log.hours = float(request.form["hours"])
    log.subject = request.form.get("subject")
    log.productivity_score = int(request.form["productivity"])
    db.session.commit()
    flash("Study log updated successfully!", "success")
    return redirect(url_for("user_study"))


@app.route("/delete_study_log/<int:id>", methods=["POST"])
@login_required
def delete_study_log(id):
    log = StudyLog.query.filter_by(id=id, user_id=current_user.id).first()
    if log is None:
        flash("Log not found or you do not have permission.", "danger")
        return redirect(url_for("user_study"))
    db.session.delete(log)
    db.session.commit()
    flash("Study log deleted.", "info")
    return redirect(url_for("user_study"))


@app.route("/user/fitness", methods=["GET", "POST"])
@login_required
def user_fitness():
    if request.method == "POST":
        db.session.add(FitnessLog(user_id=current_user.id, date=_parse_date(request.form["date"]), activity=request.form.get("activity", "Other"), duration_min=int(request.form["duration"]), calories_burned=int(request.form.get("calories", 0))))
        db.session.commit()
        flash("Fitness activity logged.", "success")
        return redirect(url_for("user_fitness"))
    logs = current_user.fitness_logs
    today = datetime.now().date()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(13, -1, -1)]
    daily_totals = defaultdict(int)
    activity_totals = defaultdict(int)
    for log in logs:
        daily_totals[log.date.strftime("%Y-%m-%d")] += log.duration_min
        activity_totals[log.activity] += log.duration_min
    total_min = sum(log.duration_min for log in logs)
    return render_template("user/fitness.html", user=current_user, logs=logs[-10:][::-1], total_min=total_min, total_cal=sum(log.calories_burned or 0 for log in logs), avg_min_per_day=round(total_min / max(1, len({log.date for log in logs})), 1), weekly_min=sum(log.duration_min for log in logs if log.date >= today - timedelta(days=today.weekday())), week_progress=0, dates=dates, daily_data=[daily_totals[d] for d in dates], activities=list(activity_totals), activity_minutes=list(activity_totals.values()), streak=0)


@app.route("/fitness/edit/<int:log_id>", methods=["POST"])
@login_required
def edit_fitness(log_id):
    log = FitnessLog.query.filter_by(id=log_id, user_id=current_user.id).first()
    if log is None:
        flash("Activity not found or you do not have permission.", "danger")
        return redirect(url_for("user_fitness"))
    log.date = _parse_date(request.form["date"])
    log.activity = request.form["activity"]
    log.duration_min = int(request.form["duration"])
    log.calories_burned = int(request.form.get("calories", 0))
    db.session.commit()
    flash("Activity updated successfully!", "success")
    return redirect(url_for("user_fitness"))


@app.route("/fitness/delete/<int:log_id>", methods=["POST"])
@login_required
def delete_fitness(log_id):
    log = FitnessLog.query.filter_by(id=log_id, user_id=current_user.id).first()
    if log is None:
        flash("Activity not found or you do not have permission.", "danger")
        return redirect(url_for("user_fitness"))
    db.session.delete(log)
    db.session.commit()
    flash("Activity deleted.", "info")
    return redirect(url_for("user_fitness"))


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        demo_user = create_default_admin()
        seed_demo_data(demo_user)
    app.run(host="127.0.0.1", port=5050, debug=True)