from collections import defaultdict
from datetime import datetime, timedelta
import os

import numpy as np
from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv()

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
    import google.generativeai as genai
except ImportError:
    genai = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if genai is not None and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
else:
    gemini_model = None


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
    return render_template("user/study.html", user=current_user, logs=logs[-10:][::-1], gpa=predict_gpa(current_user), study_dates=dates, study_hours=[by_day[d] for d in dates], subjects=list(subjects), subject_hours=list(subjects.values()), prod_dates=[], prod_avg=[], total_hours=sum(log.hours for log in logs), avg_productivity=sum(log.productivity_score or 0 for log in logs) / len(logs) if logs else 0, study_streak=0)


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
        create_default_admin()
    app.run(host="127.0.0.1", port=5050, debug=True)