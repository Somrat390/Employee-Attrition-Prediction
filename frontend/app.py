"""Employee Attrition Prediction — Streamlit frontend.

Run with:
    streamlit run app.py

Talks to the FastAPI backend over HTTP. Configure where with API_BASE_URL --
either as an OS environment variable (Docker, docker-compose, running
locally) or as a Streamlit Cloud secret (Settings -> Secrets, as
API_BASE_URL = "https://your-backend.onrender.com"). Streamlit Community
Cloud's secrets don't become OS environment variables, so both sources are
checked below rather than relying on os.environ alone.
"""

import os

import requests
import streamlit as st


def _get_config(key: str, default: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        # No secrets.toml at all (local dev), or this key isn't in it --
        # either way, fall back to a plain OS environment variable.
        return os.environ.get(key, default)


API_BASE = _get_config("API_BASE_URL", "http://127.0.0.1:8000")
API_URL = f"{API_BASE}/predict"
EXPLAIN_URL = f"{API_BASE}/explain"
WATERFALL_URL = f"{API_BASE}/explain/waterfall"
SUMMARY_URL = f"{API_BASE}/explain/summary"
PREDICTIONS_URL = f"{API_BASE}/predictions"

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="📊", layout="centered")

st.title("📊 Employee Attrition Predictor")
st.caption(
    "Enter an employee's details below to estimate their risk of leaving. "
    "Backed by the Logistic Regression + SMOTE model from the notebook."
)

with st.expander("🌍 Global feature importance (across all employees)"):
    st.caption(
        "Which features matter most to the model overall — not specific to any one employee below."
    )
    try:
        resp = requests.get(SUMMARY_URL, timeout=15)
        if resp.status_code == 200:
            st.image(resp.content, use_container_width=True)
        else:
            st.warning(f"Couldn't load global importance ({resp.status_code}).")
    except requests.exceptions.ConnectionError:
        st.warning(f"Couldn't reach the API at `{API_BASE}` to load global importance.")

# ---------------------------------------------------------------------------
# Dropdown options — must match exactly what the model was trained on
# (see EmployeeFeatures in the backend's app/schemas.py).
# ---------------------------------------------------------------------------
BUSINESS_TRAVEL_OPTIONS = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
DEPARTMENT_OPTIONS = ["Human Resources", "Research & Development", "Sales"]
EDUCATION_FIELD_OPTIONS = [
    "Human Resources",
    "Life Sciences",
    "Marketing",
    "Medical",
    "Other",
    "Technical Degree",
]
GENDER_OPTIONS = ["Female", "Male"]
JOB_ROLE_OPTIONS = [
    "Healthcare Representative",
    "Human Resources",
    "Laboratory Technician",
    "Manager",
    "Manufacturing Director",
    "Research Director",
    "Research Scientist",
    "Sales Executive",
    "Sales Representative",
]
MARITAL_STATUS_OPTIONS = ["Divorced", "Married", "Single"]
OVERTIME_OPTIONS = ["No", "Yes"]

SATISFACTION_LABELS = {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}
EDUCATION_LABELS = {
    1: "1 - Below College",
    2: "2 - College",
    3: "3 - Bachelor",
    4: "4 - Master",
    5: "5 - Doctor",
}
WORKLIFE_LABELS = {1: "1 - Bad", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}
PERFORMANCE_LABELS = {3: "3 - Excellent", 4: "4 - Outstanding"}


with st.form("employee_form"):
    st.subheader("Personal & Job Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=60, value=30)
        gender = st.selectbox("Gender", GENDER_OPTIONS)
        marital_status = st.selectbox("Marital Status", MARITAL_STATUS_OPTIONS)
        department = st.selectbox("Department", DEPARTMENT_OPTIONS)
        job_role = st.selectbox("Job Role", JOB_ROLE_OPTIONS)
        education_field = st.selectbox("Education Field", EDUCATION_FIELD_OPTIONS)
    with col2:
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=0)
        education = st.selectbox(
            "Education", list(EDUCATION_LABELS), format_func=lambda x: EDUCATION_LABELS[x], index=2
        )
        business_travel = st.selectbox("Business Travel", BUSINESS_TRAVEL_OPTIONS, index=1)
        distance_from_home = st.number_input(
            "Distance From Home (miles)", min_value=0, max_value=30, value=5
        )
        overtime = st.selectbox("OverTime", OVERTIME_OPTIONS)
        num_companies_worked = st.number_input(
            "Number of Companies Worked At", min_value=0, max_value=10, value=1
        )

    st.subheader("Compensation")
    col3, col4 = st.columns(2)
    with col3:
        monthly_income = st.number_input(
            "Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=100
        )
        daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=800)
        hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=65)
    with col4:
        monthly_rate = st.number_input(
            "Monthly Rate", min_value=2000, max_value=27000, value=14000, step=100
        )
        percent_salary_hike = st.number_input(
            "Percent Salary Hike (%)", min_value=10, max_value=25, value=15
        )
        stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=0)

    st.subheader("Tenure")
    col5, col6 = st.columns(2)
    with col5:
        total_working_years = st.number_input(
            "Total Working Years", min_value=0, max_value=40, value=8
        )
        years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5)
        years_in_current_role = st.number_input(
            "Years In Current Role", min_value=0, max_value=20, value=3
        )
    with col6:
        years_since_last_promotion = st.number_input(
            "Years Since Last Promotion", min_value=0, max_value=15, value=1
        )
        years_with_curr_manager = st.number_input(
            "Years With Current Manager", min_value=0, max_value=20, value=3
        )
        training_times_last_year = st.number_input(
            "Training Times Last Year", min_value=0, max_value=6, value=2
        )

    st.subheader("Satisfaction & Performance")
    col7, col8 = st.columns(2)
    with col7:
        job_satisfaction = st.selectbox(
            "Job Satisfaction",
            list(SATISFACTION_LABELS),
            format_func=lambda x: SATISFACTION_LABELS[x],
            index=2,
        )
        environment_satisfaction = st.selectbox(
            "Environment Satisfaction",
            list(SATISFACTION_LABELS),
            format_func=lambda x: SATISFACTION_LABELS[x],
            index=2,
        )
        relationship_satisfaction = st.selectbox(
            "Relationship Satisfaction",
            list(SATISFACTION_LABELS),
            format_func=lambda x: SATISFACTION_LABELS[x],
            index=2,
        )
    with col8:
        job_involvement = st.selectbox(
            "Job Involvement",
            list(SATISFACTION_LABELS),
            format_func=lambda x: SATISFACTION_LABELS[x],
            index=2,
        )
        work_life_balance = st.selectbox(
            "Work Life Balance",
            list(WORKLIFE_LABELS),
            format_func=lambda x: WORKLIFE_LABELS[x],
            index=2,
        )
        performance_rating = st.selectbox(
            "Performance Rating",
            list(PERFORMANCE_LABELS),
            format_func=lambda x: PERFORMANCE_LABELS[x],
            index=0,
        )

    submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")


if submitted:
    payload = {
        "Age": age,
        "DailyRate": daily_rate,
        "DistanceFromHome": distance_from_home,
        "HourlyRate": hourly_rate,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "PercentSalaryHike": percent_salary_hike,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
        "BusinessTravel": business_travel,
        "Department": department,
        "EducationField": education_field,
        "Gender": gender,
        "JobRole": job_role,
        "MaritalStatus": marital_status,
        "OverTime": overtime,
        "Education": education,
        "EnvironmentSatisfaction": environment_satisfaction,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "WorkLifeBalance": work_life_balance,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
    except requests.exceptions.ConnectionError:
        st.error(
            f"Couldn't reach the prediction API at `{API_URL}`. "
            "Make sure the FastAPI backend is running (`uvicorn app.main:app --reload` "
            "from the employee_attrition folder)."
        )
        st.stop()

    if response.status_code != 200:
        st.error(f"API returned an error ({response.status_code}): {response.text}")
        st.stop()

    result = response.json()
    prediction = result["prediction"]
    probability = result["probability_leave"]
    threshold = result["threshold_used"]

    st.divider()
    if prediction == "Leave":
        st.error("⚠️ Employee will **Leave**")
    else:
        st.success("✅ Employee will **Stay**")

    st.metric("Probability of Leaving", f"{probability * 100:.1f}%")
    st.progress(probability)
    st.caption(
        f'Decision threshold: {threshold:.2f} (probabilities at or above this are classified as "Leave")'
    )

    st.divider()
    st.subheader("Why this prediction?")
    try:
        wf_resp = requests.post(WATERFALL_URL, json=payload, timeout=15)
        if wf_resp.status_code == 200:
            st.image(wf_resp.content, use_container_width=True)
        else:
            st.warning(f"Couldn't load the explanation chart ({wf_resp.status_code}).")
    except requests.exceptions.ConnectionError:
        st.warning(f"Couldn't reach the API at `{API_BASE}` to load the explanation.")

    try:
        explain_resp = requests.post(EXPLAIN_URL, json=payload, timeout=15)
        if explain_resp.status_code == 200:
            contributions = explain_resp.json()["contributions"]
            with st.expander("See exact contribution values"):
                for c in contributions:
                    direction = "toward Leave" if c["shap_value"] > 0 else "toward Stay"
                    st.write(
                        f"**{c['feature']}** — pushes {direction} (SHAP = {c['shap_value']:+.3f})"
                    )
    except requests.exceptions.ConnectionError:
        pass  # already warned above


st.divider()
with st.expander("🗂️ Recent predictions (logged to the database)"):
    st.caption(
        "Every call to /predict is saved with a timestamp -- this shows the most recent ones."
    )
    try:
        resp = requests.get(PREDICTIONS_URL, params={"limit": 20}, timeout=10)
        if resp.status_code == 200:
            rows = resp.json()
            if not rows:
                st.info("No predictions logged yet -- submit the form above to create one.")
            else:
                table = [
                    {
                        "Time": r["timestamp"].replace("T", " ").split(".")[0],
                        "Prediction": r["prediction"],
                        "Probability": f"{r['probability_leave'] * 100:.1f}%",
                    }
                    for r in rows
                ]
                st.dataframe(table, use_container_width=True, hide_index=True)
        else:
            st.warning(f"Couldn't load prediction history ({resp.status_code}).")
    except requests.exceptions.ConnectionError:
        st.warning(f"Couldn't reach the API at `{API_BASE}` to load prediction history.")
