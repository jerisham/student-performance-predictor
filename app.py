import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Student Predictor", layout="centered")

st.markdown("""
<style>

/* Main app background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #FFDEE9, #B5FFFC);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
}

/* Text color */
html, body, [class*="css"] {
    color: #333;
}

/* Title */
.big-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #333;
}

/* Card style */
.card {
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    margin-top: 20px;
}

/* Button style */
.stButton>button {
    background: linear-gradient(to right, #ff7e5f, #feb47b);
    color: white;
    border-radius: 12px;
    padding: 10px 20px;
    font-size: 16px;
    border: none;
}

/* Slider text */
.stSlider label {
    color: #333 !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🎓✨ Smart Student Predictor</div>', unsafe_allow_html=True)
st.write("### 🌈 Adjust your habits and see your future score!")
st.write("### Adjust the inputs below:")

# Dataset
data = {
    "study_hours":    [1,   2,   3,   4,   5,   6,   7,   8,   9,   10],
    "sleep_hours":    [5,   6,   6,   7,   7,   8,   8,   7,   6,   5],
    "attendance":     [50,  60,  65,  70,  75,  80,  85,  90,  95,  100],
    "past_gpa":       [5,   5.5, 6,   6.5, 7,   7.5, 8,   8.5, 9,   9.5],
    "assignment_rate":[40,  50,  55,  60,  70,  75,  80,  85,  90,  95],
    "marks":          [30,  35,  40,  50,  55,  65,  70,  80,  85,  90]
}

df = pd.DataFrame(data)

X = df[["study_hours", "sleep_hours", "attendance", "past_gpa", "assignment_rate"]]
y = df["marks"]

model = LinearRegression()
model.fit(X, y)


def generate_study_plan(study, sleep, attendance, gpa, assignment, prediction):
    plan = "📅 **Your Smart Study Plan:**\n\n"

    if study < 5:
        plan += "- 📚 Increase study time to at least 6–7 hrs/day\n"
    else:
        plan += "- 👍 Maintain your current study routine\n"

    if sleep < 6:
        plan += "- 😴 Improve sleep to 7–8 hrs for better focus\n"
    else:
        plan += "- 🌙 Good sleep habits! Keep it up\n"

    if assignment < 70:
        plan += "- 📝 Complete assignments regularly\n"

    if attendance < 75:
        plan += "- 🏫 Improve attendance to boost performance\n"

    if prediction < 50:
        plan += "\n⚠️ You are at risk — start improving today!"
    elif prediction < 75:
        plan += "\n👍 You're doing okay — push a bit more!"
    else:
        plan += "\n🔥 Excellent work — keep going!"

    return plan


# Input sliders
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    study      = st.slider("📚 Study Hours", 0, 12, 5)
    sleep      = st.slider("😴 Sleep Hours", 0, 10, 6)
    attendance = st.slider("📅 Attendance (%)", 0, 100, 70)
    gpa        = st.slider("🎓 Past GPA", 0.0, 10.0, 6.0)
    assignment = st.slider("📝 Assignment Completion (%)", 0, 100, 60)

    st.markdown('</div>', unsafe_allow_html=True)


if st.button("🚀 Predict Performance"):

    input_data = pd.DataFrame(
        [[study, sleep, attendance, gpa, assignment]],
        columns=["study_hours", "sleep_hours", "attendance", "past_gpa", "assignment_rate"]
    )

    prediction = model.predict(input_data)[0]
    prediction = max(0, min(100, prediction))  # clamp between 0–100

    # Show predicted score
    st.metric(label="🔮 Predicted Score", value=f"{prediction:.1f} / 100")

    # Progress bar — must be between 0.0 and 1.0
    st.progress(int(prediction) / 100)

    # Feedback
    if prediction < 50:
        st.error("⚠️ At risk! Improve study consistency.")
    elif prediction < 75:
        st.warning("👍 Doing okay. Push a bit more.")
    else:
        st.success("🔥 Excellent performance!")

    # Study plan — was defined but never called before (bug fixed)
    plan_text = generate_study_plan(study, sleep, attendance, gpa, assignment, prediction)
    st.markdown("### 📋 Your Personalised Study Plan")
    st.markdown(plan_text)

    # What-if scenario
    st.markdown("### 🔮 What If You Improve?")
    new_study = min(study + 2, 12)  # cap at max slider value
    new_input = pd.DataFrame(
        [[new_study, sleep, attendance, gpa, assignment]],
        columns=["study_hours", "sleep_hours", "attendance", "past_gpa", "assignment_rate"]
    )
    new_prediction = model.predict(new_input)[0]
    new_prediction = max(0, min(100, new_prediction))
    st.write(f"If you study **2 more hours/day**, your score could become **{new_prediction:.1f}** 📈")

    # Student type
    st.markdown("### 🧠 Your Student Type")
    if study > 7 and assignment > 80:
        st.success("🔥 Consistent Performer")
    elif study < 4 and assignment < 50:
        st.error("⚠️ Last-Minute Crammer")
    else:
        st.info("👍 Balanced Learner")

    # Charts
    st.markdown("### 📊 More Insights")

    fig, ax = plt.subplots()
    ax.scatter(df["study_hours"], df["marks"], label="Dataset", color="steelblue")
    ax.scatter([study], [prediction], s=200, color="red", zorder=5, label="You")
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Marks")
    ax.set_title("Your Position vs Dataset")
    ax.legend()
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    ax2.scatter(df["sleep_hours"], df["marks"], color="mediumorchid")
    ax2.scatter([sleep], [prediction], s=200, color="red", zorder=5, label="You")
    ax2.set_xlabel("Sleep Hours")
    ax2.set_ylabel("Marks")
    ax2.set_title("Sleep vs Marks")
    ax2.legend()
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.scatter(df["attendance"], df["marks"], color="seagreen")
    ax3.scatter([attendance], [prediction], s=200, color="red", zorder=5, label="You")
    ax3.set_xlabel("Attendance (%)")
    ax3.set_ylabel("Marks")
    ax3.set_title("Attendance vs Marks")
    ax3.legend()
    st.pyplot(fig3)

    # Suggestions
    st.markdown("### 📌 Suggestions")
    if study < 5:
        st.write("• Increase study hours")
    if sleep < 6:
        st.write("• Improve sleep quality")
    if assignment < 70:
        st.write("• Complete assignments regularly")
    if attendance < 75:
        st.write("• Try to attend more classes")

    st.balloons()


