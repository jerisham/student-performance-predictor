import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = {
    "study_hours": [1,2,3,4,5,6,7,8,9,10],
    "sleep_hours": [5,6,6,7,7,8,8,7,6,5],
    "attendance": [50,60,65,70,75,80,85,90,95,100],
    "past_gpa": [5,5.5,6,6.5,7,7.5,8,8.5,9,9.5],
    "assignment_rate": [40,50,55,60,70,75,80,85,90,95],
    "marks": [30,35,40,50,55,65,70,80,85,90]
}

df = pd.DataFrame(data)

X = df[[
    "study_hours",
    "sleep_hours",
    "attendance",
    "past_gpa",
    "assignment_rate"
]]
y = df["marks"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

print(f"\n📊 Model Accuracy (R² Score): {score:.2f}")

importance = model.coef_

features = [
    "study_hours",
    "sleep_hours",
    "attendance",
    "past_gpa",
    "assignment_rate"
]

print("\n📌 Feature Importance:")
for f, imp in zip(features, importance):
    print(f"➡️ {f} impacts marks by {imp:.2f}")

print("\n🎯 Enter student details for prediction\n")

study = float(input("Study hours: "))
sleep = float(input("Sleep hours: "))
attendance = float(input("Attendance (%): "))
gpa = float(input("Past GPA (out of 10): "))
assignment = float(input("Assignment completion (%): "))

input_data = pd.DataFrame(
    [[study, sleep, attendance, gpa, assignment]],
    columns=[
        "study_hours",
        "sleep_hours",
        "attendance",
        "past_gpa",
        "assignment_rate"
    ]
)

prediction = model.predict(input_data)

print(f"\n📊 Predicted Marks: {prediction[0]:.2f}")

if prediction[0] < 50:
    print("⚠️ You are at risk. Improve consistency and assignments.")
elif prediction[0] < 75:
    print("👍 You're doing okay. Try increasing study hours.")
else:
    print("🔥 Great performance! Keep it up.")

if study < 5:
    print("📌 Suggestion: Increase study hours")

if sleep < 6:
    print("📌 Suggestion: Get better sleep")

if assignment < 70:
    print("📌 Suggestion: Complete assignments regularly")