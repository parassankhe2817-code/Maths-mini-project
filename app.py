import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# ==========================================
# PAGE TITLE
# ==========================================

st.title("📊 Student Screen Time Analysis")

st.write(
    "Predict morning alertness based on screen time and screen duration."
)


# ==========================================
# LOAD CSV
# ==========================================

df = pd.read_csv("student.csv")


# ==========================================
# TRAIN REGRESSION MODEL
# ==========================================

# Input variables
X = df[
    [
        "Screen Time Before Sleep(hours)",
        "Screen duration(hrs)"
    ]
]

# Output variable
y = df[
    "Morning Alertness(1=Tired, 10=Very Fresh)"
]


# Create model
model = LinearRegression()

# Train model
model.fit(X, y)


# ==========================================
# USER INPUT
# ==========================================

st.header("🔮 Enter Student Information")


# Input 1
screen_time = st.number_input(
    "1) Screen time before sleeping (hrs)",
    min_value=0.0,
    max_value=24.0,
    value=1.0,
    step=0.5
)


# Input 2
screen_duration = st.number_input(
    "2) Screen duration (hrs)",
    min_value=0.0,
    max_value=24.0,
    value=2.0,
    step=0.5
)


# ==========================================
# PREDICTION
# ==========================================

if st.button("🔮 Predict Morning Alertness"):

    input_data = pd.DataFrame(
        [[screen_time, screen_duration]],
        columns=[
            "Screen Time Before Sleep(hours)",
            "Screen duration(hrs)"
        ]
    )

    prediction = model.predict(input_data)[0]

    # Convert prediction to whole number
    prediction = round(prediction)

    # Keep prediction between 1 and 10
    prediction = max(1, min(10, prediction))

    st.subheader("Prediction Result")

    st.success(
        f"Predicted Morning Alertness: {prediction} / 10"
    )

    if prediction <= 3:
        st.warning("Low alertness")

    elif prediction <= 6:
        st.info("Moderate alertness")

    else:
        st.success("High alertness")


# ==========================================
# GRAPH 1
# ==========================================

st.header("📈 Screen Time Before Sleep vs Alertness")

fig1, ax1 = plt.subplots()

ax1.scatter(
    df["Screen Time Before Sleep(hours)"],
    df["Morning Alertness(1=Tired, 10=Very Fresh)"]
)

# Simple regression using screen time only
single_X = df[["Screen Time Before Sleep(hours)"]]

single_model = LinearRegression()

single_model.fit(
    single_X,
    y
)

sorted_df = df.sort_values(
    "Screen Time Before Sleep(hours)"
)

sorted_X = sorted_df[
    ["Screen Time Before Sleep(hours)"]
]

sorted_predictions = single_model.predict(
    sorted_X
)

ax1.plot(
    sorted_X[
        "Screen Time Before Sleep(hours)"
    ],
    sorted_predictions
)

ax1.set_xlabel(
    "Screen Time Before Sleep (hours)"
)

ax1.set_ylabel(
    "Morning Alertness"
)

ax1.set_title(
    "Screen Time Before Sleep vs Morning Alertness"
)

st.pyplot(fig1)


# ==========================================
# GRAPH 2
# ==========================================

st.header("📈 Screen Duration vs Alertness")

fig2, ax2 = plt.subplots()

ax2.scatter(
    df["Screen duration(hrs)"],
    df["Morning Alertness(1=Tired, 10=Very Fresh)"]
)

single_X2 = df[["Screen duration(hrs)"]]

single_model2 = LinearRegression()

single_model2.fit(
    single_X2,
    y
)

sorted_df2 = df.sort_values(
    "Screen duration(hrs)"
)

sorted_X2 = sorted_df2[
    ["Screen duration(hrs)"]
]

sorted_predictions2 = single_model2.predict(
    sorted_X2
)

ax2.plot(
    sorted_X2["Screen duration(hrs)"],
    sorted_predictions2
)

ax2.set_xlabel(
    "Screen Duration (hours)"
)

ax2.set_ylabel(
    "Morning Alertness"
)

ax2.set_title(
    "Screen Duration vs Morning Alertness"
)

st.pyplot(fig2)


# ==========================================
# BASIC STATISTICS
# ==========================================

st.header("📊 Basic Statistics")

# Show total number of students separately
total_students = len(df)

st.metric(
    "👥 Total Students",
    total_students
)

# Select useful variables
statistics_data = df[
    [
        "Age",
        "Screen Time Before Sleep(hours)",
        "Screen duration(hrs)",
        "Morning Alertness(1=Tired, 10=Very Fresh)"
    ]
]

# Calculate statistics
stats = statistics_data.describe()

# Remove count because it is shown separately
stats = stats.drop("count")

# ------------------------------------------
# Format Age
# ------------------------------------------

# Age should be whole numbers except STD
for row in stats.index:
    if row != "std":
        stats.loc[row, "Age"] = round(
            stats.loc[row, "Age"]
        )

# ------------------------------------------
# Format Morning Alertness
# ------------------------------------------

# Alertness should be whole numbers except STD
for row in stats.index:
    if row != "std":
        stats.loc[
            row,
            "Morning Alertness(1=Tired, 10=Very Fresh)"
        ] = round(
            stats.loc[
                row,
                "Morning Alertness(1=Tired, 10=Very Fresh)"
            ]
        )

# ------------------------------------------
# Round STD values
# ------------------------------------------

stats.loc["std"] = stats.loc["std"].round(2)

# Display statistics
st.dataframe(stats)