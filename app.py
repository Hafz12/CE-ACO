import streamlit as st
import pandas as pd
import numpy as np
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Multi-Objective Optimization", layout="wide")

st.title("🚍 Multi-Objective Optimization using ACO")
st.write("Objectives: **Minimize Distance and Fare**")

# =========================
# AUTO LOAD DATASET
# =========================
DATA_PATH = "delhi_metro_updated.csv"

if not os.path.exists(DATA_PATH):
    st.error("dataset.csv not found. Please place it in the project folder.")
    st.stop()

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.lower().str.replace(" ", "_")

required_cols = ["distance_km", "fare", "cost_per_passenger", "passengers"]
if not all(col in data.columns for col in required_cols):
    st.error("Dataset must contain: Distance_km, Fare, Cost_per_passenger, Passengers")
    st.stop()

distance = data["distance_km"].values
fare = data["fare"].values

# =========================
# SIDEBAR PARAMETERS
# =========================
st.sidebar.header("⚙ Optimization Parameters")

ants = st.sidebar.slider("Number of Ants", 10, 100, 30)
iterations = st.sidebar.slider("Iterations", 10, 300, 100)
evaporation = st.sidebar.slider("Evaporation Rate", 0.1, 0.9, 0.5)

st.sidebar.subheader("Objective Weights")
w_distance = st.sidebar.slider("Distance Weight", 0.0, 1.0, 0.6)
w_fare = st.sidebar.slider("Fare Weight", 0.0, 1.0, 0.4)

if w_distance + w_fare == 0:
    st.sidebar.error("At least one weight must be > 0")
    st.stop()

# =========================
# ACO CLASS
# =========================
class ACO_MultiObjective:
    def __init__(self, distance, fare, ants, iterations, w_distance, w_fare, evaporation):
        self.distance = distance
        self.fare = fare
        self.pheromone = np.ones(len(distance))
        self.ants = ants
        self.iterations = iterations
        self.w_distance = w_distance
        self.w_fare = w_fare
        self.evaporation = evaporation

    def fitness(self, i):
        return self.w_distance * self.distance[i] + self.w_fare * self.fare[i]

    def run(self):
        best_index = 0
        best_score = float("inf")
        convergence = []

        for _ in range(self.iterations):
            probs = self.pheromone / self.pheromone.sum()
            selected = np.random.choice(len(self.distance), self.ants, p=probs)

            for idx in selected:
                score = self.fitness(idx)
                if score < best_score:
                    best_score = score
                    best_index = idx

            self.pheromone *= (1 - self.evaporation)
            self.pheromone[best_index] += 1 / (best_score + 1e-6)
            convergence.append(best_score)

        return best_index, best_score, convergence

# =========================
# PARETO FRONT FUNCTION
# =========================
def pareto_front(distances, fares):
    pareto = []
    for i in range(len(distances)):
        dominated = False
        for j in range(len(distances)):
            if (
                distances[j] <= distances[i] and
                fares[j] <= fares[i] and
                (distances[j] < distances[i] or fares[j] < fares[i])
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(i)
    return pareto

pareto_indices = pareto_front(distance, fare)

# =========================
# RUN OPTIMIZATION
# =========================
if st.button("▶ Run Optimization"):
    with st.spinner("Optimizing..."):
        aco = ACO_MultiObjective(
            distance, fare,
            ants, iterations,
            w_distance, w_fare,
            evaporation
        )
        best_idx, best_score, convergence = aco.run()

    st.success("Optimization Completed ✅")

  # =====================================================
# Dataset Preview
# =====================================================
st.subheader("🗂 Dataset Preview")

with st.expander("Show dataset sample"):
    st.dataframe(data.head(10))

    # =========================
    # METRICS
    # =========================
    col1, col2, col3 = st.columns(3)
    col1.metric("Distance (km)", distance[best_idx])
    col2.metric("Fare", fare[best_idx])
    col3.metric("Fitness Score", round(best_score, 3))

    st.subheader("📊 Additional Information")
    st.write("Cost per Passenger:", data.loc[best_idx, "cost_per_passenger"])
    st.write("Passengers:", data.loc[best_idx, "passengers"])

    # =========================
    # CONVERGENCE CURVE
    # =========================
    st.subheader("📉 Convergence Curve")
    st.line_chart(convergence)

    # =========================
    # PARETO FRONT PLOT
    # =========================
    import altair as alt

    st.subheader("📈 Pareto Front (Distance vs Fare)")

    plot_df = pd.DataFrame({
        "Distance": distance,
        "Fare": fare,
        "Pareto": [
            "Pareto-optimal" if i in pareto_indices else "Dominated"
            for i in range(len(distance))
        ]
    })

    chart = alt.Chart(plot_df).mark_circle(size=80).encode(
        x="Distance",
        y="Fare",
        color=alt.Color(
            "Pareto",
            scale=alt.Scale(
                domain=["Pareto-optimal", "Dominated"],
                range=["red", "lightgray"]
            ),
            legend=alt.Legend(title="Solution Type")
        ),
        tooltip=["Distance", "Fare", "Pareto"]
    )

    st.altair_chart(chart, use_container_width=True)

    st.info("🔴 Red points represent Pareto-optimal solutions.")

# =====================================================
# Strengths & Limitations
# =====================================================
st.subheader("⚖️ Strengths vs Limitations of ACO")

colS, colL = st.columns(2)

with colS:
    st.markdown("""
    **Strengths**
    - Naturally suited for combinatorial problems
    - Strong exploration capability
    - Robust to local optima
    - Intuitive biological inspiration
    """)

with colL:
    st.markdown("""
    **Limitations**
    - Slower convergence than PSO
    - Risk of pheromone stagnation
    - Sensitive to evaporation and colony size
    """)

# =====================================================
# Conclusion
# =====================================================
st.subheader("✅ Conclusion")

st.markdown("""
The MOACO dashboard demonstrates how swarm intelligence  
can effectively address **multi-objective optimization** in urban transportation.

By visualizing Pareto fronts, convergence behavior, and efficiency metrics,  
this system enhances interpretability and supports informed decision-making.
""")

   
