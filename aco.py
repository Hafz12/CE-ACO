import numpy as np
import pandas as pd

# =========================
# LOAD DATASET
# =========================
data = pd.read_csv("delhi_metro_updated2.0 (2).csv")

# Standardize column names
data.columns = data.columns.str.lower().str.replace(" ", "_")

distance = data["distance_km"].values
fare = data["fare_rs"].values

# =========================
# ACO MULTI-OBJECTIVE CLASS
# =========================
class ACO_MultiObjective:
    def __init__(self, distance, fare,
                 ants=30, iterations=100,
                 w_distance=5.0, w_fare=0.5,
                 evaporation=0.5):

        self.distance = distance
        self.fare = fare
        self.pheromone = np.ones(len(distance))

        self.ants = ants
        self.iterations = iterations
        self.w_distance = w_distance
        self.w_fare = w_fare
        self.evaporation = evaporation

    def fitness(self, i):
        """
        Multi-objective fitness
        Minimize distance and fare
        """
        return (
            self.w_distance * self.distance[i] +
            self.w_fare * self.fare[i]
        )

    def run(self):
        best_index = 0
        best_score = float("inf")
        history = []

        for _ in range(self.iterations):
            probabilities = self.pheromone / self.pheromone.sum()
            selected = np.random.choice(
                len(self.distance),
                self.ants,
                p=probabilities
            )

            for idx in selected:
                score = self.fitness(idx)
                if score < best_score:
                    best_score = score
                    best_index = idx

            # Update pheromone
            self.pheromone *= (1 - self.evaporation)
            self.pheromone[best_index] += 1 / (best_score + 1e-6)

            history.append(best_score)

        return best_index, best_score, history

# =========================
# RUN OPTIMIZATION
# =========================
aco = ACO_MultiObjective(
    distance,
    fare,
    ants=30,
    iterations=100,
    w_distance=0.6,
    w_fare=0.4
)

best_idx, best_score, convergence = aco.run()

# =========================
# OUTPUT RESULT
# =========================
print("Best Route Found:")
print("Distance (km):", distance[best_idx])
print("Fare (Rs):", fare[best_idx])
print("Final Fitness Score:", best_score)
