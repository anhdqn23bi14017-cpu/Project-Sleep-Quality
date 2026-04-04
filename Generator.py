import numpy as np
import pandas as pd

def generate_dataset(n=100):

    sleep_time = np.random.uniform(21, 26, n) % 24
    coffee = np.random.randint(0, 5, n)
    screen_time = np.random.randint(0, 180, n)
    stress = np.random.randint(1, 11, n)
    sport = np.random.randint(0, 120, n)
    nutrition = np.random.randint(1, 11, n)

    sleep_penalty = np.abs(sleep_time - 23)

    score = (
        -0.5 * sleep_penalty
        -0.4 * coffee
        -0.3 * (screen_time / 60)
        -0.6 * stress
        +0.4 * (sport / 30)
        +0.5 * nutrition
    )

    score += np.random.normal(0, 0.8, n)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    prob = sigmoid(score)
    sleep_quality = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "sleep_time": sleep_time,
        "coffee": coffee,
        "screen_time": screen_time,
        "stress_level": stress,
        "sport_time": sport,
        "nutrition": nutrition,
        "sleep_quality": sleep_quality
    })

    return df


# 🔁 Generate new dataset each time
df = generate_dataset(100)

# ❗ overwrite file
df.to_csv("Project-Sleep-Quality/DataSet.csv", index=False)

print("✅ Dataset regenerated!")