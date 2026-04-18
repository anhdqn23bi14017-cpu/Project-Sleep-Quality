import numpy as np
import pandas as pd

def generate_dataset(n=100):
    """
    Features:
    - sleep_time (hour of day, 0-23, circular around optimal 23:00)
    - screen_time (hours per day, 0-4)
    - stress_level (1-10, 10=highest stress)
    - sport_time (hours per week, 0-5)
    - nutrition (1-10, 10=best diet)
    
    The target 'sleep_quality' is binary (0 = poor, 1 = good).
    """
    #1.Generate features with realistic distributions
    #Sleep time: normal around 23:00 (11 PM) with wrap-around
    raw_sleep = np.random.normal(23, 2, n)
    sleep_time = raw_sleep % 24

    #Screen time (hours/day) – log‑normal (many low, some high)
    screen_time = np.random.lognormal(mean=0.5, sigma=0.8, size=n)
    screen_time = np.clip(screen_time, 0, 4)
    
    #Stress level (1-10, skewed toward medium-high)
    stress = np.random.beta(a=2, b=3, size=n) * 9 + 1   #range 1-10
    
    #Sport time (hours/week) – many zero, some moderate
    sport = np.random.exponential(scale=1.5, size=n)
    sport = np.clip(activity, 0, 5)
    
    #Nutrition (1-10) – roughly normal around 6
    nutrition = np.random.normal(6, 1.5, n)
    nutrition = np.clip(nutrition, 1, 10)
    
    #2.Define relationships
    #Circular distance to optimal bedtime (23:00)
    diff = np.abs(sleep_time - 23)
    sleep_penalty = np.minimum(diff, 24 - diff)   #0 at 23, max 12 at 11 AM
    
    #U‑shape for sport: optimal around 2-3 hours/week
    activity_benefit = -0.1 * (activity - 2.5) ** 2 + 0.8   #peaks at 2.5h → +0.8
    
    #Base intercept to get realistic class balance (~65% poor, 35% good)
    base_intercept = -1.2
    
    #Raw score (log‑odds before sigmoid)
    score = (
        base_intercept
        - 0.8 * sleep_penalty          #each hour away from 23 hurts
        - 0.4 * screen_time            # per hour of screen time
        - 0.5 * (stress - 5) / 4       #stress centered at 5
        + 0.3 * sport_benefit
        + 0.4 * (nutrition - 5) / 4    #nutrition centered at 5
    )
    
    #Add realistic noise (heteroscedastic: more noise when stressed)
    noise_std = 0.5 + 0.1 * stress
    score += np.random.normal(0, noise_std, n)
    
    #3.Convert to probability and binary label
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    prob = sigmoid(score)
    sleep_quality = (prob > 0.5).astype(int)
    
    #4.Build DataFrame
    df = pd.DataFrame({
        "sleep_time": np.round(sleep_time, 1),
        "screen_time": np.round(screen_time, 1),
        "stress_level": np.round(stress, 1),
        "sport_time": np.round(sport, 1),
        "nutrition": np.round(nutrition, 1),
        "sleep_quality": sleep_quality
    })
    
    return df

#Generate and save
df = generate_dataset(n=200)
df.to_csv("Project-Sleep-Quality/DataSet.csv", index=False)

#Quick sanity check
print("Dataset generated")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Class distribution:\n{df['sleep_quality'].value_counts(normalize=True)}")
print("\nFirst 5 rows:")
print(df.head())
