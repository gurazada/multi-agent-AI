import numpy as np
import pandas as pd
import time

class RealTimeBioprocessSimulator:
    """ Agent to simulate real-time bioprocess data """
    def __init__(self):
        self.t = 0
        self.last_values = {
            'pH': 7.0,
            'temperature_C': 37.0,
            'viability_pct': 95.0,
            'viable_cell_density_million_per_mL': 0,
            'dissolved_oxygen_pct': 80.0,
            'carbon_dioxide_pct': 0.05,
            'titer_g_per_L': 0.0,
            'growth_rate_per_hr': 0.5,
            'glucose_g_per_L': 5.0,
            'lactate_g_per_L': 0.0,
            'ammonia_mM': 0.0,
            'calcium_mM': 1.8,
            'sodium_potassium_ratio': 2.0,
            'agitation_rpm': 200,
            'cell_bleed_volume_mL': 0.0
        }

    def simulate_one_step(self):
        """ Method to generate one sample at the next time point (hour) based on the previous time points """
        t = self.t
        # Generate new values using the same trends as before
        self.last_values['pH'] = 7 + 0.02 * np.sin(0.2 * t) + np.random.normal(0, 0.05)
        self.last_values['temperature_C'] += np.random.normal(0, 0.15)
        self.last_values['viability_pct'] = np.clip(95 + 5 * np.exp(-0.1 * max(0, t-30)) + np.random.normal(0, 1), 70, 100)
        self.last_values['viable_cell_density_million_per_mL'] = np.clip(30 / (1 + np.exp(-0.3 * (t-24))) + np.random.normal(0, 1), 0, None)
        self.last_values['dissolved_oxygen_pct'] = np.clip(80 - 0.5 * t + np.random.normal(0, 2), 10, 100)
        self.last_values['carbon_dioxide_pct'] = np.clip(0.05 + 0.02 * t + np.random.normal(0, 0.005), 0, None)
        self.last_values['titer_g_per_L'] = np.clip(5 / (1 + np.exp(-0.3 * (t-30))) + np.random.normal(0, 0.1), 0, None)
        self.last_values['growth_rate_per_hr'] = np.clip(0.5 * np.exp(-0.1 * t) + 0.05 * np.random.normal(0, 1), 0, None)
        self.last_values['glucose_g_per_L'] = np.clip(5 * np.exp(-0.05 * t) + np.random.normal(0, 0.1), 0, None)
        self.last_values['lactate_g_per_L'] = np.clip(2 / (1 + np.exp(-0.3 * (t-20))) + np.random.normal(0, 0.1), 0, None)
        self.last_values['ammonia_mM'] = np.clip(0.5 * np.log1p(t) + np.random.normal(0, 0.05), 0, None)
        self.last_values['calcium_mM'] = np.clip(1.8 + np.random.normal(0, 0.05), 0, None)
        self.last_values['sodium_potassium_ratio'] = np.clip(2 + 0.5 * np.sin(0.1 * t) + np.random.normal(0, 0.1), 1, 4)
        self.last_values['agitation_rpm'] = 200 + np.random.normal(0, 5)
        self.last_values['cell_bleed_volume_mL'] = np.clip(0.5 * (t - 36) + np.random.normal(0, 0.1), 0, None) if t > 36 else 0

        data = {
            'culture_time_hr': t,
            **self.last_values
        }
        self.t += 1
        return pd.DataFrame([data])

if __name__ == "__main__":
    # Example usage: stream one row per simulated hour
    sim = RealTimeBioprocessSimulator()
    for _ in range(48):
        df = sim.simulate_one_step()
        print(df)  # Replace with code to push to downstream agent
        # time.sleep(1)  # Uncomment for real-time pacing, or set to time.sleep(3600) to "run hourly"
