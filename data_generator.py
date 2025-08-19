import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import io
from PIL import Image
import traceback


def generate_batch_data(batch_length=48, seed=None):
    """ Core method to simulate and generate batch data for a bioprocess run 
        Returns dataframe with all samples (rows) and their bioprocess parameter (column) values for a batch """
    np.random.seed(seed) # Ensures variability between runs
    time_points = np.arange(batch_length + 1)

    # Simulated bioprocess values with realistic time trends
    pH = 7 + 0.02 * np.sin(0.2 * time_points) + np.random.normal(0, 0.05, batch_length + 1)
    temperature = 37 + np.random.normal(0, 0.5, batch_length + 1)
    viability = 95 + 5 * np.exp(-0.1 * (time_points - 30).clip(min=0)) + np.random.normal(0, 1, batch_length + 1)
    viability = np.clip(viability, 70, 100)
    max_vcd = 30
    vcd = max_vcd / (1 + np.exp(-0.3 * (time_points - 24))) + np.random.normal(0, 1, batch_length + 1)
    vcd = np.clip(vcd, 0, None)
    do = 80 - 0.5 * time_points + np.random.normal(0, 2, batch_length + 1)
    do = np.clip(do, 10, 100)
    co2 = 0.05 + 0.02 * time_points + np.random.normal(0, 0.005, batch_length + 1)
    co2 = np.clip(co2, 0, None)
    max_titer = 5
    titer = max_titer / (1 + np.exp(-0.3 * (time_points - 30))) + np.random.normal(0, 0.1, batch_length + 1)
    titer = np.clip(titer, 0, None)
    growth_rate = 0.5 * np.exp(-0.1 * time_points) + 0.05 * np.random.normal(0, 1, batch_length + 1)
    growth_rate = np.clip(growth_rate, 0, None)
    glucose = 5 * np.exp(-0.05 * time_points) + np.random.normal(0, 0.1, batch_length + 1)
    glucose = np.clip(glucose, 0, None)
    lactate = 2 / (1 + np.exp(-0.3 * (time_points - 20))) + np.random.normal(0, 0.1, batch_length + 1)
    lactate = np.clip(lactate, 0, None)
    ammonia = 0.5 * np.log1p(time_points) + np.random.normal(0, 0.05, batch_length + 1)
    ammonia = np.clip(ammonia, 0, None)
    calcium = 1.8 + np.random.normal(0, 0.05, batch_length + 1)
    calcium = np.clip(calcium, 0, None)
    na_k_ratio = 2 + 0.5 * np.sin(0.1 * time_points) + np.random.normal(0, 0.1, batch_length + 1)
    na_k_ratio = np.clip(na_k_ratio, 1, 4)
    agitation = 200 + np.random.normal(0, 5, batch_length + 1)
    cell_bleed = np.where(time_points > 36, 0.5 * (time_points - 36) + np.random.normal(0, 0.1, batch_length + 1), 0)
    cell_bleed = np.clip(cell_bleed, 0, None)
    culture_time = time_points

    batch_data = pd.DataFrame({
        'culture_time_hr': culture_time,
        'pH': pH,
        'temperature_C': temperature,
        'viability_pct': viability,
        'viable_cell_density_million_per_mL': vcd,
        'dissolved_oxygen_pct': do,
        'carbon_dioxide_pct': co2,
        'titer_g_per_L': titer,
        'growth_rate_per_hr': growth_rate,
        'glucose_g_per_L': glucose,
        'lactate_g_per_L': lactate,
        'ammonia_mM': ammonia,
        'calcium_mM': calcium,
        'sodium_potassium_ratio': na_k_ratio,
        'agitation_rpm': agitation,
        'cell_bleed_volume_mL': cell_bleed
    })
    return batch_data


def create_training_data(num_batches=20, batch_length=240):
    """ Method to generating large training dataset (historical data from multiple runs) 
        Returns a merged dataframe with samples from multiple runs """
    batches = [generate_batch_data(batch_length, seed=i) for i in range(num_batches)]
    for idx, batch in enumerate(batches):
        batch['batch_id'] = idx
    df = pd.concat(batches, ignore_index=True)

    ## The multi-batch sample training dataset is merged into one dataframe - df
    ## Now creating the sample Y label - Viability collapse

    # Label collapse per sample: 1 if viability < cutoff in next 24 hrs
    df['viability_collapse'] = 0
    window = 24  # 24 hours
    viability_cutoff = 93 # viability%
    for idx in range(len(df) - 1):
        current_time = df.loc[idx, 'culture_time_hr']
        current_batch = df.loc[idx, 'batch_id']
        # Find all future points within the next 24 hours
        mask = (df['batch_id'] == current_batch) & (df['culture_time_hr'] > current_time) & (df['culture_time_hr'] <= current_time + window)
        # If any future viability drops below cutoff, set target to 1
        if (df.loc[mask, 'viability_pct'] < viability_cutoff).any():
            df.loc[idx, 'viability_collapse'] = 1

    return df


def run_batch_simulation(batch_length=48):
    """ Gradio dashboard UI to simulate and generate batch data with option to choose batch length (hrs)
        Returns the data in CSV file and an image with some preliminary plots """
    try:
        df = generate_batch_data(batch_length)
        csv_path = 'batch_data.csv'
        df.to_csv(csv_path, index=False)

        fig, axs = plt.subplots(3, 1, figsize=(8, 10))
        axs[0].plot(df['culture_time_hr'], df['viability_pct'], label='Viability %')
        axs[0].plot(df['culture_time_hr'], df['viable_cell_density_million_per_mL'], label='Viable Cell Density (M/mL)')
        axs[0].legend()
        axs[0].set_title('Cell Viability & Density Over Time')

        axs[1].plot(df['culture_time_hr'], df['pH'], label='pH', color='green')
        axs[1].plot(df['culture_time_hr'], df['temperature_C'], label='Temperature (C)', color='orange')
        axs[1].legend()
        axs[1].set_title('pH and Temperature Over Time')

        axs[2].plot(df['culture_time_hr'], df['titer_g_per_L'], label='Titer (g/L)', color='red')
        axs[2].plot(df['culture_time_hr'], df['glucose_g_per_L'], label='Glucose (g/L)', color='blue')
        axs[2].legend()
        axs[2].set_title('Titer and Glucose Over Time')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf).copy() # copy to detach from buffer
        buf.close()
        plt.close(fig)  # prevent memory leaks

        return csv_path, img
    except Exception as e:
        print("Exception in run_batch_simulation:", e)
        traceback.print_exc()
        return None, None

def launch_batch_dashboard():
    """ Method to create and launch the gradio dashboard to generate batch data """
    iface = gr.Interface(
        fn=run_batch_simulation,
        inputs=gr.Slider(24, 72, step=1, label='Batch Length (hours)'),
        outputs=[gr.File(label='Download Batch Data CSV'), gr.Image(type='pil', label='Simulation Plot')],
        title='Synthetic Bioprocess Data Generator',
        description='Generates synthetic bioprocess data for a batch and visualizes key parameters.'
    )

    iface.launch(share=True, debug=True)
