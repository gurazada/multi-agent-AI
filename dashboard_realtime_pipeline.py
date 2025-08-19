import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from multi_agent_langgraph_pipeline import build_langgraph_pipeline

def create_status_image(collapse_pred):
    # Create a simple 100x100 PNG with color based on prediction
    img = Image.new('RGB', (100, 100), color='white')
    draw = ImageDraw.Draw(img)
    if collapse_pred == 1:
        # Red circle = collapse
        draw.ellipse((10, 10, 90, 90), fill='red')
    else:
        # Green circle = no collapse
        draw.ellipse((10, 10, 90, 90), fill='green')
    return img

def run_workflow(n_steps, state=None):
    """ Running the multi-agent workflow through n_steps and saving the state and history """
    if state is None or not state:
        # Reinitialize state for new session, including history
        state = {'sim': None, 'agent': None, 'sample': None, 'prediction': None, 'history': []}
    # If upgrading legacy state, ensure 'history' key exists (for robustness)
    if 'history' not in state:
        state['history'] = []

    # initiate lang chain pipeline
    chain = build_langgraph_pipeline()
    metrics = [] # This list seems intended for single step metrics, but is not used later. Let's keep it for now but note its potential redundancy.
    for _ in range(n_steps):
        state = chain.invoke(state)
        # The simulator_node is now responsible for updating state['history']
        # We still collect sample and prediction for the current step display
        sample = state.get('sample')
        prediction = state.get('prediction')
        if sample is not None:
             # The simulator_node already added the full sample to history
             # We don't need to append the dictionary here again if history is updated in simulator_node
             pass # Let simulator_node manage history

    # print(f"Current state keys: {list(state.keys())}") # debug
    # Use the history from the state which is updated by simulator_node
    df = pd.DataFrame(state.get('history', []))

    fig, ax = plt.subplots()
    if not df.empty and 'viability_pct' in df.columns: # Check for column existence
        df['viability_pct'].plot(ax=ax, label="Viability %")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Viability (%)")
        ax.legend()
    plt.close(fig)
    # Create a simple log string from the latest sample in history
    log_str = "No data yet."
    status_img = Image.new('RGB', (100, 100), color='white')  # default blank image
    if not df.empty:
        latest_row = df.iloc[-1]
        latest_viability = latest_row.get('viability_pct', 'N/A')
        latest_prediction = latest_row.get('viability_collapse_pred', 'N/A') # Assuming prediction is added to history
        # We can't reliably get the actual collapse status here unless it's in the data
        # Let's log the predicted collapse status instead if available
        log_str = f"Step {len(df) - 1}: Viability = {latest_viability:.2f} | Collapse pred = {latest_prediction}"
        status_img = create_status_image(latest_prediction)

    # Pass the updated state back to Gradio
    return fig, df.tail(10), log_str, status_img, state # Add status_image to return values

def launch_realtime_dashboard():
    """ Method to design the layout and launch the dashboard """
    with gr.Blocks() as demo:
        gr.Markdown("# Bioprocess Digital Twin — Live Multi-Agent Workflow Demo")
        with gr.Row():
            steps_slider = gr.Slider(1, 24, value=1, step=1, label="Advance simulation by N hours")
            run_btn = gr.Button("Run")
        with gr.Row():
            plot_box = gr.Plot(label="Viability (%) Over Time")
            table_box = gr.Dataframe(label="Latest Samples")
        with gr.Row():
            log_box = gr.Textbox(label="Agent Log", lines=2)
            status_image = gr.Image(label="Latest Prediction Outcome", type="pil", interactive=False) # Removed shape argument
        state_var = gr.State({}) # Gradio State component to maintain state

        run_btn.click(
            fn=run_workflow,
            inputs=[steps_slider, state_var],
            outputs=[plot_box, table_box, log_box, status_image, state_var] # Add status_image to outputs
        )

    demo.launch(share=True, debug=True)

if __name__ == "__main__":
    launch_realtime_dashboard()
