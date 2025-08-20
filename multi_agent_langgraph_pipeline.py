from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import pandas as pd # Ensure pandas is imported
import pprint

from agent_bioprocess_datastream import RealTimeBioprocessSimulator
from agent_upstream_cell_viability import UpstreamViabilityAgent

class State(TypedDict):
    """ The state object that is exchanged between multiple agents/nodes is defined here """
    sim: RealTimeBioprocessSimulator
    agent: UpstreamViabilityAgent
    sample: pd.DataFrame
    prediction: int
    history: list # Added history to the State definition

# Simulator node
def simulator_node(state: State) -> State:
    """ Method that defines the role of data simulator agentic node and how it updates the state when visited
        Returns the updated state """
    sim = state.get('sim')
    if sim is None:
        sim = RealTimeBioprocessSimulator()
        state['sim'] = sim
    sample = sim.simulate_one_step()
    state['sample'] = sample
    # Append the sample data (converted to dict) to the history list in state
    # Initialize history if it doesn't exist (though run_workflow should handle this)
    if 'history' not in state or not isinstance(state['history'], list):
         state['history'] = []

    # Add the sample and prediction to the history for plotting/logging
    # We need the prediction to be available in the history for the log_str
    # Let's add the sample data and potentially the prediction (if available from a previous step)
    # A better approach might be to add prediction in the predictor_node and then update history
    # Let's simplify for now and just add the sample dict. We'll handle prediction logging differently.
    sample_dict = sample.iloc[0].to_dict()
    state['history'].append(sample_dict) # Append the sample data to history

    return state

# Predictor node
def predictor_node(state: State) -> State:
    """ Method that defines the role of viability predictor agentic node and how it updates the state when visited
        Returns the updated state """
    agent = state.get('agent')
    if agent is None: # Initialize agent if not in state
        agent = UpstreamViabilityAgent()
        # Check if the model file exists before loading
        try:
            agent.load() # Load the saved model
            state['agent'] = agent
        except FileNotFoundError:
            print("Model file not found. Please train the agent first.")
            # Handle the error - perhaps set prediction to None or a specific error value
            state['prediction'] = -1 # Indicate error
            return state # Exit early if model not found

    sample = state.get('sample')
    if sample is None:
        print("No sample available for prediction.")
        state['prediction'] = -1 # Indicate no sample
        return state

    try:
        pred = agent.predict(sample)
        state['prediction'] = pred
         # Update the last entry in history with the prediction
        if state.get('history'):
            state['history'][-1]['viability_collapse_pred'] = pred
    except Exception as e:
        print(f"Error during prediction: {e}")
        state['prediction'] = -1 # Indicate error
        if state.get('history'):
             state['history'][-1]['viability_collapse_pred'] = -1 # Indicate error in history entry

    return state


def build_langgraph_pipeline():
    """ Method that compiles the multi-agentic pipeline end-to-end and returns the chain """
    graph = StateGraph(State)

    graph.add_node('simulator', simulator_node)
    graph.add_node('predictor', predictor_node)

    graph.add_edge(START, 'simulator')
    graph.add_edge('simulator', 'predictor')
    graph.add_edge('predictor', END)

    chain = graph.compile()
    return chain


def run_pipeline(num_hours=100):
    """ Method to invoke the multi-agent chain through N steps/hours """
    state = {}
    results = []
    chain = build_langgraph_pipeline()
    for i in range(num_hours):
        state = chain.invoke(state)
        sample = state['sample']
        prediction = state['prediction']
        results.append({
            'hour': i,
            'viability_pct': sample['viability_pct'].iloc[0],
            'viability_collapse_pred': prediction
        })
    return results


if __name__ == "__main__":
    try:
        n_hours = int(input("Enter the number of hours to simulate the bioprocess run for: "))
        print(f"Simulating bioprocess for {n_hours} hours...")
        results = run_pipeline(n_hours)
        pprint.pprint(results)
    except ValueError:
        print("Please enter a valid integer for the number of hours.")
    