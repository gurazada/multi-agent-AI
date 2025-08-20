import pprint
import subprocess
import sys

from agent_bioprocess_datastream import RealTimeBioprocessSimulator
from agent_upstream_cell_viability import UpstreamViabilityAgent
from data_generator import create_training_data
from multi_agent_langgraph_pipeline import run_pipeline
from dashboard_realtime_pipeline import launch_realtime_dashboard


def setup_environment():
    """Set up the environment and install requirements"""
    print("Setting up environment...")   
    # Install requirements
    try:
        requirements = [
            "gradio==5.42.0",
            "pandas==2.2.2",
            "numpy==2.0.2",
            "matplotlib==3.10.0",
            "pillow==11.3.0",
            "scikit-learn==1.6.1",
            "joblib==1.5.1",
            "langchain==0.3.27",
            "langgraph==0.6.5"
        ]
        
        for req in requirements:
            try:
                module_name = req.split('>=')[0].replace('-', '_')
                __import__(module_name)
                print(f"{req} - already installed")
            except ImportError:
                print(f"Installing {req}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", req])
                print(f"{req} - installed")
                
    except Exception as e:
        print(f"Error installing requirements: {e}")
        print("Please install manually using: pip install -r requirements.txt")


def test_agent_functionality():
    """ A simple way to test the agents and data exchange without LangGraph """
    print(f"Creating training dataset...")
    train_df = create_training_data()

    print(f"Initiating viability model, training and saving the model...")
    agent = UpstreamViabilityAgent()
    agent.train(train_df)

    print(f"Loading the saved model...")
    agent.load()

    print(f"Pulling one sample from Real-time bioprocess simulator & predicting viability collapse...")
    sim = RealTimeBioprocessSimulator()
    sample = sim.simulate_one_step()
    print(sample)
    collapse_pred = agent.predict(sample)
    print(f"Viability collapse predicted: {collapse_pred}")


def run_agentic_pipeline(num_hours=100):
    """ Run the multi-agent pipeline to predict viability collapse on hourly bioprocess samples """
    try:
        n_hours = int(input("Enter the number of hours to simulate the bioprocess run for: "))
        print(f"Simulating bioprocess for {n_hours} hours...")
        results = run_pipeline(n_hours)
        pprint.pprint(results)
    except ValueError:
        print("Please enter a valid integer for the number of hours.")


def launch_visualization():
    """ Launch Gradio dashboard to run the multi-agent pipeline in real-time """
    launch_realtime_dashboard()


def main():
    """Main execution function with menu interface."""
    
    # Define menu options and corresponding functions
    menu_options = {
        1: ("Setup Environment", setup_environment),
        2: ("Test Agent Functionality", test_agent_functionality),
        3: ("Run Agentic Pipeline", run_agentic_pipeline),
        4: ("Launch Visualization", launch_visualization),
        5: ("Exit", lambda: sys.exit("Goodbye!"))
    }
    
    print("Welcome to the Multi-agent Bioprocess AI Executor!")

    while True:
        try:
            # Display menu
            print("\n" + "="*40)
            for key, (description, _) in menu_options.items():
                print(f"{key}. {description}")
            print("="*40)
            
            # Get user choice
            choice = int(input("Select option (1-5): ").strip())
            
            # Execute choice
            if choice in menu_options:
                description, func = menu_options[choice]
                print(f"\n{description}...")
                func()
                
                if choice != 5:  # Don't ask to continue if exiting
                    if input("\nContinue? (y/n): ").strip().lower() not in ['y', 'yes']:
                        print("Goodbye!")
                        break
            else:
                print("Invalid option. Please choose 1-5.")
                
        except (ValueError, KeyboardInterrupt):
            print("\n Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()