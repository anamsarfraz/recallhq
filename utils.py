import json
import os


# Function to load state
def load_state(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        return {}

# Function to update state
def update_state(file_path, new_state):
    current_state = {}
    
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            current_state = json.load(f)
    
    current_state.update(new_state)
    
    with open(file_path, "w") as f:
        json.dump(current_state, f)