import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import requests

# Formats
FORMAT = {
    "task_id": [],
    "created": [],
    "process id": [],
    "job name": [],
    "accuracy": [],
    "running": [],
}

HOME_DIR = os.path.join(os.path.expanduser ('~'),'.wq')
BASE_LOG_DIR = os.path.join(HOME_DIR, "logs")
DEFAULT_LOG_DIR_OUT = f"{BASE_LOG_DIR}/stdout.txt"

def train_model(learning_rate):
    response = requests.post('http://localhost:5000/train', json={'learning_rate': learning_rate})
    data = response.json()
    return data["accuracy"]

def new_job():
    pass

def launch_command_process(command, log_filename):
    pass

def display_process_log_file(log_file_name):
    pass

def terminate_process(pid):
    pass

def main():
    st.header('MNIST tracker')
    process_df = pd.DataFrame(FORMAT) 
    st.table(process_df)
    with st.expander("New Job"):
        st.subheader('Hyperparameters')
        lr = st.number_input("learning_rate", 0.1, 1.0, 0.1 )
        if st.button("Train"):
            st.info("Tranining ...")
            command = f"python train.py -lr {lr}"
            test_command_process = launch_command_process(command, DEFAULT_LOG_DIR_OUT)
            stdout = st.empty()
        stop = st.checkbox("Stop")
        while True: 
            poll = test_command_process.poll()
            stdout.code(display_process_log_file(DEFAULT_LOG_DIR_OUT))
            if stop and poll is not None:
                terminate_process(test_command_process.pid)
                break

if __name__ == "__main__":
    main()