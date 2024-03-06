from pathlib import Path
from subprocess import Popen, PIPE
import psutil
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

def launch_command_process(command, log_filepath):
    try:
        with open(log_filepath, "w") as out:
            return Popen(command, stdout=out, stderr=out, shell=True) 
    except OSError as exc:
        raise exc

def read_log(filename: str): 
    try:
        with open(filename, "r", encoding="utf-8") as reader:
            return reader.readlines()
    except FileNotFoundError as exc:
        raise exc

def display_process_log_file(log_filename):
    try:
        return "".join(read_log(log_filename))
    except FileNotFoundError:
        return f"Waiting for {log_filename} to be created..."

def terminate_process(pid):
    try:
        parent = psutil.Process(pid)
        terminate_child_processes(parent)
        parent.terminate()
        parent.kill()
    except psutil.NoSuchProcess as exc:
        raise exc

def terminate_child_processes(parent_process: psutil.Process) -> None:
    if child_processes := parent_process.children(recursive=True):

        for child_process in child_processes:
            child_process.terminate()

        gone, alive = psutil.wait_procs(child_processes, timeout=3)

        for process in alive:
            process.terminate()

def main():
    st.header('MNIST tracker')
    process_df = pd.DataFrame(FORMAT) 
    st.table(process_df)
    with st.expander("New Job"):
        st.subheader('Hyperparameters')
        lr = st.number_input("learning_rate", 0.1, 1.0, 0.1 )
        if st.button("Train"):
            st.info("Tranining ...")
            command = f"python train.py --lr {lr}"
            popen = launch_command_process(command, DEFAULT_LOG_DIR_OUT)
            stdout = st.empty()
            stop = st.checkbox("Stop")
            while True: 
                poll = popen.poll()
                stdout.code(display_process_log_file(DEFAULT_LOG_DIR_OUT))
                if stop and poll is not None:
                    terminate_process(popen.pid)
                    break

if __name__ == "__main__":
    Path(BASE_LOG_DIR).mkdir(parents=True, exist_ok=True)
    main()