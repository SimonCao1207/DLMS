from pathlib import Path
from subprocess import Popen, PIPE
import psutil
import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import requests
from sqlalchemy import create_engine
from train import train_model
from const import * 

def submit_job(task_id, accuracy, sql_engine):
    df = pd.DataFrame({    
        "task_id" : [task_id], 
        "accuracy" : [accuracy]
    })
    df.to_sql("processes", con=sql_engine, if_exists="append", index=False)

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

def main(sql_engine):
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
            new_task_id = process_df["task_id"].max() + 1 if process_df["task_id"].values else 1

            

if __name__ == "__main__":
    Path(BASE_LOG_DIR).mkdir(parents=True, exist_ok=True)
    sql_engine = create_engine(APP_ENGINE_PATH, echo=False)
    main(sql_engine)