import json
from multiprocessing import Process
from datetime import datetime, timedelta
from pathlib import Path
import time
from subprocess import Popen, PIPE
import psutil
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from const import * 

def refresh_app(to_wait: int = 0) -> None:
    if to_wait:
        empty_slot = st.empty()

        for i in range(to_wait):
            empty_slot.write(f"Refreshing in {to_wait - i} seconds...")
            time.sleep(1)
    raise st.script_runner.RerunException(st.script_request_queue.RerunData())

def submit_job(task_id, jsonData, sql_engine):
    df = pd.DataFrame({    
        "task_id" : [task_id], 
        "learning_rate" : [jsonData["lr"]], 
        "accuracy" : [jsonData["acc"]],
        "test_loss" : [jsonData["test_loss"]], 
    })
    df.to_sql("metrics", con=sql_engine, if_exists="append", index=False)

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

def read_result(json_file):
    f = open(json_file)
    jsonData = json.load(f)
    return jsonData

def main(sql_engine):
    st.header('MNIST tracker')
    try: 
        process_df = pd.read_sql_table("metrics", con=sql_engine)
    except ValueError:
        process_df = pd.DataFrame(FORMAT) 
    process_df = process_df.astype({"task_id": int})
    st.table(process_df)
    if len(process_df): 
        if st.button("Clear table"):
            empty = pd.DataFrame(FORMAT)
            empty.to_sql("metrics", con=sql_engine, if_exists="replace", index=False)
            refresh_app()

    with st.expander("New Job"):
        st.subheader('Hyperparameters')
        lr = st.number_input("learning_rate", 0.1, 1.0, 0.1 )
        if st.button("Submit"):
            st.info("Tranining ...")
            command = f"python train.py --lr {lr}"
            popen = launch_command_process(command, DEFAULT_LOG_DIR_OUT)
            stdout = st.empty()
            while True: 
                poll = popen.poll()
                stdout.code(display_process_log_file(DEFAULT_LOG_DIR_OUT))
                if poll is not None:
                    break
            new_task_id = process_df["task_id"].max() + 1 if len(process_df["task_id"].values) else 1
            jsonData = read_result(DEFAULT_RESULT_DIR_OUT)
            jsonData["lr"] = lr
            submit_job(new_task_id, jsonData, sql_engine)
            st.success(
                f"Submitted task with task_id {new_task_id}"
            )
            refresh_app(4)

if __name__ == "__main__":
    Path(BASE_LOG_DIR).mkdir(parents=True, exist_ok=True)
    sql_engine = create_engine(APP_ENGINE_PATH, echo=False)
    main(sql_engine)