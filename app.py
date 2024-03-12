import json
import shutil
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
        "dropout_rate" : [jsonData["dropout_rate"]], 
        "num_epochs" : [jsonData["num_epochs"]],
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
    process_df = process_df.astype({"task_id": int, "num_epochs" : int})
    st.dataframe(process_df)
    if len(process_df): 
        if st.button("Clear table"):
            empty = pd.DataFrame(FORMAT)
            empty.to_sql("metrics", con=sql_engine, if_exists="replace", index=False)
            refresh_app()

    with st.expander("New job"):
        st.subheader('Hyperparameters')
        lr = st.number_input("learning_rate", 0.1, 1.0, 0.1 )
        dropout_rate = st.number_input("dropout_rate", 0.0, 1.0, 0.5)
        num_epochs = st.number_input("num_epochs", 1, 10, 5)
        if st.button("Submit"):
            # st.info("Tranining ...")
            new_task_id = process_df["task_id"].max() + 1 if len(process_df["task_id"].values) else 1
            command = f"python train.py --lr {lr} --num_epochs {num_epochs} --task_id {new_task_id} --dropout_rate {dropout_rate}"
            popen = launch_command_process(command, DEFAULT_LOG_DIR_OUT)
            stdout = st.empty()
            mask = False
            if len(process_df):
                mask = (process_df['learning_rate'] == lr) & (process_df['num_epochs'] == num_epochs) & (process_df['dropout_rate'] == dropout_rate)
            if mask.any():
                st.warning('Warning: exactly same job has been run')
            else:
                with st.spinner("Wait for training and submitting job ..."):
                    while True: 
                        poll = popen.poll()
                        stdout.code(display_process_log_file(DEFAULT_LOG_DIR_OUT))
                        if poll is not None:
                            break
                shutil.copyfile(DEFAULT_LOG_DIR_OUT, f"{BASE_LOG_DIR}/{new_task_id}_stdout.txt")       
                jsonData = read_result(DEFAULT_RESULT_DIR_OUT)
                jsonData.update({
                    "lr" : lr, 
                    "num_epochs" : num_epochs,
                    "dropout_rate" : dropout_rate
                })
                submit_job(new_task_id, jsonData, sql_engine)
                st.success(
                    f"Submitted job with task_id {new_task_id}"
                )
                refresh_app(4)

    with st.expander("Explore task"):
        explore_task_id = st.selectbox("task_id", process_df["task_id"].unique())
        if explore_task_id:
            st.write("## Task Log")
            st.code(
                display_process_log_file(
                    f"{BASE_LOG_DIR}/{explore_task_id}_stdout.txt"
                )
            )
        
            st.write("## Plot")
            st.image(f"{BASE_LOG_DIR}/{explore_task_id}.png", width=400)

if __name__ == "__main__":
    Path(BASE_LOG_DIR).mkdir(parents=True, exist_ok=True)
    sql_engine = create_engine(APP_ENGINE_PATH, echo=False)
    main(sql_engine)