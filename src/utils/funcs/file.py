
import os, base64
import json
import subprocess
from datetime import date, datetime
from decimal import Decimal

import pandas as pd
import tiktoken

import numpy as np
from tqdm import tqdm
import pickle as pkl
import uuid

import re
import unicodedata

import os
import shutil
import argparse
from pathlib import Path

def copy_file_or_directory(source: str, destination: str) -> None:
    try:
        if not os.path.exists(source):
            raise FileNotFoundError(f"PATH Not exist: {source}")
        
        os.makedirs(destination, exist_ok=True)
        
        if os.path.isfile(source):
            file_name = os.path.basename(source)
            dest_path = os.path.join(destination, file_name)
            shutil.copy2(source, dest_path)
            print(f"Copy Success: {source} -> {dest_path}")
            return
        
        if os.path.isdir(source):
            base_dir = os.path.basename(source)
            destination = os.path.join(destination, base_dir)
            for item in os.listdir(source):
                item_path = os.path.join(source, item)
                dest_item_path = os.path.join(destination, item)
                
                if os.path.isfile(item_path):
                    shutil.copy2(item_path, dest_item_path)
                    print(f"Copy Success: {item_path} -> {dest_item_path}")
                elif os.path.isdir(item_path):
                    shutil.copytree(item_path, dest_item_path, dirs_exist_ok=True)
                    print(f"Copy Success for DIR: {item_path} -> {dest_item_path}")
    except Exception as e:
        print(f"ERROR: {e}")

def clean_filename(text: str, max_length: int = 255, replacement: str = '_') -> str:
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    text = re.sub(r'[<>:"/\\|?*]', replacement, text)  
    text = re.sub(r'[\x00-\x1F\x7F]', replacement, text)  
    text = re.sub(r'^\.+$', replacement, text) 
    text = re.sub(r'\s+', ' ', text).strip() 
    
    if len(text) > max_length:
        if '.' in text:
            parts = text.rsplit('.', 1)
            name_part = parts[0][:max_length - len(parts[1]) - 1]
            text = f"{name_part}.{parts[1]}"
        else:
            text = text[:max_length]
    
    return text

def set_proxy(proxy):
    os.environ['http_proxy'] = proxy
    os.environ['https_proxy'] = proxy

def unset_proxy():
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''

def load_text_file(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        return f.read()

def save_text_file(file_path, content):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(content)

def get_unique_identifier(suffix):
    """
    Generate a unique identifier for the task.
    """
    return str(uuid.uuid4()) + "_" + suffix

def load_prompt(prompt_path):
    with open(prompt_path, "r", encoding='utf-8') as f:
        return f.read()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def delete_content_between(s, start, end):
    start_idx = s.find(start)
    end_idx = s.find(end)
    if start_idx == -1 or end_idx == -1:
        return s
    return s[:start_idx] + s[end_idx:]

def rename_file(root, include_str, replace_to):
    # rename files in root directory
    # replace include_str to replace_to
    for path, subdirs, files in os.walk(root):
        for name in files:
            if include_str in name:
                new_name = name.replace(include_str, replace_to)
                os.rename(os.path.join(path, name), os.path.join(path, new_name))

def serialize_table(df:pd.DataFrame):
    return df.to_markdown(index=False,)


def run_python(code_for_processing: str):
    try:
        result = subprocess.run(
            ["python", "-c", code_for_processing],
            capture_output=True, text=True
        )
        if result.stderr:
            return f"Error occurred: {result.stderr}"
        return result.stdout
    except Exception as e:
        return f"An error occurred: {str(e)}"
    

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(folder_path)


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        d = json.loads(line)
        data.append(d)
    return data

def save_jsonl(path, datas):
    if os.path.exists(path):
        print(f'Path: {path} already exists! Please delete it!')
        return
    
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print(f'saving jsonl object to {path}...')
    with open(path, 'a', encoding='utf-8') as f:
        for d in tqdm(datas):
            f.write(json.dumps(d, default=convert_nat) + '\n')

def save_pickle(obj, path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(path, 'wb') as f:
        pkl.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj


def cal_f1(preds, labs):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for pred, lab in zip(preds, labs):
        if pred == 1 and lab == 1:
            tp += 1
        elif pred == 1 and lab == 0:
            fp += 1
        elif pred == 0 and lab == 1:
            fn += 1
        elif pred == 0 and lab == 0:
            tn += 1
    if tp == 0:
        return 0, fp, fn, tp, tn, 0, 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    # print(f'fp: {fp}, fn: {fn}, tp: {tp}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}')
    return f1, fp, fn, tp, tn, precision, recall

def cal_tokens(s):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    codelist = encoding.encode(s)
    return len(codelist)

def open_json(path):
    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

def convert_nat(o):
    if isinstance(o, pd._libs.missing.NAType):
        return None
    elif isinstance(o, (date, datetime)):
        return o.strftime('%Y-%m-%d %H:%M:%S') if isinstance(o, datetime) else o.strftime('%Y-%m-%d')
    elif isinstance(o, Decimal):
        return float(o)
    elif isinstance(o, uuid.UUID):
        return str(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

def save_json(a, fn):

    dir_path = os.path.dirname(fn)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    try:
        b = json.dumps(a, default=convert_nat)
        with open(fn, 'w') as f2:
            f2.write(b)
    except Exception as e:
        print(f"Error saving JSON: {e}")

def execute_code_from_string(code_string, df, glo = globals(), loc = locals()):
    try:
        loc['df'] = df
        exec(code_string, glo, loc)
        return locals()['df']
    except Exception as e:
        raise ValueError(f"Error executing code: {e}")

def load_tsv(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.strip().split('\t'))
    tsv = {}
    for i in range(len(data[0])):
        tsv[data[0][i]] = [data[j][i] for j in range(1,len(data))]
        
    return tsv

def all_filepaths_in_dir(root, endswith=None):
    file_paths = []
    for subdir, dirs, files in os.walk(root):
        for file in files:
            if endswith is None or file.endswith(endswith):
                file_paths.append(os.path.join(subdir, file))
    return file_paths
