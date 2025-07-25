import re
import json
import os
import sys
import requests
from io import StringIO
import threading
from tqdm import tqdm
from multiprocessing import Pool, RLock
from datasets import load_dataset
import time
from argparse import ArgumentParser
import glob
global_settings = None
global_api = None
def timestamp() -> str:
    nowtime = time.strftime('-%Y%m%d-%H%M', time.localtime(time.time()))
    return nowtime

def save_jsonl(data: list, path: str, mode='w', add_timestamp=True, verbose=True) -> None:
    if add_timestamp:
        file_name = f"{path.replace('.jsonl', '')}{timestamp()}.jsonl"
    else:
        file_name = path
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, mode, encoding='utf-8') as f:
        if verbose:
            for line in tqdm(data, desc='Saving data'):
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        else:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')

def load_jsonl(path: str):
    try:
        with open(path, "r", encoding='utf-8') as fh:
            return [json.loads(line) for line in fh.readlines() if line]
    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        return []

def merge_batch_files(output_dir, base_filename, final_output_path):
    print("\n开始合并批次文件...")
    batch_pattern = os.path.join(output_dir, f"{base_filename}_batch_*.jsonl")
    batch_files = sorted(glob.glob(batch_pattern))
    if not batch_files:
        print("没有找到批次文件")
        return
    print(f"找到 {len(batch_files)} 个批次文件")
    all_results = []
    for batch_file in batch_files:
        print(f"正在合并: {os.path.basename(batch_file)}")
        batch_data = load_jsonl(batch_file)
        all_results.extend(batch_data)
    save_jsonl(all_results, final_output_path, mode='w', add_timestamp=False, verbose=False)
    print(f"合并完成！总共 {len(all_results)} 条记录保存到: {final_output_path}")
    print("正在删除临时批次文件...")
    for batch_file in batch_files:
        try:
            os.remove(batch_file)
            print(f"已删除: {os.path.basename(batch_file)}")
        except Exception as e:
            print(f"删除失败 {batch_file}: {e}")
class API:


    def __init__(self, api_key):
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    def get_result(self, inputs, parameters=None):
        if parameters is None:
            parameters = {}
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": parameters.get("system", "")},
                {"role": "user", "content": inputs}
            ],
            "temperature": 0.7,
            "max_tokens": 3072,
            "stop": ["<|eot_id|>"]
        }
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"API call failed: {e}")
            return None


def init_worker(settings, api):
    global global_settings, global_api
    global_settings = settings
    global_api = api


def process_wrapper(data):
    try:
        problem = data.get("problem", "")
        solution = data.get("solution", "")
        category = data.get("category", "")
        if not problem or not solution:
            print(f"Skipping item due to missing problem or solution")
            return None
        combined_text = f"Problem: {problem}\n\nSolution: {solution}"
        instruction = global_settings["instruction"].format(text=combined_text)
        system = global_settings["system"]
        parameters = {
            "system": system,
            "do_sample": False,
            "max_new_tokens": 3072,
            "stop_sequences": ['<|eot_id|>']
        }
        result = global_api.get_result(instruction, parameters)
        if result:
            result = result.replace("<|eot_id|>", "").strip("\n\t ")
        if "Computation Expression:" not in result or "Computation Result:" not in result:
            print(f"Skipping result without proper computation format")
            return None
        try:
            computation_result_section = result.split("Computation Result:")[1].strip()
            if not re.search(r"[-+]?\d+\.?\d*", computation_result_section):
                print(f"Skipping result without numerical computation")
                return None
        except IndexError:
            print(f"Error parsing computation result")
            return None
        idx = data.get("id", data.get("idx", data.get("index", None)))
        if idx is None:
            print(f"Warning: No id field found in data")
            idx = -1
        result_data = {
            "id": idx,
            "text": result,
            "category": category
        }
        return result_data
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None


def process_batch(datas, batch_size, output_dir, base_filename, settings, api):
    total_batches = (len(datas) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(datas))
        batch_data = datas[start_idx:end_idx]
        print(f"\n处理批次 {batch_idx + 1}/{total_batches} (条目 {start_idx} 到 {end_idx - 1})")
        try:
            with Pool(12, initializer=init_worker, initargs=(settings, api)) as pool:
                results = []
                for result in tqdm(pool.imap_unordered(process_wrapper, batch_data),
                                   total=len(batch_data),
                                   desc=f"批次 {batch_idx + 1}"):
                    if result is not None:
                        results.append(result)

                if results:
                    batch_out_file = os.path.join(output_dir, f'{base_filename}_batch_{batch_idx + 1}.jsonl')
                    save_jsonl(results, batch_out_file, mode='w', add_timestamp=False, verbose=False)
                    print(f"保存了 {len(results)} 条结果到批次文件")
                else:
                    print(f"批次 {batch_idx + 1} 没有有效结果")
        except Exception as e:
            print(f"批次 {batch_idx + 1} 处理失败: {str(e)}")
            continue
        if batch_idx < total_batches - 1:
            print("等待20秒后处理下一批次...")
            time.sleep(20)


def main():
    input_file = r""
    output_dir = r""
    final_output_file = ""
    batch_size = 500
    start_from = 2500
    api_key = ""
    settings = {
        "system": "You are a clever mathematical AI assistant who is good at identifying  computations and solving them using Mathematica.",
        "instruction": "You will be presented with a math problem and its solution. I need you to identify all the  computations in the solution. "
                       "For each computation that requires a scratchpad, find out the conditions needed for the computation, "
                       "the latex expression that conducts the computation, and the result of the computation. "
                       "Then generate a Mathematica code snippet for each computation that demonstrates how the result is reached. "
                       "Ensure that Mathematica code snippet includes a Print statement to output the final result. "
                       "Output each computation in the following format:\n\n"
                       "Conditions Needed:\n1. [Condition 1]\n2. [Condition 2]\n...\n\n"
                       "Computation Expression:\n$[Latex Expression]$\n\n"
                       "Computation Result:\n[Computation Result]\n\n"
                       "Mathematica Code Snippet:\n```mathematica\n[Mathematica Code]\n```\n\n"
                       "There can be more than one complex computation in the solution. "
                       "Output only the computations that requires calculation. "
                       "Do not include mathematical statements or definitions as a computation. "
                       "Make sure each snippet can be executed individually. "
                       "The problem and solution are as follows: {text}\n\nThe computations are:",
    }
    api = API(api_key)
    print(f"从文件加载数据: {input_file}")
    datas = load_jsonl(input_file)
    if not datas:
        print("没有加载到数据，退出程序")
        return
    print(f"加载了 {len(datas)} 条记录")
    if start_from is not None and start_from < len(datas):
        datas = datas[start_from:]
        print(f"从第 {start_from} 条开始处理，剩余 {len(datas)} 条记录")
    os.makedirs(output_dir, exist_ok=True)
    base_filename = "processed_math_data_remaining"
    final_output_path = os.path.join(output_dir, final_output_file)
    print(f"开始分批处理，批次大小: {batch_size}")
    process_batch(datas, batch_size, output_dir, base_filename, settings, api)
    merge_batch_files(output_dir, base_filename, final_output_path)
    print("所有处理完成！")
if __name__ == "__main__":
    main()
