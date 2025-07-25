import os
import json
from tqdm import tqdm


in_dir = ""
out_dir = ""
out_file = ""


if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def load_jsonl(in_file):
    datas = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="load"):
            datas.append(json.loads(line))
    return datas

def save_jsonl(data: list, path: str, mode='w', verbose=True) -> None:
    file_name = path
    with open(file_name, mode, encoding='utf-8') as f:
        if verbose:
            for line in tqdm(data, desc='save'):
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        else:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')


def filter_and_save_data(in_file, out_file):
    datas = load_jsonl(in_file)
    filtered_datas = [data for data in datas if data.get("t") != "u"]
    save_jsonl(filtered_datas, out_file)


def create_file_names(in_dirs, out_file):
    file_paths = []
    for in_dir in in_dirs:
        dir_list = os.listdir(in_dir)
        file_paths.extend([os.path.join(in_dir, file_name) for file_name in dir_list if file_name.endswith("jsonl")])
    return file_paths


def main():
    file_paths = create_file_names([in_dir], out_file)
    filtered_file_paths = []
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        out_path = os.path.join(out_dir, file_name)
        filter_and_save_data(file_path, out_path)
        filtered_file_paths.append(out_path)
    with open(out_file, "w") as f:
        json.dump(filtered_file_paths, f)

if __name__ == "__main__":
    main()