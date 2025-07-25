import json
import os
import re
from tqdm import tqdm


def load_jsonl(in_file):
    datas = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="load"):
            datas.append(json.loads(line))
    return datas


def save_jsonl(data: list, path: str, mode='w', verbose=True) -> None:
    with open(path, mode, encoding='utf-8') as f:
        if verbose:
            for line in tqdm(data, desc='save'):
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        else:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')


def convert_to_text(in_files, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for in_file in tqdm(in_files, desc="转换文件"):
        out_file = os.path.join(out_dir, "text_" + os.path.basename(in_file))
        if os.path.isfile(out_file):
            print(f"跳过 {os.path.basename(in_file)} - 文件已存在")
            continue

        datas = load_jsonl(in_file)
        new_datas = []

        for data in tqdm(datas, desc="processing"):
            text_parts = []

            for computation in data["computations"]:
                computation_text = []

                # 处理标题
                title = computation["title"].strip("*")
                title = re.sub(r"Computation\s*\d*", "", title).strip(" :")
                if title:
                    computation_text.append(title)

                # 处理条件
                conditions = computation.get("conditions", "").strip()
                if conditions:
                    computation_text.append(conditions)

                # 处理表达式
                expression = computation.get("expression", "").strip()
                if expression:
                    computation_text.append(expression)

                # 处理结果
                result = computation.get("result", "").strip()
                if result:
                    computation_text.append(f"Result: {result}")

                # 处理代码
                code = computation.get("code", "").strip()
                if code:
                    computation_text.append(f"```mathematica\n{code}\n```")

                # 处理执行错误
                execution_error = computation.get("execution_error", "").strip()
                if execution_error:
                    computation_text.append(f"Execution Error: {execution_error}")

                # 处理执行输出
                execution_output = computation.get("execution_output", "").strip()
                if execution_output:
                    computation_text.append(f"Execution Output: {execution_output}")

                text_parts.append("\n\n".join(computation_text))

            # 用分隔符连接所有计算
            full_text = "\n\n---\n\n".join(text_parts)

            # 创建新的数据项，保留原始字段但移除computations
            new_data = {key: value for key, value in data.items() if key != "computations"}
            new_data["text"] = full_text

            new_datas.append(new_data)

        save_jsonl(new_datas, out_file)


def main():
    in_dir = ""
    out_dir = ""

    in_files = [os.path.join(in_dir, file_name) for file_name in os.listdir(in_dir) if file_name.endswith("jsonl")]

    if not in_files:
        print(f"在目录 {in_dir} 中没有找到 .jsonl 文件")
        return

    print(f"找到 {len(in_files)} 个文件待处理")
    convert_to_text(in_files, out_dir)
    print("转换完成!")


if __name__ == "__main__":
    main()