import json
import os
import sys
import time
from tqdm import tqdm
from argparse import ArgumentParser
from mathematica_executor import MathematicaExecutor
in_dir = ""
out_dir = ""
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
def load_jsonl(in_file):
    datas = []
    error_count = 0
    total_lines = 0
    try:
        with open(in_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(tqdm(f, desc="加载数据"), 1):
                total_lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data and isinstance(data, dict):
                        datas.append(data)
                except json.JSONDecodeError as e:
                    error_count += 1
                    if error_count <= 3:
                        print(f"JSON错误 (行 {line_num}): {str(e)[:100]}")
                    if error_count > 50:
                        print("JSON错误过多，可能文件已损坏")
                        break
        if error_count > 0:
            print(f"跳过了 {error_count} 条无效数据 (总行数: {total_lines})")
            if error_count / total_lines > 0.1:
                print(f"警告: 错误率较高 ({error_count / total_lines * 100:.1f}%)，请检查文件格式")
    except FileNotFoundError:
        print(f"文件不存在: {in_file}")
        return []
    except Exception as e:
        print(f"加载文件时发生错误: {str(e)[:100]}")
        return []
    return datas
def save_jsonl(data: list, path: str, mode='w', verbose=True) -> None:
    try:
        with open(path, mode, encoding='utf-8') as f:
            if verbose:
                for line in tqdm(data, desc='保存数据'):
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
            else:
                for line in data:
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
        print(f"成功保存 {len(data)} 条数据到 {os.path.basename(path)}")
    except Exception as e:
        print(f"保存文件失败 {path}: {str(e)[:100]}")
        raise
def validate_computation_data(datas):
    valid_datas = []
    invalid_count = 0
    validation_stats = {
        "missing_idx": 0,
        "missing_computations": 0,
        "invalid_computation_structure": 0,
        "no_valid_computations": 0
    }
    for data in datas:
        if not isinstance(data, dict):
            invalid_count += 1
            continue
        if "idx" not in data:
            validation_stats["missing_idx"] += 1
            invalid_count += 1
            continue
        if "computations" not in data or not isinstance(data["computations"], list):
            validation_stats["missing_computations"] += 1
            invalid_count += 1
            continue
     valid_computations = []
        for comp in data["computations"]:
            if not isinstance(comp, dict):
                validation_stats["invalid_computation_structure"] += 1
                continue
            if "code" not in comp:
                validation_stats["invalid_computation_structure"] += 1
                continue
            required_fields = {
                "title": "",
                "conditions": "",
                "expression": "",
                "result": "",
                "execution_error": "",
                "execution_output": ""
            }

            for field, default_value in required_fields.items():
                if field not in comp:
                    comp[field] = default_value

            valid_computations.append(comp)

        if valid_computations:
            data["computations"] = valid_computations
            valid_datas.append(data)
        else:
            validation_stats["no_valid_computations"] += 1
            invalid_count += 1
    print(f"数据验证完成:")
    print(f"有效数据: {len(valid_datas)}")
    print(f"无效数据: {invalid_count}")
    if invalid_count > 0:
        print(f"验证详情:")
        for error_type, count in validation_stats.items():
            if count > 0:
                print(f"{error_type}: {count}")

    return valid_datas
def prepare_execution_data(datas):
    execution_tasks = []
    total_computations = 0
    valid_code_count = 0
    empty_code_count = 0
    for i, data in enumerate(datas):
        for j, computation in enumerate(data["computations"]):
            total_computations += 1
            code = computation.get("code", "").strip()
            if code:
                execution_tasks.append({
                    "computation": computation,
                    "data_index": i,
                    "comp_index": j,
                    "code": code
                })
                valid_code_count += 1
            else:
                empty_code_count += 1

    print(f"执行准备统计:")
    print(f"总computation数: {total_computations}")
    print(f"有代码的computation数: {valid_code_count}")
    print(f"无代码的computation数: {empty_code_count}")
    print( f"代码覆盖率: {valid_code_count / total_computations * 100:.1f}%" if total_computations > 0 else "    代码覆盖率: 0%")
    return execution_tasks
def execute_code_files(in_files, out_dir, args):
    executor = MathematicaExecutor(
        timeout=args.timeout,
        max_workers=args.max_workers,
        temp_dir=r"D:\temp",
        wolframscript_path=r"D:\wolfram14.2\wolframscript.exe"
    )
    print(f"执行器配置:")
    stats = executor.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    global_stats = {
        "total_files": len(in_files),
        "processed_files": 0,
        "failed_files": 0,
        "total_computations": 0,
        "successful_executions": 0,
        "failed_executions": 0,
        "total_time": 0
    }
    start_time = time.time()
    for in_file in tqdm(in_files, desc="处理文件"):
        base_name = os.path.splitext(os.path.basename(in_file))[0]
        output_name = base_name.replace('blocks_', '')
        out_file = os.path.join(out_dir, f"executed_{output_name}.jsonl")
        if os.path.isfile(out_file):
            print(f"\n跳过 {os.path.basename(in_file)} - 输出文件已存在")
            continue
        print(f"\n开始处理文件: {os.path.basename(in_file)}")
        file_start_time = time.time()
        try:
            datas = load_jsonl(in_file)
            if not datas:
                print(f"文件 {os.path.basename(in_file)} 中没有有效数据")
                continue
            datas = validate_computation_data(datas)
            if not datas:
                print(f"文件 {os.path.basename(in_file)} 中没有有效的computation数据")
                continue
            execution_tasks = prepare_execution_data(datas)
            global_stats["total_computations"] += sum(len(data["computations"]) for data in datas)
            if not execution_tasks:
                print("没有找到可执行的代码，跳过此文件")
                save_jsonl(datas, out_file)
                continue
            batch_size = min(5, len(execution_tasks))
            success_count = 0
            error_count = 0
            for idx in tqdm(range(0, len(execution_tasks), batch_size), desc="执行批次"):
                batch_tasks = execution_tasks[idx: idx + batch_size]
                batch_code = [task["code"] for task in batch_tasks]
                try:
                    print(f"执行批次 {idx // batch_size + 1}/{(len(execution_tasks) + batch_size - 1) // batch_size}, 代码数量: {len(batch_code)}")
                    batch_result = executor.batch_apply(batch_code)
                    for result, task in zip(batch_result, batch_tasks):
                        data_idx = task["data_index"]
                        comp_idx = task["comp_index"]
                        output = result[0].strip() if result[0] else ""
                        error_info = result[1].get("concise_exec_info", "").strip() if isinstance(result[1],                                                                                      dict) else str(
                            result[1]).strip()
                        datas[data_idx]["computations"][comp_idx]["execution_output"] = output
                        datas[data_idx]["computations"][comp_idx]["execution_error"] = error_info
                        if output and not error_info:
                            success_count += 1
                            global_stats["successful_executions"] += 1
                        else:
                            error_count += 1
                            global_stats["failed_executions"] += 1
                except Exception as e:
                    print(f"批次执行失败: {str(e)[:100]}")

                    for task in batch_tasks:
                        data_idx = task["data_index"]
                        comp_idx = task["comp_index"]
                        datas[data_idx]["computations"][comp_idx][
                            "execution_error"] = f"Batch execution failed: {str(e)[:100]}"
                        error_count += 1
                        global_stats["failed_executions"] += 1
            save_jsonl(datas, out_file)
            file_execution_time = time.time() - file_start_time
            print(f"文件处理完成:")
            print(f"执行时间: {file_execution_time:.2f} 秒")
            print(f"成功执行: {success_count}")
            print(f"执行失败: {error_count}")
            if (success_count + error_count) > 0:
                success_rate = success_count / (success_count + error_count) * 100
                print(f"成功率: {success_rate:.1f}%")
            else:
                print(f"成功率: N/A")

            global_stats["processed_files"] += 1

        except Exception as e:
            print(f"处理文件 {os.path.basename(in_file)} 时发生错误: {str(e)[:100]}")
            global_stats["failed_files"] += 1
            import traceback
            traceback.print_exc()
            continue

    global_stats["total_time"] = time.time() - start_time
    print(f"\n全部处理完成!")
    print(f"全局统计:")
    for key, value in global_stats.items():
        if "time" in key:
            print(f"{key}: {value:.2f} 秒")
        elif "rate" in key or "percentage" in key:
            print(f"{key}: {value:.1f}%")
        else:
            print(f"{key}: {value}")

    if global_stats["total_computations"] > 0:
        total_success_rate = global_stats["successful_executions"] / global_stats["total_computations"] * 100
        print(f"总体执行成功率: {total_success_rate:.1f}%")

def main():
    global in_dir, out_dir
    parser = ArgumentParser(description="执行Mathematica代码")
    parser.add_argument("--input-dir", default=in_dir, help="输入目录")
    parser.add_argument("--output-dir", default=out_dir, help="输出目录")
    parser.add_argument("--file-pattern", default="*.jsonl", help="文件模式")
    parser.add_argument("--timeout", type=int, default=120, help="执行超时时间(秒)")
    parser.add_argument("--max-workers", type=int, default=2, help="最大并发数")
    args = parser.parse_args()
    in_dir = args.input_dir
    out_dir = args.output_dir
    print("开始执行Mathematica代码...")
    print(f"输入目录: {in_dir}")
    print(f"输出目录: {out_dir}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if args.file_pattern == "*.jsonl":
        in_files = [
            os.path.join(in_dir, file_name)
            for file_name in os.listdir(in_dir)
            if file_name.endswith(".jsonl")
        ]
    else:
        import glob
        in_files = glob.glob(os.path.join(in_dir, args.file_pattern))
    if not in_files:
        print(f"在目录 {in_dir} 中没有找到匹配的文件")
        return

    print(f"找到 {len(in_files)} 个文件待处理:")
    for f in in_files:
        print(f"- {os.path.basename(f)}")
    execute_code_files(in_files, out_dir, args)


if __name__ == "__main__":
    main()
