import json
import os
import re
import time
import signal
import sys
from tqdm import tqdm
from difflib import SequenceMatcher



def signal_handler(sig, frame):
    print('\n程序被中断！正在保存当前进度...')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


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
                        print(f"  JSON错误 (行 {line_num}): {str(e)[:100]}")
                    if error_count > 20:  # 如果错误太多，停止加载
                        print("  JSON错误过多，可能文件已损坏")
                        break

        if error_count > 0:
            print(f"  跳过了 {error_count} 条无效数据 (总行数: {total_lines})")
            if total_lines > 0 and error_count / total_lines > 0.1:
                print(f"  警告: 错误率较高 ({error_count / total_lines * 100:.1f}%)，请检查文件格式")

    except FileNotFoundError:
        print(f"  文件不存在: {in_file}")
        return []
    except Exception as e:
        print(f"  加载文件时发生错误: {str(e)[:100]}")
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
        print(f"  成功保存 {len(data)} 条数据到: {os.path.basename(path)}")
    except Exception as e:
        print(f"  保存文件失败 {path}: {str(e)[:100]}")
        raise


def normalize_math_expression(expr):
    if not expr:
        return ""

    expr = str(expr).strip()

    try:

        expr = re.sub(r'\\?\[.*?\\?\]', '', expr)  # 移除 \[ \]
        expr = re.sub(r'\$[^$]*\$', '', expr)  # 移除 $ $
        expr = re.sub(r'\\?\([^)]*\\?\)', '', expr)  # 移除 \( \)


        expr = re.sub(r'\\pm', '±', expr)
        expr = re.sub(r'\\mp', '∓', expr)
        expr = re.sub(r'\\sqrt\{([^}]+)\}', r'√(\1)', expr)
        expr = re.sub(r'\\sqrt\[([^]]+)\]\{([^}]+)\}', r'(\2)^(1/\1)', expr)
        expr = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', expr)
        expr = re.sub(r'\\cdot', '*', expr)
        expr = re.sub(r'\\times', '*', expr)
        expr = re.sub(r'\bI\b', 'i', expr)
        expr = re.sub(r'\*I\b', '*i', expr)
        expr = re.sub(r'\s+', ' ', expr)
        expr = expr.strip()
    except Exception:
        pass

    return expr


def extract_polynomial_coefficients(expr):

    if not expr or len(expr) > 200:
        return {}

    try:

        expr = str(expr).lower().replace(' ', '')
        expr = re.sub(r'\*', '', expr)

        coefficients = {}


        const_matches = re.findall(r'([+-]?\d+\.?\d*)(?![a-z])', expr)
        for match in const_matches:
            if match and match not in ['+', '-']:
                if match == '+':
                    match = '1'
                elif match == '-':
                    match = '-1'
                try:
                    coefficients['const'] = float(match)
                except ValueError:
                    continue


        linear_matches = re.findall(r'([+-]?\d*\.?\d*)([a-z])(?!\^|[a-z])', expr)
        for coeff, var in linear_matches:
            try:
                if not coeff or coeff == '+':
                    coeff = 1
                elif coeff == '-':
                    coeff = -1
                else:
                    coeff = float(coeff)
                coefficients[f"{var}^1"] = coeff
            except ValueError:
                continue


        power_matches = re.findall(r'([+-]?\d*\.?\d*)([a-z])\^(\d+)', expr)
        for coeff, var, power in power_matches:
            try:
                if not coeff or coeff == '+':
                    coeff = 1
                elif coeff == '-':
                    coeff = -1
                else:
                    coeff = float(coeff)
                coefficients[f"{var}^{power}"] = coeff
            except ValueError:
                continue

    except Exception:

        return {}

    return coefficients


def polynomials_equal(expr1, expr2, tolerance=0.01):

    try:
        coeffs1 = extract_polynomial_coefficients(expr1)
        coeffs2 = extract_polynomial_coefficients(expr2)

        if not coeffs1 or not coeffs2:
            return False


        all_keys = set(coeffs1.keys()) | set(coeffs2.keys())

        for key in all_keys:
            val1 = coeffs1.get(key, 0)
            val2 = coeffs2.get(key, 0)
            if abs(val1 - val2) > tolerance:
                return False

        return True
    except Exception:
        return False


def extract_simple_numbers(text):

    if not text:
        return []

    try:

        number_patterns = [
            r'-?\d+\.?\d*(?:[eE][+-]?\d+)?',
            r'-?\d+/\d+',
        ]

        numbers = []
        for pattern in number_patterns:
            matches = re.findall(pattern, str(text))
            for match in matches:
                try:
                    if '/' in match:
                        parts = match.split('/')
                        if float(parts[1]) != 0:
                            numbers.append(float(parts[0]) / float(parts[1]))
                    else:
                        numbers.append(float(match))
                except (ValueError, ZeroDivisionError):
                    continue

        return numbers
    except Exception:
        return []


def smart_math_compare(result, output, tolerance=0.01, max_length=1000):

    if not result and not output:
        return True
    if not result or not output:
        return False

    result_str = str(result).strip()
    output_str = str(output).strip()


    if len(result_str) > max_length or len(output_str) > max_length:
        return result_str.replace(' ', '') == output_str.replace(' ', '')


    if result_str.replace(' ', '') == output_str.replace(' ', ''):
        return True

    try:
        norm_result = normalize_math_expression(result_str)
        norm_output = normalize_math_expression(output_str)
        if norm_result.replace(' ', '') == norm_output.replace(' ', ''):
            return True
    except Exception:
        pass


    try:
        if polynomials_equal(result_str, output_str, tolerance):
            return True
    except Exception:
        pass

    try:
        result_nums = extract_simple_numbers(result_str)
        output_nums = extract_simple_numbers(output_str)

        if result_nums and output_nums:
            if len(result_nums) == 1 and len(output_nums) == 1:
                return abs(result_nums[0] - output_nums[0]) <= tolerance
            if len(result_nums) == len(output_nums) and len(result_nums) <= 10:
                result_nums.sort()
                output_nums.sort()
                for r_num, o_num in zip(result_nums, output_nums):
                    if abs(r_num - o_num) > tolerance:
                        break
                else:
                    return True
    except Exception:
        pass

    try:
        result_nums = extract_simple_numbers(result_str)
        output_nums = extract_simple_numbers(output_str)
        if result_nums and output_nums:
            for r_num in result_nums[:3]:
                for o_num in output_nums[:3]:
                    if abs(r_num - o_num) <= tolerance:
                        return True
    except Exception:
        pass

    try:
        clean_result = re.sub(r'[^\w]', '', result_str.lower())
        clean_output = re.sub(r'[^\w]', '', output_str.lower())
        if clean_result and clean_output:
            similarity = SequenceMatcher(None, clean_result, clean_output).ratio()
            if similarity > 0.8:
                return True
    except Exception:
        pass

    return False


def is_number(s):
    if not s or not isinstance(s, str):
        return False

    s = s.strip()
    if not s:
        return False

    try:
        complex(s)
        return True
    except ValueError:
        if re.search(r'\d+', s):
            return True
        return False


def normalize_number_string(s):
    if not s:
        return ""

    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\.0+$', '', s)
    return s


def analyze_execution_result(computation):
    gt = computation.get("result", "").strip()
    error = computation.get("execution_error", "").strip()
    output = computation.get("execution_output", "").strip()

    analysis = {
        "has_ground_truth": bool(gt),
        "has_error": bool(error),
        "has_output": bool(output),
        "gt_is_number": is_number(gt) if gt else False,
        "output_is_number": is_number(output) if output else False,
        "gt_length": len(gt),
        "output_length": len(output),
        "error_type": "",
        "match_type": ""
    }

    if error:
        if "timeout" in error.lower():
            analysis["error_type"] = "timeout"
        elif "syntax" in error.lower() or "parse" in error.lower():
            analysis["error_type"] = "syntax"
        elif "memory" in error.lower():
            analysis["error_type"] = "memory"
        elif "file" in error.lower() or "not found" in error.lower():
            analysis["error_type"] = "system"
        else:
            analysis["error_type"] = "other"

    if analysis["has_output"] and not analysis["has_error"]:
        if analysis["has_ground_truth"]:
            if smart_math_compare(gt, output):
                analysis["match_type"] = "smart_match"
            else:
                analysis["match_type"] = "mismatch"
        else:
            analysis["match_type"] = "no_ground_truth"

    return analysis


def is_execution_correct(computation):
    analysis = analyze_execution_result(computation)
    if analysis["has_error"]:
        return False
    if not analysis["has_output"]:
        return False
    if not analysis["has_ground_truth"]:
        return True
    if analysis["match_type"] == "smart_match":
        return True
    elif analysis["match_type"] == "mismatch":
        return False

    if (analysis["output_length"] > 0 and
            analysis["output_length"] < 10000):
        return True

    return False


def categorize_computation(computation):
    analysis = analyze_execution_result(computation)

    if analysis["has_error"]:
        if analysis["error_type"] == "timeout":
            return "timeout"
        elif analysis["error_type"] == "syntax":
            return "syntax_error"
        elif analysis["error_type"] == "system":
            return "system_error"
        else:
            return "execution_error"

    if not analysis["has_output"]:
        return "no_output"

    if not analysis["has_ground_truth"]:
        return "no_ground_truth"

    if analysis["match_type"] == "smart_match":
        return "correct"
    elif analysis["match_type"] == "mismatch":
        return "incorrect"
    else:
        return "uncertain"


def execution_result_filter(in_files, out_dir):
    categories = ["correct", "wrong"]
    category_dirs = {}

    for category in categories:
        category_dir = os.path.join(out_dir, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        category_dirs[category] = category_dir

    # 全局统计
    global_stats = {
        "total_files": len(in_files),
        "processed_files": 0,
        "failed_files": 0,
        "total_data_items": 0,
        "total_computations": 0,
        "category_counts": {cat: 0 for cat in categories},
        "detailed_stats": {
            "correct": 0,
            "incorrect": 0,
            "timeout": 0,
            "syntax_error": 0,
            "system_error": 0,
            "execution_error": 0,
            "no_output": 0,
            "no_ground_truth": 0,
            "uncertain": 0
        }
    }

    start_time = time.time()

    for in_file in tqdm(in_files, desc="处理文件"):
        base_name = os.path.basename(in_file)

        output_files = {}
        for category in categories:
            output_files[category] = os.path.join(
                category_dirs[category],
                f"{category}_{base_name}"
            )

        if all(os.path.isfile(f) for f in output_files.values()):
            print(f"\n跳过 {base_name} - 输出文件已存在")
            continue

        print(f"\n处理文件: {base_name}")
        file_start_time = time.time()

        try:
            datas = load_jsonl(in_file)
            if not datas:
                print("  没有有效数据")
                continue

            global_stats["total_data_items"] += len(datas)

            categorized_data = {category: [] for category in categories}
            file_stats = {
                "total_computations": 0,
                "category_counts": {cat: 0 for cat in categories},
                "detailed_stats": {key: 0 for key in global_stats["detailed_stats"]}
            }
            processed_count = 0
            for data in tqdm(datas, desc="分析数据", leave=False):
                if "computations" not in data or not isinstance(data["computations"], list):
                    continue

                computations = data["computations"]

                if processed_count % 100 == 0:
                    print(f"    正在处理第 {processed_count} 个数据项，包含 {len(computations)} 个计算...")

                categorized_computations = {category: [] for category in categories}

                for comp_idx, computation in enumerate(computations):
                    try:
                        file_stats["total_computations"] += 1
                        global_stats["total_computations"] += 1
                        comp_start_time = time.time()

                        detailed_category = categorize_computation(computation)
                        file_stats["detailed_stats"][detailed_category] += 1
                        global_stats["detailed_stats"][detailed_category] += 1

                        if is_execution_correct(computation):
                            categorized_computations["correct"].append(computation)
                            file_stats["category_counts"]["correct"] += 1
                            global_stats["category_counts"]["correct"] += 1
                        else:
                            categorized_computations["wrong"].append(computation)
                            file_stats["category_counts"]["wrong"] += 1
                            global_stats["category_counts"]["wrong"] += 1

                        comp_time = time.time() - comp_start_time
                        if comp_time > 5:
                            print(f"      警告: 计算 {comp_idx} 处理时间较长: {comp_time:.2f}秒")

                    except Exception as e:
                        print(f"      错误: 处理计算 {comp_idx} 时出错: {str(e)[:100]}")
                        categorized_computations["wrong"].append(computation)
                        file_stats["category_counts"]["wrong"] += 1
                        global_stats["category_counts"]["wrong"] += 1
                        continue


                for category in categories:
                    if categorized_computations[category]:
                        categorized_data[category].append({
                            "idx": data.get("idx", len(categorized_data[category])),
                            "computations": categorized_computations[category]
                        })

                processed_count += 1

            file_processing_time = time.time() - file_start_time
            print(f"  处理结果:")
            print(f"    输入数据条数: {len(datas)}")
            print(f"    总computation数: {file_stats['total_computations']}")

            for category in categories:
                count = file_stats["category_counts"][category]
                percentage = count / file_stats["total_computations"] * 100 if file_stats[
                                                                                   "total_computations"] > 0 else 0
                print(f"    {category}计算数: {count} ({percentage:.1f}%)")

            print(f"    详细分类:")
            for detail_cat, count in file_stats["detailed_stats"].items():
                if count > 0:
                    percentage = count / file_stats["total_computations"] * 100
                    print(f"      {detail_cat}: {count} ({percentage:.1f}%)")

            print(f"    处理耗时: {file_processing_time:.2f}秒")
            for category in categories:
                if categorized_data[category]:
                    save_jsonl(categorized_data[category], output_files[category])
                else:
                    print(f"    {category}类别没有数据")

            global_stats["processed_files"] += 1

        except Exception as e:
            print(f"  处理文件 {base_name} 时出错: {str(e)[:100]}")
            global_stats["failed_files"] += 1
            import traceback
            traceback.print_exc()
            continue


    total_time = time.time() - start_time
    print(f"\n全部处理完成!")
    print(f"全局统计:")
    print(f"  处理时间: {total_time:.2f}秒")
    print(f"  总文件数: {global_stats['total_files']}")
    print(f"  成功处理: {global_stats['processed_files']}")
    print(f"  处理失败: {global_stats['failed_files']}")
    print(f"  总数据条数: {global_stats['total_data_items']}")
    print(f"  总computation数: {global_stats['total_computations']}")

    print(f"\n分类统计:")
    for category in categories:
        count = global_stats["category_counts"][category]
        percentage = count / global_stats["total_computations"] * 100 if global_stats["total_computations"] > 0 else 0
        print(f"  {category}: {count} ({percentage:.1f}%)")

    print(f"\n详细分类统计:")
    for detail_cat, count in global_stats["detailed_stats"].items():
        if count > 0:
            percentage = count / global_stats["total_computations"] * 100
            print(f"  {detail_cat}: {count} ({percentage:.1f}%)")

    print(f"\n输出目录: {out_dir}")


def main():
    print("开始过滤执行结果...")
    print(f"输入目录: {in_dir}")
    print(f"输出目录: {out_dir}")
    print("提示: 如果程序卡住，可以按 Ctrl+C 安全中断")

    in_files = [
        os.path.join(in_dir, file_name)
        for file_name in os.listdir(in_dir)
        if file_name.endswith(".jsonl")
    ]

    if not in_files:
        print(f"在目录 {in_dir} 中没有找到 .jsonl 文件")
        return

    print(f"找到 {len(in_files)} 个文件待处理:")
    for f in in_files:
        file_size = os.path.getsize(f) / (1024 * 1024)
        print(f"  - {os.path.basename(f)} ({file_size:.1f} MB)")

    try:
        execution_result_filter(in_files, out_dir)
    except KeyboardInterrupt:
        print("\n处理被用户中断")
    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()