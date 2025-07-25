import json
import os
from tqdm import tqdm

in_dir = ""
out_dir = ""

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def load_json(in_file):

    try:
        with open(in_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"  JSON解析错误 {in_file}: {str(e)[:100]}")
        return []
    except FileNotFoundError:
        print(f"  文件不存在: {in_file}")
        return []
    except Exception as e:
        print(f"  加载文件错误 {in_file}: {str(e)[:100]}")
        return []


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


def validate_computation_block(comp):

    required_fields = ["title", "conditions", "expression", "result", "code"]


    for field in required_fields:
        if field not in comp:
            return False, f"缺少字段: {field}"


    if not isinstance(comp["conditions"], list):
        return False, "conditions字段应该是列表"

    for field in ["title", "expression", "result", "code"]:
        if not isinstance(comp[field], str):
            return False, f"{field}字段应该是字符串"


    content_score = 0


    if comp["title"] and len(comp["title"].strip()) > 3:
        content_score += 1


    if comp["conditions"] and len(comp["conditions"]) > 0:
        valid_conditions = [c for c in comp["conditions"] if c and len(c.strip()) > 5]
        if valid_conditions:
            content_score += 1


    if comp["expression"] and len(comp["expression"].strip()) > 5:
        content_score += 1


    if comp["result"] and len(comp["result"].strip()) > 3:
        content_score += 1


    if comp["code"] and len(comp["code"].strip()) > 10:

        code_indicators = ["Print[", "=", "Solve[", "N[", "Plot[", ";"]
        if any(indicator in comp["code"] for indicator in code_indicators):
            content_score += 1

    if content_score < 3:
        return False, f"内容质量不足 (得分: {content_score}/5)"

    return True, "验证通过"


def standardize_computation_block(comp):

    standardized = {
        "title": "",
        "conditions": "",
        "expression": "",
        "result": "",
        "code": ""

    }


    title = comp.get("title", "").strip()
    title = title.strip("*").strip()
    standardized["title"] = title


    conditions = comp.get("conditions", [])
    if isinstance(conditions, list):
        condition_lines = []
        for i, condition in enumerate(conditions, 1):
            if condition and condition.strip():

                condition_text = condition.strip()
                if not condition_text.endswith('.') and not condition_text.endswith('。'):
                    condition_text += '.'
                condition_lines.append(f"{i}. {condition_text}")
        standardized["conditions"] = "\n".join(condition_lines)
    else:
        standardized["conditions"] = str(conditions).strip()


    for field in ["expression", "result", "code"]:
        standardized[field] = comp.get(field, "").strip()

    return standardized


def process_standardized_file(in_file, out_dir):

    base_name = os.path.splitext(os.path.basename(in_file))[0]

    output_name = base_name.replace('standardized_', '')
    out_file = os.path.join(out_dir, f"blocks_{output_name}.jsonl")

    if os.path.isfile(out_file):
        print(f"跳过 {os.path.basename(in_file)} - 输出文件已存在")
        return

    print(f"\n处理文件: {os.path.basename(in_file)}")


    data = load_json(in_file)
    if not data:
        print("  没有有效数据")
        return

    print(f"  加载了 {len(data)} 条记录")

    processed_data = []
    total_computations = 0
    valid_computations = 0
    invalid_computations = 0

    for item in tqdm(data, desc=f"处理 {os.path.basename(in_file)}"):
        if "idx" not in item or "computations" not in item:
            print(f"  跳过无效记录: 缺少必要字段")
            continue

        idx = item["idx"]
        computations = item["computations"]
        category = item.get("category", "")

        if not isinstance(computations, list):
            print(f"  跳过记录 {idx}: computations不是列表")
            continue

        processed_computations = []

        for comp_idx, comp in enumerate(computations):
            total_computations += 1

            is_valid, message = validate_computation_block(comp)

            if is_valid:

                standardized_comp = standardize_computation_block(comp)
                processed_computations.append(standardized_comp)
                valid_computations += 1
            else:
                invalid_computations += 1
                if invalid_computations <= 5:
                    print(f"    跳过无效计算块 (记录 {idx}, 计算块 {comp_idx}): {message}")

        if processed_computations:
            processed_item = {
                "idx": idx,
                "category": category,
                "computations": processed_computations
            }
            processed_data.append(processed_item)


    print(f"  处理完成:")
    print(f"    输入记录数: {len(data)}")
    print(f"    输出记录数: {len(processed_data)}")
    print(f"    总计算块数: {total_computations}")
    print(f"    有效计算块数: {valid_computations}")
    print(f"    无效计算块数: {invalid_computations}")

    if total_computations > 0:
        success_rate = valid_computations / total_computations * 100
        print(f"    计算块有效率: {success_rate:.1f}%")

        if success_rate < 50:
            print(f"    警告: 有效率较低，请检查数据质量")
    else:
        print(f"    计算块有效率: 0%")

    if processed_data:
        save_jsonl(processed_data, out_file)
    else:
        print("    没有有效数据，跳过保存")


def process_all_files(in_dir, out_dir):
    in_files = [
        os.path.join(in_dir, file_name)
        for file_name in os.listdir(in_dir)
        if file_name.endswith(".json")
    ]

    if not in_files:
        print(f"在 {in_dir} 中没有找到 .json 文件")
        return

    print(f"找到 {len(in_files)} 个文件待处理:")
    for f in in_files:
        print(f"  - {os.path.basename(f)}")


    total_files = len(in_files)
    successful_files = 0
    failed_files = 0

    for in_file in in_files:
        try:
            process_standardized_file(in_file, out_dir)
            successful_files += 1
        except Exception as e:
            failed_files += 1
            print(f"处理文件 {os.path.basename(in_file)} 时出错: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n全部处理完成!")
    print(f"处理统计:")
    print(f"  总文件数: {total_files}")
    print(f"  成功处理: {successful_files}")
    print(f"  处理失败: {failed_files}")
    print(f"  成功率: {successful_files / total_files * 100:.1f}%")
    print(f"输出目录: {out_dir}")


def main():

    print("开始处理标准化数据...")
    print(f"输入目录: {in_dir}")
    print(f"输出目录: {out_dir}")

    process_all_files(in_dir, out_dir)


if __name__ == "__main__":
    main()