import json
import os
import re
from tqdm import tqdm
in_dir = ""
out_dir = ""
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
def load_jsonl(in_file):
    datas = []
    error_count = 0
    with open(in_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
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
                    print(f"SON错误 (行 {line_num}): {str(e)[:100]}")
    if error_count > 0:
        print(f"跳过了 {error_count} 条无效数据")
    return datas

def save_json(data, out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
def normalize_text(text):
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text

def parse_conditions(conditions_text):
    if not conditions_text:
        return []
    conditions = []
    numbered_pattern = r'(\d+)\.\s*(.*?)(?=\n\s*\d+\.|$)'
    matches = re.findall(numbered_pattern, conditions_text, re.DOTALL)
    if matches:
        for num, condition in matches:
            condition = normalize_text(condition)
            if condition:
                conditions.append(condition)
        return conditions
    lines = conditions_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^\s*[-*•]\s*', '', line)
        line = re.sub(r'^\s*\d+\.\s*', '', line)
        line = normalize_text(line)
        if line and len(line) > 3:
            conditions.append(line)
    if not conditions and conditions_text.strip():
        conditions.append(normalize_text(conditions_text))
    return conditions


def clean_expression(expression_text):
    if not expression_text:
        return ""
    expression = expression_text.strip()
    expression = re.sub(r'\\\[\s*', '\\[\n', expression)
    expression = re.sub(r'\s*\\\]', '\n\\]', expression)
    lines = expression.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


def clean_result(result_text):
    if not result_text:
        return ""
    result = result_text.strip()
    result = re.sub(r'^(Computation Result:|Result:|Answer:)\s*', '', result, flags=re.IGNORECASE)
    return normalize_text(result)


def clean_mathematica_code(code_text):
    if not code_text:
        return ""
    code = re.sub(r'```mathematica\s*\n?', '', code_text, flags=re.IGNORECASE)
    code = re.sub(r'\n?\s*```\s*$', '', code)
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if (line.startswith('(*') and line.endswith('*)')) or \
                ('=' in line) or \
                ('Print[' in line) or \
                (line.endswith(';')):
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


def extract_title(text_block):
    if not text_block:
        return ""
    title_patterns = [
        r'###\s*Computation\s*\d*:?\s*(.*?)(?:\n|$)',
        r'\*\*([^*\n]+?)\*\*:?\s*(?:\n|$)',
        r'^\s*([^:\n]{5,80}?)[:：]?\s*(?:\n|$)',
        r'(\d+)\.\s*\*\*([^*]+?)\*\*',
        r'^([^.\n]{5,80}?)(?:\.|:)?\s*(?:\n|$)'
    ]

    for pattern in title_patterns:
        match = re.search(pattern, text_block, re.MULTILINE)
        if match:
            title = match.groups()[-1].strip()
            if (title and
                    len(title) > 3 and
                    len(title) < 100 and
                    not title.lower().startswith('condition') and
                    not title.lower().startswith('computation result')):
                return normalize_text(title)
    return ""


def parse_single_computation(text_block, index=0):
    text_block = re.sub(
        r'\*\*\s*(Conditions Needed|Computation Expression|Computation Result|Mathematica Code Snippet)\s*:\s*\*\*',
        r'\1:', text_block)
    computation = {
        "title": "",
        "conditions": [],
        "expression": "",
        "result": "",
        "code": ""
    }
    title = extract_title(text_block)
    computation["title"] = title if title else f"Computation {index + 1}"
    conditions_match = re.search(
        r'Conditions Needed:\s*(.*?)(?=\n\s*-\s*Computation Expression:|\n\s*Computation Expression:|\n\s*Computation Result:|\n\s*Mathematica Code Snippet:|\Z)',
        text_block, re.DOTALL | re.IGNORECASE)
    if conditions_match:
        conditions_text = conditions_match.group(1)
        computation["conditions"] = parse_conditions(conditions_text)
    expression_match = re.search(
        r'Computation Expression:\s*(.*?)(?=\n\s*-\s*Computation Result:|\n\s*Computation Result:|\n\s*Mathematica Code Snippet:|\Z)',
        text_block, re.DOTALL | re.IGNORECASE)
    if expression_match:
        expression_text = expression_match.group(1)
        computation["expression"] = clean_expression(expression_text)
    result_match = re.search(
        r'Computation Result:\s*(.*?)(?=\n\s*-\s*Mathematica Code Snippet:|\n\s*Mathematica Code Snippet:|\Z)',
        text_block, re.DOTALL | re.IGNORECASE)
    if result_match:
        result_text = result_match.group(1)
        computation["result"] = clean_result(result_text)
    code_match = re.search(r'Mathematica Code Snippet:\s*(.*?)(?:\Z)',
                           text_block, re.DOTALL | re.IGNORECASE)
    if code_match:
        code_text = code_match.group(1)
        computation["code"] = clean_mathematica_code(code_text)
    return computation


def standardize_text_data(text):
    if not text or not isinstance(text, str):
       return []
    computations = []
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    if "---" in text and text.count("Computation") >= 2:
        print("使用 --- 分割")
        blocks = [block.strip() for block in text.split("---") if block.strip()]
        for i, block in enumerate(blocks):
            if any(marker in block for marker in
                   ["Conditions Needed:", "Computation Expression:", "Computation Result:"]):
                comp = parse_single_computation(block, i)
                if any([comp["conditions"], comp["expression"], comp["result"], comp["code"]]):
                    computations.append(comp)
    elif "### Computation" in text:
        print("使用 ### Computation 分割")
        parts = re.split(r'(?=### Computation \d+)', text)
        for i, part in enumerate(parts):
            if "### Computation" in part:
                comp = parse_single_computation(part, i)
                if any([comp["conditions"], comp["expression"], comp["result"], comp["code"]]):
                    computations.append(comp)
 
    elif re.search(r'\d+\.\s*\*\*.*?\*\*', text):
        print("使用编号分割策略")
        parts = re.split(r'(?=\d+\.\s*\*\*)', text)
        for i, part in enumerate(parts):
            if re.match(r'\d+\.\s*\*\*', part):
                comp = parse_single_computation(part, i)
                if any([comp["conditions"], comp["expression"], comp["result"], comp["code"]]):
                    computations.append(comp)
    else:
        print("使用单块处理策略")
        if any(marker in text for marker in ["Conditions Needed:", "Computation Expression:", "Computation Result:"]):
            comp = parse_single_computation(text, 0)
            if any([comp["conditions"], comp["expression"], comp["result"], comp["code"]]):
                computations.append(comp)
    return computations


def validate_computation(comp):
    score = 0

    if comp["title"] and len(comp["title"]) > 5:
        score += 1
    if comp["conditions"] and len(comp["conditions"]) > 0:
        total_length = sum(len(c) for c in comp["conditions"])
        if total_length > 20:
            score += 2
    if comp["expression"] and len(comp["expression"]) > 10:
        score += 2
    if comp["result"] and len(comp["result"]) > 5:
        score += 1
    if comp["code"] and len(comp["code"]) > 15:
        if "Print[" in comp["code"] or "=" in comp["code"]:
            score += 2
    return score >= 4


def process_files(in_files, out_dir):
    for in_file in tqdm(in_files, desc="处理文件"):
        base_name = os.path.splitext(os.path.basename(in_file))[0]
        out_file = os.path.join(out_dir, f"standardized_{base_name}.json")
        if os.path.isfile(out_file):
            print(f"跳过 {in_file} - 输出文件已存在")
            continue
        print(f"\n处理文件: {os.path.basename(in_file)}")
        datas = load_jsonl(in_file)
        if not datas:
            print("没有有效数据")
            continue
        print(f"加载了 {len(datas)} 条记录")
        standardized_data = []
        for data in tqdm(datas, desc=f"标准化 {os.path.basename(in_file)}"):
            if "text" not in data or not data["text"]:
                continue
            if "idx" not in data:
                print(f"警告：记录缺少idx字段，跳过: {data}")
                continue
            idx = data["idx"]
            category = data.get("category", "")
            computations = standardize_text_data(data["text"])
            valid_computations = []
            for comp in computations:
                if validate_computation(comp):
                    valid_computations.append(comp)
            if valid_computations:
                standardized_item = {
                    "idx": idx,
                    "category": category,
                    "computations": valid_computations
                }
                standardized_data.append(standardized_item)
        print(f"标准化完成:")
        print(f"输入记录: {len(datas)}")
        print(f"输出记录: {len(standardized_data)}")
        print(f"成功率: {len(standardized_data) / len(datas) * 100:.1f}%")
        if standardized_data:
            save_json(standardized_data, out_file)
            print(f"保存到: {out_file}")
        else:
            print("没有有效数据，跳过保存")

def main():
    in_files = [os.path.join(in_dir, file_name) for file_name in os.listdir(in_dir) if file_name.endswith("jsonl")]

    if not in_files:
        print(f"在 {in_dir} 中没有找到 .jsonl 文件")
        return
    print(f"找到 {len(in_files)} 个文件待处理")
    print("开始标准化数据...")
    process_files(in_files, out_dir)
    print("\n标准化完成！")
    print(f"标准化数据保存在: {out_dir}")


if __name__ == "__main__":
    main()
