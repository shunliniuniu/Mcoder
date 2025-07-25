import json
import os
data_folder = ""
code_folder = ""
output_folder = ""
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"解析JSON行时出错: {e} - 行内容: {line}")
    return data

def save_jsonl(data, out_file):

    with open(out_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def load_problems(data_folder):
    data_file = os.path.join(data_folder, "test.jsonl")
    try:
        problems = load_jsonl(data_file)
        problem_dict = {
            (item["idx"], item.get("category", "")): item["problem"]
            for item in problems
        }
        print(f"从test.jsonl加载了{len(problem_dict)}个问题")
        return problem_dict
    except Exception as e:
        print(f"加载问题数据时出错: {e}")
        exit(1)


def load_code_data(code_folder):
    code_files = [f for f in os.listdir(code_folder) if f.endswith(".jsonl")]
    if not code_files:
        print("在代码文件夹中未找到.jsonl文件")
        exit(1)
    code_file = os.path.join(code_folder, code_files[0])
    try:
        code_data = load_jsonl(code_file)
        print(f"从{os.path.basename(code_file)}加载了{len(code_data)}个代码数据项")
        return code_data
    except Exception as e:
        print(f"加载代码数据时出错: {e}")
        exit(1)

def merge_data(problem_dict, code_data):
    merged_data = []
    matched_count = 0
    unmatched_count = 0
    category_mismatch_count = 0
    for code_item in code_data:
        try:
            idx = code_item["idx"]
            code_category = code_item.get("category", "")
            lookup_key = (idx, code_category)
            if lookup_key in problem_dict:
                merged_item = {
                    "idx": idx,
                    "category": code_category,
                    "problem": problem_dict[lookup_key],
                    "answer": code_item["text"]
                }
                merged_data.append(merged_item)
                matched_count += 1
            else:
                if any(key[0] == idx for key in problem_dict.keys()):
                    category_mismatch_count += 1
                    problem_category = [key[1] for key in problem_dict.keys() if key[0] == idx][0]
                    print(f"category不匹配: idx={idx}, 代码category='{code_category}', 问题category='{problem_category}'")
                else:
                    print(f"未找到匹配的问题: {idx}")
                    unmatched_count += 1
        except KeyError as e:
            print(f"数据项缺少必要字段: {e} - 跳过此项")
    return merged_data, matched_count, unmatched_count, category_mismatch_count


def main():
    problem_dict = load_problems(data_folder)
    code_data = load_code_data(code_folder)
    merged_data, matched_count, unmatched_count, category_mismatch_count = merge_data(problem_dict, code_data)
    output_file = os.path.join(output_folder, "merged_data.jsonl")
    save_jsonl(merged_data, output_file)
    print("\n合并完成，结果汇总:")
    print(f"匹配的数据项数量: {matched_count}")
    print(f"未匹配的数据项数量: {unmatched_count}")
    print(f"category不匹配的数量: {category_mismatch_count}")
    print(f"总合并数据项数量: {len(merged_data)}")
    print(f"合并后的数据已保存到: {output_file}")
    # 打印前3个合并后的数据项作为示例
    print("\n合并数据示例(前3项):")
    for i, item in enumerate(merged_data[:3], 1):
        print(f"示例 {i}:")
        print(f"idx: {item['idx']}")
        print(f"category: {item.get('category', 'N/A')}")
        print(f"problem: {item['problem'][:50]}...")
        print(f"nswer: {item['answer'][:50]}..." if isinstance(item['answer'],
                                                                  str) else f"  answer长度: {len(item['answer'])}")


if __name__ == "__main__":
    main()
