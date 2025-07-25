import json
import requests
from tqdm import tqdm
import os
class AutoMathEvaluator:
    def __init__(self, api_key):
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"}

        self.system_prompt = """你是一个数学答案验证专家，请按以下规则处理：
1. 如果回答中已明确给出具体数值，直接验证该数值对应的选项是不是正确答案
2. 否则提取并执行所有Mathematica代码计算正确答案
3. 最终只需返回是否匹配正确选项（True/False）"""

        self.user_prompt = """请验证以下回答是否正确：
问题：{question}
选项：{choices}
正确答案：{ground_truth}

回答内容：
{model_response}

请按以下格式响应：
```json
{{
  "match": "True/False",
  "method": "direct/extracted_code"
}}
```"""

    def evaluate(self, sample):
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.user_prompt.format(
                            question=sample["question"],
                            choices=sample["question"].split("Answer Choices:")[-1],
                            ground_truth=sample["ground_truth"],
                            model_response=sample["model_response"]
                        )}
                    ],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "max_tokens": 100
                },
                timeout=20
            ).json()

            result = json.loads(response["choices"][0]["message"]["content"])
            return result["match"] == "True"
        except Exception as e:
            print(f"评估失败（样本ID:{sample.get('sample_id')}）: {str(e)}")
            return False


def main():
    # 配置
    CONFIG = {
        "api_key": "",
        "data_path": "",
        "result_file": "auto_evaluation_results.json"
    }

    if not os.path.exists(CONFIG["data_path"]):
        print(f"文件不存在: {CONFIG['data_path']}")
        return

    with open(CONFIG["data_path"], 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "responses" in data:
        samples = data["responses"]
    else:
        samples = data
    evaluator = AutoMathEvaluator(CONFIG["api_key"])
    results = []

    for sample in tqdm(samples, desc="自动评估"):
        if isinstance(sample, dict):
            results.append({
                "sample_id": sample.get("sample_id"),
                "is_correct": evaluator.evaluate(sample),
                "ground_truth": sample.get("ground_truth")
            })
        else:
            print(f"警告：跳过非字典样本 {sample}")
    correct = sum(1 for r in results if r["is_correct"])
    total = len(results)

    print(f"\n评估完成：正确 {correct}/{total} (准确率 {correct / total * 100:.1f}%)")
    with open(os.path.join(os.path.dirname(CONFIG["data_path"]), CONFIG["result_file"]), 'w') as f:
        json.dump({
            "statistics": {"correct": correct, "total": total, "accuracy": correct / total * 100},
            "details": results
        }, f, indent=2)


if __name__ == "__main__":
    main()