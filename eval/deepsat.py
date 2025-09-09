import os
import json
import torch
import argparse
import time
from tqdm import tqdm
from typing import List, Dict, Optional
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAT_DATA_PATH = "sat.json"

DEEPSEEKMATH_PROMPT = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\\"', '"').replace("\\'", "'")
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\xa0', ' ').replace('\u200b', '').replace('\ufffd', '')
    text = ' '.join(text.split()).strip()
    return text
def load_sat_dataset(data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"SAT数据集文件不存在: {data_path}")
    logger.info(f"加载SAT数据集从: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cleaned_data = []
    for sample in data:
        if 'question' in sample and 'answer' in sample:
            cleaned_sample = {
                'question': clean_text(sample['question']),
                'answer': sample['answer'].upper().strip()
            }
            cleaned_data.append(cleaned_sample)
    if max_samples and len(cleaned_data) > max_samples:
        cleaned_data = cleaned_data[:max_samples]
        logger.info(f"限制评估样本数为: {max_samples}")
    logger.info(f"加载SAT数据集: {len(cleaned_data)} 条样本")
    return cleaned_data


def load_model_and_tokenizer(base_model_path: str, adapter_path: str = None):
    logger.info(f"加载基础模型: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left",
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=False,
    )
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    if adapter_path and os.path.exists(adapter_path):
        logger.info(f"加载LoRA适配器: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("LoRA适配器加载完成")
    model.eval()
    logger.info(f"模型加载完成，使用设备: {model.device}")
    return model, tokenizer


def batch_generate_responses(
        model,
        tokenizer,
        samples: List[Dict],
        batch_size: int = 4,
        max_new_tokens: int = 500
) -> List[Dict]:
    questions = [sample['question'] for sample in samples]
    ground_truths = [sample['answer'] for sample in samples]
    prompts = [DEEPSEEKMATH_PROMPT.format(question=q) for q in questions]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
        pad_to_multiple_of=8
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,  
        num_beams=1,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.0,
    )
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask', None),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
            )
        results = []
        for i, output in enumerate(outputs.sequences):
            generated_tokens = output[inputs['input_ids'].shape[1]:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            result_dict = {
                'sample_id': i,
                'question': questions[i],
                'ground_truth': ground_truths[i],
                'model_response': generated_text,
            }
            results.append(result_dict)
        return results
    except Exception as e:
        logger.error(f"批量生成答案时出错: {e}")
        return [{
            'sample_id': i,
            'question': questions[i],
            'ground_truth': ground_truths[i],
            'model_response': "",
            'error': str(e)
        } for i in range(len(questions))]


def main():
    parser = argparse.ArgumentParser(description="SAT数据集模型回答")
    parser.add_argument("--base_model_path", type=str, default="/mnt/d/deepseek7b",
                        help="基础模型路径")
    parser.add_argument("--adapter_path", type=str, default="/mnt/d/deepseekmath_mathcode_latex",
                        help="LoRA适配器路径")
    parser.add_argument("--output_dir", type=str, default="./sat_responses",
                        help="输出目录")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大样本数")
    parser.add_argument("--max_new_tokens", type=int, default=500,
                        help="最大生成token数")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批处理大小")
    args = parser.parse_args()
    start_time = time.time()
    
    os.makedirs(args.output_dir, exist_ok=True)
    try:
       
        logger.info("配置:")
        logger.info(f"基础模型: {args.base_model_path}")
        logger.info(f"适配器: {args.adapter_path}")
        logger.info(f"批处理大小: {args.batch_size}")
        logger.info(f"最大新token数: {args.max_new_tokens}")
        dataset = load_sat_dataset(SAT_DATA_PATH, args.max_samples)
        model, tokenizer = load_model_and_tokenizer(
            args.base_model_path,
            args.adapter_path
        )
        all_results = []
        logger.info("开始批量生成模型回答...")
        with tqdm(total=len(dataset), desc="处理样本", unit="sample") as pbar:
            for i in range(0, len(dataset), args.batch_size):
                batch = dataset[i:i + args.batch_size]
                batch_results = batch_generate_responses(
                    model, tokenizer, batch, args.batch_size, args.max_new_tokens
                )
                for j, result in enumerate(batch_results):
                    result['sample_id'] = i + j

                all_results.extend(batch_results)
                pbar.update(len(batch))
                pbar.set_postfix({
                    'batch': f"{i // args.batch_size + 1}/{(len(dataset) - 1) // args.batch_size + 1}",
                    'samples': f"{len(all_results)}/{len(dataset)}"
                })
        total_time = time.time() - start_time
        results_file = os.path.join(args.output_dir, "sat_model_responses.json")
        output_data = {
            'metadata': {
                'total_samples': len(all_results),
                'model_path': args.base_model_path,
                'adapter_path': args.adapter_path,
                'generation_config': {
                    'max_new_tokens': args.max_new_tokens,
                    'batch_size': args.batch_size,
                    'decoding_strategy': 'greedy'
                },
                'total_time': total_time,
                'processing_speed': f"{len(all_results) / (total_time / 3600):.1f} samples/hour"
            },
            'responses': all_results
        }
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"模型回答收集完成！")
        logger.info(f"总样本数: {len(all_results)}")
        logger.info(f"总耗时: {total_time:.1f}秒 ({total_time / 60:.1f}分钟)")
        logger.info(f"处理速度: {len(all_results) / (total_time / 3600):.1f} 样本/小时")
        logger.info(f"结果保存在: {results_file}")
    except KeyboardInterrupt:
        logger.info("处理被用户中断")
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
