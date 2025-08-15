import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.6"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

from collections import defaultdict
import copy
import json
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import gc

import torch
import torch.distributed as dist
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    default_data_collator
)
from datasets import load_dataset, Dataset
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

DEEPSEEKMATH_ASSISTANT_PROMPT_DICT = {
    "prompt_with_instruction": (
        "User: {question}\n"
        "Please reason step by step using LaTeX for mathematical expressions "
        "and provide Mathematica code for computational steps where appropriate. "
        "Put your final answer within \\boxed{{}}.\n\nA: "
    )
   
}


def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        print(f"Initializing distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        print("No distributed training environment detected, running on single GPU")
        return 0, 1, 0


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=r"/mnt/d/deepseek7b")
    trust_remote_code: Optional[bool] = field(default=True)
    use_flash_attention: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    eval_dataset_size: int = field(default=500)
    max_train_samples: Optional[int] = field(default=4900)
    max_eval_samples: Optional[int] = field(default=500)
    source_max_len: int = field(default=660)
    target_max_len: int = field(default=448)
    dataset: str = field(default='merged_data.jsonl')
    dataset_format: Optional[str] = field(default='question-answer')
    use_math_assistant_prompt: bool = field(default=True)
    use_chinese_prompt: bool = field(default=False)
    validation_split_ratio: float = field(default=0.1)


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    train_on_source: Optional[bool] = field(default=False)
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    fp16_opt_level: str = field(default=None)
    fp16_full_eval: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    gradient_clip_mode: str = field(default="norm")
    gradient_clip_value: float = field(default=1.0)
    bits: int = field(default=4)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    lora_r: int = field(default=24)
    lora_alpha: float = field(default=48)
    lora_dropout: float = field(default=0.1)
    max_memory_MB: int = field(default=22000)
    output_dir: str = field(default='./deepseekmath_assistant_output')
    optim: str = field(default='paged_adamw_32bit')
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=4)
    max_steps: int = field(default=700)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=2e-4)
    remove_unused_columns: bool = field(default=False)
    max_grad_norm: float = field(default=0.5)
    gradient_checkpointing: bool = field(default=True)
    do_train: bool = field(default=True)
    lr_scheduler_type: str = field(default='cosine')
    warmup_steps: int = field(default=100)
    logging_steps: int = field(default=10)
    group_by_length: bool = field(default=True)
    save_strategy: str = field(default='steps')
    save_steps: int = field(default=150)
    save_total_limit: int = field(default=2)
    eval_strategy: str = field(default='steps')
    eval_steps: int = field(default=150)
    do_eval: bool = field(default=True)
    evaluation_strategy: str = field(default='steps')
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default='eval_loss')
    greater_is_better: bool = field(default=False)
    ddp_find_unused_parameters: bool = field(default=False)
    ddp_timeout: int = field(default=3600)
    dataloader_num_workers: int = field(default=2)
    dataloader_drop_last: bool = field(default=True)
    report_to: str = field(default='none')
    full_finetune: bool = field(default=False)
    adam8bit: bool = field(default=False)
    seed: int = field(default=42)


@dataclass
class GenerationArguments:
    max_new_tokens: Optional[int] = field(default=384)
    min_new_tokens: Optional[int] = field(default=None)
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    use_cache: Optional[bool] = field(default=True)
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    repetition_penalty: Optional[float] = field(default=1.0)


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    print(f"Found linear layers: {lora_module_names}")
    return list(lora_module_names)


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.is_local_process_zero:
            print('Saving PEFT checkpoint...')
            if state.best_model_checkpoint is not None:
                checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
            else:
                checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)
            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        if state.is_local_process_zero:
            touch(join(args.output_dir, 'completed'))
            self.save_model(args, state, kwargs)


def get_accelerate_model(args, checkpoint_dir):
    max_memory_MB = args.max_memory_MB
    n_gpus = torch.cuda.device_count()
    max_memory = f'{max_memory_MB}MB'

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory}
        print(f"Local rank: {local_rank}, Device map: {device_map}")
    else:
        max_memory = {i: max_memory for i in range(n_gpus)}
        device_map = "auto"

    print(f'Loading DeepSeekMath base model {args.model_name_or_path}...')
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float32
    print(f"Using compute dtype: {compute_dtype}")

    clear_memory()

    model_kwargs = {
        'cache_dir': args.cache_dir,
        'device_map': device_map,
        'max_memory': max_memory,
        'quantization_config': BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        'torch_dtype': compute_dtype,
        'trust_remote_code': args.trust_remote_code,
        'use_cache': False,
        'attn_implementation': "eager",
    }

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **model_kwargs
        )
        print(f" DeepSeekMath model loaded successfully with {compute_dtype} precision")
    except Exception as e:
        print(f"Failed to load DeepSeekMath model: {e}")
        raise e

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.torch_dtype = compute_dtype
    model.config.use_cache = False

    if hasattr(model.config, 'pretraining_tp'):
        model.config.pretraining_tp = 1
    if not args.full_finetune:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not args.full_finetune:
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
        else:
            print(f'Adding LoRA modules for DeepSeekMath...')
            modules = find_all_linear_names(args, model)
            print(f"Target modules for LoRA: {modules}")
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if hasattr(module, 'to'):
                module = module.to(compute_dtype)
        if 'norm' in name:
            if hasattr(module, 'to'):
                module = module.to(compute_dtype)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight') and hasattr(module, 'to'):
                module = module.to(compute_dtype)

    clear_memory()
    return model


def print_trainable_parameters(args, model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    if args.bits == 4:
        trainable_params /= 2

    print(
        f"Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {100 * trainable_params / all_param:.2f}%")


def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer: transformers.PreTrainedTokenizer,
                                         model: transformers.PreTrainedModel):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def extract_deepseekmath_dataset(example, use_math_assistant_prompt=True, use_chinese_prompt=False):
    if use_chinese_prompt:
        prompt_format = DEEPSEEKMATH_ASSISTANT_PROMPT_DICT["prompt_chinese"]
    else:
        prompt_format = DEEPSEEKMATH_ASSISTANT_PROMPT_DICT["prompt_with_instruction"]

    formatted_input = prompt_format.format(question=example['question'])
    return {
        'input': formatted_input,
        'output': example['answer'] + ""
    }


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = [example['input'] for example in instances]
        targets = [example['output'] for example in instances]

        tokenized_sources_with_prompt = self.tokenizer(
            sources, max_length=self.source_max_len, truncation=True, add_special_tokens=True, padding=False,
        )
        tokenized_targets = self.tokenizer(
            targets, max_length=self.target_max_len, truncation=True, add_special_tokens=False, padding=False,
        )

        input_ids = []
        labels = []
        max_length = self.source_max_len + self.target_max_len

        for tokenized_source, tokenized_target in zip(tokenized_sources_with_prompt['input_ids'],
                                                      tokenized_targets['input_ids']):
            if not self.predict_with_generate:
                combined_ids = tokenized_source + tokenized_target
                if len(combined_ids) > max_length:
                    combined_ids = combined_ids[:max_length]
                    tokenized_target = combined_ids[len(tokenized_source):]

                input_ids.append(torch.tensor(combined_ids))
               
                labels.append(torch.tensor(copy.deepcopy(combined_ids)))
            else:
                input_ids.append(torch.tensor(tokenized_source))

        if input_ids:
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = pad_sequence(labels, batch_first=True,
                                  padding_value=IGNORE_INDEX) if not self.predict_with_generate else None

        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict


def local_dataset(dataset_name):
    if not os.path.exists(dataset_name):
        raise FileNotFoundError(f"Dataset file not found: {dataset_name}")

    print(f"Loading dataset from {dataset_name}")

    if dataset_name.endswith('.json'):
        with open(dataset_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        full_dataset = Dataset.from_list(data)
    elif dataset_name.endswith('.jsonl'):
        data = []
        with open(dataset_name, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        full_dataset = Dataset.from_list(data)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    print(f"Dataset loaded: {len(full_dataset)} examples")
    return {"train": full_dataset}


def split_dataset(dataset, eval_size=500, seed=42):
    dataset = dataset.shuffle(seed=seed)

    total_size = len(dataset)
    if eval_size >= total_size:
        raise ValueError(f"Validation size ({eval_size}) >= total dataset size ({total_size})")

    train_size = total_size - eval_size
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, total_size))

    print(f"Dataset split: Train={len(train_dataset)}, Validation={len(eval_dataset)}")
    return train_dataset, eval_dataset


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    dataset_dict = local_dataset(args.dataset)
    dataset = dataset_dict['train']
    sample_data = dataset[0]
    print(f"Sample data keys: {sample_data.keys()}")
    if 'question' in sample_data and 'answer' in sample_data:
        print("Converting question/answer format to DeepSeekMath assistant format with LaTeX/Mathematica support...")
        dataset = dataset.map(
            lambda x: extract_deepseekmath_dataset(x, args.use_math_assistant_prompt, args.use_chinese_prompt),
            num_proc=4,
            remove_columns=['question', 'answer'] + (
                [col for col in dataset.column_names if col not in ['question', 'answer']])
        )
    else:
        raise ValueError(f"Unsupported data format. Expected 'question' and 'answer' fields, got: {sample_data.keys()}")

    print(f"Dataset columns: {dataset.column_names}")
    print(f"Sample input: {dataset[0]['input'][:300]}...")
    print(f"Sample output: {dataset[0]['output'][:300]}...")
    train_dataset, eval_dataset = split_dataset(dataset, eval_size=args.eval_dataset_size, seed=args.seed)
    if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
        print(f"Limited training dataset to {args.max_train_samples} samples")

    if args.group_by_length:
        train_dataset = train_dataset.map(
            lambda x: {'length': len(x['input']) + len(x['output'])},
            num_proc=4
        )

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )

    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=None,
        data_collator=data_collator
    )


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed:
            return None, True
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0:
            return None, is_completed
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed
    return None, False


def train():
    rank, world_size, local_rank = setup_distributed()
    hfparser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GenerationArguments))
    model_args, data_args, training_args, generation_args, extra_args = hfparser.parse_args_into_dataclasses(
        return_remaining_strings=True)

    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))
    set_seed(args.seed)
    clear_memory()
    model = get_accelerate_model(args, checkpoint_dir=None)

    print_trainable_parameters(args, model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    clear_memory()
    data_module = make_data_module(tokenizer=tokenizer, args=args)

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
    )

    trainer.add_callback(SavePeftModelCallback)

    if args.do_train:
        clear_memory()

        # 检查是否有checkpoint可以恢复
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif training_args.output_dir is not None:
            last_checkpoint, is_completed = get_last_checkpoint(training_args.output_dir)
            if is_completed:
                print("Training already completed!")
                return
            checkpoint = last_checkpoint

        try:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)

            metrics = train_result.metrics
            trainer.save_state()
            trainer.save_model()
            summary = {
                "model": args.model_name_or_path,
                "dataset_size": len(data_module["train_dataset"]) + len(data_module["eval_dataset"]),
                "train_samples": len(data_module["train_dataset"]),
                "eval_samples": len(data_module["eval_dataset"]),
                "final_metrics": metrics,
                "lora_config": {
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout
                }
            }

            with open(join(args.output_dir, 'training_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise e


if __name__ == "__main__":
    train()
