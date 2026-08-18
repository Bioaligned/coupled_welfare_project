"""QLoRA CPT for the coupled-welfare corpus (Arm M / Arm N).

Recipe follows the Andy D2 bioalignment run: r=32 alpha=64, 1 epoch, lr=1e-4,
max_grad_norm=0.3, 4-bit NF4, target all-linear. Continued-pretraining objective
(plain causal LM on the doc text). Pod-ready (offline volume cache + the
ALL_PARALLEL_STYLES patch).

  python train_cpt.py --base Qwen/Qwen2.5-7B-Instruct --train /root/cw_train/cw_train.jsonl \
      --out /root/cw_train/armM_qwen7b [--tokenizer_id ...]
"""
import os, argparse
os.environ.setdefault("HF_HUB_CACHE", "/workspace/models")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import transformers.modeling_utils as _mu
if getattr(_mu, "ALL_PARALLEL_STYLES", None) is None:
    _mu.ALL_PARALLEL_STYLES = ["colwise", "rowwise", "colwise_rep", "rowwise_rep",
        "local_colwise", "local_rowwise", "local", "gather",
        "local_packed_rowwise", "sequence_parallel", "replicate"]
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer_id", default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--max_steps", type=int, default=-1)  # for sanity checks
    a = ap.parse_args()

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer_id or a.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=bnb,
        device_map="cuda", trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha,
        target_modules="all-linear", lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=a.train, split="train")
    ds = ds.map(lambda ex: tok(ex["text"], truncation=True, max_length=a.maxlen),
                remove_columns=ds.column_names)
    args = TrainingArguments(output_dir=a.out, num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=2, gradient_accumulation_steps=8, learning_rate=a.lr,
        bf16=True, logging_steps=10, save_strategy="no", lr_scheduler_type="cosine",
        warmup_ratio=0.03, optim="paged_adamw_8bit", max_grad_norm=0.3, report_to="none")
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
    model.save_pretrained(a.out + "/adapter")
    tok.save_pretrained(a.out + "/adapter")
    print("SAVED_ADAPTER", a.out + "/adapter")


if __name__ == "__main__":
    main()
