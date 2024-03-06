from datasets import load_dataset, DatasetDict
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments, PreTrainedTokenizerFast, RobertaForMaskedLM, RobertaConfig, MistralForCausalLM, MistralConfig, LlamaForCausalLM, LlamaConfig, GPT2LMHeadModel, GPT2Config
import transformers
from pathlib import Path
#from tokenizers.processors import BertProcessing
import tokenizers
import os
from datetime import datetime

start_time = datetime.utcnow()
start_time_str = start_time.strftime('%Y-%m-%d-%H-%M-%S')

parser = ArgumentParser(prog="Trainer")
parser.add_argument("-t", "--token-dataset", type=str, default="dataset", help="Dataset dir to train the tokenizer")
parser.add_argument("-d", "--dataset", type=str, default="dataset", help="Dataset dir to train the model")
parser.add_argument("-v", "--valid-dataset", type=str, default="dataset", help="Dataset dir to perform validation")
parser.add_argument("-o", "--output-dir", type=str, default=f"result-{start_time_str}", help="Where to write the resulting models to")
parser.add_argument("-n", "--name", type=str, default="custom-model", help="Name of the model")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume from")
parser.add_argument("--steps", type=int, default=500, help="Number of steps")
parser.add_argument("--train-pct", type=int, default=100, help="How much of the training data to train in percent")
parser.add_argument("--validate-every", type=int, default=10, help="How many steps to train before validating")
parser.add_argument("--save-every", type=int, default=10, help="How many steps to train before saving")
args = parser.parse_args()

if not (args.checkpoint and args.output_dir):

    print("Preparing to train tokenizer")

    paths = [str(path) for path in Path(args.token_dataset).glob("*.txt")]
    tokenizer = tokenizers.ByteLevelBPETokenizer()

    print("Training tokenizer")

    tokenizer.train(
        files=paths,
        vocab_size=64_000,
        min_frequency=2,
        special_tokens=[
            "<s>", "<pad>", "</s>", "<unk>", "<mask>"
        ]
    )

    print("Saving tokenizer")
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_model(args.output_dir, args.name)
    tokenizer.save(f"{args.output_dir}/tokenizer.json")

context_length = 256

print("Loading newly trained tokenizer")
#tokenizer = tokenizers.implementations.ByteLevelBPETokenizer(
#    f"{args.output_dir}/{args.name}-vocab.json",
#    f"{args.output_dir}/{args.name}-merges.txt",
#)
tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{args.output_dir}/tokenizer.json")
#tokenizer.post_procesor = BertProcessing(
#    ("</s>", tokenizer.token_to_id("</s>")),
#    ("<s>", tokenizer.token_to_id("<s>")),
#)
#tokenizer.enable_truncation(max_length=context_length)
#tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '<pad>'})

print("Preparing datasets for model training")
train_dataset = load_dataset("text", data_dir="dataset", sample_by="line", split=f"train[:{args.train_pct}]")
validation_dataset = load_dataset("text", data_dir="dataset", sample_by="line", split="train[:10%]")

# use .shuffle().select(range(500)) to limit

print("Processing datasets")
def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        padding='max_length',
        max_length=context_length,
        return_overflowing_tokens=False,
        return_length=True,
    )
    return outputs


#tokenized_datasets = raw_datasets.map(
#    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
#)
tokenized_train_ds = train_dataset.map(tokenize, batched=True)
tokenized_validation_ds = validation_dataset.map(tokenize, batched=True)

print("Preparing to train the model *drumroll*")

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

model = GPT2LMHeadModel(
    config=GPT2Config()
)

transformers.logging.set_verbosity_info()

train_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    save_total_limit=2,
    save_strategy='steps',
    max_steps=args.steps,
    eval_steps=args.validate_every,
#    logging_steps=500,
    gradient_accumulation_steps=8,
#    num_train_epochs=1,
    weight_decay=0.1,
    warmup_ratio=0.2,
    lr_scheduler_type="linear",
    learning_rate=5e-4,
    save_steps=args.save_every,
    bf16=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_validation_ds,
    args=train_args,
    data_collator=data_collator,
)

print("Starting trainer")
trainer.train(
    resume_from_checkpoint=f"{args.output_dir}/{args.checkpoint}" if args.checkpoint else None
)

print("Saving model")
trainer.save_model(output_dir=f"{args.output_dir}/model")


