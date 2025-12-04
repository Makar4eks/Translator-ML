import os, random, shutil, numpy as np, pandas as pd, torch
import sacrebleu as sb
from multiprocessing import freeze_support
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Helsinki-NLP/opus-mt-en-ru"
max_source_length = 256
max_target_length = 256
train_val_split = 0.05

per_device_train_batch_size = 4
per_device_eval_batch_size = 8
gradient_accumulation_steps = 4
num_train_epochs = 5
learning_rate = 3e-5
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
fp16 = torch.cuda.is_available()
output_dir = "./results_en_ru"

os.makedirs(output_dir, exist_ok=True)

df = pd.read_parquet("train-00000-of-00001.parquet")
df['en_text'] = df['translation'].apply(lambda x: x.get('en') if isinstance(x, dict) else None)
df['ru_text'] = df['translation'].apply(lambda x: x.get('ru') if isinstance(x, dict) else None)
df = df.dropna(subset=['en_text','ru_text']).reset_index(drop=True)

train_df, val_df = train_test_split(df[['en_text','ru_text']], test_size=train_val_split, random_state=SEED)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

def preprocess_batch(batch):
    sources = batch["en_text"]
    targets = batch["ru_text"]
    model_inputs = tokenizer(sources, max_length=max_source_length, truncation=True, padding=False)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True, padding=False)
    labels_input_ids = labels["input_ids"]
    labels_input_ids = [
        [(t if t != tokenizer.pad_token_id else -100) for t in seq]
        for seq in labels_input_ids
    ]
    model_inputs["labels"] = labels_input_ids
    return model_inputs

tokenized = dataset.map(preprocess_batch, batched=True, remove_columns=dataset["train"].column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds)
    if preds.dtype.kind == "f" or preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.asarray(labels)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    bleu = sb.corpus_bleu(decoded_preds, [decoded_labels])
    return {"bleu": float(bleu.score)}

def translate(texts, batch_size=16, num_beams=5, max_length=max_target_length, length_penalty=1.3):
    model.eval()
    preds = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_source_length).to(device)
        with torch.no_grad():
            generated = model.generate(**tokens, num_beams=num_beams, max_length=max_length, early_stopping=True)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        preds.extend([d.strip() for d in decoded])
    return preds

def main():
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        fp16=fp16,
        predict_with_generate=True,
        generation_num_beams=5,
        generation_max_length=max_target_length,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=200,
        dataloader_num_workers=0,
        label_smoothing_factor=0.0,
        remove_unused_columns=False,
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model("./en_ru_model")
    tokenizer.save_pretrained("./en_ru_model")

    val_texts = val_df['en_text'].tolist()
    val_refs = val_df['ru_text'].tolist()

    preds = translate(val_texts, batch_size=32)
    bleu = sb.corpus_bleu(preds, [val_refs]).score
    print("BLEU:", bleu)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

if __name__ == "__main__":
    freeze_support()
    main()
