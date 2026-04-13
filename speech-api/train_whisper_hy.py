#!/usr/bin/env python3
"""Fine-tune Whisper Large V3 Turbo on Armenian (Common Voice 20)."""

import os
import torch
from dataclasses import dataclass
from typing import Any

import evaluate
import numpy as np
from datasets import Audio, DatasetDict, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

MODEL_ID = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3-turbo")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/models/whisper-hy-finetuned")
DATASET = "Chillarmo/common_voice_20_armenian"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "5"))
LR = float(os.environ.get("LR", "1e-5"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "1000"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "1000"))
MAX_AUDIO_LENGTH = 30.0
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"


def load_data():
    print(f"Loading {DATASET}...")
    cv = DatasetDict()
    cv["train"] = load_dataset(DATASET, split="train")
    cv["test"] = load_dataset(DATASET, split="test")
    keep_cols = {"audio", "sentence"}
    remove = [c for c in cv["train"].column_names if c not in keep_cols]
    cv = cv.remove_columns(remove)
    cv = cv.cast_column("audio", Audio(sampling_rate=16000))

    if SMOKE_TEST:
        cv["train"] = cv["train"].select(range(min(20, len(cv["train"]))))
        cv["test"] = cv["test"].select(range(min(10, len(cv["test"]))))

    print(f"Train: {len(cv['train'])} samples, Test: {len(cv['test'])} samples")
    return cv


def make_preprocess_fn(processor):
    def prepare(batch):
        audio = batch["audio"]
        inputs = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        )
        batch["input_features"] = inputs.input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        return batch
    return prepare


def filter_long_audio(example):
    return example["input_length"] < MAX_AUDIO_LENGTH


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def compute_metrics_fn(processor, metric):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    return compute_metrics


def main():
    print(f"Model: {MODEL_ID}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Smoke test: {SMOKE_TEST}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    resume_ckpt = os.environ.get("RESUME_CKPT")
    if not resume_ckpt and os.path.isdir(OUTPUT_DIR):
        ckpts = sorted(
            [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        if ckpts:
            resume_ckpt = os.path.join(OUTPUT_DIR, ckpts[-1])

    load_from = resume_ckpt if resume_ckpt and os.path.isdir(resume_ckpt) else MODEL_ID
    print(f"Loading model from: {load_from}")

    processor = WhisperProcessor.from_pretrained(
        MODEL_ID, language="Armenian", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        load_from, torch_dtype=torch.float32
    )
    model.generation_config.language = "armenian"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    cv = load_data()
    prepare = make_preprocess_fn(processor)

    num_workers = 1

    print("Preprocessing training data...")
    cv["train"] = cv["train"].map(prepare, remove_columns=["audio", "sentence"], num_proc=num_workers)
    cv["train"] = cv["train"].filter(filter_long_audio)
    cv["train"] = cv["train"].remove_columns(["input_length"])

    print("Preprocessing test data...")
    cv["test"] = cv["test"].map(prepare, remove_columns=["audio", "sentence"], num_proc=num_workers)
    cv["test"] = cv["test"].filter(filter_long_audio)
    cv["test"] = cv["test"].remove_columns(["input_length"])

    print(f"After filtering: Train={len(cv['train'])}, Test={len(cv['test'])}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")

    eval_steps = 2 if SMOKE_TEST else EVAL_STEPS
    save_steps = 2 if SMOKE_TEST else SAVE_STEPS
    epochs = 1 if SMOKE_TEST else EPOCHS

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1 if SMOKE_TEST else GRAD_ACCUM,
        learning_rate=LR,
        warmup_steps=0 if SMOKE_TEST else 500,
        num_train_epochs=epochs,
        gradient_checkpointing=True,
        bf16=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=1 if SMOKE_TEST else 50,
        report_to=["tensorboard"],
        push_to_hub=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=cv["train"],
        eval_dataset=cv["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn(processor, metric),
        processing_class=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train()

    import shutil
    for d in os.listdir(OUTPUT_DIR):
        dp = os.path.join(OUTPUT_DIR, d)
        if d.startswith("checkpoint-") and os.path.isdir(dp):
            shutil.rmtree(dp)
            print(f"Removed {dp}")

    print("Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print("Running final evaluation...")
    results = trainer.evaluate()
    print(f"Final WER: {results['eval_wer']:.2f}%")
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
