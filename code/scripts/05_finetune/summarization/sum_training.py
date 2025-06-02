# Import core libraries
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset, load_from_disk, Dataset
import evaluate

# Utilities
import pandas as pd
import matplotlib.pyplot as plt
import jsonlines
import numpy as np
from tqdm import tqdm
import os

# Environment checks
print(f" Torch version: {torch.__version__}")

import transformers
print(f" Transformers version: {transformers.__version__}")

# Check if CUDA is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Using device: {device}")

# =========================
#  Model Selection & Checkpoint Configuration
# =========================

# Choose model type: either 'biot5' or 'biobart'
model_choice = "biot5"  # Change this to "biobart" if needed

# Define the corresponding HuggingFace checkpoint
if model_choice == "biot5":
    model_checkpoint = "QizhiPei/biot5-base"
elif model_choice == "biobart":
    model_checkpoint = "GanjinZero/biobart-base"
elif model_choice == "biov2bart":
    model_checkpoint = "GanjinZero/biobart-v2-base"
else:
    raise ValueError(" Invalid model choice! Use 'biot5', 'biobart' or biov2bart.")

print(f" Selected model checkpoint: {model_checkpoint}")

# =========================
# Load Tokenizer & Model
# =========================

# Load the tokenizer corresponding to the selected model
# Responsible for converting text to input token IDs (and back)
#tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Load the pretrained Seq2Seq model (e.g., T5 or BART)
# This model is designed for generation tasks like summarization
#model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)

print(f" Model and tokenizer loaded to device: {device}")

# =========================
#  Load Tokenized Datasets
# =========================
# If you already have tokenized datasets saved from a previous run,
# you can skip the tokenization step and just load them here.

# Define paths to the saved tokenized datasets
if model_choice == "biot5":
    train_path = "../../../../data/tokenized/biot5_sum/train"
    validation_path = "../../../../data/tokenized/biot5_sum/val"
elif model_choice == "biobart":
    train_path = "../../../../data/tokenized/biobart_sum/train"
    validation_path = "../../../../data/tokenized/biobart_sum/val"
elif model_choice == "biov2bart":
    train_path = "../../../../data/tokenized/biov2bart_sum/train"
    validation_path = "../../../../data/tokenized/biov2bart_sum/val"


# Load the datasets
train_dataset = load_from_disk(train_path)
validation_dataset = load_from_disk(validation_path)

print(f" Loaded tokenized datasets: {len(train_dataset)} train / {len(validation_dataset)} validation")

if model_choice == "biot5":
    out_dir="../../../../models/biot5_sum"
elif model_choice == "biobart":
    out_dir="../../../../models/biobart_sum"
elif model_choice == "biov2bart":
    out_dir="../../../../models/biov2bart_sum"
    # Where to save model checkpoints and logs

# Define the training arguments
training_args = Seq2SeqTrainingArguments(

    output_dir= out_dir,
    # Where to save model checkpoints and logs

    eval_strategy="epoch",       # Evaluate after each epoch
    save_strategy="epoch",             # Save a checkpoint after each epoch

    learning_rate=1e-5,                # Small learning rate for fine-tuning
    per_device_train_batch_size=16,    # Training batch size per device
    per_device_eval_batch_size=16,     # Evaluation batch size per device
    gradient_accumulation_steps=2,     # Gradient accumulation for effective batch size

    num_train_epochs=25,              # Number of total training epochs
    weight_decay=0.01,                 # Regularization to prevent overfitting

    save_total_limit=3,                # Keep only the 2 most recent checkpoints

    predict_with_generate=True,        # Use generation during validation for summarization
    generation_max_length=256,         # Max length for generated sequences

    logging_dir="../../../../outputs/05_fine_tuning/summarization/biot5_sum",
    logging_strategy="epoch",          # Log once per epoch

    load_best_model_at_end=True,       # Automatically load best checkpoint (lowest eval loss)
    metric_for_best_model="eval_rougeL",
    greater_is_better=True,

    report_to="none",                  # No external logging (e.g., wandb)
    fp16=True                          # Enable mixed-precision training if using CUDA
)

print(" Training arguments configured.")


rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")
# bertscore = evaluate.load("bertscore")  # Uncomment if used

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)

    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    bleu = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
    )
    result["bleu"] = round(bleu["bleu"], 4)

    # Optional BERTScore (comment out if compute-constrained)
    # bs = bertscore.compute(
    #     predictions=decoded_preds,
    #     references=decoded_labels,
    #     lang="en"
    # )
    # result["bertscore_f1"] = round(np.mean(bs["f1"]), 4)

    return {key: round(value, 4) for key, value in result.items()}

# ============================
# Trainer Initialization
# ============================

trainer = Seq2SeqTrainer(
    model=model,  # The Seq2Seq model to be fine-tuned (e.g., T5, BART)
    args=training_args,  # Training hyperparameters and behavior (batch size, epochs, save strategy etc.)

    train_dataset=train_dataset,         # Tokenized training data
    eval_dataset=validation_dataset,     # Tokenized validation data

    tokenizer=tokenizer,                 # Tokenizer used to preprocess data and decode predictions

    compute_metrics=compute_metrics,     # Custom metric function to evaluate the model (e.g., ROUGE)

    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3)
        # Early stopping to prevent overfitting.
        # Training stops if validation loss does not improve for 3 consecutive evaluations.
    ]
)

# ===============================
# Start Fine-Tuning
# ===============================
train_result = trainer.train(resume_from_checkpoint=False)  # Starts the training loop using the trainer configuration

# ===============================
# Save Final Model & Tokenizer
# ===============================
# Save both the trained model and tokenizer for later use
if model_choice == "biot5":
    trainer.save_model("../../../../models/biot5_sum_final")
    tokenizer.save_pretrained("../../../../models/biot5_sum_final")
elif model_choice == "biobart":
    trainer.save_model("../../../../models/biobart_sum_final")
    tokenizer.save_pretrained("../../../../models/biobart_sum_final")
elif model_choice == "biov2bart":
    trainer.save_model("../../../../models/biov2bart_sum_final")
    trainer.save_pretrained("../../../../models/biov2bart_sum_final")

print(" Fine-tuning completed and the final model has been saved.")

# 📂 Paths for Saving Outputs
logs_dir = f"../../../../data/plots/summarization/{model_choice}"
images_dir = os.path.join(logs_dir, "images")
os.makedirs(images_dir, exist_ok=True)

CSV_LOG_PATH = os.path.join(logs_dir, "metrics_logs.csv")
PLOT_LOSS_PATH = os.path.join(images_dir, "loss_plot.png")
PLOT_ROUGE_PATH = os.path.join(images_dir, "rouge_plot.png")
PLOT_BLEU_PATH = os.path.join(images_dir, "bleu_plot.png")  # 🔥 New

# 📊 Convert Trainer Logs to DataFrame
log_df = pd.DataFrame(trainer.state.log_history)

# ✅ Keep only rows that include 'epoch' (skip early stopping, lr logs etc.)
log_df = log_df[log_df["epoch"].notnull()].copy()

# 🧮 Group by epoch and average values (to plot 1 point per epoch)
grouped_df = log_df.groupby("epoch").mean(numeric_only=True).reset_index()

# 💾 Save full raw log history
log_df.to_csv(CSV_LOG_PATH, index=False)

# 📈 Plot: Training & Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(grouped_df["epoch"], grouped_df["loss"], label="Train Loss", marker="o")
plt.plot(grouped_df["epoch"], grouped_df["eval_loss"], label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.savefig(PLOT_LOSS_PATH)
plt.show()

# 📈 Plot: ROUGE Scores
plt.figure(figsize=(10, 6))
plt.plot(grouped_df["epoch"], grouped_df["eval_rouge1"], label="ROUGE-1", marker="o")
plt.plot(grouped_df["epoch"], grouped_df["eval_rouge2"], label="ROUGE-2", marker="o")
plt.plot(grouped_df["epoch"], grouped_df["eval_rougeL"], label="ROUGE-L", marker="o")
plt.xlabel("Epoch")
plt.ylabel("ROUGE Score")
plt.title("ROUGE Scores over Epochs")
plt.legend()
plt.grid()
plt.savefig(PLOT_ROUGE_PATH)
plt.show()

# 📈 Plot: BLEU Score
if "eval_bleu" in grouped_df.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(grouped_df["epoch"], grouped_df["eval_bleu"], label="BLEU", marker="o", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU Score")
    plt.title("BLEU Score over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(PLOT_BLEU_PATH)
    plt.show()
else:
    print("⚠️ 'eval_bleu' not found in logs. Skipping BLEU plot.")

# 📝 Save file paths
print(f"📊 Metrics CSV saved at: {CSV_LOG_PATH}")
print(f"🖼️ Loss plot saved at: {PLOT_LOSS_PATH}")
print(f"🖼️ ROUGE plot saved at: {PLOT_ROUGE_PATH}")
print(f"🖼️ BLEU plot saved at: {PLOT_BLEU_PATH}")
