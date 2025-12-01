# HW5/scripts/train_gpt2_qa.py
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai-community/gpt2"
CKPT_DIR = "HW5/checkpoints/gpt2_qa"

# Smaller + faster config
MAX_LEN = 128          # shorter sequences
BATCH_SIZE = 2         # smaller batch to reduce memory & compute
EPOCHS = 1             # single epoch
TRAIN_SUBSET = 2000    # use only first 2000 train examples
VAL_SUBSET = 500       # use only first 500 val examples
MAX_TRAIN_STEPS = 200  # stop each epoch after this many batches (None = no limit)


def prepare_data(tokenizer):
    """
    Load SQuAD and create a small subset suitable for a quick fine-tune run.
    We build texts as:
        "Question: ...\\nContext: ...\\nAnswer: ..."
    and use standard causal LM labels = input_ids.
    """
    dataset = load_dataset("rajpurkar/squad")

    # Take a subset to keep training fast
    train_raw = dataset["train"].select(range(min(TRAIN_SUBSET, len(dataset["train"]))))
    val_raw = dataset["validation"].select(range(min(VAL_SUBSET, len(dataset["validation"]))))

    def build_text(batch):
        questions = batch["question"]
        contexts = batch["context"]
        answers_list = batch["answers"]

        texts = []
        for q, c, ans in zip(questions, contexts, answers_list):
            # Use first answer if there are multiple
            if ans["text"]:
                a = ans["text"][0]
            else:
                a = ""
            text = f"Question: {q.strip()}\nContext: {c.strip()}\nAnswer: {a.strip()}"
            texts.append(text)

        enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    train_tok = train_raw.map(
        build_text,
        batched=True,
        remove_columns=train_raw.column_names,
    )
    val_tok = val_raw.map(
        build_text,
        batched=True,
        remove_columns=val_raw.column_names,
    )

    train_tok.set_format(type="torch")
    val_tok.set_format(type="torch")

    return train_tok, val_tok


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # GPT-2 has no pad token by default; use eos_token as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds = prepare_data(tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(1, EPOCHS + 1):
        # ---- training loop ----
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            if MAX_TRAIN_STEPS is not None and step > MAX_TRAIN_STEPS:
                break

            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 20 == 0:
                print(f"[Epoch {epoch} Step {step}] loss = {loss.item():.4f}")

        avg_train_loss = total_loss / max(1, min(len(train_loader), MAX_TRAIN_STEPS or len(train_loader)))
        print(f"[Epoch {epoch}] avg train loss: {avg_train_loss:.4f}")

        # ---- validation loop ----
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                val_steps += 1
        avg_val_loss = val_loss / max(1, val_steps)
        print(f"[Epoch {epoch}] avg val loss: {avg_val_loss:.4f}")

    # Save fine-tuned model and tokenizer
    model.save_pretrained(CKPT_DIR)
    tokenizer.save_pretrained(CKPT_DIR)
    print(f"Saved fine-tuned model to {CKPT_DIR}")


if __name__ == "__main__":
    main()
