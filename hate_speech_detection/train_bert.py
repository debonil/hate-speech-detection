import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def train():
    # Load the dataset
    dataset_path = "data\SBIC.v2.agg.tst.csv"
    df = pd.read_csv(dataset_path)

    # Preprocess the dataset: assuming 'hasBiasedImplication' column has values 0 (no bias) and 1 (social bias)
    df = df[['post', 'hasBiasedImplication']]
    df.columns = ['text', 'label']

    # Split the dataset into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create PyTorch Datasets
    train_dataset = CustomDataset(
        train_df["text"].to_numpy(),
        train_df["label"].to_numpy(),
        tokenizer,
        max_length=128,
    )
    val_dataset = CustomDataset(
        val_df["text"].to_numpy(),
        val_df["label"].to_numpy(),
        tokenizer,
        max_length=128,
    )

    # Initialize the model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="trained_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Fine-tune the model
    trainer.train()
    trainer.save_model("trained_model")

if __name__ == "__main__":
    train()
