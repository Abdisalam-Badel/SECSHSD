import pandas as pd
import re
import logging
from transformers import RobertaTokenizer, RobertaForSequenceClassification,Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


df = pd.read_csv('data/test_data.csv', encoding='Windows-1252')
df = df.dropna()


"""
Basic preprocessing
"""

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


df['Text'] = df['Text'].apply(preprocess_text)
train_data, test_data = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)
X_train = train_data['Text'].tolist()
y_train = train_data['Label'].tolist()
X_test = test_data['Text'].tolist()
y_test = test_data['Label'].tolist()



class hatesomDataset(Dataset):
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


        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


"""
for RoBERTa use: roberta-base, RobertaTokenizer, and RobertaForSequenceClassification
"""
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
train_dataset = hatesomDataset(X_train, y_train, tokenizer, max_length=128)
test_dataset = hatesomDataset(X_test, y_test, tokenizer, max_length=128)


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    lr_scheduler_type="linear",
    warmup_steps=500
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)


trainer.train()
Som_bert_predictions = trainer.predict(test_dataset)
Som_bert_predictions = torch.argmax(torch.tensor(Som_bert_predictions.predictions), axis=1)
logger.info("BERT Classification Report:")
logger.info(classification_report(y_test, Som_bert_predictions.numpy(), zero_division=1, digits=4))
