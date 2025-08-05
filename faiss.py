"""Faiss Retrival"""
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import difflib
import os
import re
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


train_data = pd.read_csv('data/train.csv', encoding='Windows-1252')
test_data = pd.read_csv('data/test_data.csv', encoding='Windows-1252')


train_data = train_data.dropna()
test_data = test_data.dropna()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text
train_data['Text'] = train_data['Text'].apply(preprocess_text)
test_data['Text'] = test_data['Text'].apply(preprocess_text)



model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
logger.info("Encoding training data...")
embeddings_train = model.encode(train_data['Text'].tolist(), normalize_embeddings=True)
logger.info("Encoding test data...")
embeddings_test = model.encode(test_data['Text'].tolist(), normalize_embeddings=True)
data = embeddings_train.shape[1]
my_index = faiss.IndexFlatIP(data)
my_index.add(np.ascontiguousarray(embeddings_train.astype('float32')))

"""When using the Openai api_key, uncomment this one"""

# deepseek_client = OpenAI(
#     api_key=" "
# )
""" When using the deepseek api_key, use this one """

deepseek_client = OpenAI(api_key=" ")
k = 3


predictions = []

logger.info("Processing test data...")
for text, embedding in zip(test_data['Text'], embeddings_test):
    Document, Index = my_index.search(np.ascontiguousarray([embedding]).astype('float32'), k)
    with open('data/hate', 'r', encoding='utf-8') as f2:
        hate_words = [line.strip() for line in f2.readlines()]
    retrieved_texts = [train_data['Text'].iloc[idx] for idx in Index[0]]
    prompt = "Classify the input text as hate or normal, example the sentence 'she is dhilo baahan' is Classified as 'hate'. If classifying the Input Text is challenging, use Retrieved Examples and Example Words as additional information to improve accuracy. Please strictly return only 1 for hate and 0 for normal.\n\n"
    word = max(hate_words, key=lambda w: difflib.SequenceMatcher(None, w, text).ratio())
    for i, retrieved_text in enumerate(retrieved_texts, 1):
        prompt += f"Retrieved Examples{i}: Text: \"{retrieved_text}\"\n"
        prompt += f"Example Words: {word}"
    prompt += f"\nInput Text: \"{text}\"\nClass:"
    response = deepseek_client.chat.completions.create(
        model='deepseek-chat',
        messages=[
            {'role': 'system', 'content': 'You are a Somali-English code-mixed hate speech detection assistant.'},
            {'role': 'user', 'content': prompt}
        ],
    )

    #%% These are the models we used:
    # 1. gpt-4o-mini-2024-07-18    # 2. gpt-4o-2024-08-06       # 3. gpt-3.5-turbo-0125   #deepseek-chat

    prediction_text = response.choices[0].message.content.strip().lower()
    if 'hate' in prediction_text:
        predictions.append('1')
    else:
        predictions.append('0')
results_df = pd.DataFrame({
    'Text': test_data['Text'],
    'predicted_Class': predictions
})

#%%  We save the results
os.makedirs('faiss.csv', exist_ok=True)
output_path = 'my_result/faiss/deepseek-chat.csv'
results_df.to_csv(output_path, index=False)
logger.info(f"Predictions saved to {output_path}")
