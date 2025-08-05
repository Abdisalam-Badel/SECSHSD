import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import csv
import logging
import os
import re
from sklearn.metrics.pairwise import cosine_similarity
import difflib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


som_train = pd.read_csv('data/train.csv', encoding='Windows-1252')
som_test = pd.read_csv('data/test_data.csv', encoding='Windows-1252')


train_data = som_train.dropna()
test_data = som_test.dropna()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

train_data['Text'] = train_data['Text'].apply(preprocess_text)
test_data['Text'] = test_data['Text'].apply(preprocess_text)
train_corpus = [text.split() for text in train_data['Text']]
bm25 = BM25Okapi(train_corpus)

""" 
Our embedding model 
"""
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL)

""" 
Save chunks and embeddings for future use.
"""
som_CHUNKS = "my_result/hybrid/hate_ch.csv"
som_EMBEDDINGS = "my_result/hybrid/hate_emb.csv"

def save_chunks_embeddings(chunks, embeddings, chunks_file, embeddings_file):
    with open(chunks_file, 'w', newline='', encoding='utf-8') as ch_file, open(embeddings_file, 'w', newline='', encoding='utf-8') as emb_file:
        chunk_writer = csv.writer(ch_file)
        embedding_writer = csv.writer(emb_file)
        for chunk, embedding in zip(chunks, embeddings):
            chunk_writer.writerow([chunk])
            embedding_writer.writerow(embedding.tolist())


def load_chunks_embeddings(chunks_file, embeddings_file):
    chunks = []
    embeddings = []
    with open(chunks_file, 'r', encoding='utf-8') as ch_file, open(embeddings_file, 'r', encoding='utf-8') as emb_file:
        chunk_reader = csv.reader(ch_file)
        embedding_reader = csv.reader(emb_file)
        for row in chunk_reader:
            chunks.append(row[0])
        for row in embedding_reader:
            embeddings.append([float(value) for value in row])
    return chunks, np.array(embeddings)


if os.path.exists(som_CHUNKS) and os.path.exists(som_EMBEDDINGS):
    logger.info("Loading embeddings...")
    chunks, sentence_embeddings = load_chunks_embeddings(som_CHUNKS, som_EMBEDDINGS)
else:
    logger.info("Generating embeddings...")
    chunks = train_data['Text'].tolist()
    sentence_embeddings = model.encode(chunks)
    save_chunks_embeddings(chunks, sentence_embeddings, som_CHUNKS, som_EMBEDDINGS)

    """ 
    Top 3 relevant rows
    """
def retrieve_examples(input, k=3):
    input = preprocess_text(input)
    input_embedding = model.encode([input])[0]
    input_tokens = input.split()
    bm25_scores = bm25.get_scores(input_tokens)
    cosine_similarities = cosine_similarity([input_embedding], sentence_embeddings)[0]


    """ 
    This is score of variability calculation
    """
    bm25_range = np.max(bm25_scores) - np.min(bm25_scores)
    cosine_range = np.max(cosine_similarities) - np.min(cosine_similarities)

    """ 
    We take the average scores
    """
    # epsilon = 1e-8
    # normalized_bm25_scores = (bm25_scores - np.min(bm25_scores)) / (bm25_range + epsilon)
    # normalized_cosine_similarities = (cosine_similarities - np.min(cosine_similarities)) / (cosine_range + epsilon)

    """ 
    When runing this code, if you face any warnings, please uncomment and use the above three lines of code instead, and put the below two lines under comment.
    """
    normalized_bm25_scores = (bm25_scores - np.min(bm25_scores)) / (bm25_range)
    normalized_cosine_similarities = (cosine_similarities - np.min(cosine_similarities)) / (cosine_range)
    
    """ 
    Normalizing the scores
    """
    weight = 0.5
    combined_scores = weight * normalized_cosine_similarities + (1 - weight) * normalized_bm25_scores
    top_k_indices = np.argsort(combined_scores)[::-1][:k]
    retrieved_examples = [chunks[idx] for idx in top_k_indices]
    return retrieved_examples


""" 
For Chatgpt API use this 
"""
deepseek_client = OpenAI(
 api_key=" "
 )

""" 
For deepseek-V3 uncomment this 
"""
#deepseek_client = OpenAI(
    #api_key=" ", base_url="   ")


with open('hateful-words/hate', 'r', encoding='utf-8') as f2:
         hate_words = [line.strip() for line in f2.readlines()]
       
def som_hybrid(text, retrieved_examples):
    prompt = "Classify the input text as hate or normal, example the sentence 'she is dhilo baahan' is Classified as 'hate'. If classifying the Input Text is challenging, use Retrieved Examples and Example Words as additional information to improve accuracy. Please strictly return only 1 for hate and 0 for normal.\n\n"
    word = max(hate_words, key=lambda w: difflib.SequenceMatcher(None, w, text).ratio())
    prompt += f"Example Words: {word}\n"
    for i, example_text in enumerate(retrieved_examples, 1):
        prompt += f"Retrieved Examples {i}: Text: \"{example_text}\"\n"
    prompt += f"\nInput Text: \"{text}\"\nClass:"
    response = deepseek_client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content':'You are a Somali-English code-mixed hate speech detection assistant.'},
            {'role': 'user', 'content': prompt}
        ],
    )
    return response.choices[0].message.content.strip()

# %% These are the models we used:
# 1. gpt-4o-mini-2024-07-18    # 2. gpt-4o-2024-08-06       # 3. gpt-3.5-turbo-0125   # 4. deepseek-chat

predictions = []

""" For every row, except the first rom, retrieve the top relevant rows to the current row. """
for _, row in test_data.iterrows():
    retrieved = retrieve_examples(row['Text'], k=3)
    pred = som_hybrid(row['Text'], retrieved)
    predictions.append(pred)
results_df = pd.DataFrame({
    'Text': test_data['Text'],
    'predicted_Class': predictions
})
results_df.to_csv('Topk-values/deepseek-chat33.csv', index=False)

