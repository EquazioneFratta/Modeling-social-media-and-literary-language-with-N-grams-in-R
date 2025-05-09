import pandas as pd
import matplotlib.pyplot as plt
from nltk.util import ngrams
from collections import Counter
import re
import os

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)

    text = text.lower()

    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_with_punctuation(text):
    tokens = []
    pattern = r'[\w\']+|[.,!?;:()\[\]{}]'
    tokens = re.findall(pattern, text)
    return tokens

def generate_ngrams(tokens, n):
    stop_words = {'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una',
                 'e', 'ed', 'ma', 'se', 'perché', 'che', 'di', 'a', 'da',
                 'in', 'con', 'su', 'per', 'tra', 'fra', 'è', 'sono', 'sei',
                 'del', 'della', 'dei', 'degli', 'delle', 'nel', 'nella',
                 'nei', 'negli', 'nelle', 'sul', 'sulla', 'sui', 'sugli',
                 'sulle', 'non', 'si', 'ci', 'mi', 'ti', 'vi', 'lo', 'la',
                 'li', 'le', 'gli', 'ne', 'al', 'dal', 'del', 'più', 'che',
                 'chi', 'cui', 'ha', 'ho', 'hai', 'hanno', 'come', 'dove',
                 'quando', 'però', 'quindi', 'questo', 'questa', 'questi',
                 'queste', 'quello', 'quella', 'quelli', 'quelle'}
    
    punctuation = {'.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}'}
    
    filtered_tokens = []
    for token in tokens:
        if token in punctuation:
            filtered_tokens.append(token)
        elif token not in stop_words and len(token) > 1:
            filtered_tokens.append(token)
    
    return list(ngrams(filtered_tokens, n))

def plot_ngrams(ngram_freq, title, n, username):
    items = ngram_freq.most_common(20)
    phrases = [' '.join(gram) for gram, _ in items]
    counts = [count for _, count in items]

    plt.figure(figsize=(10, 6))
    plt.barh(phrases[::-1], counts[::-1], color='skyblue')
    plt.title(f"{title} per @{username}")
    plt.xlabel("Frequenza")
    plt.tight_layout()
    plt.savefig(f"{username}_{title.lower().replace(' ', '_')}.png")
    plt.show()

def analyze_ngrams_from_csv(username):
    filename = f"{username}_tweets.csv"
    if not os.path.exists(filename):
        print(f"File {filename} non trovato. Salto {username}.")
        return

    df = pd.read_csv(filename)
    
    all_tokens = []
    for _, row in df.iterrows():
        if 'text' in row and isinstance(row['text'], str):
            cleaned_text = clean_text(row['text'])
            tokens = tokenize_with_punctuation(cleaned_text)
            all_tokens.extend(tokens)
    
    bigrams = generate_ngrams(all_tokens, 2)
    trigrams = generate_ngrams(all_tokens, 3)

    bigram_freq = Counter(bigrams)
    trigram_freq = Counter(trigrams)

    plot_ngrams(bigram_freq, "Top 20 Bigrammi", 2, username)
    plot_ngrams(trigram_freq, "Top 20 Trigrammi", 3, username)

if __name__ == "__main__":
    usernames = ["matteosalvinimi", "GiorgiaMeloni"]
    for username in usernames:
        analyze_ngrams_from_csv(username)