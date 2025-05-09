import os
import pandas as pd
import matplotlib.pyplot as plt

from lib import clean_text, tokenize_with_punctuation, filter_tokens
from nltk.util import ngrams
from collections import Counter

    # return list(ngrams(filtered_tokens, n))

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
    # plt.show()

def analyze_ngrams_from_csv(username):
    filename = f"../{username}_tweets.csv"
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
    
    bigrams = list(ngrams(filter_tokens(all_tokens),2))
    trigrams = list(ngrams(filter_tokens(all_tokens),3))

    bigram_freq = Counter(bigrams)
    trigram_freq = Counter(trigrams)

    plot_ngrams(bigram_freq, "Top 20 Bigrammi", 2, username)
    plot_ngrams(trigram_freq, "Top 20 Trigrammi", 3, username)

if __name__ == "__main__":
    usernames = ["matteosalvinimi", "GiorgiaMeloni"]
    for username in usernames:
        analyze_ngrams_from_csv(username)