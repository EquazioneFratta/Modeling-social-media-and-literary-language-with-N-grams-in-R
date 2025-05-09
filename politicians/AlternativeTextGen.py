import pandas as pd
import random
import math

from lib import clean_text, tokenize_with_punctuation, filter_tokens
from nltk.util import ngrams
from collections import defaultdict, Counter


def load_and_tokenize_text(username):
    df = pd.read_csv(f"../{username}_tweets.csv")
    all_tokens = []
    for _, row in df.iterrows():
        if 'text' in row and isinstance(row['text'], str):
            cleaned = clean_text(row['text'])
            tokens = tokenize_with_punctuation(cleaned)
            if tokens:
                all_tokens.extend(['<s>'] + tokens + ['</s>'])
    return all_tokens

def compute_mle_probs(tokens):
    unigrams = Counter(tokens)
    bigrams = list(ngrams(tokens, 2))
    bigram_counts = Counter(bigrams)
    mle_probs = {}
    vocab_size = len(unigrams)
    for (w1, w2), count in bigram_counts.items():
        # Laplace smoothing
        mle_probs[(w1, w2)] = (count + 1) / (unigrams[w1] + vocab_size)
    return mle_probs, unigrams, bigram_counts


def sentence_probability(sentence, probs, unigrams, vocab_size):
    tokens = tokenize_with_punctuation(clean_text(sentence))
    tokens = ['<s>'] + tokens + ['</s>']
    bigrams = list(ngrams(tokens, 2))
    log_prob = 0
    for (w1, w2) in bigrams:
        prob = probs.get((w1, w2), 1 / (unigrams[w1] + vocab_size))
        log_prob += math.log(prob)
    return log_prob


def perplexity(log_prob, n):
    return math.exp(-log_prob / n)


def generate_sentence(probs, unigrams, max_length=15):
    sentence = ['<s>']
    while len(sentence) < max_length:
        candidates = [(w2, p) for (w1, w2), p in probs.items() if w1 == sentence[-1]]
        if not candidates:
            break
        next_word = random.choices([w for w, _ in candidates], weights=[p for _, p in candidates])[0]
        sentence.append(next_word)
        if next_word == '</s>':
            break
    return format_sentence_with_punctuation(sentence[1:-1])


def format_sentence_with_punctuation(tokens):
    text = ""
    for i, token in enumerate(tokens):
        if token in {'.', ',', '!', '?', ';', ':', ')', ']', '}'}:
            text = text.rstrip() + token + " "
        elif token in {'(', '[', '{'}:
            text += token
        elif i > 0 and tokens[i-1] in {'(', '[', '{'}:
            text += token + " "
        else:
            text += token + " "
    return text.strip()


if __name__ == "__main__":
    username = "matteosalvinimi"
    tokens = load_and_tokenize_text(username)
    tokens = filter_tokens(tokens)
    mle_probs, unigrams, bigram_counts = compute_mle_probs(tokens)
    vocab_size = len(unigrams)

    print("\nEsempi di frasi generate con probabilità calcolate (MLE + smoothing):")
    for _ in range(5):
        print("-", generate_sentence(mle_probs, unigrams))

    # Calcola probabilità e perplexity di una frase esistente
    test_sentence = "L'Italia ha bisogno di sicurezza."
    log_p = sentence_probability(test_sentence, mle_probs, unigrams, vocab_size)
    perp = perplexity(log_p, len(test_sentence.split()))
    print(f"\nFrase di test: '{test_sentence}'")
    print(f"Log-probabilità: {log_p:.4f}")
    print(f"Perplessità: {perp:.2f}")
