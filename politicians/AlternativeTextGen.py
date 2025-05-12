import pandas as pd
import random
import math

from lib import clean_text, tokenize_with_punctuation, filter_tokens
from nltk.util import ngrams
from collections import defaultdict, Counter


def load_and_tokenize_text(username):
    df = pd.read_csv(f"{username}_tweets.csv")
    all_tokens = []
    for _, row in df.iterrows():
        if 'text' in row and isinstance(row['text'], str):
            cleaned = clean_text(row['text'])
            tokens = tokenize_with_punctuation(cleaned)
            if tokens:
                all_tokens.extend(['<s>'] + tokens + ['</s>']) #TODO:capire
    return all_tokens

def compute_mle_probs_no_smoothing(tokens):
    # Conta gli unigrammi, bigrammi e trigrammi
    unigrams = Counter(tokens)
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))

    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)

    bigram_probs = {}
    trigram_probs = {}

    # Probabilità MLE per i bigrammi: P(w2 | w1) = count(w1, w2) / count(w1)
    for (w1, w2), count in bigram_counts.items():
        if unigrams[w1] > 0:
            bigram_probs[(w1, w2)] = count / unigrams[w1]

    # Probabilità MLE per i trigrammi: P(w3 | w1, w2) = count(w1, w2, w3) / count(w1, w2)
    for (w1, w2, w3), count in trigram_counts.items():
        if bigram_counts[(w1, w2)] > 0:
            trigram_probs[(w1, w2, w3)] = count / bigram_counts[(w1, w2)]

    return bigram_probs, trigram_probs, unigrams, bigram_counts, trigram_counts


#def sentence_probability(sentence, probs, unigrams, vocab_size):
#    tokens = tokenize_with_punctuation(clean_text(sentence))
#    tokens = ['<s>'] + tokens + ['</s>']
#    bigrams = list(ngrams(tokens, 2))
#    log_prob = 0
#    for (w1, w2) in bigrams:
#        prob = probs.get((w1, w2), 1 / (unigrams[w1] + vocab_size))
#        log_prob += math.log(prob)
#    return log_prob


#def perplexity(log_prob, n):
#    return math.exp(-log_prob / n)


def generate_sentence(probs, max_length=30):
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

def generate_sentence_trigram(trigram_probs, bigram_probs, max_length=30):
    sentence = ['<s>']
    # Seleziona il secondo token basandosi sui bigrammi che seguono '<s>'
    starters = [(w2, p) for (w1, w2), p in bigram_probs.items() if w1 == '<s>']
    if not starters:
        return ""
    
    second_word = random.choices(
        [w for w, _ in starters],
        weights=[p for _, p in starters]
    )[0]
    sentence.append(second_word)
    
    
    while len(sentence) < int(max_length):
        w1, w2 = sentence[-2], sentence[-1]
        # Trova tutti i trigrammi che seguono la coppia (w1, w2)
        candidates = [(w3, p) for (c1, c2, w3), p in trigram_probs.items() if c1 == w1 and c2 == w2]
        if not candidates:
            break
        # Seleziona il prossimo token in base alle probabilità
        next_word = random.choices(
            [w for w, _ in candidates],
            weights=[p for _, p in candidates]
        )[0]
        sentence.append(next_word)
        if next_word == '</s>':
            break
    
    # Rimuove i token di inizio e fine frase e restituisce la frase
    return format_sentence_with_punctuation(sentence[1:-1])


def generate_sentence_trigram_on_steroids(trigram_probs, bigram_probs, max_length=30):
    sentence = ['<s>']
    
    # Serve un secondo token iniziale
    # Scegli un primo bigramma che comincia con <s>
    starters = [(w2, p) for (w1, w2), p in bigram_probs.items() if w1 == '<s>']
    if not starters:
        return ""  # Nessun inizio possibile
    first_word = random.choices([w for w, _ in starters], weights=[p for _, p in starters])[0]
    sentence.append(first_word)
    
    while len(sentence) < max_length:
        w1, w2 = sentence[-2], sentence[-1]
        candidates = [(w3, p) for (c1, c2, w3), p in trigram_probs.items() if c1 == w1 and c2 == w2]
        
        if not candidates:
            # Fallback: cerca bigrammi a partire da w2
            candidates = [(w2_next, p) for (prev, w2_next), p in bigram_probs.items() if prev == w2]
            if not candidates:
                break
        
        next_word = random.choices([w for w, _ in candidates], weights=[p for _, p in candidates])[0]
        sentence.append(next_word)
        if next_word == '</s>':
            break
    
    return format_sentence_with_punctuation(sentence[1:-1])  # Rimuove <s> e </s>


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
    username = "GiorgiaMeloni"
    tokens = load_and_tokenize_text(username)
    tokens = filter_tokens(tokens)
    bigram_probs, trigram_probs, unigrams, bigram_counts, trigram_counts = compute_mle_probs_no_smoothing(tokens)
    vocab_size = len(unigrams)

    print("\nEsempi di frasi generate con probabilità calcolate (Bigrams with MLE):")
    for _ in range(5):
        print("-", generate_sentence(bigram_probs))
    print("\nEsempi di frasi generate con probabilità calcolate (Trigrams with MLE):")
    for _ in range(5):
        print("-", generate_sentence_trigram(trigram_probs, bigram_probs))
    print("\nEsempi di frasi generate con probabilità calcolate (Trigrams with MLE on steroids):")
    for _ in range(5):
        print("-", generate_sentence_trigram_on_steroids(trigram_probs, bigram_probs))
    # Calcola probabilità e perplexity di una frase esistente
    #test_sentence = "L'Italia ha bisogno di sicurezza."
    #log_p = sentence_probability(test_sentence, mle_probs, unigrams, vocab_size)
    #perp = perplexity(log_p, len(test_sentence.split()))
    #print(f"\nFrase di test: '{test_sentence}'")
    #print(f"Log-probabilità: {log_p:.4f}")
    #print(f"Perplessità: {perp:.2f}")
