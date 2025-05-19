import pandas as pd
import random
import math
import re
from collections import defaultdict, Counter
from nltk.util import ngrams

from lib import clean_text, tokenize_with_punctuation, filter_tokens


def load_and_tokenize_text(username):
    df = pd.read_csv(f"{username}_tweets.csv")
    all_tokens = []
    
    for _, row in df.iterrows():
        if 'text' in row and isinstance(row['text'], str):
            # Tratta i ":" iniziali - rimuovili solo se sono all'inizio del tweet
            text = row['text']
            text = re.sub(r'^:\s*', '', text).strip()
            
            cleaned = clean_text(text)
            tokens = tokenize_with_punctuation(cleaned)
            
            if tokens:
                all_tokens.extend(['<s>'] + tokens + ['</s>'])
    
    return all_tokens


def analyze_start_end_words(tokens):
    """
    Analizza i token per identificare le parole che iniziano e finiscono le frasi.
    Esclude i segni di punteggiatura come parole di inizio o fine frase.
    """
    start_words = Counter()
    end_words = Counter()
    punctuation = ['.', ',', '!', '?', ':', ';', '...']
    
    for i in range(len(tokens) - 1):
        if tokens[i] == '<s>' and i + 1 < len(tokens):
            # Evita che la punteggiatura sia considerata una parola di inizio
            if tokens[i + 1] not in punctuation:
                start_words[tokens[i + 1]] += 1
        elif tokens[i + 1] == '</s>' and i >= 0:
            end_words[tokens[i]] += 1
    
    return start_words, end_words


def compute_mle_probs_with_improved_smoothing(tokens, smoothing_factor=0.1):
    """
    Calcola le probabilità con Laplace smoothing migliorato.
    Un fattore di smoothing più basso (0.1 invece di 1) può dare probabilità più realistiche.
    """
    unigrams = Counter(tokens)
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))

    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)

    # Rimuovi combinazioni di punteggiatura non valide
    punctuation = ['.', ',', '!', '?', ':', ';', '...']
    filtered_bigram_counts = Counter()
    filtered_trigram_counts = Counter()
    
    for (w1, w2), count in bigram_counts.items():
        # Evita sequenze di punteggiatura consecutiva
        if not (w1 in punctuation and w2 in punctuation):
            filtered_bigram_counts[(w1, w2)] = count
    
    for (w1, w2, w3), count in trigram_counts.items():
        # Evita sequenze di punteggiatura consecutiva
        if not ((w1 in punctuation and w2 in punctuation) or 
                (w2 in punctuation and w3 in punctuation)):
            filtered_trigram_counts[(w1, w2, w3)] = count

    bigram_probs = {}
    trigram_probs = {}

    # Dimensione del vocabolario
    V = len(unigrams)

    # Probabilità MLE con Laplace smoothing migliorato per i bigrammi
    for (w1, w2), count in filtered_bigram_counts.items():
        bigram_probs[(w1, w2)] = (count + smoothing_factor) / (unigrams[w1] + smoothing_factor * V)

    # Probabilità MLE con Laplace smoothing migliorato per i trigrammi
    for (w1, w2, w3), count in filtered_trigram_counts.items():
        bigram = (w1, w2)
        bigram_count = filtered_bigram_counts[bigram] if bigram in filtered_bigram_counts else 0
        trigram_probs[(w1, w2, w3)] = (count + smoothing_factor) / (bigram_count + smoothing_factor * V)

    return bigram_probs, trigram_probs, unigrams, filtered_bigram_counts, filtered_trigram_counts


def select_next_word_weighted(candidates, weights):
    """
    Seleziona la prossima parola in base ai pesi (probabilità).
    Questo rende la selezione più realistica rispetto alla scelta casuale.
    """
    if not candidates:
        return None
    return random.choices(candidates, weights=weights, k=1)[0]


def generate_sentence_with_start_end(bigram_probs, start_words, end_words, max_length=30):
    """
    Genera una frase con bigrammi, con selezione pesata delle parole.
    """
    if not start_words:
        return ""
    
    # Seleziona una parola iniziale pesata in base alla frequenza
    candidates = list(start_words.keys())
    weights = [start_words[word] for word in candidates]
    first_word = select_next_word_weighted(candidates, weights) 
    
    sentence = [first_word]
    current_word = first_word
    punctuation = ['.', ',', '!', '?', ':', ';', '...']
    
    for _ in range(max_length):
        # Trova tutte le possibili parole successive
        next_words = []
        next_weights = []
        
        for (w1, w2), prob in bigram_probs.items():
            if w1 == current_word:
                # Evita punteggiatura consecutiva
                if not (current_word in punctuation and w2 in punctuation):
                    next_words.append(w2)
                    next_weights.append(prob)
        
        if not next_words:
            break
        
        # Verso la fine della frase, favorisci le parole di fine
        if len(sentence) > max_length * 0.7:
            end_candidates = [w for w in next_words if w in end_words]
            if end_candidates and random.random() > 0.3:  # 70% di probabilità di usare una parola di fine
                end_weights = [end_words[w] for w in end_candidates]
                next_word = select_next_word_weighted(end_candidates, end_weights)
            else:
                next_word = select_next_word_weighted(next_words, next_weights)
        else:
            next_word = select_next_word_weighted(next_words, next_weights)
        
        sentence.append(next_word)
        if next_word == '</s>':
            break
        current_word = next_word
    
    return post_process_text(sentence)


def generate_sentence_trigram(trigram_probs, bigram_probs, start_words, end_words, max_length=30):
    """
    Genera una frase con trigrammi, con fallback ai bigrammi quando necessario.
    """
    if not start_words:
        return ""
    
    # Seleziona una parola iniziale pesata
    candidates = list(start_words.keys())
    weights = [start_words[word] for word in candidates]
    first_word = select_next_word_weighted(candidates, weights)
    
    sentence = ['<s>', first_word]  # Inizia con <s> per chiarezza
    
    # Trova tutte le possibili seconde parole basate sul primo token
    second_word_candidates = []
    second_word_weights = []
    
    for (w1, w2), prob in bigram_probs.items():
        if w1 == first_word:
            second_word_candidates.append(w2)
            second_word_weights.append(prob)
    
    if not second_word_candidates:
        return post_process_text([first_word])
    
    second_word = select_next_word_weighted(second_word_candidates, second_word_weights)
    sentence.append(second_word)
    
    current_bigram = (first_word, second_word)
    punctuation = ['.', ',', '!', '?', ':', ';', '...']
    
    for _ in range(max_length - 2):  # -2 perché abbiamo già aggiunto 2 parole
        # Trova tutte le possibili parole successive basate sul trigramma
        next_words = []
        next_weights = []
        
        # Prova con i trigrammi
        for (w1, w2, w3), prob in trigram_probs.items():
            if (w1, w2) == current_bigram:
                # Evita punteggiatura consecutiva
                if not (current_bigram[1] in punctuation and w3 in punctuation):
                    next_words.append(w3)
                    next_weights.append(prob)
        
        # Se non ci sono trigrammi, prova con i bigrammi
        if not next_words:
            for (w1, w2), prob in bigram_probs.items():
                if w1 == current_bigram[1]:
                    if not (current_bigram[1] in punctuation and w2 in punctuation):
                        next_words.append(w2)
                        next_weights.append(prob)
        
        if not next_words:
            break
        
        # Verso la fine della frase, favorisci le parole di fine
        if len(sentence) > max_length * 0.7:
            end_candidates = [w for w in next_words if w in end_words]
            if end_candidates and random.random() > 0.3:  # 70% di probabilità di usare una parola di fine
                end_weights = [end_words[w] for w in end_candidates]
                next_word = select_next_word_weighted(end_candidates, end_weights)
            else:
                next_word = select_next_word_weighted(next_words, next_weights)
        else:
            next_word = select_next_word_weighted(next_words, next_weights)
        
        sentence.append(next_word)
        if next_word == '</s>':
            break
        
        # Aggiorna il bigram corrente
        current_bigram = (current_bigram[1], next_word)
    
    return post_process_text(sentence)


def calculate_perplexity(test_tokens, bigram_probs, unigrams, smoothing_factor=0.1):
    """
    Calcola perplexity con smoothing migliorato e gestione di OOV.
    """
    N = len(test_tokens)
    sum_log_prob = 0.0
    V = len(unigrams)
    
    for i in range(1, N):
        w1, w2 = test_tokens[i - 1], test_tokens[i]
        
        # Gestione di parole fuori vocabolario (OOV)
        if (w1, w2) in bigram_probs:
            prob = bigram_probs[(w1, w2)]
        elif w1 in unigrams:
            # Backoff a unigram con smoothing
            prob = smoothing_factor / (unigrams[w1] + smoothing_factor * V)
        else:
            # Parola OOV, usa una probabilità uniforme
            prob = 1 / V
        
        sum_log_prob += math.log2(prob)  # Usando log base 2 invece di log naturale

    # Calcola perplexity con sicurezza per evitare valori estremi
    avg_log_prob = sum_log_prob / N
    perplexity = pow(2, -avg_log_prob)
    
    return min(perplexity, 10000)  # Limita valori estremi


def calculate_perplexity_trigram(test_tokens, bigram_probs, trigram_probs, unigrams, smoothing_factor=0.1):
    """
    Calcola perplexity per il modello trigram con smoothing e gestione OOV migliorati.
    """
    N = len(test_tokens)
    sum_log_prob = 0.0
    V = len(unigrams)
    
    for i in range(2, N):
        w1, w2, w3 = test_tokens[i - 2], test_tokens[i - 1], test_tokens[i]
        
        # Modello a cascata con backoff
        if (w1, w2, w3) in trigram_probs:
            prob = trigram_probs[(w1, w2, w3)]
        elif (w2, w3) in bigram_probs:
            prob = bigram_probs[(w2, w3)] * 0.4  # Penalità per backoff
        elif w3 in unigrams:
            prob = (unigrams[w3] / sum(unigrams.values())) * 0.1  # Penalità maggiore
        else:
            # Parola OOV
            prob = smoothing_factor / V
        
        sum_log_prob += math.log2(prob)  # Usando log base 2
    
    avg_log_prob = sum_log_prob / N
    perplexity = pow(2, -avg_log_prob)
    
    return min(perplexity, 10000)  # Limita valori estremi


def post_process_text(tokens):
    """
    Applica una serie di euristiche migliorate per la formattazione del testo.
    """
    # Rimuovi i tag speciali
    tokens = [t for t in tokens if t not in ['<s>', '</s>']]
    
    if not tokens:
        return ""
    
    text = ""
    punctuation = ['.', ',', '!', '?', ':', ';', '...']
    
    for i, token in enumerate(tokens):
        # Gestione della punteggiatura - evita spazi prima
        if token in punctuation:
            text = text.rstrip() + token + " "
        # Gestione di simboli speciali
        elif token in ["€", "$", "°C", "°F", "%", "°", "n°", "km/h"]:
            if i > 0 and tokens[i-1].isdigit():
                text = text.rstrip() + token + " "
            else:
                text += token + " "
        # Gestione delle parentesi
        elif token in ["(", "[", "{"]:
            text += token
        elif token in [")", "]", "}"]:
            text = text.rstrip() + token + " "
        # Gestione degli apostrofi
        elif token.startswith("'") and len(token) > 1:
            text = text.rstrip() + token + " "
        # Gestione normale per le altre parole
        else:
            text += token + " "
    
    # Pulizia finale
    text = text.strip()
    
    # Correggi eventuali spazi duplicati
    text = re.sub(r'\s+', ' ', text)
    
    # Inserisce maiuscola
    
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]
    
    return text


if __name__ == "__main__":
    username = "GiorgiaMeloni"  # Puoi cambiare l'username
    tokens = load_and_tokenize_text(username)
    tokens = filter_tokens(tokens)
    
    start_words, end_words = analyze_start_end_words(tokens)
    
    # Utilizza lo smoothing migliorato
    bigram_probs, trigram_probs, unigrams, bigram_counts, trigram_counts = \
        compute_mle_probs_with_improved_smoothing(tokens, smoothing_factor=0.1)
    
    vocab_size = len(unigrams)
    
    print(f"\nVocabolario: {vocab_size} parole uniche")
    print(f"Bigrammi unici: {len(bigram_counts)}")
    print(f"Trigrammi unici: {len(trigram_counts)}")

    print("\nEsempi di frasi generate con Bigrammi migliorati:")
    for _ in range(5):
        print("-", generate_sentence_with_start_end(bigram_probs, start_words, end_words))

    print("\nEsempi di frasi generate con Trigrammi migliorati:")
    for _ in range(5):
        print("-", generate_sentence_trigram(trigram_probs, bigram_probs, start_words, end_words))

    # Calcola perplexity di una frase esistente
    test_sentence_raw = "Questa è una frase di test"
    test_sentence = test_sentence_raw.split()
    test_sentence = filter_tokens(test_sentence)
    
    print(f"\nFrase di test utilizzata: '{test_sentence_raw}'")
    print(f"Token dopo filtro: {test_sentence}")

    perplexity_bigram = calculate_perplexity(test_sentence, bigram_probs, unigrams)
    print(f"Perplexity della frase di test (Bigram): {perplexity_bigram:.2f}")

    perplexity_trigram = calculate_perplexity_trigram(test_sentence, bigram_probs, trigram_probs, unigrams)
    print(f"Perplexity della frase di test (Trigram): {perplexity_trigram:.2f}")
    
    # Calcola anche perplexity su un esempio reale dal dataset
    if len(tokens) > 20:
        real_example = tokens[10:20]  # Prendiamo un esempio reale dai dati
        real_text = post_process_text(real_example)
        print(f"\nEsempio reale dal dataset: '{real_text}'")
        
        perplexity_real_bigram = calculate_perplexity(real_example, bigram_probs, unigrams)
        print(f"Perplexity dell'esempio reale (Bigram): {perplexity_real_bigram:.2f}")
        
        if len(real_example) >= 3:
            perplexity_real_trigram = calculate_perplexity_trigram(real_example, bigram_probs, trigram_probs, unigrams)
            print(f"Perplexity dell'esempio reale (Trigram): {perplexity_real_trigram:.2f}")