import pandas as pd
import random
import math
import re
from collections import defaultdict, Counter
from nltk.util import ngrams

from lib import clean_text, tokenize_with_nltk, filter_tokens


def load_and_tokenize_text(username):
    df = pd.read_csv(f"{username}_tweets.csv")
    all_tokens = []
    
    for _, row in df.iterrows(): 
        if 'text' in row and isinstance(row['text'], str):
            text = row['text']
            
            # Pulisci il testo mantenendo la punteggiatura originale
            cleaned = clean_text(text)
            
            
                
            tokens = tokenize_with_nltk(cleaned)
            
            if tokens:
                # Filtra solo token molto corti ma mantieni la punteggiatura
                filtered_tokens = filter_tokens(tokens)
                if filtered_tokens:
                    all_tokens.extend(['<s>'] + filtered_tokens + ['</s>'])
            #stampa tutti i token per debug
            print(f"Tokenizzati: {filtered_tokens}")

    return all_tokens


def analyze_start_end_words(tokens):
    """
    Analizza i token per identificare le parole che iniziano e finiscono le frasi.
    Esclude i segni di punteggiatura come parole di inizio o fine frase.
    """
    start_words = Counter()
    end_words = Counter()
    punctuation = {'.', ',', '!', '?', ':', ';', '...', '(', ')', '[', ']', '{', '}', '"', "'"}
    
    for i in range(len(tokens) - 1):
        if tokens[i] == '<s>' and i + 1 < len(tokens):
            # Evita che la punteggiatura sia considerata una parola di inizio
            if tokens[i + 1] not in punctuation:
                start_words[tokens[i + 1]] += 1
        elif tokens[i + 1] == '</s>' and i >= 0:
            # Evita che la punteggiatura sia considerata una parola di fine
            if tokens[i] not in punctuation:
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
    punctuation = {'.', ',', '!', '?', ':', ';', '...', '(', ')', '[', ']', '{', '}', '"', "'"}
    filtered_bigram_counts = Counter()
    filtered_trigram_counts = Counter()
    
    for (w1, w2), count in bigram_counts.items():
        # Evita sequenze di punteggiatura consecutiva non naturali
        if not (w1 in punctuation and w2 in punctuation and w1 != '.' and w2 != '.'):
            filtered_bigram_counts[(w1, w2)] = count
    
    for (w1, w2, w3), count in trigram_counts.items():
        # Evita sequenze di punteggiatura consecutiva non naturali
        punctuation_count = sum([w in punctuation for w in [w1, w2, w3]])
        if punctuation_count < 2:  # Massimo una parola di punteggiatura nel trigram
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
        if bigram_count > 0:
            trigram_probs[(w1, w2, w3)] = (count + smoothing_factor) / (bigram_count + smoothing_factor * V)

    return bigram_probs, trigram_probs, unigrams, filtered_bigram_counts, filtered_trigram_counts


def select_next_word_weighted(candidates, weights):
    """
    Seleziona la prossima parola in base ai pesi (probabilità).
    Questo rende la selezione più realistica rispetto alla scelta casuale.
    """
    if not candidates:
        return None
    
    # Normalizza i pesi per evitare problemi numerici
    total_weight = sum(weights)
    if total_weight <= 0:
        return random.choice(candidates)
    
    normalized_weights = [w / total_weight for w in weights]
    return random.choices(candidates, weights=normalized_weights, k=1)[0]


def generate_sentence_with_start_end(bigram_probs, start_words, end_words, max_length=25):
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
    punctuation = {'.', ',', '!', '?', ':', ';', '...', '(', ')', '[', ']', '{', '}', '"', "'"}
    
    for step in range(max_length):
        # Trova tutte le possibili parole successive
        next_words = []
        next_weights = []
        
        for (w1, w2), prob in bigram_probs.items():
            if w1 == current_word:
                next_words.append(w2)
                next_weights.append(prob)
        
        if not next_words:
            break
        
        # Verso la fine della frase, favorisci le parole di fine
        if len(sentence) > max_length * 0.6:  # Inizia prima a cercare finali
            end_candidates = [w for w in next_words if w in end_words]
            if end_candidates and random.random() > 0.4:  # 60% di probabilità di usare una parola di fine
                end_weights = [end_words[w] * next_weights[next_words.index(w)] for w in end_candidates]
                next_word = select_next_word_weighted(end_candidates, end_weights)
            else:
                next_word = select_next_word_weighted(next_words, next_weights)
        else:
            next_word = select_next_word_weighted(next_words, next_weights)
        
        sentence.append(next_word)
        
        # Termina se raggiungiamo un punto fermo o end-of-sentence
        if next_word == '</s>' or (next_word == '.' and random.random() > 0.3):
            break
            
        current_word = next_word
    
    return post_process_text(sentence)


def generate_sentence_trigram(trigram_probs, bigram_probs, start_words, end_words, max_length=25):
    """
    Genera una frase con trigrammi, con fallback ai bigrammi quando necessario.
    """
    if not start_words:
        return ""
    
    # Seleziona una parola iniziale pesata
    candidates = list(start_words.keys())
    weights = [start_words[word] for word in candidates]
    first_word = select_next_word_weighted(candidates, weights)
    
    sentence = [first_word]
    
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
    punctuation = {'.', ',', '!', '?', ':', ';', '...', '(', ')', '[', ']', '{', '}', '"', "'"}
    
    for step in range(max_length - 2):  # -2 perché abbiamo già aggiunto 2 parole
        # Trova tutte le possibili parole successive basate sul trigramma
        next_words = []
        next_weights = []
        
        # Prova con i trigrammi
        for (w1, w2, w3), prob in trigram_probs.items():
            if (w1, w2) == current_bigram:
                next_words.append(w3)
                next_weights.append(prob)
        
        # Se non ci sono trigrammi, prova con i bigrammi
        if not next_words:
            for (w1, w2), prob in bigram_probs.items():
                if w1 == current_bigram[1]:
                    next_words.append(w2)
                    next_weights.append(prob * 0.7)  # Penalizza leggermente il backoff
        
        if not next_words:
            break
        
        # Verso la fine della frase, favorisci le parole di fine
        if len(sentence) > max_length * 0.6:
            end_candidates = [w for w in next_words if w in end_words]
            if end_candidates and random.random() > 0.4:
                end_weights = [end_words[w] * next_weights[next_words.index(w)] for w in end_candidates]
                next_word = select_next_word_weighted(end_candidates, end_weights)
            else:
                next_word = select_next_word_weighted(next_words, next_weights)
        else:
            next_word = select_next_word_weighted(next_words, next_weights)
        
        sentence.append(next_word)
        
        # Termina se raggiungiamo un punto fermo o end-of-sentence
        if next_word == '</s>' or (next_word == '.' and random.random() > 0.3):
            break
        
        # Aggiorna il bigram corrente
        current_bigram = (current_bigram[1], next_word)
    
    return post_process_text(sentence)


def calculate_perplexity(test_tokens, bigram_probs, unigrams, smoothing_factor=0.1):
    """
    Calcola perplexity con smoothing migliorato e gestione di OOV.
    """
    if len(test_tokens) < 2:
        return float('inf')
    
    N = len(test_tokens) - 1  # Numero di bigrammi
    sum_log_prob = 0.0
    V = len(unigrams)
    
    for i in range(1, len(test_tokens)):
        w1, w2 = test_tokens[i - 1], test_tokens[i]
        
        # Gestione di parole fuori vocabolario (OOV)
        if (w1, w2) in bigram_probs:
            prob = bigram_probs[(w1, w2)]
        elif w1 in unigrams:
            # Backoff a unigram con smoothing
            prob = smoothing_factor / (unigrams[w1] + smoothing_factor * V)
        else:
            # Parola OOV, usa una probabilità uniforme
            prob = 1.0 / V
        
        # Evita log di 0
        prob = max(prob, 1e-10)
        sum_log_prob += math.log2(prob)

    # Calcola perplexity
    avg_log_prob = sum_log_prob / N
    perplexity = pow(2, -avg_log_prob)
    
    return min(perplexity, 100000)  # Limita valori estremi


def calculate_perplexity_trigram(test_tokens, bigram_probs, trigram_probs, unigrams, smoothing_factor=0.1):
    """
    Calcola perplexity per il modello trigram con smoothing e gestione OOV migliorati.
    """
    if len(test_tokens) < 3:
        return float('inf')
    
    N = len(test_tokens) - 2  # Numero di trigrammi
    sum_log_prob = 0.0
    V = len(unigrams)
    
    for i in range(2, len(test_tokens)):
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
        
        # Evita log di 0
        prob = max(prob, 1e-10)
        sum_log_prob += math.log2(prob)
    
    avg_log_prob = sum_log_prob / N
    perplexity = pow(2, -avg_log_prob)
    
    return min(perplexity, 100000)  # Limita valori estremi


def post_process_text(tokens):
    """
    Applica una serie di euristiche migliorate per la formattazione del testo.
    """
    # Rimuovi i tag speciali
    tokens = [t for t in tokens if t not in ['<s>', '</s>']]
    
    if not tokens:
        return ""
    
    text = ""
    punctuation_no_space_before = {'.', ',', '!', '?', ':', ';', ')', ']', '}', "'"}
    punctuation_no_space_after = {'(', '[', '{'}
    
    for i, token in enumerate(tokens):
        # Gestione della punteggiatura - evita spazi prima di certi segni
        if token in punctuation_no_space_before:
            text = text.rstrip() + token + " "
        elif token in punctuation_no_space_after:
            text += token
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
    
    # Inserisce maiuscola iniziale
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    return text


if __name__ == "__main__":
    username = "GiorgiaMeloni"  # Scegliere username
    
    print(f"Caricamento e tokenizzazione del testo per {username}...")
    tokens = load_and_tokenize_text(username)
    
    print(f"Numero totale di token (inclusi <s> e </s>): {len(tokens)}")
    
    # Mostra alcuni esempi di token per debug
    print(f"Primi 20 token: {tokens[:20]}")
    
    if len(tokens) < 10:
        print("ERRORE: Troppo pochi token trovati. Verificare la pulizia del testo.")
        exit(1)
    
    start_words, end_words = analyze_start_end_words(tokens)
    
    print(f"Parole di inizio frase: {len(start_words)} uniche")
    print(f"Parole di fine frase: {len(end_words)} uniche")
    print(f"Top 10 parole di inizio: {start_words.most_common(10)}")
    print(f"Top 10 parole di fine: {end_words.most_common(10)}")
    
    # Utilizza lo smoothing migliorato
    bigram_probs, trigram_probs, unigrams, bigram_counts, trigram_counts = \
        compute_mle_probs_with_improved_smoothing(tokens, smoothing_factor=0.1)
    
    vocab_size = len(unigrams)
    
    print(f"\nStatistiche del modello:")
    print(f"Vocabolario: {vocab_size} parole uniche")
    print(f"Bigrammi unici: {len(bigram_counts)}")
    print(f"Trigrammi unici: {len(trigram_counts)}")
    print(f"Top 10 parole più frequenti: {unigrams.most_common(10)}")

    print("\nEsempi di frasi generate con Bigrammi migliorati:")
    for i in range(5):
        sentence = generate_sentence_with_start_end(bigram_probs, start_words, end_words)
        print(f"{i+1}. {sentence}")

    print("\nEsempi di frasi generate con Trigrammi migliorati:")
    for i in range(5):
        sentence = generate_sentence_trigram(trigram_probs, bigram_probs, start_words, end_words)
        print(f"{i+1}. {sentence}")

    # Calcola perplexity di una frase di test
    test_sentence_raw = "L'Italia è una grande nazione."
    test_tokens = ['<s>'] + tokenize_with_nltk(test_sentence_raw.lower()) + ['</s>']
    test_tokens = filter_tokens(test_tokens)
    
    print(f"\nFrase di test utilizzata: '{test_sentence_raw}'")
    print(f"Token dopo filtro: {test_tokens}")

    if len(test_tokens) >= 2:
        perplexity_bigram = calculate_perplexity(test_tokens, bigram_probs, unigrams)
        print(f"Perplexity della frase di test (Bigram): {perplexity_bigram:.2f}")

    if len(test_tokens) >= 3:
        perplexity_trigram = calculate_perplexity_trigram(test_tokens, bigram_probs, trigram_probs, unigrams)
        print(f"Perplexity della frase di test (Trigram): {perplexity_trigram:.2f}")
    
    # Calcola anche perplexity su un esempio reale dal dataset
    if len(tokens) > 20:
        # Trova una sequenza che non contenga solo tag speciali
        real_example = []
        for i in range(10, min(25, len(tokens))):
            if tokens[i] not in ['<s>', '</s>']:
                real_example.append(tokens[i])
                if len(real_example) >= 8:
                    break
        
        if real_example:
            real_example = ['<s>'] + real_example + ['</s>']
            real_text = post_process_text(real_example)
            print(f"\nEsempio reale dal dataset: '{real_text}'")
            
            if len(real_example) >= 2:
                perplexity_real_bigram = calculate_perplexity(real_example, bigram_probs, unigrams)
                print(f"Perplexity dell'esempio reale (Bigram): {perplexity_real_bigram:.2f}")
            
            if len(real_example) >= 3:
                perplexity_real_trigram = calculate_perplexity_trigram(real_example, bigram_probs, trigram_probs, unigrams)
                print(f"Perplexity dell'esempio reale (Trigram): {perplexity_real_trigram:.2f}")