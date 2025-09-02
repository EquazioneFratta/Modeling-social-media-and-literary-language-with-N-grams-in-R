import pandas as pd
import argparse
import random
import re
from collections import Counter
from nltk.util import ngrams
from pathlib import Path
import matplotlib.pyplot as plt

from lib import clean_text, tokenize_with_nltk, filter_tokens, evaluate_sentence

BAD_END_TOKENS = {
    'il','lo','la','i','gli','le','un','uno','una','di','a','da','in','con','su','per','tra','fra',
    'e','o','che','se','ma','non','mi','ti','si','ci','vi','ne','al','del','della','dello','dei','degli','delle',
    'nel','nella','nello','sul','sulla','sullo','col','coi','dagli','dai','dalle','dalla','allo','alla','agli','alle'
}

PUNCTUATION = {'.', ',', '!', '?', ':', ';', '...', '(', ')', '[', ']', '{', '}', '"', "'"}

def load_and_tokenize_from_file(input_path: str):
    
    path = Path(input_path)
    df = pd.read_excel(path)
    texts = df[df.columns[0]].astype(str)
    all_tokens = []
    punctuation_ends = {'.', '!', '?'}

    for raw_text in texts:
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue

        cleaned = clean_text(raw_text)
        tokens = tokenize_with_nltk(cleaned)
        filtered_tokens = filter_tokens(tokens)

        # Divide in frasi e inserisce tag <s> ... </s>
        sentence_tokens = []
        for token in filtered_tokens:
            sentence_tokens.append(token)
            if token in punctuation_ends:
                if sentence_tokens:
                    all_tokens.extend(['<s>'] + sentence_tokens + ['</s>'])
                sentence_tokens = []
        if sentence_tokens:
            all_tokens.extend(['<s>'] + sentence_tokens + ['</s>'])

    return all_tokens


def analyze_start_end_words(tokens):
    
    start_words = Counter()
    end_words = Counter()
        
    for i in range(len(tokens) - 1):
        if tokens[i] == '<s>' and i + 1 < len(tokens):
            # Evita che la punteggiatura sia considerata una parola di inizio
            if tokens[i + 1] not in PUNCTUATION:
                start_words[tokens[i + 1]] += 1
        elif tokens[i + 1] == '</s>' and i >= 0:
            if tokens[i] not in PUNCTUATION and tokens[i] not in BAD_END_TOKENS:
                end_words[tokens[i]] += 1
    
    return start_words, end_words


def compute_mle_probs_with_improved_smoothing(tokens, smoothing_factor=0.1):
    
    unigrams = Counter(tokens)
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))

    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)

    filtered_bigram_counts = Counter()
    filtered_trigram_counts = Counter()
    
    for (w1, w2), count in bigram_counts.items():
        # Evita sequenze di punteggiatura consecutiva non naturali
        if not (w1 in PUNCTUATION and w2 in PUNCTUATION and w1 != '.' and w2 != '.'):
            filtered_bigram_counts[(w1, w2)] = count
    
    for (w1, w2, w3), count in trigram_counts.items():
        # Evita sequenze di punteggiatura consecutiva non naturali
        punctuation_count = sum([w in PUNCTUATION for w in [w1, w2, w3]])
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
    # Gestione robusta di liste vuote o pesi non validi
    if not candidates or not weights:
        return None
    total_weight = sum(weights)
    if total_weight <= 0:
        # Fallback: scelta uniforme se i pesi non sono utilizzabili
        return random.choice(candidates)
    normalized_weights = [w / total_weight for w in weights]
    return random.choices(candidates, weights=normalized_weights, k=1)[0]


def generate_sentence_with_start_end(bigram_probs, start_words, end_words, max_length=30):
       
    # Seleziona una parola iniziale pesata in base alla frequenza
    candidates = list(start_words.keys())
    weights = [start_words[word] for word in candidates]
    first_word = select_next_word_weighted(candidates, weights)
    if first_word is None:
        if not candidates:
            return ""
        first_word = random.choice(candidates)
    
    sentence = ['<s>', first_word]
    current_word = first_word

    allowed_end_tokens = {w for w in end_words.keys() if w not in BAD_END_TOKENS}

    for _ in range(max_length):
        # Trova tutte le possibili parole successive
        next_words = []
        next_weights = []
        
        for (w1, w2), prob in bigram_probs.items():
            if w1 == current_word:
                next_words.append(w2)
                next_weights.append(prob)
        
        if not next_words:
            break
        
        # Favorisce le parole di fine
        if len(sentence) > max_length * 0.7:  
            # Crea una lista di possibili parole successive che sono anche parole di fine ammesse
            end_candidates = [w for w in next_words if w in allowed_end_tokens]
            if end_candidates and random.random() > 0.3:  
                end_weights = [end_words[w] * next_weights[next_words.index(w)] for w in end_candidates]
                next_word = select_next_word_weighted(end_candidates, end_weights)
            else:
                next_word = select_next_word_weighted(next_words, next_weights)
        else:
            next_word = select_next_word_weighted(next_words, next_weights)
        if next_word is None:
            break
        
        sentence.append(next_word)
        
        # Termina se raggiungiamo un punto fermo o end-of-sentence
        if next_word == '</s>' or (next_word == '.' and random.random() > 0.3):
            break
            
        current_word = next_word
    
    return post_process_text(sentence)


def generate_sentence_trigram(trigram_probs, start_words, end_words, max_length=30):
    
    # Seleziona una parola iniziale pesata
    candidates = list(start_words.keys())
    weights = [start_words[word] for word in candidates]
    first_word = select_next_word_weighted(candidates, weights)
    if first_word is None:
        if not candidates:
            return ""
        first_word = random.choice(candidates)
    
    sentence = ['<s>', first_word]
    
    # Trova tutte le possibili seconde parole basate sul primo token
    second_word_candidates = []
    second_word_weights = []
    for (w1, w2, w3), prob in trigram_probs.items():
        if w1 == '<s>' and w2 == first_word:
            second_word_candidates.append(w3)
            second_word_weights.append(prob)
    ## Se non ci sono trigrammi per la seconda parola, usa tutti i possibili w3 (con probabilità >0)
    if not second_word_candidates:
        # Considera tutti i possibili w3 dati (w1, w2) = ('<s>', first_word)
        for (w1, w2, w3), prob in trigram_probs.items():
            if w1 == '<s>' and w2 == first_word:
                second_word_candidates.append(w3)
                second_word_weights.append(prob)

    if not second_word_candidates:
        all_w3 = set(w3 for (_, _, w3) in trigram_probs.keys())
        second_word_candidates = list(all_w3)
        second_word_weights = [1.0 for _ in second_word_candidates]
    
    second_word = select_next_word_weighted(second_word_candidates, second_word_weights)
    if second_word is None:
        if not second_word_candidates:
            return post_process_text(sentence)
        second_word = random.choice(second_word_candidates)
    sentence.append(second_word)
    
    current_words = (first_word, second_word)
    end_punctuation = {'.', ',', '!', '?', ':', ';', '...'}
    
    allowed_end_tokens = {w for w in end_words.keys() if w not in BAD_END_TOKENS}

    for _ in range(max_length - 2):  # -2 perché abbiamo già aggiunto 2 parole
        # Trova tutte le possibili parole successive basate solo sul trigramma
        next_words = []
        next_weights = []
        for (w1, w2, w3), prob in trigram_probs.items():
            if (w1, w2) == current_words:
                # Evita punteggiatura consecutiva
                if not (current_words[1] in end_punctuation and w3 in end_punctuation):
                    next_words.append(w3)
                    next_weights.append(prob)
        # Se non ci sono trigrammi, scegli dal vocabolario dei terzi elementi o termina
        if not next_words:
            all_w3 = set(w3 for (_, _, w3) in trigram_probs.keys())
            if all_w3:
                next_words = list(all_w3)
                next_weights = [1.0 for _ in next_words]
            else:
                break
        # Verso la fine della frase, favorisci le parole di fine
        if len(sentence) > max_length * 0.7:
            end_candidates = [w for w in next_words if w in allowed_end_tokens]
            if end_candidates and random.random() > 0.3:
                end_weights = [end_words[w] * next_weights[next_words.index(w)] for w in end_candidates]
                next_word = select_next_word_weighted(end_candidates, end_weights)
            else:
                next_word = select_next_word_weighted(next_words, next_weights)
        else:
            next_word = select_next_word_weighted(next_words, next_weights)
        if next_word is None:
            break
        
        sentence.append(next_word)
        
        # Termina se raggiungiamo un punto fermo o end-of-sentence
        if next_word == '</s>' or (next_word == '.' and random.random() > 0.3):
            break
        
        current_words = (current_words[1], next_word)
    return post_process_text(sentence)


def post_process_text(tokens):
    
    # Rimuovi i tag speciali
    tokens = [t for t in tokens if t not in ['<s>', '</s>']]
    
    text = ""
    punctuation_no_space_before = {'.', ',', '!', '?', ':', ';', ')', ']', '}', "'"}
    punctuation_no_space_after = {'(', '[', '{'}
    
    for _, token in enumerate(tokens):
        # Gestione della punteggiatura - evita spazi prima di certi segni
        if token in punctuation_no_space_before:
            text = text.rstrip() + token + " "
        elif token in punctuation_no_space_after:
            text += token
        ## Gestione degli apostrofi
        elif token.startswith("'") and len(token) > 1:
            text = text.rstrip() + token + " "
        ## Gestione normale per le altre parole
        else:
            text += token + " "
    
    # Pulizia finale
    text = text.strip()
    
    text = re.sub(r"\b(" + '|'.join(sorted(BAD_END_TOKENS, key=len, reverse=True)) + r")\s*$", '', text, flags=re.IGNORECASE)

    # Correggi eventuali spazi duplicati
    text = re.sub(r'\s+', ' ', text)
    
    # Inserisce maiuscola iniziale
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    # Garantisci una punteggiatura finale se assente
    if text and text[-1] not in {'.', '!', '?'}:
        text = text + '.'
    
    return text


def plot_top_ngrams(tokens, n=2, top_k=20, include_special=False, include_punct=False, title=None, save_path=None):
    
    punct = {'.', ',', '!', '?', ';', ':', '...', '(', ')', '[', ']', '{', '}', '"', "'"}

    toks = tokens if include_special else [t for t in tokens if t not in {'<s>', '</s>'}]
    if not include_punct:
        toks = [t for t in toks if t not in punct]

    counts = Counter(ngrams(toks, n))
    most_common = counts.most_common(top_k)
    if not most_common:
        print("Nessun n-gram disponibile per il grafico.")
        return

    labels = [' '.join(ng) for ng, _ in most_common]
    values = [c for _, c in most_common]

    plt.figure(figsize=(10, max(4, 0.4 * len(labels))))
    plt.barh(range(len(labels)), values, color='#4C78A8')
    plt.yticks(range(len(labels)), labels)
    plt.gca().invert_yaxis()
    plt.xlabel('Frequenza')
    plt.title(title or f'Top {top_k} {"bi" if n==2 else "tri"}grammi')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Grafico salvato in: {save_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", dest="user", type=str, default="GiorgiaMeloni")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / f"data/{args.user}_tweets.xlsx"
    if not file_path.exists():
        print(f"ERRORE: file non trovato: {file_path}")
        raise SystemExit(1)

    files_to_process = [str(file_path)]

    project_root = Path(__file__).resolve().parents[1]

    for input_file in files_to_process:
        file_path = Path(input_file)
        user_label = file_path.stem.replace("_tweets", "")

        print(f"Caricamento e tokenizzazione del testo da {input_file}...")
        tokens = load_and_tokenize_from_file(str(file_path))

        print(f"Numero totale di token (inclusi <s> e </s>): {len(tokens)}")
        

        start_words, end_words = analyze_start_end_words(tokens)
        
        bigram_probs, trigram_probs, unigrams, bigram_counts, trigram_counts = compute_mle_probs_with_improved_smoothing(tokens, smoothing_factor=0.1)

        print(f"\n[{user_label}] Statistiche del modello:")
        print(f"Vocabolario: {len(unigrams)} parole uniche")
        print(f"Bigrammi unici: {len(bigram_counts)}")
        print(f"Trigrammi unici: {len(trigram_counts)}")
        if(args.user =="GiorgiaMeloni"):
            frase = "Continueremo a lavorare a palazzo chigi per il governo"
        else:
            frase = "Il nostro paese ha fatto un passo avanti con il ponte sullo stretto"
        perp_bi, perp_tri = evaluate_sentence(frase, bigram_probs, trigram_probs, unigrams, bigram_counts)
    
        print(f"Frase: {frase}")
        print(f"Perplexity bigram: {perp_bi:.2f}")
        print(f"Perplexity trigram: {perp_tri:.2f}")
        frase = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        perp_bi, perp_tri = evaluate_sentence(frase, bigram_probs, trigram_probs, unigrams, bigram_counts)
    
        print(f"Frase: {frase}")
        print(f"Perplexity bigram: {perp_bi:.2f}")
        print(f"Perplexity trigram: {perp_tri:.2f}")

        plot_top_ngrams(
            tokens,
            n=2,
            top_k=20,
            title=f"{user_label} - Top 20 bigrammi",
            save_path=str(project_root / f"{user_label}_top_20_bigrammi.png")
        )
        plot_top_ngrams(
            tokens,
            n=3,
            top_k=20,
            title=f"{user_label} - Top 20 trigrammi",
            save_path=str(project_root / f"{user_label}_top_20_trigrammi.png")
        )

        print("\nEsempi di frasi generate con Bigrammi:")
        for i in range(3):
            print(f"- {generate_sentence_with_start_end(bigram_probs, start_words, end_words, max_length=100)}")

        print("\nEsempi di frasi generate con Trigrammi:")
        for i in range(3):
            print(f"- {generate_sentence_trigram(trigram_probs, start_words, end_words, max_length=100)}")
