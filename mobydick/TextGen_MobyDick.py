import re
import random
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from nltk.util import ngrams
from lib import clean_text, tokenize_english, filter_tokens, evaluate_sentence

BAD_END_TOKENS = {
    'the','a','an','and','or','but','of','to','in','on','for','with','at','by','from','as','that','this','these','those',
    'is','are','was','were','be','am','been','being','do','does','did','so','not','if','than','then',
    'i','you','he','she','it','we','they','me','him','her','us','them','my','your','his','its','our','their'
}

PUNCTUATION = {'.', ',', '!', '?', ':', ';', '...', '(', ')', '[', ']', '{', '}', '"', "'"}

def load_and_tokenize_mobydick(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    cleaned = clean_text(raw_text)
    tokens = tokenize_english(cleaned)
    filtered_tokens = filter_tokens(tokens)

    # Inserisce <s> e </s> ai segmenti separati da punti o simili
    segmented = []
    sentence_tokens = []
    for token in filtered_tokens:
        sentence_tokens.append(token)
        if token in ['.', '!', '?']:
            segmented.extend(['<s>'] + sentence_tokens + ['</s>'])
            sentence_tokens = []
    if sentence_tokens:
        segmented.extend(['<s>'] + sentence_tokens + ['</s>'])

    return segmented


def analyze_start_end_words(tokens):
    
    start_words = Counter()
    end_words = Counter()

    for i in range(len(tokens) - 1):
        if tokens[i] == '<s>' and i + 1 < len(tokens):
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
        # Evita sequenze di punteggiatura consecutiva
        if not (w1 in PUNCTUATION and w2 in PUNCTUATION):
            filtered_bigram_counts[(w1, w2)] = count
    
    for (w1, w2, w3), count in trigram_counts.items():
        # Evita sequenze di punteggiatura consecutiva 
        punctuation_count = int(w1 in PUNCTUATION) + int(w2 in PUNCTUATION) + int(w3 in PUNCTUATION)
        if punctuation_count < 2:
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
    
    if not candidates or not weights:
        return None
    total_weight = sum(weights)
    if total_weight <= 0:
        return random.choice(candidates)
    normalized = [w / total_weight for w in weights]
    return random.choices(candidates, weights=normalized, k=1)[0]


def generate_sentence_with_start_end(bigram_probs, start_words, end_words, max_length=30):
    
    if not start_words:
        return ""
    
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
        next_words = []
        next_weights = []
        for (w1, w2), prob in bigram_probs.items():
            if w1 == current_word:
                if not (current_word in PUNCTUATION and w2 in PUNCTUATION):
                    next_words.append(w2)
                    next_weights.append(prob)
        if not next_words:
            break
        
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
        if next_word == '</s>' or (next_word == '.' and random.random() > 0.3):
            break
        current_word = next_word
    
    return post_process_text(sentence)


def post_process_text(tokens):
   
    tokens = [t for t in tokens if t not in ['<s>', '</s>']]
    if not tokens:
        return ""
    
    text = ""
    punctuation_no_space_before = {'.', ',', '!', '?', ':', ';', ')', ']', '}', "'"}
    punctuation_no_space_after = {'(', '[', '{'}
    
    for i, token in enumerate(tokens):
        if token in punctuation_no_space_before:
            text = text.rstrip() + token + " "
        elif token in punctuation_no_space_after:
            text += token
        elif token.startswith("'") and len(token) > 1:
            text = text.rstrip() + token + " "
        else:
            text += token + " "
    
    text = text.strip()
 
    if BAD_END_TOKENS:
        pattern = r"\b(" + '|'.join(sorted(BAD_END_TOKENS, key=len, reverse=True)) + r")\s*$"
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\s+', ' ', text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in {'.', '!', '?'}:
        text = text + '.'
    return text



def generate_sentence_trigram(trigram_probs, bigram_probs, start_words, end_words, max_length=30):
    
    if not start_words:
        return ""
    
    # Seleziona una parola iniziale pesata
    candidates = list(start_words.keys())
    weights = [start_words[word] for word in candidates]
    first_word = select_next_word_weighted(candidates, weights)
    
    sentence = ['<s>', first_word]  
    
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
    end_punctuation = {'.', ',', '!', '?', ':', ';', '...'}
    allowed_end_tokens = {w for w in end_words.keys() if w not in BAD_END_TOKENS}
    
    for _ in range(max_length - 2):  # -2 perché abbiamo già aggiunto 2 parole
        # Trova tutte le possibili parole successive basate sul trigramma
        next_words = []
        next_weights = []
        
        # Prova con i trigrammi
        for (w1, w2, w3), prob in trigram_probs.items():
            if (w1, w2) == current_bigram:
                # Evita punteggiatura consecutiva
                if not (current_bigram[1] in end_punctuation and w3 in end_punctuation):
                    next_words.append(w3)
                    next_weights.append(prob)
        
        # Se non ci sono trigrammi, prova con i bigrammi
        if not next_words:
            for (w1, w2), prob in bigram_probs.items():
                if w1 == current_bigram[1]:
                    if not (current_bigram[1] in end_punctuation and w2 in end_punctuation):
                        next_words.append(w2)
                        next_weights.append(prob * 0.7)  # Penalizza leggermente il backoff
        
        if not next_words:
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
        
        sentence.append(next_word)
        if next_word == '</s>' or (next_word == '.' and random.random() > 0.3):
            break
        
        current_bigram = (current_bigram[1], next_word)
    
    return post_process_text(sentence)

def plot_top_ngrams(tokens, n=2, top_k=20, include_special=False, title=None, save_path=None):
    
    toks = tokens if include_special else [t for t in tokens if t not in {'<s>', '</s>'}]

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
    filepath = str((Path(__file__).resolve().parent / 'moby-dick-clean.txt'))
    tokens = load_and_tokenize_mobydick(filepath)

    start_words, end_words = analyze_start_end_words(tokens)

    bigram_probs, trigram_probs, unigrams, bigram_counts, trigram_counts = \
        compute_mle_probs_with_improved_smoothing(tokens, smoothing_factor=0.1)

    vocab_size = len(unigrams)
    print(f"Vocabolario: {vocab_size} parole uniche")
    print(f"Bigrammi unici: {len(bigram_counts)}")
    print(f"Trigrammi unici: {len(trigram_counts)}")

    project_root = Path(__file__).resolve().parents[1]
    plot_top_ngrams(
        tokens,
        n=2,
        top_k=20,
        title='Moby Dick - Top 20 bigrammi',
        save_path=str(project_root / 'MobyDick_top_20_bigrammi.png')
    )
    plot_top_ngrams(
        tokens,
        n=3,
        top_k=20,
        title='Moby Dick - Top 20 trigrammi',
        save_path=str(project_root / 'MobyDick_top_20_trigrammi.png')
    )

    # Valutazione perplexity su frasi di esempio
    sample_sentence = "It is one of the white whale in the sea"
    perp_bi, perp_tri = evaluate_sentence(sample_sentence, bigram_probs, trigram_probs, unigrams, bigram_counts)
    print(f"\nFrase: {sample_sentence}")
    print(f"Perplexity bigram: {perp_bi:.2f}")
    print(f"Perplexity trigram: {perp_tri:.2f}")

    sample_sentence = "The whale swims in the vast ocean under the cold sky with a chainsaw."
    perp_bi, perp_tri = evaluate_sentence(sample_sentence, bigram_probs, trigram_probs, unigrams, bigram_counts)
    print(f"Frase: {sample_sentence}")
    print(f"Perplexity bigram: {perp_bi:.2f}")
    print(f"Perplexity trigram: {perp_tri:.2f}")

    print("\nEsempi di frasi generate con Bigrammi:")
    for _ in range(3):
        print("-", generate_sentence_with_start_end(bigram_probs, start_words, end_words))

    print("\nEsempi di frasi generate con Trigrammi:")
    for _ in range(3):
        print("-", generate_sentence_trigram(trigram_probs, bigram_probs, start_words, end_words))



