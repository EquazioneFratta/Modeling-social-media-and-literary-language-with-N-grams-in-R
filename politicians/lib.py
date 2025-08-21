import re
import nltk
import math
from collections import Counter
# Decommenta per il primo utilizzo
# nltk.download('punkt')

def clean_text(text):
  
    extracted_text = text
    # Rimuovi link HTTP/HTTPS
    extracted_text = re.sub(r'https?://\S+', '', extracted_text)
    extracted_text = re.sub(r'www\.\S+', '', extracted_text)
    
    # Rimuovi pattern di link comuni
    extracted_text = re.sub(r't\.co/\S+', '', extracted_text)
    
    
    # Emoticons
    extracted_text = re.sub(r'[\U0001F600-\U0001F64F]', '', extracted_text)
    # Simboli e pittogrammi
    extracted_text = re.sub(r'[\U0001F300-\U0001F5FF]', '', extracted_text)
    # Trasporti e mappe
    extracted_text = re.sub(r'[\U0001F680-\U0001F6FF]', '', extracted_text)
    # Simboli vari
    extracted_text = re.sub(r'[\U00002600-\U000027BF]', '', extracted_text)
    # Indicatori regionali (bandiere)
    extracted_text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', extracted_text)
    
    # Rimuovi caratteri di controllo e spazi strani
    extracted_text = re.sub(r'[\u200b-\u200d\ufeff]', '', extracted_text)
    
    # Normalizza apostrofi tipografici in apostrofo ASCII
    extracted_text = extracted_text.replace('\u2019', "'").replace('\u2018', "'")
    extracted_text = extracted_text.replace('\u02BC', "'").replace('\u2032', "'").replace('\uFF07', "'")
    
    # Pulisci spazi multipli ma mantieni la punteggiatura
    extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
    
    # Converti a lowercase per consistenza
    extracted_text = extracted_text.lower()
    
    return extracted_text


def tokenize_with_nltk(text):

    # Segmentazione in frasi
    sentences = nltk.sent_tokenize(text, language='italian')
    tokens = []
    
    for sentence in sentences:
        # Tokenizzazione delle parole mantenendo la punteggiatura
        words = nltk.word_tokenize(sentence, language='italian')
        tokens.extend(words)
    
    return tokens
    

def filter_tokens(tokens):
    
    # Punteggiatura da mantenere (escludiamo parentesi, che verranno rimosse)
    important_punctuation = {'.', ',', '!', '?', ';', ':'}
    quote_tokens = {'"', '“', '”', '«', '»', '„', '‟'}
    paren_tokens = {'(', ')', '[', ']', '{', '}'}

    def normalize_apostrophes(s: str) -> str:
        return s.replace('\u2019', "'").replace('\u2018', "'").replace('\u02BC', "'").replace('\u2032', "'").replace('\uFF07', "'")

    filtered_tokens = []
    i = 0
    while i < len(tokens):
        raw = tokens[i]
        i += 1
        if not raw or not raw.strip():
            continue

        token = normalize_apostrophes(raw.strip())

        # Merge pattern "X'" + "parola" => "X'parola"
        if token.endswith("'") and i < len(tokens):
            nxt_raw = tokens[i]
            nxt = normalize_apostrophes(nxt_raw.strip()) if nxt_raw else ''
            if nxt and re.match(r"^[a-zA-ZÀ-ÿàèéìíîòóùúü]+(?:-[a-zA-ZÀ-ÿàèéìíîòóùúü]+)*$", nxt):
                token = token + nxt
                i += 1

        token = token.lower()

        # Rimuovi doppi apici e parentesi
        if token in quote_tokens or token in paren_tokens:
            continue

        # Mantieni punteggiatura importante
        if token in important_punctuation:
            filtered_tokens.append(token)
            continue

        # Rimuovi solo menzioni e hashtag
        if token.startswith('@') or token.startswith('#'):
            continue

        # Mantieni parole (anche 1 lettera), numeri e parole con apostrofi/trattini
        if re.match(r"^[a-zA-ZÀ-ÿàèéìíîòóùúü'\-]+$", token):
            filtered_tokens.append(token)
            continue

        if token.isdigit():
            filtered_tokens.append(token)
            continue

        # Tieni tutto il resto così com'è
        filtered_tokens.append(token)

    return filtered_tokens





def calculate_bigram_perplexity(sentence_tokens, bigram_probs, unigrams, smoothing_factor=0.1):
    
    
    log_prob_sum = 0.0
    N = len(sentence_tokens) - 1
    V = len(unigrams)
    
    for i in range(len(sentence_tokens) - 1):
        w1, w2 = sentence_tokens[i], sentence_tokens[i + 1]
        bigram = (w1, w2)
        
        if bigram in bigram_probs:
            prob = bigram_probs[bigram]
        else:
            # Smoothing per bigrammi non visti
            unigram_count = unigrams.get(w1, 0)
            prob = smoothing_factor / (unigram_count + smoothing_factor * V)
        
        log_prob_sum += math.log2(max(prob, 1e-10))
    
    return 2 ** (-log_prob_sum / N)


def calculate_trigram_perplexity(sentence_tokens, trigram_probs, bigram_counts, smoothing_factor=0.1):
    """Calcola perplexity trigramma di una frase."""
    if len(sentence_tokens) < 3:
        return float('inf')
    
    log_prob_sum = 0.0
    N = len(sentence_tokens) - 2
    V = len(set(w for _, _, w in trigram_probs.keys()))
    
    for i in range(len(sentence_tokens) - 2):
        w1, w2, w3 = sentence_tokens[i], sentence_tokens[i + 1], sentence_tokens[i + 2]
        trigram = (w1, w2, w3)
        
        if trigram in trigram_probs:
            prob = trigram_probs[trigram]
        else:
            # Smoothing per trigrammi non visti
            bigram = (w1, w2)
            bigram_count = bigram_counts.get(bigram, 0)
            prob = smoothing_factor / (bigram_count + smoothing_factor * V)
        
        log_prob_sum += math.log2(max(prob, 1e-10))
    
    return 2 ** (-log_prob_sum / N)


def evaluate_sentence(sentence, bigram_probs, trigram_probs, unigrams, bigram_counts):
    
    cleaned = clean_text(sentence)
    tokens = tokenize_with_nltk(cleaned)
    filtered_tokens = filter_tokens(tokens)
    sentence_tokens = ['<s>'] + filtered_tokens + ['</s>']
    
    # Calcola perplexity
    bigram_perp = calculate_bigram_perplexity(sentence_tokens, bigram_probs, unigrams)
    trigram_perp = calculate_trigram_perplexity(sentence_tokens, trigram_probs, bigram_counts)
    
    return bigram_perp, trigram_perp



    