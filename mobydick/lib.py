import re
import nltk
import math
from collections import Counter

def clean_text(text):
    
    if not text or not isinstance(text, str):
        return ""


    # Normalize whitespace and lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def filter_tokens(tokens):
    
    if not tokens:
        return []

    important_punctuation = {'.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '"', "'"}
    allowed_short_words = {
        'a', 'an', 'the', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'us', 'him', 'her', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
        'am', 'is', 'are', 'be', 'was', 'were', 'to', 'of', 'in', 'on', 'at', 'by', 'as', 'or', 'and', 'but', 'not',
        'do', 'did', 'does', 'so', 'if', 'no', 'yes', 'up', 'out', 'for', 'from', 'with', 'this', 'that'
    }

    filtered_tokens = []
    for token in tokens:
        if not token or not token.strip():
            continue
        token = token.strip().lower()

        if token in important_punctuation:
            filtered_tokens.append(token)
        elif len(token) > 1 or token in allowed_short_words:
            # English letters, apostrophes, and hyphens
            if re.match(r"^[a-zA-Z][a-zA-Z\-'’]*$", token) or token in allowed_short_words:
                filtered_tokens.append(token)
        elif token.isdigit() and len(token) >= 2:
            filtered_tokens.append(token)

    return filtered_tokens

def tokenize_english(text):
   
    if not text:
        return []

    sentences = nltk.sent_tokenize(text, language='english')
    tokens = []
    for sentence in sentences:
        tokens.extend(nltk.word_tokenize(sentence, language='english'))
    return tokens
    


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
    tokens = tokenize_english(cleaned)
    filtered_tokens = filter_tokens(tokens)
    sentence_tokens = ['<s>'] + filtered_tokens + ['</s>']
    bigram_perp = calculate_bigram_perplexity(sentence_tokens, bigram_probs, unigrams)
    trigram_perp = calculate_trigram_perplexity(sentence_tokens, trigram_probs, bigram_counts)
    return bigram_perp, trigram_perp
