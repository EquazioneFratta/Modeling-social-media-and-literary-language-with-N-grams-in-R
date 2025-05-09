import pandas as pd
import re
from nltk.util import ngrams
from collections import defaultdict, Counter
import random

def clean_text(text):
    """Pulisce il testo rimuovendo URL, menzioni e hashtag"""
    # Rimuovi URL, menzioni e hashtag
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    # Rimuovi numeri
    text = re.sub(r'\d+', '', text)
    # Converti in minuscolo
    text = text.lower()
    # Normalizza spazi bianchi prima di tokenizzare
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_with_punctuation(text):
    """Tokenizza il testo mantenendo la punteggiatura come token separati"""
    # Utilizziamo una regex per separare parole e punteggiatura
    pattern = r'[\w\']+|[.,!?;:()\[\]{}]'
    tokens = re.findall(pattern, text)
    return tokens

def load_and_tokenize_text(username):
    df = pd.read_csv(f"{username}_tweets.csv")
    
    all_tokens = []
    for _, row in df.iterrows():
        if 'text' in row and isinstance(row['text'], str):
            cleaned_text = clean_text(row['text'])
            tokens = tokenize_with_punctuation(cleaned_text)
            all_tokens.extend(tokens)
    
    return all_tokens

def filter_tokens(tokens):
    """Filtra i token rimuovendo stopword ma mantenendo punteggiatura"""
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
    
    # Considera la punteggiatura come token indipendenti
    filtered_tokens = []
    for token in tokens:
        if token in punctuation:
            filtered_tokens.append(token)
        elif token not in stop_words and len(token) > 1:
            filtered_tokens.append(token)
    
    return filtered_tokens

def build_ngram_models(tokens):
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))

    # Modelli di transizione
    bigram_model = defaultdict(list)
    trigram_model = defaultdict(list)

    for w1, w2 in bigrams:
        bigram_model[w1].append(w2)

    for w1, w2, w3 in trigrams:
        trigram_model[(w1, w2)].append(w3)

    return bigram_model, trigram_model

def generate_sentence_with_model(model, is_trigram=True, max_length=40):
    """Genera una frase usando il modello n-gram specificato fino a un punto o max_length token"""
    end_punctuation = {'.', '!', '?'}
    
    if is_trigram:
        # Per il modello trigram, iniziamo con una coppia di parole
        # Escludiamo le coppie che iniziano con punteggiatura o includono fine frase
        valid_starts = [key for key in model.keys() 
                      if not any(p in key[0] for p in end_punctuation) 
                      and not any(p in key[1] for p in end_punctuation)]
        
        if not valid_starts:
            valid_starts = list(model.keys())
        
        start = random.choice(valid_starts)
        sentence = list(start)  # Converte la tupla in lista
        
        while len(sentence) < max_length:
            current_key = (sentence[-2], sentence[-1])
            
            if current_key not in model:
                break
                
            next_token = random.choice(model[current_key])
            sentence.append(next_token)
            
            # Interrompi se abbiamo raggiunto un segno di fine frase
            if next_token in end_punctuation:
                break
    else:
        # Per il modello bigram, iniziamo con una singola parola
        # Escludiamo le parole che sono punteggiatura di fine frase
        valid_starts = [word for word in model.keys() 
                      if word not in end_punctuation]
        
        if not valid_starts:
            valid_starts = list(model.keys())
            
        start_word = random.choice(valid_starts)
        sentence = [start_word]
        
        while len(sentence) < max_length:
            current_word = sentence[-1]
            
            if current_word not in model:
                break
                
            next_token = random.choice(model[current_word])
            sentence.append(next_token)
            
            # Interrompi se abbiamo raggiunto un segno di fine frase
            if next_token in end_punctuation:
                break
    
    # Formatta la frase per la visualizzazione
    formatted_sentence = format_sentence_with_punctuation(sentence)
    return formatted_sentence

def format_sentence_with_punctuation(tokens):
    """Formatta una lista di token in una frase leggibile"""
    # Separa i token in parole e punteggiatura
    words = []
    current_word = ""
    
    for i, token in enumerate(tokens):
        if token in {'.', ',', '!', '?', ';', ':', ')', ']', '}'}:
            # Punteggiatura che va attaccata alla parola precedente
            if current_word:
                words.append(current_word)
                current_word = ""
            words.append(token)
        elif token in {'(', '[', '{'}:
            # Punteggiatura che va attaccata alla parola successiva
            if current_word:
                words.append(current_word)
                current_word = ""
            words.append(token)
        else:
            # Parola normale
            if current_word:
                words.append(current_word)
            current_word = token
    
    if current_word:
        words.append(current_word)
    
    # Formatta la frase con la punteggiatura corretta
    text = ""
    for i, word in enumerate(words):
        if word in {'.', ',', '!', '?', ';', ':', ')', ']', '}'}:
            # Nessuno spazio prima della punteggiatura
            text = text.rstrip() + word + " "
        elif word in {'(', '[', '{'}:
            # Nessuno spazio dopo la parentesi aperta
            text += word
        elif i > 0 and words[i-1] in {'(', '[', '{'}:
            # Nessuno spazio dopo la parentesi aperta
            text += word + " "
        else:
            text += word + " "
    
    return text.strip()

def generate_sentences(username, n_sentences=5, use_trigram=True):
    """Genera n_sentences frasi usando il modello di n-grammi specificato"""
    # Carica e tokenizza il testo
    tokens = load_and_tokenize_text(username)
    
    # Filtra i token (rimuovi stopwords, mantieni punteggiatura)
    filtered_tokens = filter_tokens(tokens)
    
    # Costruisci i modelli di n-grammi
    bigram_model, trigram_model = build_ngram_models(filtered_tokens)
    
    # Seleziona il modello da usare
    model = trigram_model if use_trigram else bigram_model
    
    print(f"\nGenerazione frasi per {username} usando {'trigrammi' if use_trigram else 'bigrammi'}:\n{'='*50}")
    for i in range(n_sentences):
        sentence = generate_sentence_with_model(model, is_trigram=use_trigram)
        print(f"- {sentence}")
    print()

# Esempio di utilizzo
if __name__ == "__main__":
    usernames = ["matteosalvinimi", "GiorgiaMeloni"]
    
    for username in usernames:
        # Genera 5 frasi usando trigrammi
        generate_sentences(username, n_sentences=5, use_trigram=True)

        # Genera 3 frasi usando bigrammi
        generate_sentences(username, n_sentences=3, use_trigram=False)