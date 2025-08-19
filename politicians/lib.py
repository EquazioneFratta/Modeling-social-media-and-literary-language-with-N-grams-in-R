import re
import nltk

# Decommenta per il primo utilizzo
# nltk.download('punkt')

def clean_text(text):
    """
    Pulisce il testo mantenendo la struttura originale e la punteggiatura.
    Estrae solo il contenuto tra virgolette se presente, altrimenti usa tutto il testo.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Prima prova a estrarre il testo tra virgolette usando caratteri ASCII sicuri
    extracted_text = None
    
    # Pattern per virgolette normali
    match = re.search(r'"([^"]*)"', text)
    if match and len(match.group(1).strip()) > 5:
        extracted_text = match.group(1)
    
    # Se non troviamo virgolette normali, prova altri delimitatori comuni
    if not extracted_text:
        # Cerca contenuto dopo i due punti seguiti da virgolette o spazio
        match = re.search(r':\s*["\']?([^"\']*)["\']?$', text)
        if match and len(match.group(1).strip()) > 10:
            extracted_text = match.group(1)
    
    # Se non troviamo pattern specifici, usa tutto il testo
    if not extracted_text or len(extracted_text.strip()) < 5:
        extracted_text = text
        
        # Rimuovi pattern comuni di introduzione ai tweet
        prefixes_to_remove = [
            r'^[^a-zA-Z]*(?:Oggi|Ieri|Stamattina|Stasera)\s+dalle?\s+ore\s+\d+[^.]*[.]?\s*',
            r'^[^a-zA-Z]*Il mio\s+[^.]*[.]?\s*',
            r'^[^a-zA-Z]*La mia\s+[^.]*[.]?\s*',
            r'^[^a-zA-Z]*Dal\s+[^.]*[.]?\s*',
            r'^[^a-zA-Z]*Dalla\s+[^.]*[.]?\s*'
        ]
        
        for prefix in prefixes_to_remove:
            extracted_text = re.sub(prefix, '', extracted_text, flags=re.IGNORECASE)
            if len(extracted_text.strip()) < len(text) * 0.3:  # Se rimuove troppo, ripristina
                extracted_text = text
                break
    
    if not extracted_text or len(extracted_text.strip()) < 3:
        return ""
    
    # Rimuovi link HTTP/HTTPS
    extracted_text = re.sub(r'https?://\S+', '', extracted_text)
    extracted_text = re.sub(r'www\.\S+', '', extracted_text)
    
    # Rimuovi pattern di link comuni
    extracted_text = re.sub(r't\.co/\S+', '', extracted_text)
    
    # Rimuovi emoji usando range Unicode sicuri
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
    
    # Pulisci spazi multipli ma mantieni la punteggiatura
    extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
    
    # Converti a lowercase per consistenza
    extracted_text = extracted_text.lower()
    
    return extracted_text


def tokenize_with_nltk(text):
    """
    Tokenizza il testo mantenendo la punteggiatura come token separati.
    """
    if not text:
        return []
    
    try:
        # Segmentazione in frasi
        sentences = nltk.sent_tokenize(text, language='italian')
        tokens = []
        
        for sentence in sentences:
            # Tokenizzazione delle parole mantenendo la punteggiatura
            words = nltk.word_tokenize(sentence, language='italian')
            tokens.extend(words)
        
        return tokens
    except Exception as e:
        print(f"Errore nella tokenizzazione: {e}")
        # Fallback a tokenizzazione semplice
        return simple_tokenize(text)


def simple_tokenize(text):
    """
    Tokenizzazione di fallback semplice che mantiene la punteggiatura.
    """
    if not text:
        return []
    
    # Aggiungi spazi attorno alla punteggiatura
    text = re.sub(r'([.!?;:,])', r' \1 ', text)
    text = re.sub(r'([()[\]{}])', r' \1 ', text)
    
    # Dividi per spazi e filtra token vuoti
    tokens = [token for token in text.split() if token.strip()]
    
    return tokens


def filter_tokens(tokens):
    """
    Filtra i token mantenendo parole significative e punteggiatura importante.
    """
    if not tokens:
        return []
    
    # Punteggiatura da mantenere
    important_punctuation = {'.', ',', '!', '?', ';', ':', '(', ')', '[', ']'}
    
    # Parole corte significative da mantenere
    allowed_short_words = {
        'è', 'e', 'a', 'o', 'i', 'il', 'la', 'lo', 'le', 'di', 'da', 'in', 'su', 'per', 'con', 'tra', 'fra',
        'si', 'ci', 'vi', 'mi', 'ti', 'ne', 'se', 'ma', 'no', 'sì', 'me', 'te', 'noi', 'voi', 'lui', 'lei',
        'un', 'una', 'uno', 'del', 'dei', 'nel', 'sul', 'col', 'dal', 'al', 'più', 'già', 'mai', 'poi',
        'non', 'chi', 'che', 'cui', 'come', 'dove', 'quando', 'qui', 'qua', 'lì', 'là', 'su', 'giù', 'va',
        'ho', 'hai', 'ha', 'abbiamo', 'avete', 'hanno', 'sono', 'sei', 'siamo', 'siete', 'era', 'ero'
    }
    
    filtered_tokens = []
    for token in tokens:
        # Salta token vuoti
        if not token or not token.strip():
            continue
            
        token = token.strip().lower()
        
        # Mantieni punteggiatura importante
        if token in important_punctuation:
            filtered_tokens.append(token)
        # Mantieni parole di lunghezza > 1 o parole corte significative
        elif len(token) > 1 or token in allowed_short_words:
            # Filtra solo caratteri alfabetici italiani, apostrofi e trattini
            if re.match(r'^[a-zA-ZÀ-ÿàèéìíîòóùúü\'\-]+$', token) or token in allowed_short_words:
                filtered_tokens.append(token)
        # Mantieni anche numeri se significativi
        elif token.isdigit() and len(token) >= 2:
            filtered_tokens.append(token)
        #stampa i token che non sono stati filtrati
        else:
            print(f"Token filtrato: {token}")
    return filtered_tokens