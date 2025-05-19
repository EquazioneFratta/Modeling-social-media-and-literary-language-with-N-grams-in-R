import re

def clean_text(text):
    #rimuove URL, menzioni, hashtag e numeri
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)

    text = text.lower()
    
    #rimuove spazi 
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_with_punctuation(text):
    #separa parole e punteggiatura
    pattern = r'[\w\']+|[.,!?;:()\[\]{}]'
    tokens = re.findall(pattern, text)
    return tokens


def filter_tokens(tokens):
    #stop_words = {'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una',
    #             'e', 'ed', 'ma', 'se', 'perché', 'che', 'di', 'a', 'da',
    #             'in', 'con', 'su', 'per', 'tra', 'fra', 'è', 'sono', 'sei',
    #             'del', 'della', 'dei', 'degli', 'delle', 'nel', 'nella',
    #             'nei', 'negli', 'nelle', 'sul', 'sulla', 'sui', 'sugli',
    #             'sulle', 'non', 'si', 'ci', 'mi', 'ti', 'vi', 'lo', 'la',
    #             'li', 'le', 'gli', 'ne', 'al', 'dal', 'del', 'più', 'che',
    #             'chi', 'cui', 'ha', 'ho', 'hai', 'hanno', 'come', 'dove',
    #             'quando', 'però', 'quindi', 'questo', 'questa', 'questi',
    #             'queste', 'quello', 'quella', 'quelli', 'quelle'}
    
    punctuation = {'.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}'}
    
    filtered_tokens = []
    for token in tokens:
        if token in punctuation:
            filtered_tokens.append(token)
        elif len(token) > 1:
            filtered_tokens.append(token)
    
    return filtered_tokens

def tokenize_english(text):
    """
    Tokenizza testo inglese mantenendo le contrazioni come hasn't, I'm, they'll.
    Separa punteggiatura ma non spezza le parole con apostrofo.
    """
    pattern = r"\w+(?:'\w+)?|[.,!?;:()\[\]{}]"
    return re.findall(pattern, text)
