import re
import nltk

def clean_text(text):
    
    if not text or not isinstance(text, str):
        return ""


    # Normalize whitespace and lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def tokenize_with_punctuation(text):
  
    pattern = r"\w+(?:'\w+)?|[.,!?;:()\[\]{}]"
    return re.findall(pattern, text)


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
    try:
        sentences = nltk.sent_tokenize(text, language='english')
        tokens = []
        for sentence in sentences:
            tokens.extend(nltk.word_tokenize(sentence, language='english'))
        return tokens
    except Exception:
        return tokenize_with_punctuation(text)
