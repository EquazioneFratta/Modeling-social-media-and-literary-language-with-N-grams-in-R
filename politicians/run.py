import requests
import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.util import ngrams
from collections import Counter
import time
from datetime import datetime

def setup_bearer_auth(bearer_token):
    """Configura l'autenticazione Bearer per l'API v2"""
    return {"Authorization": f"Bearer {bearer_token}"}

def get_user_id(username, bearer_token):
    """Ottiene l'ID utente dal nome utente utilizzando l'API v2"""
    url = f"https://api.twitter.com/2/users/by/username/{username}"
    headers = setup_bearer_auth(bearer_token)
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Errore nell'ottenere l'ID utente per {username}: {response.status_code}")
        print(response.text)
        return None
    
    data = response.json()
    return data.get('data', {}).get('id')

def get_user_tweets(user_id, bearer_token, max_results=100, max_pages=10):
    """Estrae tweet dall'utente specificato usando l'API v2"""
    url = f"https://api.twitter.com/2/users/{user_id}/tweets"
    headers = setup_bearer_auth(bearer_token)
    params = {
        "max_results": max_results,
        "tweet.fields": "created_at,text",
        "exclude": "retweets,replies"
    }
    
    all_tweets = []
    next_token = None
    pages = 0
    
    while pages < max_pages:
        if next_token:
            params["pagination_token"] = next_token
            
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Errore nell'estrazione dei tweet: {response.status_code}")
            print(response.text)
            break
            
        data = response.json()
        
        # Estrai i tweet
        tweets = data.get('data', [])
        if not tweets:
            break
            
        for tweet in tweets:
            created_at = tweet.get('created_at')
            text = tweet.get('text')
            all_tweets.append((created_at, text))
        
        # Verifica se ci sono altri tweet
        meta = data.get('meta', {})
        next_token = meta.get('next_token')
        
        if not next_token:
            break
            
        pages += 1
        # Aggiungi un ritardo per rispettare i limiti di frequenza dell'API
        time.sleep(1)
    
    return all_tweets

def clean_text(text):
    """Pulisce il testo rimuovendo URL, menzioni, hashtag e caratteri speciali"""
    # Rimuovi URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Rimuovi menzioni (@username)
    text = re.sub(r'@\w+', '', text)
    # Rimuovi hashtag
    text = re.sub(r'#\w+', '', text)
    # Rimuovi caratteri speciali e numeri
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Converti in minuscolo
    text = text.lower()
    # Rimuovi spazi multipli
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_ngrams(text, n):
    """Genera n-grammi dal testo dato"""
    # Tokenizzazione semplice
    tokens = text.split()
    
    # Lista base di stopwords italiane
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
    
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    
    # Genera n-grammi
    n_grams = list(ngrams(tokens, n))
    return n_grams

def analyze_tweets_v2(bearer_token, usernames, max_results=100, max_pages=5):
    """Estrae e analizza i tweet degli utenti specificati usando l'API v2"""
    results = {}
    
    # Per ogni utente
    for username in usernames:
        print(f"Analisi dei tweet di {username}...")
        
        # Ottieni l'ID utente
        user_id = get_user_id(username, bearer_token)
        if not user_id:
            print(f"Impossibile trovare l'ID utente per {username}. Passaggio al prossimo utente.")
            continue
            
        print(f"ID utente per {username}: {user_id}")
        
        # Estrai i tweet
        tweets = get_user_tweets(user_id, bearer_token, max_results, max_pages)
        
        if not tweets:
            print(f"Nessun tweet trovato per {username}. Passaggio al prossimo utente.")
            continue
            
        print(f"Estratti {len(tweets)} tweet per {username}")
        
        # Prepara dataframe
        df = pd.DataFrame(tweets, columns=['created_at', 'text'])
        
        # Pulisci i testi
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        # Concatena tutti i testi per l'analisi degli n-grammi
        all_text = ' '.join(df['cleaned_text'].tolist())
        
        # Genera bigrammi e trigrammi
        bigrams = generate_ngrams(all_text, 2)
        trigrams = generate_ngrams(all_text, 3)
        
        # Conta le frequenze
        bigram_freq = Counter(bigrams)
        trigram_freq = Counter(trigrams)
        
        # Prendi i top 20
        top_bigrams = bigram_freq.most_common(20)
        top_trigrams = trigram_freq.most_common(20)
        
        # Salva risultati
        results[username] = {
            'tweets': df,
            'bigrams': top_bigrams,
            'trigrams': top_trigrams
        }
        
        # Visualizza i risultati
        print(f"\nTop 20 bigrammi per {username}:")
        for bg, count in top_bigrams:
            print(f"{' '.join(bg)}: {count}")
        
        print(f"\nTop 20 trigrammi per {username}:")
        for tg, count in top_trigrams:
            print(f"{' '.join(tg)}: {count}")
        
        # Salva in CSV
        df.to_csv(f"{username}_tweets.csv", index=False)
        
        # Crea e salva i grafici per bigrammi
        plt.figure(figsize=(12, 8))
        bg_labels = [' '.join(bg[0]) for bg in top_bigrams]
        bg_values = [bg[1] for bg in top_bigrams]
        plt.barh(bg_labels, bg_values)
        plt.title(f"Top 20 bigrammi per {username}")
        plt.xlabel("Frequenza")
        plt.tight_layout()
        plt.savefig(f"{username}_bigrams.png")
        
        # Crea e salva i grafici per trigrammi
        plt.figure(figsize=(12, 8))
        tg_labels = [' '.join(tg[0]) for tg in top_trigrams]
        tg_values = [tg[1] for tg in top_trigrams]
        plt.barh(tg_labels, tg_values)
        plt.title(f"Top 20 trigrammi per {username}")
        plt.xlabel("Frequenza")
        plt.tight_layout()
        plt.savefig(f"{username}_trigrams.png")
    
    return results

# Esempio di utilizzo
if __name__ == "__main__":
    # Inserisci qui il tuo bearer token dell'API v2
    bearer_token = "AAAAAAAAAAAAAAAAAAAAABI%2B1AEAAAAAgyoRcjP0XkSh2%2F4wlPFIKiOZOi4%3DKtj4HFtQBVkZYUAaEQ2Wyzjt7vYAJ45S9FCwYkSdmb8exieWie"
    
    # Usernames da analizzare
    usernames = ["matteosalvinimi", "GiorgiaMeloni"]
    
    # Esegui l'analisi con API v2
    results = analyze_tweets_v2(bearer_token, usernames)
    
    print("\nAnalisi completata! I risultati sono stati salvati in file CSV e immagini PNG.")