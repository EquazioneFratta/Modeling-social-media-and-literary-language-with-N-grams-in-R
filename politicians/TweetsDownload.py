import time
import requests
import pandas as pd

from lib import clean_text
from nltk.util import ngrams
from collections import Counter

def setup_bearer_auth(bearer_token):
    return {"Authorization": f"Bearer {bearer_token}"}

def get_user_id(username, bearer_token):
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
        
        tweets = data.get('data', [])
        if not tweets:
            break
            
        for tweet in tweets:
            created_at = tweet.get('created_at')
            text = tweet.get('text')
            all_tweets.append((created_at, text))
        
        meta = data.get('meta', {})
        next_token = meta.get('next_token')
        
        if not next_token:
            break
            
        pages += 1
        time.sleep(1)
    
    return all_tweets

def generate_ngrams(text, n):
    tokens = text.split()
    
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
    
    n_grams = list(ngrams(tokens, n))
    return n_grams

def analyze_tweets_v2(bearer_token, usernames, max_results=100, max_pages=5):
    results = {}
    
    for username in usernames:
        
        user_id = get_user_id(username, bearer_token)
        if not user_id:
            print(f"Impossibile trovare l'ID utente per {username}. Passaggio al prossimo utente.")
            continue
        
        tweets = get_user_tweets(user_id, bearer_token, max_results, max_pages)
        
        if not tweets:
            print(f"Nessun tweet trovato per {username}. Passaggio al prossimo utente.")
            continue
            
        print(f"Estratti {len(tweets)} tweet per {username}")
        
        df = pd.DataFrame(tweets, columns=['created_at', 'text'])
        
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        all_text = ' '.join(df['cleaned_text'].tolist())
        
        bigrams = generate_ngrams(all_text, 2)
        trigrams = generate_ngrams(all_text, 3)
        
        bigram_freq = Counter(bigrams)
        trigram_freq = Counter(trigrams)
        
        top_bigrams = bigram_freq.most_common(20)
        top_trigrams = trigram_freq.most_common(20)
        
        results[username] = {
            'tweets': df,
            'bigrams': top_bigrams,
            'trigrams': top_trigrams
        }
        
        print(f"\nTop 20 bigrammi per {username}:")
        for bg, count in top_bigrams:
            print(f"{' '.join(bg)}: {count}")
        
        print(f"\nTop 20 trigrammi per {username}:")
        for tg, count in top_trigrams:
            print(f"{' '.join(tg)}: {count}")
        
        df.to_csv(f"{username}_tweets.csv", index=False)
        
        
    return results

if __name__ == "__main__":
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAJxk1AEAAAAApXbavtsUNRBehh1gMrMjKMg3Qx4%3DdFx4pQqLfwIXlPhqwFA9HptfYoBdjq76lkCSf1X749R3hN5PCY"
    
    usernames = ["matteosalvinimi", "GiorgiaMeloni"]

    for username in usernames:
        try:
            df = pd.read_csv(f"{username}_tweets.csv")
        except FileNotFoundError:
            results = analyze_tweets_v2(bearer_token, [username])
