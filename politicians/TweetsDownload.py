import time
import requests
import pandas as pd
from datetime import datetime
import json

def setup_bearer_auth(bearer_token):
    return {"Authorization": f"Bearer {bearer_token}"}

def get_user_id(username, bearer_token):
    """Ottiene l'ID utente dal nome utente"""
    url = f"https://api.twitter.com/2/users/by/username/{username}"
    headers = setup_bearer_auth(bearer_token)
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"Errore nell'ottenere l'ID utente per {username}: {response.status_code}")
            return None
        
        data = response.json()
        user_data = data.get('data')
        
        if user_data:
            return user_data.get('id')
        else:
            print(f"Utente {username} non trovato")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Errore di connessione per {username}: {e}")
        return None

def download_tweets_exact_count(user_id, username, bearer_token, target_count=250):
    """Scarica esattamente il numero specificato di tweet"""
    url = f"https://api.twitter.com/2/users/{user_id}/tweets"
    headers = setup_bearer_auth(bearer_token)
    
    # Parametri minimali - solo quello che serve
    params = {
        "max_results": 100,  # Massimo consentito dall'API
        "tweet.fields": "created_at,text",
        "exclude": "retweets,replies"  # Esclude retweet e risposte
    }
    
    all_tweets = []
    next_token = None
    total_requests = 0
    
    print(f"Scaricamento {target_count} tweet per {username}...")
    
    while len(all_tweets) < target_count:
        if next_token:
            params["pagination_token"] = next_token
            
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            total_requests += 1
            
            if response.status_code == 429:  # Rate limit
                print("Rate limit raggiunto. Attesa...")
                time.sleep(900)  # 15 minuti
                continue
                
            if response.status_code != 200:
                print(f"Errore API: {response.status_code}")
                break
                
            data = response.json()
            tweets = data.get('data', [])
            
            if not tweets:
                print(f"Non ci sono più tweet disponibili. Trovati solo {len(all_tweets)} tweet.")
                break
            
            # Aggiungi tweet fino al limite
            remaining_needed = target_count - len(all_tweets)
            tweets_to_add = tweets[:remaining_needed]
            
            for tweet in tweets_to_add:
                tweet_data = {
                    'created_at': tweet.get('created_at'),
                    'text': tweet.get('text', '').strip()
                }
                # Accetta tutti i tweet (solo controllo che non sia completamente vuoto)
                if tweet_data['text']:
                    all_tweets.append(tweet_data)
            
            print(f"Progresso: {len(all_tweets)}/{target_count} tweet")
            
            # Se abbiamo raggiunto il target, fermati
            if len(all_tweets) >= target_count:
                break
            
            # Gestione paginazione
            meta = data.get('meta', {})
            next_token = meta.get('next_token')
            
            if not next_token:
                print(f"Fine dei tweet disponibili. Scaricati {len(all_tweets)} tweet.")
                break
            
            # Pausa per rispettare rate limits
            time.sleep(1)
            
        except requests.exceptions.RequestException as e:
            print(f"Errore di connessione: {e}")
            time.sleep(5)
            continue
        except json.JSONDecodeError as e:
            print(f"Errore parsing JSON: {e}")
            break
    
    print(f"Download completato per {username}: {len(all_tweets)} tweet, {total_requests} richieste API")
    return all_tweets

def save_training_dataset(tweets, username):
    """Salva il dataset in formato CSV ottimizzato per ML"""
    if not tweets:
        print(f"Nessun tweet da salvare per {username}")
        return False
    
    try:
        df = pd.DataFrame(tweets)
        
        # Converti timestamp in formato data leggibile
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['date'] = df['created_at'].dt.date
        
        # Dataset finale: solo data e testo
        final_df = df[['date', 'text']].copy()
        
        # Nome file semplice
        filename = f"{username}_training_data.csv"
        
        # Salva con encoding UTF-8 per caratteri speciali
        final_df.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"✅ Dataset salvato: {filename}")
        print(f"📊 Righe nel dataset: {len(final_df)}")
        
        # Statistiche rapide
        if len(final_df) > 0:
            print(f"📅 Periodo: {final_df['date'].min()} - {final_df['date'].max()}")
            avg_length = final_df['text'].str.len().mean()
            print(f"📝 Lunghezza media testo: {avg_length:.1f} caratteri")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore nel salvataggio per {username}: {e}")
        return False

def download_training_data(bearer_token, usernames, tweets_per_user=50):
    """Funzione principale per scaricare dati di training"""
    total_tweets_needed = tweets_per_user * len(usernames)
    
    print(f"🚀 Inizio download di {tweets_per_user} tweet per {len(usernames)} utenti")
    print(f"📊 Totale tweet da scaricare: {total_tweets_needed}")
    
    # Controllo limite mensile
    if total_tweets_needed > 100:
        print(f"⚠️  ATTENZIONE: Stai per scaricare {total_tweets_needed} tweet")
        print(f"🚨 Il tuo limite mensile è 100 tweet")
        print(f"💡 Suggerimento: riduci tweets_per_user a {100 // len(usernames)} per utente")
        
        response = input("Vuoi continuare comunque? (y/n): ")
        if response.lower() != 'y':
            print("❌ Download annullato")
            return [], []
    
    successful = []
    failed = []
    tweets_downloaded = 0
    
    for i, username in enumerate(usernames, 1):
        print(f"\n{'='*60}")
        print(f"👤 Utente {i}/{len(usernames)}: {username}")
        print(f"📊 Tweet scaricati finora: {tweets_downloaded}/100 (limite mensile)")
        print(f"{'='*60}")
        
        # Calcola quanti tweet possiamo ancora scaricare
        remaining_limit = 100 - tweets_downloaded
        tweets_to_download = min(tweets_per_user, remaining_limit)
        
        if tweets_to_download <= 0:
            print("🚨 Limite mensile raggiunto! Impossibile scaricare altri tweet.")
            failed.append(f"{username} (limite raggiunto)")
            continue
        
        if tweets_to_download < tweets_per_user:
            print(f"⚠️  Posso scaricare solo {tweets_to_download} tweet per {username} (limite residuo)")
        
        # Ottieni user ID
        user_id = get_user_id(username, bearer_token)
        if not user_id:
            failed.append(username)
            continue
        
        # Scarica i tweet disponibili
        tweets = download_tweets_exact_count(user_id, username, bearer_token, tweets_to_download)
        tweets_downloaded += len(tweets)
        
        if len(tweets) == 0:
            failed.append(username)
            continue
        
        # Salva il dataset
        if save_training_dataset(tweets, username):
            successful.append(f"{username} ({len(tweets)} tweet)")
        else:
            failed.append(username)
        
        # Controlla se abbiamo raggiunto il limite
        if tweets_downloaded >= 100:
            print("🚨 Limite mensile di 100 tweet raggiunto!")
            break
        
        # Pausa tra utenti
        if i < len(usernames) and tweets_downloaded < 100:
            print("⏳ Pausa di 10 secondi...")
            time.sleep(10)
    
    # Riepilogo finale
    print(f"\n{'='*60}")
    print("📋 RIEPILOGO FINALE")
    print(f"{'='*60}")
    print(f"📊 Tweet totali scaricati: {tweets_downloaded}/100")
    print(f"📊 Limite residuo questo mese: {100 - tweets_downloaded}")
    print(f"✅ Download riusciti: {len(successful)}")
    for user in successful:
        print(f"   - {user}")
    
    if failed:
        print(f"❌ Download falliti: {len(failed)}")
        for user in failed:
            print(f"   - {user}")
    
    return successful, failed

def test_api_connection(bearer_token):
    """Testa la connessione all'API con diversi endpoint"""
    print("🔍 Test connessione API...")
    headers = setup_bearer_auth(bearer_token)
    
    # Test 1: Endpoint più semplice
    print("Test 1: Verifica token base...")
    url1 = "https://api.twitter.com/2/users/by/username/twitter"
    try:
        response = requests.get(url1, headers=headers, timeout=30)
        print(f"Status code: {response.status_code}")
        print(f"Risposta: {response.text[:200]}...")
        
        if response.status_code == 200:
            print("✅ Token valido!")
        elif response.status_code == 401:
            print("❌ Token non valido o scaduto")
            return False
        elif response.status_code == 403:
            print("❌ Token valido ma permessi insufficienti")
            print("Possibili cause:")
            print("- Account Twitter gratuito (serve piano a pagamento)")
            print("- App non ha i permessi necessari")
            print("- Limitazioni geografiche")
            
            # Prova un test più semplice
            print("\nTest 2: Verifica informazioni account...")
            url2 = "https://api.twitter.com/2/users/me"
            response2 = requests.get(url2, headers=headers, timeout=30)
            print(f"Status /users/me: {response2.status_code}")
            
            if response2.status_code == 200:
                print("✅ Accesso al proprio profilo funziona")
                user_info = response2.json()
                print(f"Account: @{user_info.get('data', {}).get('username', 'N/A')}")
            elif response2.status_code == 403:
                print("❌ Anche l'accesso al proprio profilo è bloccato")
                print("Questo suggerisce limitazioni dell'account o del piano")
            
            return False
        else:
            print(f"❌ Errore imprevisto: {response.status_code}")
            print(f"Dettagli: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Errore connessione: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Configurazione
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAJxk1AEAAAAApXbavtsUNRBehh1gMrMjKMg3Qx4%3DdFx4pQqLfwIXlPhqwFA9HptfYoBdjq76lkCSf1X749R3hN5PCY"
    usernames = ["matteosalvinimi", "GiorgiaMeloni"]
    tweets_per_user = 50  # Ridotto per rispettare il limite di 100/mese
    
    print("🤖 Tweet Downloader per ML Training")
    print(f"🎯 Target: {tweets_per_user} tweet per utente")
    print(f"📋 Totale: {tweets_per_user * len(usernames)} tweet")
    print(f"⚠️  Limite mensile: 100 tweet")
    
    # Test connessione
    if test_api_connection(bearer_token):
        # Avvio download
        successful, failed = download_training_data(bearer_token, usernames, tweets_per_user)
        
        if successful:
            print(f"\n🎉 Download completato! Controlla i file CSV generati.")
        else:
            print(f"\n😞 Nessun download riuscito.")
    else:
        print("❌ Impossibile connettersi all'API. Controlla il token.")