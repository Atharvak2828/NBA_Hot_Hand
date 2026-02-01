import pandas as pd
import numpy as np
import kagglehub
import os
from hmmlearn import hmm

def process_everything():
    print("🚀 Fetching and Processing...")
    path = kagglehub.dataset_download("mexwell/nba-shots")
    all_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    
    players = ["LeBron James", "Kobe Bryant", "Kevin Durant", "James Harden", 
               "Russell Westbrook", "Stephen Curry", "Carmelo Anthony", "Dirk Nowitzki", 
               "Chris Paul", "Dwyane Wade", "DeMar DeRozan", "LaMarcus Aldridge", 
               "Damian Lillard", "Paul George", "Joe Johnson"]

    df_list = []
    for f in all_files:
        temp = pd.read_csv(f)
        temp.columns = [c.upper() for c in temp.columns]
        # Adding Game ID to estimate games played
        subset = temp[temp['PLAYER_NAME'].isin(players)][['PLAYER_NAME', 'SHOT_MADE', 'SHOT_TYPE', 'GAME_DATE', 'GAME_ID']]
        df_list.append(subset)
    
    df = pd.concat(df_list).sort_values(['PLAYER_NAME', 'GAME_DATE'])
    df['SHOT_MADE'] = df['SHOT_MADE'].astype(int)

    final_results = []
    for player in players:
        print(f"🧠 Analyzing {player}...")
        p_df = df[df['PLAYER_NAME'] == player].copy()
        X = p_df['SHOT_MADE'].values.reshape(-1, 1)
        
        if len(X) > 100:
            model = hmm.CategoricalHMM(n_components=3, n_iter=200, random_state=42)
            model.fit(X)
            states = model.predict(X)
            p_df['HMM_STATE'] = states
            
            state_means = p_df.groupby('HMM_STATE')['SHOT_MADE'].mean()
            hot_state_label = state_means.idxmax()
            p_df['IS_HOT_ZONE'] = (p_df['HMM_STATE'] == hot_state_label).astype(int)
        else:
            p_df['IS_HOT_ZONE'] = 0
            
        final_results.append(p_df)

    os.makedirs('data/processed', exist_ok=True)
    pd.concat(final_results).to_parquet('data/processed/nba_data_optimized.parquet')
    print("✅ Done! Optimized Parquet created.")

if __name__ == "__main__":
    process_everything()