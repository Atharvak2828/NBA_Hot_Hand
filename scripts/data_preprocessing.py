import pandas as pd
import kagglehub
import os
import shutil

# 1. Setup Folders
os.makedirs('data/processed', exist_ok=True)

# 2. Define the Top 15 high-volume shooters
TOP_15 = [
    "LeBron James", "Kobe Bryant", "Kevin Durant", "James Harden", 
    "Russell Westbrook", "Stephen Curry", "Carmelo Anthony", "Dirk Nowitzki", 
    "Chris Paul", "Dwyane Wade", "DeMar DeRozan", "LaMarcus Aldridge", 
    "Damian Lillard", "Paul George", "Joe Johnson"
]

def download_and_clean():
    print("Fetching data from Kaggle...")
    # This pulls the latest version of the NBA Shots 04-25 dataset
    path = kagglehub.dataset_download("mexwell/nba-shots")
    
    # The dataset usually contains multiple CSVs (one per year)
    # For a beginner project, let's combine the last 5-10 years to keep it fast
    all_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    combined_list = []

    print(f"Found {len(all_files)} season files. Filtering top players...")
    
    for file in all_files:
        file_path = os.path.join(path, file)
        temp_df = pd.read_csv(file_path)
        
        # Standardize column names (Kaggle datasets can be messy)
        temp_df.columns = [c.upper() for c in temp_df.columns]
        
        # Filter
        filtered = temp_df[temp_df['PLAYER_NAME'].isin(TOP_15)]
        combined_list.append(filtered[['PLAYER_NAME', 'SHOT_MADE', 'SHOT_TYPE', 'GAME_DATE']])

    # Merge everything into one clean file
    final_df = pd.concat(combined_list, ignore_index=True)
    
    # Sort by date so the HMM sees the shots in order
    final_df['GAME_DATE'] = pd.to_datetime(final_df['GAME_DATE'])
    final_df = final_df.sort_values(['PLAYER_NAME', 'GAME_DATE'])
    
    final_df.to_csv('data/processed/top_15_shots.csv', index=False)
    print("Done! Data saved to data/processed/top_15_shots.csv")

if __name__ == "__main__":
    download_and_clean()