import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="NBA Hot Hand Dashboard", layout="wide")

@st.cache_data
def load_fast_data():
    return pd.read_parquet('data/processed/nba_data_optimized.parquet')

try:
    df = load_fast_data()
except:
    st.error("Please run 'python process_data.py' first!")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("NBA Analytics")
player = st.sidebar.selectbox("Select Legend", sorted(df['PLAYER_NAME'].unique()))
shot_type = st.sidebar.radio("Shot Type", ["All", "2PT Field Goal", "3PT Field Goal"])

# Filter
view_df = df[df['PLAYER_NAME'] == player].copy()
if shot_type != "All":
    view_df = view_df[view_df['SHOT_TYPE'] == shot_type]

# --- 1. TOP STATS BAR (New!) ---
st.title(f"🏀 {player}: Career Profile")

# Calculations for the top bar
total_shots = len(view_df)
total_makes = view_df['SHOT_MADE'].sum()
est_games = view_df['GAME_ID'].nunique()
# Est Points: 2s vs 3s logic
pts_2 = (view_df[view_df['SHOT_TYPE'] == '2PT Field Goal']['SHOT_MADE'].sum() * 2)
pts_3 = (view_df[view_df['SHOT_TYPE'] == '3PT Field Goal']['SHOT_MADE'].sum() * 3)
total_pts = pts_2 + pts_3

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Shot Attempts", f"{total_shots:,}")
m2.metric("Total FGs Made", f"{total_makes:,}")
m3.metric("Est. Games Tracked", f"{est_games:,}")
m4.metric("Est. Points (No FTs)", f"{total_pts:,}")

st.divider()

# --- 2. HMM ACCURACIES ---
st.subheader("🔥 The 'Hot Hand' Discovery")

total_avg = view_df['SHOT_MADE'].mean()
hot_zone_df = view_df[view_df['IS_HOT_ZONE'] == 1]
hot_zone_eff = hot_zone_df['SHOT_MADE'].mean() if not hot_zone_df.empty else total_avg
hot_hand_effect = hot_zone_eff - total_avg

c1, c2, c3 = st.columns(3)
c1.metric("Average Efficiency", f"{total_avg:.1%}", help="Lifetime average for this player")
c2.metric("Hot State Efficiency", f"{hot_zone_eff:.1%}", help="Avg accuracy while HMM detects 'The Zone'")
c3.metric("Hot Hand Effect", f"{hot_hand_effect:+.1%}", delta=f"{hot_hand_effect:.1%}")

# --- 3. THE GRAPH ---
# --- THE GRAPH (Fixed X-Axis) ---
st.subheader("📈 Shooting Probability Trajectory")

# 1. Calculate the values
view_df['career_prob'] = view_df['SHOT_MADE'].expanding().mean()
view_df['current_form'] = view_df['SHOT_MADE'].rolling(100).mean()

# 2. Add a 'Shot Number' column to use as the X-axis
view_df['Shot Number'] = np.arange(1, len(view_df) + 1)

# 3. Slice the data for speed, but KEEP the 'Shot Number'
chart_data = view_df[['Shot Number', 'career_prob', 'current_form']].iloc[::20]

# 4. Tell Streamlit to use 'Shot Number' as the X-axis
st.line_chart(chart_data, x='Shot Number')

st.caption("The X-axis now shows the actual shot count in the player's career.")