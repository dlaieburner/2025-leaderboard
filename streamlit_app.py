#!/usr/bin/env python3



# Usage:
# Local testing:
# $ ./streamlit_app.py --no-streamlit

# Streamlit mode (also works when deployed):
# $ streamlit run streamlit_app.py

# When deployed to Streamlit Cloud, it'll ignore the argparse stuff and just run normally.



import pandas as pd
from datetime import datetime
import argparse


def style_leaderboard(df_display):
    return df_display.style \
        .set_properties(**{'background-color': '#1a472a'}, subset=['Params ↓', 'ms/Sample ↓']) \
        .set_properties(**{'background-color': '#2d5a5a'}, subset=['MSE ↓', 'SSIM ↑']) \
        .set_properties(**{'background-color': '#1e3a5f'}, subset=['Entropy ↓', 'KL Div ↓', 'Confidence ↑']) \
        .set_properties(**{'background-color': '#4a235a'}, subset=['Overall Score ↓', 'Rank ↓'])


def display_leaderboard(use_streamlit=True):
    """Display leaderboard, optionally using Streamlit."""
    
    df = pd.read_csv('leaderboard.csv')
    df = df.drop(columns=['names'], errors='ignore')  # keep names private
    # Drop individual rank columns - users can sort interactively
    rank_cols = [col for col in df.columns if col.endswith('_rank') and col != 'overall_rank']
    df_display = df.drop(columns=rank_cols)
    
    if not use_streamlit:
        print("\n" + "="*100)
        print("LEADERBOARD (Raw CSV columns)")
        print("="*100)
        print(df.to_string(index=False))

    # Create actual rank column (1, 2, 3, etc.) based on overall_rank (which is the score)
    df_display = df_display.sort_values('overall_rank')
    df_display.insert(len(df_display.columns), 'rank_position', range(1, len(df_display) + 1))

    # Convert time_per_sample to milliseconds for better readability
    df_display['time_per_sample'] *= 1000

    
    # Rename columns for display
    df_display = df_display.rename(columns={
        'team': 'Team Name',
        'latent_shape': 'Latent Dim',
        'overall_rank': 'Overall Score ↓',
        'rank_position': 'Rank ↓',
        'total_params': 'Params ↓',
        'time_per_sample': 'ms/Sample ↓',
        'mse': 'MSE ↓',
        'ssim': 'SSIM ↑',
        'entropy': 'Entropy ↓',
        'kl_div_classes': 'KL Div ↓',
        'gen_confidence': 'Confidence ↑'
    })

    
    if not use_streamlit:
        print("\n" + "="*100)
        print("LEADERBOARD (Display columns)")
        print("="*100)
        print(df_display.to_string(index=False))
        return df_display
    
    # Streamlit mode
    import streamlit as st

    st.set_page_config(layout="wide", page_title="DLAIE Leaderboard")

    # Remove default padding
    st.markdown("""
        <style>
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("🏆 2025 DLAIE Latent Flow Matching Leaderboard")

    st.markdown('''
        <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 20px;">
            <div style="flex: 1; min-width: 250px;">
                <h4><a href="https://github.com/DLAIE/2025-LeaderboardContest">Contest Page</a> | 
                <a href="https://github.com/drscotthawley/DLAIE">DLAIE Course Page</a></h4>
            </div>
            <div style="flex: 0 0 auto;">
                <img src="https://raw.githubusercontent.com/dlaieburner/2025-leaderboard/refs/heads/main/flow_anim_3d.gif" height="120">
            </div>
        </div>
        ''', unsafe_allow_html=True)

    
    # Get latest submission time from the data
    latest_update = pd.to_datetime(df['time_stamp']).max()
    st.caption(f"Last updated: {latest_update.strftime('%Y-%m-%d %H:%M')}")

    df_display = style_leaderboard(df_display)
    numeric_cols = ['ms/Sample ↓', 'MSE ↓', 'SSIM ↑', 'Entropy ↓', 'KL Div ↓', 'Confidence ↑', 'Overall Score ↓']
    st.dataframe(df_display, use_container_width=True, hide_index=True,
                 column_config={col: st.column_config.NumberColumn(format="%.4f") for col in numeric_cols})
    

    st.markdown("### Prizes from:")

    st.markdown('''
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <img src="https://raw.githubusercontent.com/dlaieburner/2025-leaderboard/refs/heads/main/wandb_logo.png" height="100">
        <img src="https://raw.githubusercontent.com/dlaieburner/2025-leaderboard/refs/heads/main/coreweave_logo.jpg" height="100">
            <img src="https://raw.githubusercontent.com/dlaieburner/2025-leaderboard/refs/heads/main/bdaic_logo.png" height="100">
    </div>
    ''', unsafe_allow_html=True)

    #st.markdown("---")
    st.markdown('')
    st.markdown('')
    st.markdown("Powered by @drscotthawley/[botograder](https://github.com/drscotthawley/botograder)")

    st.markdown("""
        ### 📊 Leaderboard Metrics

        **<span style="background-color:#2d7a4d; padding:2px 8px;">  </span>&nbsp; Model Efficiency**
        - **Params ↓**: Total parameters in VAE + Flow model
        - **ms/Sample ↓**: Generation time per sample in milliseconds

        **<span style="background-color:#4d8a8a; padding:2px 8px;">  </span>&nbsp; VAE Reconstruction Quality**
        - **MSE ↓**: Mean squared error between original and reconstructed images
        - **SSIM ↑**: Structural similarity index (perceptual quality)

        **<span style="background-color:#3d5a8f; padding:2px 8px;">  </span>&nbsp; Generated Sample Quality** (using pretrained ResNet classifier)
        - **Entropy ↓**: Classifier uncertainty (over all classes) on generated images (lower = more realistic)
        - **KL Div ↓**: Divergence between generated and real class distributions
        - **Confidence ↑**: Classifier confidence (max) on generated samples

        **<span style="background-color:#7a3d8a; padding:2px 8px;">  </span>&nbsp; Final Score**
        - **Overall Score ↓**: Average rank across all metrics
        - **Rank ↓**: Final leaderboard position
        """, unsafe_allow_html=True)

    return df_display

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display leaderboard")
    parser.add_argument('--no-streamlit', action='store_true', 
                       help='Run in CLI mode for local testing')
    args = parser.parse_args()
    
    display_leaderboard(use_streamlit=not args.no_streamlit)


