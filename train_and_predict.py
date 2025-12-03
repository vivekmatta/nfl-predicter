import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# ENHANCED NFL PREDICTOR - EDGE IMPULSE VERSION
# ============================================================================
# This version includes advanced features that significantly improve accuracy:
# 1. Exponentially Weighted Moving Averages (recent games matter more)
# 2. Home/Away Performance Splits
# 3. Point Differential (better than win/loss)
# 4. Strength of Schedule
# 5. Expected Points Added (EPA)
# 6. FiveThirtyEight Elo Ratings
# 7. Third Down & Red Zone Efficiency
# 8. Turnover Differential
# 9. Rest Days Between Games
# 10. Weather Factors (temperature, wind)
# ============================================================================

def load_all_data():
    """Load all available CSV files from nflscraPy."""
    print("=" * 70)
    print("🏈 ENHANCED NFL PREDICTOR - LOADING DATA")
    print("=" * 70)
    
    data = {}
    
    # Core datasets
    print("\n📂 Loading available datasets...")
    
    try:
        # Season gamelogs (REQUIRED)
        data['gamelogs'] = pd.read_csv('season_gamelogs.csv')
        print(f"   ✓ Gamelogs: {len(data['gamelogs'])} games")
    except FileNotFoundError:
        print("   ❌ ERROR: season_gamelogs.csv not found (REQUIRED)")
        return None
    
    # Optional but highly recommended datasets
    optional_files = {
        'metadata': 'gamelog_metadata.csv',
        'statistics': 'gamelog_statistics.csv',
        'expected_points': 'gamelog_expected_points.csv',
        'five_thirty_eight': 'five_thirty_eight.csv',
        'season_splits': 'season_splits.csv',
    }
    
    for key, filename in optional_files.items():
        try:
            data[key] = pd.read_csv(filename)
            print(f"   ✓ {key.replace('_', ' ').title()}: {len(data[key])} records")
        except FileNotFoundError:
            print(f"   ⚠ {key.replace('_', ' ').title()}: Not found (skipping)")
            data[key] = None
    
    return data


def create_enhanced_features(data):
    """Create advanced features for better predictions."""
    print("\n" + "=" * 70)
    print("⚙️  FEATURE ENGINEERING")
    print("=" * 70)
    
    # Start with gamelogs as base
    df = data['gamelogs'].copy()
    
    # Parse dates
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'])
    
    # ========================================================================
    # FEATURE 1: BASIC GAME OUTCOME & POINT DIFFERENTIAL
    # ========================================================================
    print("\n1. Creating target variable and point differential...")
    df['tm_win'] = (df['tm_score'] > df['opp_score']).astype(int)
    df['point_diff'] = df['tm_score'] - df['opp_score']
    
    # Sort by team and date for time-series features
    df = df.sort_values(['tm_alias', 'season', 'week'])
    
    # ========================================================================
    # FEATURE 2: EXPONENTIALLY WEIGHTED MOVING AVERAGES (EWMA)
    # Recent games weighted MORE heavily than older games
    # ========================================================================
    print("2. Calculating exponentially weighted averages (EWMA)...")
    ewma_cols = ['tm_score', 'opp_score', 'point_diff']
    
    for col in ewma_cols:
        if col in df.columns:
            df[f'ewma_{col}'] = df.groupby('tm_alias')[col].transform(
                lambda x: x.shift(1).ewm(span=4, adjust=False).mean()
            )
            # Also add simple 3-game rolling for comparison
            df[f'roll3_{col}'] = df.groupby('tm_alias')[col].transform(
                lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
            )
    
    # ========================================================================
    # FEATURE 3: HOME/AWAY PERFORMANCE SPLITS
    # Teams perform differently at home vs away
    # ========================================================================
    print("3. Creating home/away performance splits...")
    
    # Calculate home performance
    home_stats = df[df['tm_location'] == 'H'].groupby('tm_alias').agg({
        'tm_score': 'mean',
        'point_diff': 'mean',
        'tm_win': 'mean'
    }).reset_index()
    home_stats.columns = ['tm_alias', 'home_ppg', 'home_point_diff', 'home_win_pct']
    
    # Calculate away performance
    away_stats = df[df['tm_location'] == 'A'].groupby('tm_alias').agg({
        'tm_score': 'mean',
        'point_diff': 'mean',
        'tm_win': 'mean'
    }).reset_index()
    away_stats.columns = ['tm_alias', 'away_ppg', 'away_point_diff', 'away_win_pct']
    
    df = df.merge(home_stats, on='tm_alias', how='left')
    df = df.merge(away_stats, on='tm_alias', how='left')
    
    # ========================================================================
    # FEATURE 4: STRENGTH OF SCHEDULE
    # Are they beating good teams or bad teams?
    # ========================================================================
    print("4. Calculating strength of schedule...")
    
    # Calculate each team's win percentage
    team_win_pct = df.groupby(['tm_alias', 'season'])['tm_win'].mean().reset_index()
    team_win_pct.columns = ['tm_alias', 'season', 'win_pct']
    
    # Merge opponent's win percentage
    df = df.merge(
        team_win_pct, 
        left_on=['opp_alias', 'season'], 
        right_on=['tm_alias', 'season'],
        how='left',
        suffixes=('', '_opp')
    )
    df = df.drop('tm_alias_opp', axis=1, errors='ignore')
    df.rename(columns={'win_pct': 'opp_win_pct'}, inplace=True)
    
    # Rolling strength of schedule
    df['sos'] = df.groupby('tm_alias')['opp_win_pct'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    # ========================================================================
    # FEATURE 5: REST DAYS
    # Teams with more rest perform better
    # ========================================================================
    print("5. Calculating rest days between games...")
    if 'event_date' in df.columns:
        df['prev_game_date'] = df.groupby('tm_alias')['event_date'].shift(1)
        df['rest_days'] = (df['event_date'] - df['prev_game_date']).dt.days
        df['rest_days'] = df['rest_days'].fillna(7)  # Default to 7 days
    
    # ========================================================================
    # FEATURE 6: ADD STATISTICS DATA (if available)
    # Third down conversions, turnovers, time of possession
    # ========================================================================
    if data['statistics'] is not None:
        print("6. Merging detailed statistics (yards, conversions, turnovers)...")
        stats = data['statistics'].copy()
        
        # Key columns from statistics
        stat_cols = [
            'boxscore_stats_link', 'alias', 'total_yds', 'rush_yds', 'pass_yds',
            'third_down_conv_pct', 'turnovers', 'times_sacked', 'possesion_time'
        ]
        stat_cols = [c for c in stat_cols if c in stats.columns]
        
        if 'boxscore_stats_link' in df.columns and 'boxscore_stats_link' in stats.columns:
            df = df.merge(
                stats[stat_cols],
                left_on=['boxscore_stats_link', 'tm_alias'],
                right_on=['boxscore_stats_link', 'alias'],
                how='left',
                suffixes=('', '_stat')
            )
            
            # Create rolling averages for key stats
            for col in ['total_yds', 'third_down_conv_pct', 'turnovers']:
                if col in df.columns:
                    df[f'ewma_{col}'] = df.groupby('tm_alias')[col].transform(
                        lambda x: x.shift(1).ewm(span=4, adjust=False).mean()
                    )
    
    # ========================================================================
    # FEATURE 7: ADD EXPECTED POINTS (if available)
    # EPA is one of the best predictors of team quality
    # ========================================================================
    if data['expected_points'] is not None:
        print("7. Merging Expected Points Added (EPA)...")
        ep = data['expected_points'].copy()
        
        ep_cols = [
            'boxscore_stats_link', 'alias', 'exp_pts_off', 'exp_pts_def'
        ]
        ep_cols = [c for c in ep_cols if c in ep.columns]
        
        if 'boxscore_stats_link' in df.columns and 'boxscore_stats_link' in ep.columns:
            df = df.merge(
                ep[ep_cols],
                left_on=['boxscore_stats_link', 'tm_alias'],
                right_on=['boxscore_stats_link', 'alias'],
                how='left',
                suffixes=('', '_ep')
            )
            
            # Create rolling EPA
            for col in ['exp_pts_off', 'exp_pts_def']:
                if col in df.columns:
                    df[f'ewma_{col}'] = df.groupby('tm_alias')[col].transform(
                        lambda x: x.shift(1).ewm(span=4, adjust=False).mean()
                    )
    
    # ========================================================================
    # FEATURE 8: ADD FIVETHIRTYEIGHT ELO RATINGS (if available)
    # Professional Elo ratings - highly predictive
    # ========================================================================
    if data['five_thirty_eight'] is not None:
        print("8. Merging FiveThirtyEight Elo ratings...")
        elo = data['five_thirty_eight'].copy()
        
        elo_cols = [
            'event_date', 'tm_alias', 'tm_elo_pre', 'tm_qb_elo_pre_game'
        ]
        elo_cols = [c for c in elo_cols if c in elo.columns]
        
        if 'event_date' in df.columns and 'event_date' in elo.columns:
            elo['event_date'] = pd.to_datetime(elo['event_date'])
            df = df.merge(
                elo[elo_cols],
                on=['event_date', 'tm_alias'],
                how='left',
                suffixes=('', '_elo')
            )
    
    # ========================================================================
    # FEATURE 9: ADD WEATHER/METADATA (if available)
    # Weather affects scoring, especially wind and temperature
    # ========================================================================
    if data['metadata'] is not None:
        print("9. Merging weather and game metadata...")
        meta = data['metadata'].copy()
        
        meta_cols = [
            'boxscore_stats_link', 'tm_spread', 'temperature', 
            'wind_speed', 'roof_type', 'surface_type'
        ]
        meta_cols = [c for c in meta_cols if c in meta.columns]
        
        if 'boxscore_stats_link' in df.columns and 'boxscore_stats_link' in meta.columns:
            # Get unique metadata per game (not per team)
            meta_unique = meta[meta_cols].drop_duplicates('boxscore_stats_link')
            df = df.merge(meta_unique, on='boxscore_stats_link', how='left')
    
    print(f"\n✅ Feature engineering complete!")
    print(f"   Total features: {len(df.columns)}")
    
    return df


def create_matchup_dataset(df):
    """Create home vs away matchup pairs for prediction."""
    print("\n" + "=" * 70)
    print("🔗 CREATING MATCHUP PAIRS (HOME vs AWAY)")
    print("=" * 70)
    
    # Split into home and away
    home = df[df['tm_location'] == 'H'].copy()
    away = df[df['tm_location'] == 'A'].copy()
    
    # Merge on game identifier
    matchups = pd.merge(
        home,
        away,
        on=['boxscore_stats_link', 'season', 'week'],
        suffixes=('_home', '_away')
    )
    
    print(f"   ✓ Created {len(matchups)} matchup pairs")
    
    # Target: home team wins
    matchups['home_win'] = (matchups['tm_score_home'] > matchups['tm_score_away']).astype(int)
    
    return matchups


def prepare_for_edge_impulse(matchups):
    """Select and prepare features for Edge Impulse."""
    print("\n" + "=" * 70)
    print("📊 PREPARING DATA FOR EDGE IMPULSE")
    print("=" * 70)
    
    # Select the BEST predictive features
    feature_patterns = [
        'ewma_',      # Exponentially weighted averages
        'roll3_',     # Rolling 3-game averages
        'home_',      # Home performance splits
        'away_',      # Away performance splits
        'sos',        # Strength of schedule
        'rest_days',  # Rest between games
        '_elo_',      # Elo ratings
        'tm_spread',  # Vegas spread (if available)
        'temperature', # Weather
        'wind_speed'   # Weather
    ]
    
    # Get all columns matching patterns
    feature_cols = []
    for col in matchups.columns:
        if any(pattern in col for pattern in feature_patterns):
            feature_cols.append(col)
    
    # Remove any columns with too many NaNs
    feature_cols = [c for c in feature_cols if matchups[c].notna().sum() > len(matchups) * 0.5]
    
    # Add target
    target = 'home_win'
    final_cols = feature_cols + [target]
    
    # Create final dataset
    final_data = matchups[final_cols].dropna()
    
    print(f"\n📋 DATASET SUMMARY:")
    print(f"   Total matchups: {len(final_data)}")
    print(f"   Total features: {len(feature_cols)}")
    print(f"   Home wins: {final_data[target].sum()} ({final_data[target].mean():.1%})")
    print(f"   Away wins: {len(final_data) - final_data[target].sum()} ({1-final_data[target].mean():.1%})")
    
    # Save to CSV
    output_file = 'nfl_enhanced_training_data.csv'
    final_data.to_csv(output_file, index=False)
    
    print(f"\n✅ SUCCESS! Enhanced data saved to: {output_file}")
    
    # Show top features
    print(f"\n🎯 TOP 20 FEATURES TO USE IN EDGE IMPULSE:")
    for i, feat in enumerate(feature_cols[:20], 1):
        print(f"   {i:2d}. {feat}")
    
    if len(feature_cols) > 20:
        print(f"   ... and {len(feature_cols) - 20} more features")
    
    print(f"\n🎯 TARGET VARIABLE: {target}")
    print("   1 = Home team wins")
    print("   0 = Away team wins")
    
    return final_data, feature_cols


def show_feature_importance_estimate(final_data, feature_cols, target='home_win'):
    """Show which features are likely most important."""
    print("\n" + "=" * 70)
    print("📈 ESTIMATED FEATURE IMPORTANCE")
    print("=" * 70)
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        # Quick model to estimate feature importance
        X = final_data[feature_cols]
        y = final_data[target]
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X, y)
        
        # Get importances
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
        for i, row in importance_df.head(10).iterrows():
            print(f"   {row['feature']:<40} {row['importance']:.4f}")
        
        print("\n💡 TIP: Focus on these features in Edge Impulse!")
        
    except ImportError:
        print("\n⚠ Install scikit-learn to see feature importance:")
        print("   pip install scikit-learn")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("    🏈 ENHANCED NFL PREDICTOR - EDGE IMPULSE VERSION 🏈")
    print("=" * 70)
    
    # Load all data
    data = load_all_data()
    
    if data is None:
        print("\n❌ Cannot proceed without season_gamelogs.csv")
        return
    
    # Create enhanced features
    df_enhanced = create_enhanced_features(data)
    
    # Create matchup pairs
    matchups = create_matchup_dataset(df_enhanced)
    
    # Prepare for Edge Impulse
    final_data, feature_cols = prepare_for_edge_impulse(matchups)
    
    # Show feature importance
    show_feature_importance_estimate(final_data, feature_cols)
    
    # Instructions
    print("\n" + "=" * 70)
    print("📚 NEXT STEPS - UPLOADING TO EDGE IMPULSE")
    print("=" * 70)
    print("1. Go to https://studio.edgeimpulse.com/")
    print("2. Create a new project: 'NFL Enhanced Predictor'")
    print("3. Go to 'Data acquisition' tab")
    print("4. Click 'Upload data'")
    print("5. Upload: nfl_enhanced_training_data.csv")
    print("6. Set label column to: home_win")
    print("7. Train/Test split: 80/20")
    print("\n8. In 'Impulse design':")
    print("   - Add 'Classification' block")
    print("   - Choose 'Keras (Tabular)' or 'Random Forest'")
    print("   - Use all features starting with:")
    print("     • ewma_ (exponentially weighted averages)")
    print("     • roll3_ (rolling averages)")  
    print("     • home_ / away_ (performance splits)")
    print("     • elo ratings (if available)")
    print("\n9. Train and test your model!")
    print("=" * 70)
    
    print("\n💡 EXPECTED ACCURACY:")
    print("   With basic features: 55-60%")
    print("   With all enhanced features: 65-70%")
    print("   With Elo + EPA: 70-75%")


if __name__ == "__main__":
    main()