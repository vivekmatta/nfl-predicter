import nflscraPy
import pandas as pd
import time
from datetime import datetime

# ============================================================================
# NFL DATA DOWNLOADER - COMPREHENSIVE HISTORICAL DATA
# ============================================================================
# This script downloads ALL available data from nflscraPy (2000-2024)
# WARNING: This will take SEVERAL HOURS due to rate limiting (3.5-5.5s per request)
# 
# Pro Football Reference Rate Limits:
# - Each function sleeps 3.5-5.5 seconds between requests
# - DO NOT reduce these delays or you'll get banned
# - For 24 seasons × multiple datasets, expect 4-8 hours total
# ============================================================================

def download_season_gamelogs(start_year=2000, end_year=2024):
    """
    Download all game logs from 2000 to 2024.
    This is the foundation dataset - contains basic game results.
    """
    print("=" * 70)
    print("📥 DOWNLOADING SEASON GAMELOGS (2000-2024)")
    print("=" * 70)
    print("⏱️  Estimated time: 30-45 minutes")
    print("=" * 70)
    
    all_gamelogs = []
    
    for year in range(start_year, end_year + 1):
        try:
            print(f"\n📅 Fetching {year} season...")
            gamelogs = nflscraPy._gamelogs(year)
            
            if gamelogs is not None and len(gamelogs) > 0:
                all_gamelogs.append(gamelogs)
                print(f"   ✅ Downloaded {len(gamelogs)} games from {year}")
            else:
                print(f"   ⚠️  No data for {year}")
                
        except Exception as e:
            print(f"   ❌ Error downloading {year}: {e}")
            continue
    
    # Combine all seasons
    if all_gamelogs:
        combined = pd.concat(all_gamelogs, ignore_index=True)
        output_file = 'season_gamelogs.csv'
        combined.to_csv(output_file, index=False)
        print(f"\n✅ SUCCESS! Saved {len(combined)} games to: {output_file}")
        return combined
    else:
        print("\n❌ No data downloaded")
        return None


def download_gamelog_metadata(gamelogs_df):
    """
    Download metadata (spreads, weather, stadium) for each game.
    This contains Vegas spreads, temperature, wind, etc.
    """
    print("\n" + "=" * 70)
    print("📥 DOWNLOADING GAME METADATA (Spreads, Weather, Stadium)")
    print("=" * 70)
    
    if gamelogs_df is None:
        print("❌ Need gamelogs first! Run download_season_gamelogs() first.")
        return None
    
    # Get unique boxscore links
    unique_games = gamelogs_df['boxscore_stats_link'].unique()
    total_games = len(unique_games)
    
    print(f"📊 Total unique games to fetch: {total_games}")
    print(f"⏱️  Estimated time: {(total_games * 4.5) / 60:.1f} minutes")
    print("=" * 70)
    
    all_metadata = []
    
    for i, link in enumerate(unique_games, 1):
        try:
            if i % 50 == 0:
                print(f"   Progress: {i}/{total_games} ({i/total_games*100:.1f}%)")
            
            metadata = nflscraPy._gamelog_metadata(link)
            
            if metadata is not None and len(metadata) > 0:
                all_metadata.append(metadata)
                
        except Exception as e:
            if i % 100 == 0:
                print(f"   ⚠️  Error on game {i}: {e}")
            continue
    
    if all_metadata:
        combined = pd.concat(all_metadata, ignore_index=True)
        output_file = 'gamelog_metadata.csv'
        combined.to_csv(output_file, index=False)
        print(f"\n✅ SUCCESS! Saved {len(combined)} metadata records to: {output_file}")
        return combined
    else:
        print("\n❌ No metadata downloaded")
        return None


def download_gamelog_statistics(gamelogs_df):
    """
    Download detailed statistics (yards, conversions, turnovers) for each game.
    This contains total yards, 3rd down %, time of possession, etc.
    """
    print("\n" + "=" * 70)
    print("📥 DOWNLOADING GAME STATISTICS (Yards, Conversions, Turnovers)")
    print("=" * 70)
    
    if gamelogs_df is None:
        print("❌ Need gamelogs first! Run download_season_gamelogs() first.")
        return None
    
    unique_games = gamelogs_df['boxscore_stats_link'].unique()
    total_games = len(unique_games)
    
    print(f"📊 Total unique games to fetch: {total_games}")
    print(f"⏱️  Estimated time: {(total_games * 4.5) / 60:.1f} minutes")
    print("=" * 70)
    
    all_statistics = []
    
    for i, link in enumerate(unique_games, 1):
        try:
            if i % 50 == 0:
                print(f"   Progress: {i}/{total_games} ({i/total_games*100:.1f}%)")
            
            statistics = nflscraPy._gamelog_statistics(link)
            
            if statistics is not None and len(statistics) > 0:
                all_statistics.append(statistics)
                
        except Exception as e:
            if i % 100 == 0:
                print(f"   ⚠️  Error on game {i}: {e}")
            continue
    
    if all_statistics:
        combined = pd.concat(all_statistics, ignore_index=True)
        output_file = 'gamelog_statistics.csv'
        combined.to_csv(output_file, index=False)
        print(f"\n✅ SUCCESS! Saved {len(combined)} statistics records to: {output_file}")
        return combined
    else:
        print("\n❌ No statistics downloaded")
        return None


def download_expected_points(gamelogs_df):
    """
    Download Expected Points Added (EPA) data.
    EPA is one of the BEST predictors of team quality.
    """
    print("\n" + "=" * 70)
    print("📥 DOWNLOADING EXPECTED POINTS (EPA)")
    print("=" * 70)
    
    if gamelogs_df is None:
        print("❌ Need gamelogs first! Run download_season_gamelogs() first.")
        return None
    
    unique_games = gamelogs_df['boxscore_stats_link'].unique()
    total_games = len(unique_games)
    
    print(f"📊 Total unique games to fetch: {total_games}")
    print(f"⏱️  Estimated time: {(total_games * 4.5) / 60:.1f} minutes")
    print("=" * 70)
    
    all_expected_points = []
    
    for i, link in enumerate(unique_games, 1):
        try:
            if i % 50 == 0:
                print(f"   Progress: {i}/{total_games} ({i/total_games*100:.1f}%)")
            
            expected_points = nflscraPy._gamelog_expected_points(link)
            
            if expected_points is not None and len(expected_points) > 0:
                all_expected_points.append(expected_points)
                
        except Exception as e:
            if i % 100 == 0:
                print(f"   ⚠️  Error on game {i}: {e}")
            continue
    
    if all_expected_points:
        combined = pd.concat(all_expected_points, ignore_index=True)
        output_file = 'gamelog_expected_points.csv'
        combined.to_csv(output_file, index=False)
        print(f"\n✅ SUCCESS! Saved {len(combined)} EPA records to: {output_file}")
        return combined
    else:
        print("\n❌ No expected points downloaded")
        return None


def download_five_thirty_eight():
    """
    Download FiveThirtyEight Elo ratings (1970-Present).
    Professional-grade team strength ratings.
    """
    print("\n" + "=" * 70)
    print("📥 DOWNLOADING FIVETHIRTYEIGHT ELO RATINGS")
    print("=" * 70)
    print("⏱️  Estimated time: 5-10 seconds")
    print("=" * 70)
    
    try:
        elo_data = nflscraPy._five_thirty_eight()
        
        if elo_data is not None and len(elo_data) > 0:
            output_file = 'five_thirty_eight.csv'
            elo_data.to_csv(output_file, index=False)
            print(f"✅ SUCCESS! Saved {len(elo_data)} Elo records to: {output_file}")
            return elo_data
        else:
            print("❌ No Elo data downloaded")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading Elo data: {e}")
        return None


def download_season_splits(gamelogs_df, start_year=2000, end_year=2024):
    """
    Download season splits (performance by situation).
    This shows how teams perform in different situations (home/away, by quarter, etc.)
    """
    print("\n" + "=" * 70)
    print("📥 DOWNLOADING SEASON SPLITS (Situational Performance)")
    print("=" * 70)
    
    if gamelogs_df is None:
        print("❌ Need gamelogs first! Run download_season_gamelogs() first.")
        return None
    
    # Get unique teams
    unique_teams = pd.concat([
        gamelogs_df['tm_alias'],
        gamelogs_df['opp_alias']
    ]).unique()
    
    total_requests = len(unique_teams) * (end_year - start_year + 1) * 2  # For + Against
    
    print(f"📊 Total teams: {len(unique_teams)}")
    print(f"📊 Total seasons: {end_year - start_year + 1}")
    print(f"📊 Total requests: {total_requests}")
    print(f"⏱️  Estimated time: {(total_requests * 4.5) / 60:.1f} minutes")
    print("=" * 70)
    
    all_splits = []
    request_count = 0
    
    for year in range(start_year, end_year + 1):
        print(f"\n📅 Fetching {year} season splits...")
        
        for team in unique_teams:
            for split_type in ['For', 'Against']:
                try:
                    request_count += 1
                    
                    if request_count % 100 == 0:
                        print(f"   Progress: {request_count}/{total_requests} ({request_count/total_requests*100:.1f}%)")
                    
                    splits = nflscraPy._season_splits(year, team, split_type)
                    
                    if splits is not None and len(splits) > 0:
                        all_splits.append(splits)
                        
                except Exception as e:
                    continue
    
    if all_splits:
        combined = pd.concat(all_splits, ignore_index=True)
        output_file = 'season_splits.csv'
        combined.to_csv(output_file, index=False)
        print(f"\n✅ SUCCESS! Saved {len(combined)} split records to: {output_file}")
        return combined
    else:
        print("\n❌ No splits downloaded")
        return None


# ============================================================================
# MAIN EXECUTION WITH SMART OPTIONS
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("    🏈 NFL DATA DOWNLOADER - COMPREHENSIVE EDITION 🏈")
    print("=" * 70)
    print("\n⚠️  IMPORTANT WARNINGS:")
    print("   • This will take 4-8 HOURS total")
    print("   • DO NOT interrupt or you'll have to restart")
    print("   • DO NOT reduce rate limits or you'll get IP banned")
    print("   • Make sure you have stable internet connection")
    print("=" * 70)
    
    print("\n📋 DOWNLOAD OPTIONS:")
    print("   1. Quick Start (Gamelogs + Metadata only) - ~1 hour")
    print("   2. Standard (Gamelogs + Metadata + Statistics) - ~2-3 hours")
    print("   3. Full Dataset (Everything) - ~6-8 hours")
    print("   4. Custom Selection")
    print("   5. Exit")
    
    choice = input("\n👉 Enter your choice (1-5): ").strip()
    
    # Year range
    print("\n📅 YEAR RANGE:")
    print("   Available: 2000-2024")
    start_year = int(input("   Start year (default 2000): ").strip() or "2000")
    end_year = int(input("   End year (default 2024): ").strip() or "2024")
    
    start_time = datetime.now()
    
    if choice == '1':
        # Quick Start
        print("\n🚀 QUICK START MODE")
        gamelogs = download_season_gamelogs(start_year, end_year)
        if gamelogs is not None:
            download_gamelog_metadata(gamelogs)
    
    elif choice == '2':
        # Standard
        print("\n📊 STANDARD MODE")
        gamelogs = download_season_gamelogs(start_year, end_year)
        if gamelogs is not None:
            download_gamelog_metadata(gamelogs)
            download_gamelog_statistics(gamelogs)
    
    elif choice == '3':
        # Full Dataset
        print("\n💎 FULL DATASET MODE")
        gamelogs = download_season_gamelogs(start_year, end_year)
        
        if gamelogs is not None:
            download_gamelog_metadata(gamelogs)
            download_gamelog_statistics(gamelogs)
            download_expected_points(gamelogs)
            download_five_thirty_eight()
            
            # Ask about season splits (takes longest)
            splits_choice = input("\n⚠️  Download season splits? This takes 2-4 hours. (y/n): ").strip().lower()
            if splits_choice == 'y':
                download_season_splits(gamelogs, start_year, end_year)
    
    elif choice == '4':
        # Custom
        print("\n🎯 CUSTOM SELECTION MODE")
        
        gamelogs = None
        if input("   Download Gamelogs? (y/n): ").strip().lower() == 'y':
            gamelogs = download_season_gamelogs(start_year, end_year)
        
        if gamelogs is not None:
            if input("   Download Metadata (spreads, weather)? (y/n): ").strip().lower() == 'y':
                download_gamelog_metadata(gamelogs)
            
            if input("   Download Statistics (yards, turnovers)? (y/n): ").strip().lower() == 'y':
                download_gamelog_statistics(gamelogs)
            
            if input("   Download Expected Points (EPA)? (y/n): ").strip().lower() == 'y':
                download_expected_points(gamelogs)
            
            if input("   Download FiveThirtyEight Elo? (y/n): ").strip().lower() == 'y':
                download_five_thirty_eight()
            
            if input("   Download Season Splits? (SLOW - 2-4 hrs) (y/n): ").strip().lower() == 'y':
                download_season_splits(gamelogs, start_year, end_year)
    
    else:
        print("\n👋 Goodbye!")
        return
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("🎉 DOWNLOAD COMPLETE!")
    print("=" * 70)
    print(f"⏱️  Total time: {duration}")
    print("\n📂 Files created:")
    print("   • season_gamelogs.csv (if selected)")
    print("   • gamelog_metadata.csv (if selected)")
    print("   • gamelog_statistics.csv (if selected)")
    print("   • gamelog_expected_points.csv (if selected)")
    print("   • five_thirty_eight.csv (if selected)")
    print("   • season_splits.csv (if selected)")
    
    print("\n📚 NEXT STEPS:")
    print("   1. Run the enhanced predictor script")
    print("   2. It will automatically load all available CSV files")
    print("   3. Upload the output to Edge Impulse")
    print("=" * 70)


if __name__ == "__main__":
    main()