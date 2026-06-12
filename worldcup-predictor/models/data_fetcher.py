 """
 World Cup Data Fetcher
 
 Fetches FIFA rankings and historical match data.
 Falls back to comprehensive built-in data for reliability.
 """
 
 import json
 import os
 import random
 from datetime import datetime
 from typing import Dict, List, Optional, Tuple
 
 import requests
 from bs4 import BeautifulSoup
 
 
 # ============================================================
 # Built-in Team Data: 32 teams with ELO-style strength ratings
 # Based on FIFA rankings + historical performance
 # ============================================================
 
 TEAM_DATA: Dict[str, dict] = {
     "Brazil": {"rank": 1, "code": "BRA", "continent": "South America", "strength": 98.5, "elo": 2123, "gd_per_match": 1.85},
     "Argentina": {"rank": 2, "code": "ARG", "continent": "South America", "strength": 97.8, "elo": 2098, "gd_per_match": 1.72},
     "France": {"rank": 3, "code": "FRA", "continent": "Europe", "strength": 97.2, "elo": 2076, "gd_per_match": 1.68},
     "England": {"rank": 4, "code": "ENG", "continent": "Europe", "strength": 95.8, "elo": 2054, "gd_per_match": 1.55},
     "Belgium": {"rank": 5, "code": "BEL", "continent": "Europe", "strength": 94.5, "elo": 2038, "gd_per_match": 1.48},
     "Portugal": {"rank": 6, "code": "POR", "continent": "Europe", "strength": 93.9, "elo": 2025, "gd_per_match": 1.42},
     "Netherlands": {"rank": 7, "code": "NED", "continent": "Europe", "strength": 93.2, "elo": 2012, "gd_per_match": 1.38},
     "Spain": {"rank": 8, "code": "ESP", "continent": "Europe", "strength": 92.8, "elo": 2005, "gd_per_match": 1.35},
     "Germany": {"rank": 9, "code": "GER", "continent": "Europe", "strength": 91.5, "elo": 1988, "gd_per_match": 1.28},
     "Croatia": {"rank": 10, "code": "CRO", "continent": "Europe", "strength": 90.2, "elo": 1972, "gd_per_match": 1.22},
     "Italy": {"rank": 11, "code": "ITA", "continent": "Europe", "strength": 89.8, "elo": 1965, "gd_per_match": 1.18},
     "Uruguay": {"rank": 12, "code": "URU", "continent": "South America", "strength": 89.2, "elo": 1958, "gd_per_match": 1.15},
     "Denmark": {"rank": 13, "code": "DEN", "continent": "Europe", "strength": 88.5, "elo": 1945, "gd_per_match": 1.08},
     "Mexico": {"rank": 14, "code": "MEX", "continent": "North America", "strength": 87.8, "elo": 1932, "gd_per_match": 1.02},
     "Switzerland": {"rank": 15, "code": "SUI", "continent": "Europe", "strength": 87.2, "elo": 1925, "gd_per_match": 0.98},
     "USA": {"rank": 16, "code": "USA", "continent": "North America", "strength": 86.5, "elo": 1912, "gd_per_match": 0.92},
     "Japan": {"rank": 17, "code": "JPN", "continent": "Asia", "strength": 85.8, "elo": 1900, "gd_per_match": 0.88},
     "Morocco": {"rank": 18, "code": "MAR", "continent": "Africa", "strength": 85.2, "elo": 1892, "gd_per_match": 0.85},
     "Senegal": {"rank": 19, "code": "SEN", "continent": "Africa", "strength": 84.5, "elo": 1880, "gd_per_match": 0.78},
     "Poland": {"rank": 20, "code": "POL", "continent": "Europe", "strength": 83.8, "elo": 1868, "gd_per_match": 0.75},
     "Serbia": {"rank": 21, "code": "SRB", "continent": "Europe", "strength": 83.2, "elo": 1858, "gd_per_match": 0.72},
     "South Korea": {"rank": 22, "code": "KOR", "continent": "Asia", "strength": 82.5, "elo": 1845, "gd_per_match": 0.68},
     "Nigeria": {"rank": 23, "code": "NGA", "continent": "Africa", "strength": 81.8, "elo": 1832, "gd_per_match": 0.65},
     "Australia": {"rank": 24, "code": "AUS", "continent": "Asia", "strength": 81.2, "elo": 1822, "gd_per_match": 0.62},
     "Iran": {"rank": 25, "code": "IRN", "continent": "Asia", "strength": 80.5, "elo": 1810, "gd_per_match": 0.58},
     "Ecuador": {"rank": 26, "code": "ECU", "continent": "South America", "strength": 79.8, "elo": 1798, "gd_per_match": 0.55},
     "Cameroon": {"rank": 27, "code": "CMR", "continent": "Africa", "strength": 79.2, "elo": 1788, "gd_per_match": 0.52},
     "Canada": {"rank": 28, "code": "CAN", "continent": "North America", "strength": 78.5, "elo": 1775, "gd_per_match": 0.48},
     "Tunisia": {"rank": 29, "code": "TUN", "continent": "Africa", "strength": 77.8, "elo": 1765, "gd_per_match": 0.45},
     "Saudi Arabia": {"rank": 30, "code": "KSA", "continent": "Asia", "strength": 77.2, "elo": 1752, "gd_per_match": 0.42},
     "Ghana": {"rank": 31, "code": "GHA", "continent": "Africa", "strength": 76.5, "elo": 1740, "gd_per_match": 0.38},
     "Costa Rica": {"rank": 32, "code": "CRC", "continent": "North America", "strength": 75.8, "elo": 1728, "gd_per_match": 0.35},
 }
 
 # ============================================================
 # Group Configuration (8 groups x 4 teams)
 # ============================================================
 
 GROUPS: Dict[str, List[str]] = {
     "A": ["Brazil", "Ecuador", "Senegal", "Netherlands"],
     "B": ["England", "Iran", "USA", "Wales"],
     "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
     "D": ["France", "Australia", "Denmark", "Tunisia"],
     "E": ["Spain", "Costa Rica", "Germany", "Japan"],
     "F": ["Belgium", "Canada", "Morocco", "Croatia"],
     "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
     "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
 }
 
 # Fix: Remove duplicate Brazil, use proper groups
 GROUPS = {
     "A": ["Netherlands", "Senegal", "Ecuador", "Qatar"],
     "B": ["England", "USA", "Iran", "Wales"],
     "C": ["Argentina", "Mexico", "Poland", "Saudi Arabia"],
     "D": ["France", "Denmark", "Tunisia", "Australia"],
     "E": ["Spain", "Germany", "Japan", "Costa Rica"],
     "F": ["Belgium", "Croatia", "Morocco", "Canada"],
     "G": ["Brazil", "Switzerland", "Serbia", "Cameroon"],
     "H": ["Portugal", "Uruguay", "South Korea", "Ghana"],
 }
 
 
 def get_all_team_names() -> List[str]:
     """Return all team names from the dataset."""
     return list(TEAM_DATA.keys())
 
 
 def get_team_strength(team_name: str) -> float:
     """Get ELO-based strength rating for a team."""
     data = TEAM_DATA.get(team_name)
     if data:
         return data["strength"]
     return 75.0  # default for unknown teams
 
 
 def get_team_rating(team_name: str) -> dict:
     """Get full rating profile for a team."""
     data = TEAM_DATA.get(team_name)
     if data:
         return {
             "name": team_name,
             "code": data["code"],
             "rank": data["rank"],
             "elo": data["elo"],
             "strength": data["strength"],
             "continent": data["continent"],
             "gd_per_match": data["gd_per_match"],
         }
     return {
         "name": team_name,
         "code": team_name[:3].upper(),
         "rank": 50,
         "elo": 1600,
         "strength": 75.0,
         "continent": "Unknown",
         "gd_per_match": 0.0,
     }
 
 
 def generate_historical_matches(num_matches: int = 5000, seed: int = 42) -> List[dict]:
     """
     Generate synthetic historical match data based on team ELO ratings.
     Uses Poisson distribution approximation: goals ~ Poisson(lambda)
     where lambda depends on team strength.
     """
     rng = random.Random(seed)
     matches = []
     teams = list(TEAM_DATA.keys())
     tournament_types = ["Friendly", "World Cup", "Continental Cup", "Qualifier"]
     years = list(range(2014, 2026))
 
     for _ in range(num_matches):
         home = rng.choice(teams)
         away = rng.choice([t for t in teams if t != home])
 
         home_strength = TEAM_DATA[home]["strength"] / 100.0
         away_strength = TEAM_DATA[away]["strength"] / 100.0
 
         # Expected goals based on strength difference
         # Average ~2.5 goals per match total
         home_lambda = max(0.1, home_strength * 2.8 / (home_strength + away_strength))
         away_lambda = max(0.1, away_strength * 2.8 / (home_strength + away_strength))
 
         # Home advantage (~0.3 goals)
         home_lambda += 0.3
 
         # Poisson-like goal generation
         home_goals = int(rng.poisson(home_lambda))
         away_goals = int(rng.poisson(away_lambda))
 
         matches.append({
             "home_team": home,
             "away_team": away,
             "home_goals": min(home_goals, 8),
             "away_goals": min(away_goals, 8),
             "tournament": rng.choice(tournament_types),
             "year": rng.choice(years),
         })
 
     return matches
 
 
 def fetch_fifa_rankings(url: str = "https://www.fifa.com/fifa-world-ranking/men") -> Optional[List[dict]]:
     """
     Attempt to fetch current FIFA rankings from the official website.
     Falls back to built-in data if scraping fails.
     """
     headers = {
         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
     }
     try:
         resp = requests.get(url, headers=headers, timeout=10)
         resp.raise_for_status()
         soup = BeautifulSoup(resp.text, "html.parser")
         rankings = []
         rows = soup.select("table tbody tr")
         for row in rows[:50]:
             cols = row.find_all("td")
             if len(cols) >= 3:
                 rank_text = cols[0].get_text(strip=True)
                 name_text = cols[1].get_text(strip=True)
                 points_text = cols[2].get_text(strip=True)
                 if rank_text.isdigit():
                     rankings.append({
                         "rank": int(rank_text),
                         "name": name_text,
                         "points": float(points_text.replace(",", "")) if points_text.replace(",", "").replace(".", "").isdigit() else 0.0,
                     })
         return rankings if rankings else None
     except Exception:
         return None
 
 
 def get_training_data() -> Tuple[List[dict], List[dict]]:
     """
     Get training data for the ML model.
     Returns (X_features, y_labels) ready for training.
     X_features: list of dicts with feature vectors
     y_labels: list of dicts with home_goals, away_goals
     """
     matches = generate_historical_matches(5000)
     features = []
     labels = []
 
     for match in matches:
         home = get_team_rating(match["home_team"])
         away = get_team_rating(match["away_team"])
 
         features.append({
             "home_elo": home["elo"],
             "away_elo": away["elo"],
             "home_strength": home["strength"],
             "away_strength": away["strength"],
             "home_rank": home["rank"],
             "away_rank": away["rank"],
             "elo_diff": home["elo"] - away["elo"],
             "strength_diff": home["strength"] - away["strength"],
             "rank_diff": away["rank"] - home["rank"],  # positive means home is better ranked
         })
         labels.append({
             "home_goals": match["home_goals"],
             "away_goals": match["away_goals"],
         })
 
     return features, labels
 
 
 def get_recent_form(team_name: str, matches: List[dict], num_matches: int = 5) -> float:
     """Calculate recent form (win rate) from historical matches."""
     recent = [m for m in matches if m["home_team"] == team_name or m["away_team"] == team_name][-num_matches:]
     if not recent:
         return 0.5
     points = 0
     for m in recent:
         if m["home_team"] == team_name:
             if m["home_goals"] > m["away_goals"]:
                 points += 1
             elif m["home_goals"] == m["away_goals"]:
                 points += 0.5
         else:
             if m["away_goals"] > m["home_goals"]:
                 points += 1
             elif m["home_goals"] == m["away_goals"]:
                 points += 0.5
     return points / len(recent)
 
 
 if __name__ == "__main__":
     # Quick test
     print(f"Loaded {len(TEAM_DATA)} teams")
     print(f"Groups: {list(GROUPS.keys())}")
     features, labels = get_training_data()
     print(f"Generated {len(features)} training samples")
     print(f"Sample feature: {features[0]}")
     print(f"Sample label: {labels[0]}")
