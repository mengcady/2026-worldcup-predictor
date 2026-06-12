"""
World Cup Tournament Simulation

Handles group stage, knockout bracket, and match simulation.
Uses the ML predictor to determine match outcomes.
"""

import random
import threading
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from .data_fetcher import GROUPS, TEAM_DATA, get_team_rating
from .predictor import ScorePredictor


class TeamStats:
    """Represents a team's group stage statistics."""

    def __init__(self, name: str):
        self.name = name
        self.played = 0
        self.won = 0
        self.drawn = 0
        self.lost = 0
        self.goals_for = 0
        self.goals_against = 0
        self.points = 0
        self.group_rank = 0

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against

    def record_match(self, goals_for: int, goals_against: int):
        """Record a match result."""
        self.played += 1
        self.goals_for += goals_for
        self.goals_against += goals_against
        if goals_for > goals_against:
            self.won += 1
            self.points += 3
        elif goals_for == goals_against:
            self.drawn += 1
            self.points += 1
        else:
            self.lost += 1

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "code": TEAM_DATA.get(self.name, {}).get("code", self.name[:3].upper()),
            "played": self.played,
            "won": self.won,
            "drawn": self.drawn,
            "lost": self.lost,
            "goals_for": self.goals_for,
            "goals_against": self.goals_against,
            "goal_diff": self.goal_diff,
            "points": self.points,
            "rank": self.group_rank,
        }


class Match:
    """Represents a single match."""

    def __init__(self, home_team: str, away_team: str, stage: str = "group",
                 group: str = "", match_id: str = ""):
        self.home_team = home_team
        self.away_team = away_team
        self.stage = stage
        self.group = group
        self.match_id = match_id
        self.home_goals = None
        self.away_goals = None
        self.home_expected = 0.0
        self.away_expected = 0.0
        self.win_prob = 0.0
        self.draw_prob = 0.0
        self.lose_prob = 0.0
        self.most_likely_score = ""
        self.score_probs = {}
        self.is_finished = False
        self.minute = 0
        self.events: List[dict] = []  # goals, cards, etc.
        self.goal_timeline: List[dict] = []

    @property
    def score_text(self) -> str:
        if self.home_goals is None:
            return "-"
        return f"{self.home_goals}-{self.away_goals}"

    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_code": TEAM_DATA.get(self.home_team, {}).get("code", self.home_team[:3].upper()),
            "away_code": TEAM_DATA.get(self.away_team, {}).get("code", self.away_team[:3].upper()),
            "home_strength": TEAM_DATA.get(self.home_team, {}).get("strength", 75),
            "away_strength": TEAM_DATA.get(self.away_team, {}).get("strength", 75),
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "stage": self.stage,
            "group": self.group,
            "is_finished": self.is_finished,
            "minute": self.minute,
            "home_expected": round(self.home_expected, 2),
            "away_expected": round(self.away_expected, 2),
            "win_prob": self.win_prob,
            "draw_prob": self.draw_prob,
            "lose_prob": self.lose_prob,
            "most_likely_score": self.most_likely_score,
            "events": self.events,
            "goal_timeline": self.goal_timeline,
        }


class Tournament:
    """
    Full World Cup tournament simulation.
    Manages group stage, knockout rounds, and live match simulation.
    """

    # Knockout bracket pairings for Round of 16
    R16_PAIRINGS = [
        ("A", 1, "B", 2), ("C", 1, "D", 2),
        ("E", 1, "F", 2), ("G", 1, "H", 2),
        ("B", 1, "A", 2), ("D", 1, "C", 2),
        ("F", 1, "E", 2), ("H", 1, "G", 2),
    ]

    def __init__(self, predictor: Optional[ScorePredictor] = None):
        self.predictor = predictor or ScorePredictor()
        self.groups: Dict[str, Dict[str, TeamStats]] = {}
        self.group_matches: List[Match] = []
        self.knockout_matches: Dict[str, List[Match]] = {
            "round_of_16": [],
            "quarter_final": [],
            "semi_final": [],
            "third_place": [],
            "final": [],
        }
        self.current_stage = "group"
        self.current_match_index = 0
        self.all_matches: List[Match] = []
        self.winner = None
        self.running = False
        self._sim_thread = None
        self._callbacks: List[Callable] = []

    def on_update(self, callback: Callable):
        """Register a callback for match updates."""
        self._callbacks.append(callback)

    def _notify(self, event_type: str, data: dict):
        """Notify all callbacks of an event."""
        for cb in self._callbacks:
            try:
                cb(event_type, data)
            except Exception:
                pass

    def build_group_stage(self):
        """Build all group stage matches."""
        self.group_matches = []
        self.groups = {}
        match_id = 0

        for group_name, teams in GROUPS.items():
            self.groups[group_name] = {t: TeamStats(t) for t in teams}
            # Round-robin: each team plays every other team
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    match_id += 1
                    match = Match(
                        home_team=teams[i],
                        away_team=teams[j],
                        stage="group",
                        group=group_name,
                        match_id=f"G{match_id:02d}"
                    )
                    self.group_matches.append(match)

        self.all_matches = list(self.group_matches)
        return self.group_matches

    def predict_match(self, match: Match):
        """Run prediction for a match."""
        home = get_team_rating(match.home_team)
        away = get_team_rating(match.away_team)

        if self.predictor and self.predictor.is_trained:
            pred = self.predictor.predict(home, away)
        else:
            from .predictor import ELOBasedPredictor
            pred = ELOBasedPredictor.predict(home, away)

        match.home_expected = pred["expected_home_goals"]
        match.away_expected = pred["expected_away_goals"]
        match.win_prob = pred["win_probability"]
        match.draw_prob = pred["draw_probability"]
        match.lose_prob = pred["lose_probability"]
        match.most_likely_score = pred["most_likely_score"]
        match.score_probs = pred.get("score_probabilities", {})

    def predict_all_matches(self):
        """Run prediction for all matches."""
        for match in self.all_matches:
            self.predict_match(match)

    def simulate_match_result(self, match: Match) -> Tuple[int, int]:
        """
        Determine actual match result based on predicted probabilities.
        Uses the score probability distribution to randomly select outcome.
        """
        if not match.score_probs:
            self.predict_match(match)

        # Weighted random selection based on score probabilities
        scores = list(match.score_probs.keys())
        weights = [match.score_probs[s] for s in scores]
        total = sum(weights)
        if total <= 0:
            weights = [1.0] * len(scores)
            total = sum(weights)
        probs = [w / total for w in weights]

        chosen = random.choices(scores, weights=probs, k=1)[0]
        home_g, away_g = map(int, chosen.split("-"))
        return home_g, away_g

    def _generate_match_events(self, match: Match, home_goals: int, away_goals: int):
        """Generate minute-by-minute goal timeline for a match."""
        total_goals = home_goals + away_goals
        if total_goals == 0:
            return []

        # Distribute goals across 90 minutes
        minutes_pool = list(range(1, 91))
        # Weight toward later minutes (more goals in second half)
        weights = [1.0 + (m / 90) * 2 for m in minutes_pool]
        chosen_minutes = random.choices(minutes_pool, weights=weights, k=total_goals)
        chosen_minutes.sort()

        events = []
        goal_idx = 0
        for _ in range(home_goals):
            minute = chosen_minutes[goal_idx]
            events.append({
                "type": "goal",
                "team": "home",
                "minute": minute,
                "scorer": f"Player {goal_idx + 1}",
            })
            goal_idx += 1
        for _ in range(away_goals):
            minute = chosen_minutes[goal_idx]
            events.append({
                "type": "goal",
                "team": "away",
                "minute": minute,
                "scorer": f"Player {goal_idx + 1}",
            })
            goal_idx += 1

        # Sort events by minute
        events.sort(key=lambda e: e["minute"])
        return events

    def simulate_group_stage(self, live: bool = True):
        """
        Simulate all group stage matches.
        If live=True, simulates with time delay and callbacks.
        """
        self.current_stage = "group"

        for i, match in enumerate(self.group_matches):
            self.current_match_index = i
            self.predict_match(match)

            home_g, away_g = self.simulate_match_result(match)
            match.home_goals = home_g
            match.away_goals = away_g
            match.is_finished = True
            match.minute = 90

            # Generate events
            match.goal_timeline = self._generate_match_events(match, home_g, away_g)
            match.events = [{"type": "full_time", "minute": 90, "home": home_g, "away": away_g}]

            # Update group standings
            group = match.group
            self.groups[group][match.home_team].record_match(home_g, away_g)
            self.groups[group][match.away_team].record_match(away_g, home_g)

            self._notify("match_result", match.to_dict())

            if live:
                time.sleep(0.5)  # Brief delay between matches

        # Sort groups
        for group_name in self.groups:
            standings = sorted(
                self.groups[group_name].values(),
                key=lambda s: (s.points, s.goal_diff, s.goals_for),
                reverse=True
            )
            for rank, team in enumerate(standings, 1):
                team.group_rank = rank
                self.groups[group_name][team.name] = team

        self._notify("group_stage_complete", self.get_group_standings())

    def get_group_standings(self) -> Dict[str, List[dict]]:
        """Get current group standings."""
        standings = {}
        for group_name, teams in self.groups.items():
            sorted_teams = sorted(
                teams.values(),
                key=lambda s: (s.points, s.goal_diff, s.goals_for),
                reverse=True
            )
            standings[group_name] = [t.to_dict() for t in sorted_teams]
        return standings

    def _get_knockout_teams(self) -> List[str]:
        """Get qualified teams (top 2 from each group)."""
        qualified = []
        for group_name in sorted(self.groups.keys()):
            standings = sorted(
                self.groups[group_name].values(),
                key=lambda s: (s.points, s.goal_diff, s.goals_for),
                reverse=True
            )
            qualified.append((group_name, standings[0].name, standings[1].name))
        return qualified

    def build_knockout_stage(self) -> Dict[str, List[Match]]:
        """
        Build knockout bracket based on group stage results.
        Standard World Cup Round of 16 pairings.
        """
        qualified = self._get_knockout_teams()
        group_winners = {g: w for g, w, _ in qualified}
        group_runners = {g: r for g, _, r in qualified}

        self.knockout_matches = {k: [] for k in self.knockout_matches}

        # Round of 16
        r16_matches = []
        for gw_group, gw_rank, gr_group, gr_rank in self.R16_PAIRINGS:
            winner = group_winners[gw_group]
            runner = group_runners[gr_group]
            match = Match(winner, runner, stage="round_of_16", match_id=f"R16_{len(r16_matches)+1:02d}")
            r16_matches.append(match)

        self.knockout_matches["round_of_16"] = r16_matches
        return self.knockout_matches

    def simulate_knockout_stage(self, live: bool = True):
        """Simulate all knockout rounds."""
        # Round of 16 -> Quarter Finals -> Semi Finals -> Final
        round_names = [
            ("round_of_16", "quarter_final"),
            ("quarter_final", "semi_final"),
            ("semi_final", "third_place_final"),
        ]
        self.current_stage = "round_of_16"

        for from_round, to_round in round_names:
            winners = []
            for i, match in enumerate(self.knockout_matches[from_round]):
                self.current_match_index = i
                self.predict_match(match)

                home_g, away_g = self.simulate_match_result(match)
                match.home_goals = home_g
                match.away_goals = away_g
                match.is_finished = True
                match.minute = 90 + (30 if home_g == away_g else 0)  # extra time if draw

                match.goal_timeline = self._generate_match_events(match, home_g, away_g)
                match.events = [{"type": "full_time", "minute": match.minute, "home": home_g, "away": away_g}]

                if home_g == away_g:
                    # Penalty shootout
                    pen_winner = random.choice([match.home_team, match.away_team])
                    match.events.append({
                        "type": "penalties",
                        "winner": pen_winner,
                        "score": f"{random.randint(3, 5)}-{random.randint(2, 4)}",
                    })

                winner = match.home_team if home_g > away_g else (match.away_team if away_g > home_g else random.choice([match.home_team, match.away_team]))
                winners.append(winner)

                self._notify("match_result", match.to_dict())
                if live:
                    time.sleep(0.5)

            # Build next round
            if to_round == "third_place_final":
                # Semi final winners go to final, losers to third place match
                self.knockout_matches["semi_final"] = [
                    Match(winners[0], winners[1], stage="semi_final", match_id="SF1"),
                    Match(winners[2], winners[3], stage="semi_final", match_id="SF2"),
                ]
            elif to_round == "final":
                pass  # Will be built below

            self._notify(f"{from_round}_complete", {"winners": winners})
            self.current_stage = to_round

        # Semi finals
        semis = self.knockout_matches["semi_final"]
        sf_winners = []
        sf_losers = []
        for match in semis:
            self.predict_match(match)
            hg, ag = self.simulate_match_result(match)
            match.home_goals = hg
            match.away_goals = ag
            match.is_finished = True
            match.minute = 90
            match.goal_timeline = self._generate_match_events(match, hg, ag)
            match.events = [{"type": "full_time", "minute": 90, "home": hg, "away": ag}]
            winner = match.home_team if hg > ag else match.away_team
            loser = match.away_team if hg > ag else match.home_team
            sf_winners.append(winner)
            sf_losers.append(loser)
            self._notify("match_result", match.to_dict())
            if live:
                time.sleep(0.5)

        # Third place match
        third_place = Match(sf_losers[0], sf_losers[1], stage="third_place", match_id="3P")
        self.knockout_matches["third_place"] = [third_place]
        self.predict_match(third_place)
        hg, ag = self.simulate_match_result(third_place)
        third_place.home_goals = hg
        third_place.away_goals = ag
        third_place.is_finished = True
        third_place.minute = 90
        third_place.goal_timeline = self._generate_match_events(third_place, hg, ag)
        third_place.events = [{"type": "full_time", "minute": 90, "home": hg, "away": ag}]
        self._notify("match_result", third_place.to_dict())
        if live:
            time.sleep(0.5)

        # Final
        final = Match(sf_winners[0], sf_winners[1], stage="final", match_id="FIN")
        self.knockout_matches["final"] = [final]
        self.predict_match(final)
        hg, ag = self.simulate_match_result(final)
        final.home_goals = hg
        final.away_goals = ag
        final.is_finished = True
        if hg == ag:
            # Extra time and penalties in final
            final.minute = 120
            pen_winner = random.choice([final.home_team, final.away_team])
            final.events = [
                {"type": "extra_time", "minute": 120},
                {"type": "penalties", "winner": pen_winner, "score": f"{random.randint(3,5)}-{random.randint(2,4)}"},
            ]
            self.winner = pen_winner
        else:
            final.minute = 90
            final.goal_timeline = self._generate_match_events(final, hg, ag)
            final.events = [{"type": "full_time", "minute": 90, "home": hg, "away": ag}]
            self.winner = final.home_team if hg > ag else final.away_team

        final.home_goals = hg
        final.away_goals = ag
        final.is_finished = True
        self._notify("match_result", final.to_dict())
        self._notify("tournament_complete", {"winner": self.winner})
        self.current_stage = "complete"

    def run_full_tournament(self, live: bool = True):
        """Run full tournament: group stage -> knockout stage."""
        self.build_group_stage()
        self.predict_all_matches()
        self.simulate_group_stage(live=live)
        self.build_knockout_stage()
        self.simulate_knockout_stage(live=live)

    def get_bracket_data(self) -> dict:
        """Get bracket data for frontend display."""
        bracket = {}
        for round_name, matches in self.knockout_matches.items():
            bracket[round_name] = [m.to_dict() for m in matches]
        return bracket

    def get_tournament_summary(self) -> dict:
        """Get complete tournament state for frontend."""
        return {
            "current_stage": self.current_stage,
            "group_standings": self.get_group_standings(),
            "group_matches": [m.to_dict() for m in self.group_matches],
            "knockout_matches": self.get_bracket_data(),
            "winner": self.winner,
        }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    def callback(event_type, data):
        print(f"[{event_type}] {type(data).__name__}")

    tourney = Tournament()
    tourney.on_update(callback)
    tourney.run_full_tournament(live=False)

    print(f"\nTournament complete! Winner: {tourney.winner}")
    for group, standings in tourney.get_group_standings().items():
        print(f"\nGroup {group}:")
        for s in standings:
            print(f"  {s['rank']}. {s['name']} - {s['points']}pts ({s['goal_diff']:+d})")
    print(f"\nKnockout bracket:")
    for round_name, matches in tourney.knockout_matches.items():
        print(f"  {round_name}:")
        for m in matches:
            print(f"    {m.home_team} {m.score_text} {m.away_team}")
