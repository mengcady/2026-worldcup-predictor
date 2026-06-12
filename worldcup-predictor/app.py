"""
World Cup Score Prediction System

Flask + Flask-SocketIO web application for real-time
World Cup match simulation and score prediction.
"""

import json
import os
import threading
import time
import uuid
from datetime import datetime

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

from models.data_fetcher import (
    GROUPS, TEAM_DATA, get_all_team_names, get_team_rating, get_training_data,
)
from models.predictor import ScorePredictor, ELOBasedPredictor
from models.tournament import Tournament, Match

# ============================================================
# Application Setup
# ============================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24).hex()
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Global state
predictor = ScorePredictor()
tournament = Tournament(predictor)
model_trained = False
tournament_running = False
training_thread = None
sim_thread = None


# ============================================================
# Model Training
# ============================================================

def train_model():
    """Train the ML model in background thread."""
    global predictor, model_trained

    socketio.emit("training_status", {"status": "started", "message": "正在收集训练数据..."})
    time.sleep(0.5)

    features, labels = get_training_data()
    socketio.emit("training_status", {
        "status": "training",
        "message": f"Training on {len(features)} samples..."
    })

    try:
        metrics = predictor.train(features, labels, verbose=False)
        model_trained = True
        socketio.emit("training_status", {
            "status": "complete",
            "message": f"模型训练完成！误差: {metrics.get("rf_home_mae", 0):.3f}",
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
        })
    except Exception as e:
        socketio.emit("training_status", {
            "status": "error",
            "message": f"训练失败: {str(e)}",
        })


# ============================================================
# Tournament Simulation
# ============================================================

def run_tournament_simulation():
    """Run full tournament simulation, emitting events via SocketIO."""
    global tournament, tournament_running, predictor

    tournament_running = True
    tournament = Tournament(predictor)

    def socket_callback(event_type, data):
        socketio.emit("tournament_update", {"type": event_type, "data": data})

    tournament.on_update(socket_callback)
    tournament.build_group_stage()
    tournament.predict_all_matches()

    # Emit initial predictions
    socketio.emit("tournament_update", {
        "type": "predictions",
        "data": [m.to_dict() for m in tournament.group_matches],
    })

    # Simulate group stage with live updates
    for i, match in enumerate(tournament.group_matches):
        tournament.predict_match(match)
        home_g, away_g = tournament.simulate_match_result(match)

        # Simulate live match progression
        match.home_goals = 0
        match.away_goals = 0

        # Build goal timeline
        goal_timeline = []
        total_goals = home_g + away_g

        import random
        if total_goals > 0:
            minutes_pool = list(range(5, 91))
            weights = [1.0 + (m / 90) * 2 for m in minutes_pool]
            chosen = sorted(random.choices(minutes_pool, weights=weights, k=total_goals))
            hg_count, ag_count = 0, 0
            for minute in chosen:
                if hg_count < home_g and (ag_count >= away_g or random.random() < home_g / max(1, total_goals)):
                    goal_timeline.append({"minute": minute, "team": "home", "scorer": f"Player {hg_count+1}"})
                    hg_count += 1
                else:
                    goal_timeline.append({"minute": minute, "team": "away", "scorer": f"Player {ag_count+1}"})
                    ag_count += 1
            goal_timeline.sort(key=lambda e: e["minute"])

        # Simulate minute-by-minute
        timeline_idx = 0
        match_status = {
            "match_id": match.match_id,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "home_code": get_team_rating(match.home_team).get("code", match.home_team[:3].upper()),
            "away_code": get_team_rating(match.away_team).get("code", match.away_team[:3].upper()),
            "away_code": get_team_rating(match.away_team).get("code", match.away_team[:3].upper()),
            "home_cn": TEAM_DATA.get(match.home_team, {}).get("name_cn", match.home_team),
            "away_cn": TEAM_DATA.get(match.away_team, {}).get("name_cn", match.away_team),
            "expected_home": round(match.home_expected, 2),
            "expected_away": round(match.away_expected, 2),
            "most_likely_score": match.most_likely_score,
            "win_prob": match.win_prob,
            "draw_prob": match.draw_prob,
            "lose_prob": match.lose_prob,
        }

        for minute in range(0, 91, 5):
            current_hg = 0
            current_ag = 0
            while timeline_idx < len(goal_timeline) and goal_timeline[timeline_idx]["minute"] <= minute:
                g = goal_timeline[timeline_idx]
                if g["team"] == "home":
                    current_hg += 1
                else:
                    current_ag += 1
                # Emit goal event
                socketio.emit("live_update", {
                    **match_status,
                    "minute": g["minute"],
                    "home_goals": current_hg if g["team"] == "home" else current_hg,
                    "away_goals": current_ag if g["team"] == "away" else current_ag,
                    "goal": g,
                    "event": "goal",
                    "score": f"{current_hg}-{current_ag}",
                })
                time.sleep(0.3)
                timeline_idx += 1

            if minute < 90:
                socketio.emit("live_update", {
                    **match_status,
                    "minute": minute,
                    "home_goals": current_hg,
                    "away_goals": current_ag,
                    "event": "progress",
                    "score": f"{current_hg}-{current_ag}",
                })
                time.sleep(0.3)

        # Final result
        match.home_goals = home_g
        match.away_goals = away_g
        match.is_finished = True
        match.minute = 90
        match.goal_timeline = goal_timeline

        # Update group standings
        tournament.groups[match.group][match.home_team].record_match(home_g, away_g)
        tournament.groups[match.group][match.away_team].record_match(away_g, home_g)

        socketio.emit("match_result", match.to_dict())
        socketio.emit("group_update", tournament.get_group_standings())
        time.sleep(0.5)

    # Sort groups
    for group_name in tournament.groups:
        standings = sorted(
            tournament.groups[group_name].values(),
            key=lambda s: (s.points, s.goal_diff, s.goals_for),
            reverse=True
        )
        for rank, team in enumerate(standings, 1):
            team.group_rank = rank

    socketio.emit("group_stage_complete", tournament.get_group_standings())

    # Build and simulate knockout stage
    tournament.build_knockout_stage()

    # Round of 16
    round_names = [
        ("round_of_16", "Round of 16"),
        ("quarter_final", "Quarter-Final"),
        ("semi_final", "Semi-Final"),
    ]

    for rn, rlabel in round_names:
        for match in tournament.knockout_matches[rn]:
            tournament.predict_match(match)
            hg, ag = tournament.simulate_match_result(match)

            match.home_goals = hg
            match.away_goals = ag
            match.is_finished = True
            match.minute = 90 + (30 if hg == ag else 0)

            socketio.emit("knockout_result", {
                "round": rn,
                "round_label": rlabel,
                "match": match.to_dict(),
            })
            time.sleep(0.5)

        socketio.emit("knockout_round_complete", {"round": rn})

    # Semi finals
    r16_winners = [m.home_team if (m.home_goals or 0) > (m.away_goals or 0) else m.away_team
                   for m in tournament.knockout_matches["round_of_16"]]
    qf_winners = [m.home_team if (m.home_goals or 0) > (m.away_goals or 0) else m.away_team
                  for m in tournament.knockout_matches["quarter_final"]]

    tournament.knockout_matches["semi_final"] = [
        Match(qf_winners[0], qf_winners[1], stage="semi_final", match_id="SF1"),
        Match(qf_winners[2], qf_winners[3], stage="semi_final", match_id="SF2"),
    ]

    for match in tournament.knockout_matches["semi_final"]:
        tournament.predict_match(match)
        hg, ag = tournament.simulate_match_result(match)
        match.home_goals = hg
        match.away_goals = ag
        match.is_finished = True
        match.minute = 90
        socketio.emit("knockout_result", {
            "round": "semi_final",
            "round_label": "Semi-Final",
            "match": match.to_dict(),
        })
        time.sleep(0.5)

    sf_winners = [m.home_team if (m.home_goals or 0) > (m.away_goals or 0) else m.away_team
                  for m in tournament.knockout_matches["semi_final"]]
    sf_losers = [m.away_team if (m.home_goals or 0) > (m.away_goals or 0) else m.home_team
                 for m in tournament.knockout_matches["semi_final"]]

    # Third place
    third = Match(sf_losers[0], sf_losers[1], stage="third_place", match_id="3P")
    tournament.predict_match(third)
    hg, ag = tournament.simulate_match_result(third)
    third.home_goals = hg
    third.away_goals = ag
    third.is_finished = True
    third.minute = 90
    tournament.knockout_matches["third_place"] = [third]
    socketio.emit("knockout_result", {
        "round": "third_place",
        "round_label": "Third Place",
        "match": third.to_dict(),
    })
    time.sleep(0.5)

    # Final
    final = Match(sf_winners[0], sf_winners[1], stage="final", match_id="FIN")
    tournament.predict_match(final)
    hg, ag = tournament.simulate_match_result(final)
    final.home_goals = hg
    final.away_goals = ag
    final.is_finished = True
    final.minute = 90 if hg != ag else 120
    tournament.knockout_matches["final"] = [final]
    tournament.winner = final.home_team if (hg or 0) > (ag or 0) else final.away_team
    socketio.emit("knockout_result", {
        "round": "final",
        "round_label": "Final",
        "match": final.to_dict(),
    })
    time.sleep(0.5)

    socketio.emit("tournament_complete", {"winner": tournament.winner})
    tournament_running = False


# ============================================================
# Flask Routes
# ============================================================


@app.route("/")
def index():
    """Main page."""
    return render_template("index.html")


@app.route("/api/teams")
def api_teams():
    """Get all team data."""
    teams = {}
    for name, data in TEAM_DATA.items():
        teams[name] = get_team_rating(name)
    return jsonify(teams)


@app.route("/api/groups")
def api_groups():
    """Get group configuration."""
    return jsonify(GROUPS)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict a match between two teams."""
    data = request.get_json()
    home = data.get("home", "")
    away = data.get("away", "")

    if home not in TEAM_DATA or away not in TEAM_DATA:
        return jsonify({"error": "Team not found"}), 404

    home_rating = get_team_rating(home)
    away_rating = get_team_rating(away)

    if predictor.is_trained:
        pred = predictor.predict(home_rating, away_rating)
    else:
        pred = ELOBasedPredictor.predict(home_rating, away_rating)

    return jsonify(pred)


@app.route("/api/status")
def api_status():
    """Get system status."""
    return jsonify({
        "model_trained": model_trained,
        "tournament_running": tournament_running,
        "teams_available": len(TEAM_DATA),
        "groups": len(GROUPS),
    })


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Trigger model retraining."""
    global training_thread
    if training_thread and training_thread.is_alive():
        return jsonify({"status": "already_training"})
    training_thread = threading.Thread(target=train_model, daemon=True)
    training_thread.start()
    return jsonify({"status": "training_started"})


@app.route("/api/start_tournament", methods=["POST"])
def api_start_tournament():
    """Start tournament simulation."""
    global sim_thread, tournament_running
    if tournament_running:
        return jsonify({"status": "already_running"})
    sim_thread = threading.Thread(target=run_tournament_simulation, daemon=True)
    sim_thread.start()
    return jsonify({"status": "simulation_started"})


# ============================================================
# SocketIO Events
# ============================================================

@socketio.on("connect")
def handle_connect():
    """Client connected."""
    emit("connected", {
        "status": "ok",
        "model_trained": model_trained,
        "teams": list(TEAM_DATA.keys()),
        "teams_cn": {k: v.get("name_cn", k) for k, v in TEAM_DATA.items()},
        "groups": GROUPS,
    })


@socketio.on("request_training")
def handle_request_training():
    """Client requests model training."""
    global training_thread
    if training_thread and training_thread.is_alive():
        emit("training_status", {"status": "already_training"})
        return
    training_thread = threading.Thread(target=train_model, daemon=True)
    training_thread.start()


@socketio.on("request_prediction")
def handle_request_prediction(data):
    """Client requests prediction for a match."""
    home = data.get("home", "")
    away = data.get("away", "")
    if home in TEAM_DATA and away in TEAM_DATA:
        home_r = get_team_rating(home)
        away_r = get_team_rating(away)
        if predictor.is_trained:
            pred = predictor.predict(home_r, away_r)
        else:
            pred = ELOBasedPredictor.predict(home_r, away_r)
        emit("prediction_result", pred)


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  FIFA 世界杯比分预测系统")
    print("=" * 60)
    print()
    print(f'  已加载球队: {len(TEAM_DATA)}')
    print(f'  小组数: {len(GROUPS)}')
    print()
    print("  正在自动训练模型...")
    training_thread = threading.Thread(target=train_model, daemon=True)
    training_thread.start()
    print()
    print("  Server starting at http://127.0.0.1:5000")
    print("=" * 60)
    socketio.run(app, host="127.0.0.1", port=5000, debug=True, allow_unsafe_werkzeug=True)
