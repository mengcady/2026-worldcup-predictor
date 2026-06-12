/**
* World Cup Predictor - Frontend Application
* Handles WebSocket communication and UI updates.
*/

(function () {
   "use strict";

   // ============================================================
   // State
   // ============================================================

   let socket = null;
   let state = {
       tournamentRunning: false,
       modelTrained: false,
       currentTab: "groups",
       matchResults: {},
       groupStandings: {},
       knockoutMatches: {},
       winner: null,
   };

   // ============================================================
   // DOM References
   // ============================================================

   const $ = (sel) => document.querySelector(sel);
   const $$ = (sel) => document.querySelectorAll(sel);

   const dom = {
       statusBadge: $("#status-badge"),
       statusText: $("#status-text"),
       trainBtn: $("#train-btn"),
       simulateBtn: $("#simulate-btn"),
       predictBtn: $("#predict-btn"),
       homeSelect: $("#home-team"),
       awaySelect: $("#away-team"),
       predictionResult: $("#prediction-result"),
       groupsTab: $("#tab-groups"),
       knockoutTab: $("#tab-knockout"),
       groupsContent: $("#groups-content"),
       knockoutContent: $("#knockout-content"),
       groupsContainer: $("#groups-container"),
       matchesContainer: $("#matches-container"),
       bracketContainer: $("#bracket-container"),
       winnerBanner: $("#winner-banner"),
       winnerName: $("#winner-name"),
       trainingLog: $("#training-log"),
   };

   // ============================================================
   // SocketIO Connection
   // ============================================================

   function connectSocket() {
       socket = io();

       socket.on("connect", function () {
           console.log("[WS] Connected");
           addLog("Connected to server");
       });

       socket.on("connected", function (data) {
           console.log("[WS] Handshake received", data);
           state.modelTrained = data.model_trained || false;
           if (state.modelTrained) {
               setStatus("ready", "Model Ready");
           } else {
               setStatus("training", "Training model...");
               socket.emit("request_training");
           }
           populateTeamSelects(data.teams || []);
       });

       socket.on("training_status", function (data) {
           handleTrainingStatus(data);
       });

       socket.on("tournament_update", function (data) {
           console.log("[WS] Tournament update:", data.type);
       });

       socket.on("live_update", function (data) {
           handleLiveUpdate(data);
       });

       socket.on("match_result", function (data) {
           handleMatchResult(data);
       });

       socket.on("group_update", function (data) {
           renderGroupTables(data);
       });

       socket.on("group_stage_complete", function (data) {
           renderGroupTables(data);
           addLog("Group stage complete!");
       });

       socket.on("knockout_result", function (data) {
           handleKnockoutResult(data);
       });

       socket.on("knockout_round_complete", function (data) {
           addLog("Knockout round complete: " + data.round);
       });

       socket.on("tournament_complete", function (data) {
           state.tournamentRunning = false;
           state.winner = data.winner;
           dom.simulateBtn.disabled = false;
           showWinner(data.winner);
           addLog("Tournament complete! Winner: " + data.winner);
       });

       socket.on("prediction_result", function (data) {
           renderPredictionResult(data);
       });

       socket.on("disconnect", function () {
           console.log("[WS] Disconnected");
           setStatus("training", "Disconnected - reconnecting...");
       });
   }

   // ============================================================
   // Status & Logging
   // ============================================================

   function setStatus(type, text) {
       dom.statusBadge.className = "status-badge " + type;
       dom.statusText.textContent = text;
   }

   function addLog(message) {
       var entry = document.createElement("div");
       entry.className = "log-entry";
       var time = new Date().toLocaleTimeString();
       entry.innerHTML = '<span class="time">[' + time + ']</span> ' + message;
       dom.trainingLog.appendChild(entry);
       dom.trainingLog.scrollTop = dom.trainingLog.scrollHeight;
       if (dom.trainingLog.children.length > 50) {
           dom.trainingLog.removeChild(dom.trainingLog.firstChild);
       }
   }

   // ============================================================
   // Training
   // ============================================================

   function handleTrainingStatus(data) {
       if (data.status === "started") {
           setStatus("training", "Training...");
           dom.trainingLog.classList.add("visible");
           addLog("Training started: " + data.message);
       } else if (data.status === "training") {
           addLog(data.message);
       } else if (data.status === "complete") {
           state.modelTrained = true;
           setStatus("ready", "Model Ready");
           addLog("Training complete! " + (data.message || ""));
           dom.trainBtn.disabled = false;
       } else if (data.status === "error") {
           setStatus("ready", "Training Error");
           addLog("Error: " + data.message);
           dom.trainBtn.disabled = false;
       } else if (data.status === "already_training") {
           addLog("Already training in progress");
       }
   }

   function requestTraining() {
       dom.trainBtn.disabled = true;
       socket.emit("request_training");
   }

   // ============================================================
   // Team Selects
   // ============================================================

   function populateTeamSelects(teams) {
       var sorted = teams.slice().sort();
       sorted.forEach(function (team) {
           var opt1 = document.createElement("option");
           opt1.value = team;
           opt1.textContent = team;
           dom.homeSelect.appendChild(opt1);
           var opt2 = document.createElement("option");
           opt2.value = team;
           opt2.textContent = team;
           dom.awaySelect.appendChild(opt2);
       });
       // Set defaults
       if (sorted.length >= 2) {
           dom.homeSelect.value = sorted[0];
           dom.awaySelect.value = sorted[1];
       }
   }

   // ============================================================
   // Prediction
   // ============================================================

   function requestPrediction() {
       var home = dom.homeSelect.value;
       var away = dom.awaySelect.value;
       if (!home || !away || home === away) {
           dom.predictionResult.classList.remove("visible");
           return;
       }
       socket.emit("request_prediction", { home: home, away: away });
   }

   function renderPredictionResult(data) {
       dom.predictionResult.classList.add("visible");

       // Score
       dom.predictionResult.querySelector(".predicted-score").textContent =
           data.most_likely_score || "-";

       // Expected goals
       dom.predictionResult.querySelector(".exp-home").textContent =
           data.expected_home_goals.toFixed(2);
       dom.predictionResult.querySelector(".exp-away").textContent =
           data.expected_away_goals.toFixed(2);

       // Probability bar
       var total = data.win_probability + data.draw_probability + data.lose_probability;
       var wp = ((data.win_probability / total) * 100).toFixed(1);
       var dp = ((data.draw_probability / total) * 100).toFixed(1);
       var lp = ((data.lose_probability / total) * 100).toFixed(1);

       dom.predictionResult.querySelector(".seg-win").style.width = wp + "%";
       dom.predictionResult.querySelector(".seg-draw").style.width = dp + "%";
       dom.predictionResult.querySelector(".seg-lose").style.width = lp + "%";

       dom.predictionResult.querySelector(".p-win").textContent =
           data.home_team + " " + wp + "%";
       dom.predictionResult.querySelector(".p-draw").textContent = "Draw " + dp + "%";
       dom.predictionResult.querySelector(".p-lose").textContent =
           data.away_team + " " + lp + "%";

       // Score heatmap
       renderScoreHeatmap(data.score_probabilities);
   }

   function renderScoreHeatmap(scoreProbs) {
       var container = dom.predictionResult.querySelector(".score-grid");
       container.innerHTML = "";

       var values = Object.values(scoreProbs);
       var maxVal = Math.max.apply(null, values);

       for (var hg = 0; hg <= 6; hg++) {
           for (var ag = 0; ag <= 6; ag++) {
               var key = hg + "-" + ag;
               var prob = scoreProbs[key] || 0;
               var cell = document.createElement("div");
               cell.className = "score-cell";
               cell.textContent = prob > 0 ? prob.toFixed(1) + "%" : "-";

               if (prob >= maxVal * 0.8) {
                   cell.classList.add("max", "high");
               } else if (prob >= maxVal * 0.5) {
                   cell.classList.add("high");
               } else if (prob >= maxVal * 0.2) {
                   cell.classList.add("medium");
               } else if (prob > 0) {
                   cell.classList.add("low");
               } else {
                   cell.classList.add("very-low");
               }

               cell.title = hg + "-" + ag + ": " + prob.toFixed(2) + "%";
               container.appendChild(cell);
           }
       }
   }

   // ============================================================
   // Live Match Updates
   // ============================================================

   function handleLiveUpdate(data) {
       var card = document.getElementById("match-" + data.match_id);
       if (!card) return;

       var scoreEl = card.querySelector(".match-score");
       var minuteEl = card.querySelector(".match-minute");

       if (data.event === "goal") {
           // Flash animation
           card.style.borderColor = "#22c55e";
           setTimeout(function () {
               card.style.borderColor = "";
           }, 1000);
           addLog("GOAL! " + data.score + " (" + data.minute + "')");
       }

       if (data.event === "goal" || data.event === "progress") {
           if (scoreEl) {
               scoreEl.textContent = data.home_goals + "-" + data.away_goals;
               scoreEl.classList.add("live");
           }
           if (minuteEl) {
               minuteEl.textContent = data.minute + "'";
           }
       }
   }

   // ============================================================
   // Match Results (Group Stage)
   // ============================================================

   function createMatchCard(match) {
       var card = document.createElement("div");
       card.className = "match-card";
       card.id = "match-" + match.match_id;

       var groupLabel = match.group ? "Group " + match.group : match.stage;

       card.innerHTML = [
           '<div class="match-header">',
           '  <span>' + groupLabel + '</span>',
           '  <span class="match-minute">' + (match.is_finished ? "FT" : "0'") + '</span>',
           '</div>',
           '<div class="match-teams">',
           '  <div class="team-side home">',
           '    <span class="code">' + (match.home_code || match.home_team.slice(0, 3).toUpperCase()) + '</span>',
           '    <span class="name">' + match.home_team + '</span>',
           '  </div>',
           '  <div class="match-score' + (match.is_finished ? '' : ' live') + '">',
           '    ' + (match.home_goals != null ? match.home_goals + '-' + match.away_goals : '-'),
           '  </div>',
           '  <div class="team-side away">',
           '    <span class="name">' + match.away_team + '</span>',
           '    <span class="code">' + (match.away_code || match.away_team.slice(0, 3).toUpperCase()) + '</span>',
           '  </div>',
           '</div>',
       ].join("");

       // Prediction bar
       if (match.win_prob != null) {
           var bar = document.createElement("div");
           bar.className = "prediction-bar";
           bar.innerHTML = [
               '<div class="bar-home" style="width:' + match.win_prob + '%"></div>',
               '<div class="bar-draw" style="width:' + match.draw_prob + '%"></div>',
               '<div class="bar-away" style="width:' + match.lose_prob + '%"></div>',
           ].join("");
           card.appendChild(bar);

           var labels = document.createElement("div");
           labels.className = "proba-labels";
           labels.innerHTML = [
               '<span>' + match.home_team + ' ' + match.win_prob + '%</span>',
               '<span>Draw ' + match.draw_prob + '%</span>',
               '<span>' + match.lose_prob + '% ' + match.away_team + '</span>',
           ].join("");
           card.appendChild(labels);

           var exp = document.createElement("div");
           exp.className = "expected-goals";
           exp.innerHTML = [
               '<span>xG: ' + (match.home_expected || 0).toFixed(2) + '</span>',
               '<span>Most Likely: ' + (match.most_likely_score || "-") + '</span>',
               '<span>xG: ' + (match.away_expected || 0).toFixed(2) + '</span>',
           ].join("");
           card.appendChild(exp);
       }

       return card;
   }

   function handleMatchResult(data) {
       var existing = document.getElementById("match-" + data.match_id);
       if (existing) {
           existing.replaceWith(createMatchCard(data));
       } else {
           dom.matchesContainer.appendChild(createMatchCard(data));
       }
       state.matchResults[data.match_id] = data;
   }

   // ============================================================
   // Group Tables
   // ============================================================

   function renderGroupTables(data) {
       dom.groupsContainer.innerHTML = "";
       var groups = Object.keys(data).sort();

       groups.forEach(function (groupName) {
           var teams = data[groupName];
           var card = document.createElement("div");
           card.className = "group-card";

           var header = document.createElement("div");
           header.className = "group-header";
           header.innerHTML = '<h3>Group ' + groupName + '</h3>';
           card.appendChild(header);

           var table = document.createElement("table");
           table.className = "group-table";
           table.innerHTML = [
               '<thead><tr>',
               '<th>Team</th><th>P</th><th>W</th><th>D</th><th>L</th>',
               '<th>GF</th><th>GA</th><th>GD</th><th>Pts</th>',
               '</tr></thead>',
           ].join("");

           var tbody = document.createElement("tbody");
           teams.forEach(function (team, idx) {
               var tr = document.createElement("tr");
               var isQualified = idx < 2;
               if (isQualified) tr.classList.add("qualified");
               tr.innerHTML = [
                   '<td><span class="team-name"><span class="team-code">' +
                       (team.code || "") + '</span>' + team.name + '</span></td>',
                   '<td>' + team.played + '</td>',
                   '<td>' + team.won + '</td>',
                   '<td>' + team.drawn + '</td>',
                   '<td>' + team.lost + '</td>',
                   '<td>' + team.goals_for + '</td>',
                   '<td>' + team.goals_against + '</td>',
                   '<td>' + (team.goal_diff > 0 ? "+" : "") + team.goal_diff + '</td>',
                   '<td><strong>' + team.points + '</strong></td>',
               ].join("");
               tbody.appendChild(tr);
           });

           table.appendChild(tbody);
           card.appendChild(table);
           dom.groupsContainer.appendChild(card);
       });

       state.groupStandings = data;
   }

   // ============================================================
   // Knockout Bracket
   // ============================================================

   function handleKnockoutResult(data) {
       var round = data.round;
       var match = data.match;

       if (!state.knockoutMatches[round]) {
           state.knockoutMatches[round] = [];
       }
       state.knockoutMatches[round].push(match);
       renderBracket();
   }

   function renderBracket() {
       var bracket = state.knockoutMatches;
       dom.bracketContainer.innerHTML = "";

       var rounds = [
           { key: "round_of_16", label: "Round of 16" },
           { key: "quarter_final", label: "Quarter-Finals" },
           { key: "semi_final", label: "Semi-Finals" },
           { key: "third_place", label: "3rd Place" },
           { key: "final", label: "Final" },
       ];

       var bracketDiv = document.createElement("div");
       bracketDiv.className = "bracket";

       rounds.forEach(function (r) {
           var matches = bracket[r.key] || [];
           var roundDiv = document.createElement("div");
           roundDiv.className = "bracket-round";

           var title = document.createElement("div");
           title.className = "bracket-round-title";
           title.textContent = r.label;
           roundDiv.appendChild(title);

           if (matches.length === 0) {
               if (r.key === "final" || r.key === "third_place") {
                   var placeholders = r.key === "final" ? 1 : 1;
                   for (var i = 0; i < placeholders; i++) {
                       var ph = document.createElement("div");
                       ph.className = "bracket-match";
                       ph.innerHTML = '<div class="b-team pending"><span>TBD</span></div>';
                       roundDiv.appendChild(ph);
                   }
               } else {
                   var placeholderCount = r.key === "round_of_16" ? 8 :
                       r.key === "quarter_final" ? 4 : 2;
                   for (var j = 0; j < placeholderCount; j++) {
                       var ph2 = document.createElement("div");
                       ph2.className = "bracket-match";
                       ph2.style.opacity = "0.3";
                       ph2.innerHTML = '<div class="b-team pending"><span>TBD</span></div>';
                       roundDiv.appendChild(ph2);
                   }
               }
           } else {
               matches.forEach(function (m) {
                   var md = document.createElement("div");
                   md.className = "bracket-match";
                   if (m.stage === "final") md.classList.add("bracket-final");

                   var homeWon = m.home_goals != null && m.away_goals != null &&
                       m.home_goals >= m.away_goals;

                   md.innerHTML = [
                       '<div class="b-team' + (m.home_goals != null ? (homeWon ? ' winner' : ' loser') : ' pending') + '">',
                       '  <span>' + m.home_team + '</span>',
                       '  <span class="score">' + (m.home_goals != null ? m.home_goals : "?") + '</span>',
                       '</div>',
                       '<div class="b-team' + (m.away_goals != null ? (!homeWon ? ' winner' : ' loser') : ' pending') + '">',
                       '  <span>' + m.away_team + '</span>',
                       '  <span class="score">' + (m.away_goals != null ? m.away_goals : "?") + '</span>',
                       '</div>',
                   ].join("");

                   if (m.stage === "final" && state.winner) {
                       var label = document.createElement("div");
                       label.style.cssText =
                           "text-align:center;font-size:9px;color:#f59e0b;margin-top:4px;font-weight:600;";
                       label.textContent = "Champion";
                       md.appendChild(label);
                   }

                   roundDiv.appendChild(md);
               });
           }

           bracketDiv.appendChild(roundDiv);
       });

       dom.bracketContainer.appendChild(bracketDiv);
   }

   // ============================================================
   // Winner Banner
   // ============================================================

   function showWinner(winner) {
       dom.winnerBanner.classList.add("visible");
       dom.winnerName.textContent = winner + " 馃弳";
   }

   // ============================================================
   // Tournament Simulation
   // ============================================================

   function startSimulation() {
       if (state.tournamentRunning) return;
       state.tournamentRunning = true;
       dom.simulateBtn.disabled = true;
       dom.winnerBanner.classList.remove("visible");
       state.knockoutMatches = {};
       state.matchResults = {};
       state.winner = null;

       dom.matchesContainer.innerHTML =
           '<div class="loading-text"><span class="spinner"></span> Simulating matches...</div>';
       dom.groupsContainer.innerHTML =
           '<div class="loading-text"><span class="spinner"></span> Waiting for results...</div>';
       dom.bracketContainer.innerHTML =
           '<div class="loading-text">Waiting for group stage...</div>';

       addLog("Starting tournament simulation...");

       fetch("/api/start_tournament", { method: "POST" })
           .then(function (r) { return r.json(); })
           .then(function (data) {
               addLog("Simulation: " + data.status);
           })
           .catch(function (err) {
               addLog("Error starting simulation: " + err);
               state.tournamentRunning = false;
               dom.simulateBtn.disabled = false;
           });
   }

   // ============================================================
   // Tab Switching
   // ============================================================

   function switchTab(tab) {
       state.currentTab = tab;
       $$(".tab").forEach(function (t) { t.classList.remove("active"); });
       $$(".tab-content").forEach(function (c) { c.classList.remove("active"); });

       if (tab === "groups") {
           dom.groupsTab.classList.add("active");
           dom.groupsContent.classList.add("active");
       } else {
           dom.knockoutTab.classList.add("active");
           dom.knockoutContent.classList.add("active");
           renderBracket();
       }
   }

   // ============================================================
   // Initialization
   // ============================================================

   function init() {
       // Tab switching
       dom.groupsTab.addEventListener("click", function () { switchTab("groups"); });
       dom.knockoutTab.addEventListener("click", function () { switchTab("knockout"); });

       // Training
       dom.trainBtn.addEventListener("click", requestTraining);

       // Prediction
       dom.predictBtn.addEventListener("click", requestPrediction);
       dom.homeSelect.addEventListener("change", requestPrediction);
       dom.awaySelect.addEventListener("change", requestPrediction);

       // Simulation
       dom.simulateBtn.addEventListener("click", startSimulation);

       // Connect
       connectSocket();
   }

   // Start when DOM is ready
   if (document.readyState === "loading") {
       document.addEventListener("DOMContentLoaded", init);
   } else {
       init();
   }
})();
