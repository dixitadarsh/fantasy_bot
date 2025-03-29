import os
import json
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary, PULP_CBC_CMD

# === Data Loading & Processing Functions ===

def load_json_data(folder_path):
    """Load all JSON files from a folder into a list."""
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    data_list = []
    for file in json_files:
        try:
            with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                data_list.append(data)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return data_list

def extract_player_stats(match_data):
    """
    Processes match data to accumulate batting and bowling stats per player.
    Also extracts basic match context (from innings order) and tracks team performance bonus.
    """
    player_stats = {}
    team_performance = {}  # To track team bonus (e.g. for chasing success)

    for match in match_data:
        info = match.get('info', {})
        # Note: toss info is not used here because we now let the user provide batting_first.
        target = match.get('target', None)

        innings = match.get('innings', [])
        inning_context = {}
        if target and len(innings) >= 2:
            # Assume first innings is defending; second innings is chasing.
            inning_context[list(innings[0].keys())[0]] = 'defending'
            inning_context[list(innings[1].keys())[0]] = 'chasing'
        else:
            for inning in innings:
                inning_context[list(inning.keys())[0]] = 'neutral'

        for inning in innings:
            for team_name, details in inning.items():
                context = inning_context.get(team_name, 'neutral')
                team_performance.setdefault(team_name, {'matches': 0, 'bonus': 0})
                team_performance[team_name]['matches'] += 1
                if context == 'chasing':
                    team_performance[team_name]['bonus'] += 1

                # Batting extraction
                batting = details.get('batting', {})
                batting_order = details.get('batting_order', {})  # e.g., {"Player A": 1, ...}
                for player, stats in batting.items():
                    if player not in player_stats:
                        player_stats[player] = {
                            'runs': 0, 'wickets': 0, 'balls': 0, 'matches': 0,
                            'strike_rate': 0, 'economy': 0,
                            'batting_order': batting_order.get(player, 999),
                            'role_counts': {'bat': 0, 'bowl': 0},
                            'team': team_name,
                            'contexts': []
                        }
                    runs = stats.get('R', 0)
                    balls = stats.get('B', 0)
                    player_stats[player]['runs'] += runs
                    if balls > 0:
                        player_stats[player]['strike_rate'] += (runs / balls) * 100
                    player_stats[player]['balls'] += balls
                    player_stats[player]['matches'] += 1
                    player_stats[player]['role_counts']['bat'] += 1
                    player_stats[player]['contexts'].append(context)

                # Bowling extraction
                bowling = details.get('bowling', {})
                for player, stats in bowling.items():
                    if player not in player_stats:
                        player_stats[player] = {
                            'runs': 0, 'wickets': 0, 'balls': 0, 'matches': 0,
                            'strike_rate': 0, 'economy': 0,
                            'batting_order': 999,
                            'role_counts': {'bat': 0, 'bowl': 0},
                            'team': team_name,
                            'contexts': []
                        }
                    wickets = stats.get('W', 0)
                    runs_given = stats.get('R', 0)
                    overs = stats.get('O', 0)
                    player_stats[player]['wickets'] += wickets
                    if overs > 0:
                        player_stats[player]['economy'] += (runs_given / overs)
                    player_stats[player]['role_counts']['bowl'] += 1
                    if player_stats[player]['matches'] == 0:
                        player_stats[player]['matches'] += 1
                    player_stats[player]['contexts'].append(context)

    # Finalize averages for strike rate and economy.
    for player, stats in player_stats.items():
        m = stats['matches']
        if m > 0:
            stats['strike_rate'] = stats['strike_rate'] / m
            stats['economy'] = stats['economy'] / m if stats['economy'] > 0 else 0
    return player_stats, team_performance

def compute_stadium_pitch_factors(match_data):
    """
    Computes the average total runs per innings for each stadium from historical match records.
    Returns a mapping: stadium -> pitch factor, where:
      - "batting" if average >= 165,
      - "bowling" if average <= 145,
      - "neutral" otherwise.
    Adjust thresholds as needed.
    """
    stadium_scores = {}
    stadium_counts = {}
    for match in match_data:
        info = match.get('info', {})
        stadium = info.get('venue', 'Unknown')
        innings = match.get('innings', [])
        for inning in innings:
            for team, details in inning.items():
                inning_total = 0
                batting = details.get('batting', {})
                for player, stats in batting.items():
                    inning_total += stats.get('R', 0)
                stadium_scores[stadium] = stadium_scores.get(stadium, 0) + inning_total
                stadium_counts[stadium] = stadium_counts.get(stadium, 0) + 1
    pitch_map = {}
    for stadium in stadium_scores:
        avg_score = stadium_scores[stadium] / stadium_counts[stadium]
        if avg_score >= 165:
            pitch_map[stadium] = "batting"
        elif avg_score <= 145:
            pitch_map[stadium] = "bowling"
        else:
            pitch_map[stadium] = "neutral"
    return pitch_map

def assign_role(stats):
    """Assign a role based on a player's batting and bowling contributions."""
    runs = stats.get('runs', 0)
    wickets = stats.get('wickets', 0)
    if stats['role_counts']['bat'] > 0 and stats['role_counts']['bowl'] == 0:
        return 'batsman'
    if stats['role_counts']['bowl'] > 0 and stats['role_counts']['bat'] == 0:
        return 'bowler'
    if runs > 0 and wickets > 0:
        ratio = runs / wickets
        if ratio >= 50:
            return 'batsman'
        elif ratio <= 20:
            return 'bowler'
        else:
            return 'allrounder'
    return 'unknown'

# --- Updated Scoring Function ---
def compute_fantasy_score(stats, pitch_factor, batting_first, recent_form, team_bonus):
    """
    Compute a fantasy score for a player using:
      - Basic performance: runs, wickets, strike rate, economy.
      - Pitch factor adjustment (from historical records).
      - Match context derived from user input: players from the batting_first team are considered defending.
      - Recent form multiplier.
      - Batting order adjustment: openers get a boost.
      - Team bonus from historical performance.
    """
    runs = stats.get('runs', 0)
    wickets = stats.get('wickets', 0)
    sr = stats.get('strike_rate', 0)
    eco = stats.get('economy', 0)
    batting_order = stats.get('batting_order', 999)

    # Base score.
    score = runs * 0.5 + wickets * 10 + sr * 0.2 - eco * 0.2

    # Pitch factor adjustment.
    if pitch_factor == 'batting' and stats['role_counts']['bat'] > 0:
        score += 5
    elif pitch_factor == 'bowling' and stats['role_counts']['bowl'] > 0:
        score += 5

    # Match context based on batting_first.
    if stats.get('team', '').lower() == batting_first.lower():
        # For defending (batting first) boost openers.
        if batting_order <= 3:
            score *= 1.05
        elif batting_order <= 6:
            score *= 1.03
        # Additional bonus for being in the batting-first team.
        score += 3
    else:
        # For chasing, apply a milder boost.
        if batting_order <= 3:
            score *= 1.03
        elif batting_order <= 6:
            score *= 1.01

    # Apply recent form multiplier.
    score *= recent_form

    # Add team bonus.
    score += team_bonus

    return score

def prepare_player_dataframe(player_stats, team_performance, pitch_factor, batting_first, recent_form):
    """Build a DataFrame with player info, assigned roles, and computed fantasy scores."""
    data = []
    for player, stats in player_stats.items():
        role = assign_role(stats)
        team = stats.get('team', 'Unknown')
        team_bonus = team_performance.get(team, {}).get('bonus', 0)
        score = compute_fantasy_score(stats, pitch_factor, batting_first, recent_form, team_bonus)
        data.append({
            'player': player,
            'runs': stats.get('runs', 0),
            'wickets': stats.get('wickets', 0),
            'matches': stats.get('matches', 0),
            'strike_rate': stats.get('strike_rate', 0),
            'economy': stats.get('economy', 0),
            'batting_order': stats.get('batting_order', 999),
            'role': role,
            'fantasy_score': score,
            'team': team
        })
    df = pd.DataFrame(data)
    return df

# === Team Selection Using Linear Programming ===

def select_team_lp(df):
    """
    Uses linear programming to select 11 players maximizing total fantasy score.
    Ensures at least one wicketkeeper is selected if any exist.
    """
    players = df.index.tolist()
    x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in players}
    prob = LpProblem("Fantasy_Team_Selection", LpMaximize)
    prob += lpSum(df.loc[i, "fantasy_score"] * x[i] for i in players)
    prob += lpSum(x[i] for i in players) == 11

    wk_indices = df[df['player'].str.contains("keeper", case=False, na=False)].index.tolist()
    if wk_indices:
        prob += lpSum(x[i] for i in wk_indices) >= 1

    prob.solve(PULP_CBC_CMD(msg=0))
    selected_players = [df.loc[i, "player"] for i in players if x[i].varValue == 1]
    return selected_players

def select_best_teams(player_stats, team_performance, num_teams, pitch_factor, batting_first, recent_form):
    """
    Generates multiple candidate teams by repeatedly solving the LP problem.
    """
    teams = []
    df_all = prepare_player_dataframe(player_stats, team_performance, pitch_factor, batting_first, recent_form)
    df_all = df_all.reset_index(drop=True)
    
    for t in range(num_teams):
        if len(df_all) < 11:
            break  # Not enough players remain.
        team = select_team_lp(df_all)
        teams.append(team)
        df_all = df_all[~df_all['player'].isin(team)]
        df_all = df_all.reset_index(drop=True)
    
    return teams

# === Telegram Bot Command Handlers ===

# Data folder paths.
IPL_FOLDER = 'ipl_json_extracted'
ALL_FOLDER = 'all_json_extracted'

def prepare_data():
    ipl_matches = load_json_data(IPL_FOLDER)
    all_matches = load_json_data(ALL_FOLDER)
    combined_matches = ipl_matches + all_matches
    player_stats, team_performance = extract_player_stats(combined_matches)
    # Compute dynamic pitch factors from historical records.
    stadium_pitch_map = compute_stadium_pitch_factors(combined_matches)
    return player_stats, team_performance, stadium_pitch_map

# Load data on startup.
PLAYER_STATS, TEAM_PERFORMANCE, STADIUM_PITCH_MAP = prepare_data()

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome to the Fantasy Team Bot!\n"
        "Use the /teams command with the following parameters:\n"
        "/teams <stadium> <team1> <team2> <batting_first>\n"
        "Example:\n"
        '/teams "Wankhede Stadium" "Mumbai Indians" "Chennai Super Kings" "Mumbai Indians"'
    )

async def teams_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Expecting four arguments: stadium, team1, team2, batting_first.
    if len(context.args) < 4:
        await update.message.reply_text(
            "Please provide all parameters:\n"
            "/teams <stadium> <team1> <team2> <batting_first>"
        )
        return

    stadium = context.args[0]
    team1 = context.args[1]
    team2 = context.args[2]
    batting_first = context.args[3]

    # Use our dynamically computed pitch factor.
    pitch_factor = STADIUM_PITCH_MAP.get(stadium, "neutral")

    # For this example, we use a default recent form multiplier.
    recent_form = 1.0

    # Generate teams (six candidate combinations).
    num_teams = 6
    teams = select_best_teams(PLAYER_STATS, TEAM_PERFORMANCE, num_teams, pitch_factor, batting_first, recent_form)

    # Format the response.
    response = f"Pitch factor for {stadium} (from records): {pitch_factor}\n"
    response += "Here are your fantasy team combinations:\n"
    for idx, team in enumerate(teams, 1):
        response += f"\nTeam {idx}:\n" + ", ".join(team) + "\n"
    await update.message.reply_text(response)

# === Main Function to Start the Bot ===

def main():
    token = os.getenv("7643786273:AAHyiehHwFt5GjoTqDrDgk6v_vD3QXhnJ8s")  # Ensure this environment variable is set.
    application = ApplicationBuilder().token(token).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("teams", teams_command))

    application.run_polling()

if __name__ == "__main__":
    main()
