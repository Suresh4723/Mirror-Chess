import requests
import json

username = "Suresh_Gundumogula"

headers = {"User-Agent": "Mozilla/5.0"}

archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"

archives = requests.get(archives_url, headers=headers).json()["archives"]

all_games = []

for url in archives:
    games = requests.get(url, headers=headers).json()["games"]
    all_games.extend(games)

cleaned_games = []

for game in all_games:

    if game["white"]["username"].lower() == username.lower():
        my_color = "white"
    else:
        my_color = "black"

    cleaned_games.append({
        "my_color": my_color,
        "pgn": game["pgn"]
    })

with open("cleaned_games.json", "w") as f:
    json.dump(cleaned_games, f)

print("Cleaned games saved:", len(cleaned_games))