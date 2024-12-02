import pandas as pd
import tbaapiv3client as tba
import requests
import AwardReference as awardref
import time
import requests_cache


print("Starting Data Collection")

data = pd.read_csv("team_events.csv")
team_year_data = pd.read_csv("team_years.csv")

request_session = requests_cache.CachedSession("tba_cache")

TBA_KEY = "iGhMvxiLLPojQXjY6Uvfj2mSrXQ9ZJUurqJKGQ6ZscI0THwc6ZouVkC7RF1918pr"


def get_world_championship_events(data):
    data = data[data["week"] == 8]
    return data


def get_einstein(data):
    data = data[data["type"] == "einstein"]
    return data


def get_championship_division(data):
    data = data[data["type"] == "champs_div"]
    return data


# TBA Getters
def get_awards(team_number, year):
    # print("Getting Awards for " + str(team_number) + " in " + str(year))
    awards = request_session.get(
        "https://www.thebluealliance.com/api/v3/team/frc"
        + str(team_number)
        + "/awards/"
        + str(year),
        headers={"X-TBA-Auth-Key": TBA_KEY},
    ).json()
    return awards


def count_blue_banners(awards):
    banners = 0
    for award in awards:
        if award["award_type"] in awardref.BLUE_BANNER_AWARDS:
            banners += 1
    return banners


def get_elim_progression(entry):
    return int(entry["count"]) - int(entry["qual_count"])


# Print all headers
print(data.columns)

print("Filtering Data")
worlds = get_world_championship_events(data)

# filter entries with empty norm_epa, or count
worlds = worlds.dropna(subset=["norm_epa", "count", "rank", "qual_count", "state"])

# Filter year to only after 2018
worlds = worlds[worlds["year"] >= 2018]

# Get random entries
worlds = worlds.sample(n=1200)


print("Filtered Length: " + str(len(worlds)))

print("Getting Awards")
# get first entry
first_entry = worlds.iloc[23]
print(first_entry)
awards = get_awards(first_entry["team"], first_entry["year"])

print("Getting Blue Banners")
print(count_blue_banners(awards))

# Create new dataframe
processing_data = pd.DataFrame(
    columns=[
        "team",
        "year",
        "norm_epa",
        "blue_banners",
        "rank",
        "elim_progression",
        "state",
        "state_epa_rank",
        "state_epa_percentile",
        "total_epa_rank",
        "total_epa_percentile",
        "worlds_winrate",
        "worlds_qual_winrate",
    ]
)
count = 0

# Start Time
start_time = time.time()
curr_loop_time = time.time()
# Loop through all entries
for index, entry in worlds.iterrows():
    count += 1
    if count % 50 == 0:
        print("Processing Entry " + str(count) + " of " + str(len(worlds)))
        now = time.time()
        print("Loop Rate: " + str(50 / (now - curr_loop_time)) + " entries per second")

        rate = count / (now - start_time)
        print("Total Rate: " + str(rate) + " entries per second")
        print("Remaining Time: " + str((len(worlds) - count) / rate) + " seconds")
        curr_loop_time = now

    if entry["type"] == "einstein":

        current_entry = processing_data[processing_data["team"] == entry["team"]]

        if len(current_entry) == 0:
            continue
        current_entry = current_entry.iloc[0]
        current_entry["elim_progression"] += entry["count"]
        continue

    awards = get_awards(entry["team"], entry["year"])
    blue_banners = count_blue_banners(awards)
    elim_progression = get_elim_progression(entry)

    table_object = {
        "team": entry["team"],
        "year": entry["year"],
        "norm_epa": entry["norm_epa"],
        "blue_banners": blue_banners,
        "rank": entry["rank"],
        "elim_progression": elim_progression,
        "state": entry["state"],
        "worlds_winrate": entry["winrate"],
        "worlds_qual_winrate": entry["qual_winrate"],
    }

    # Team Year Entry
    team_year_entry = team_year_data[
        (team_year_data["team"] == entry["team"])
        & (team_year_data["year"] == entry["year"])
    ]
    if len(team_year_entry) != 0:
        team_year_entry = team_year_entry.iloc[0]
        table_object["state_epa_rank"] = team_year_entry["state_epa_rank"]
        table_object["state_epa_percentile"] = team_year_entry["state_epa_percentile"]
        table_object["total_epa_rank"] = team_year_entry["total_epa_rank"]
        table_object["total_epa_percentile"] = team_year_entry["total_epa_percentile"]

    processing_data = processing_data._append(
        table_object,
        ignore_index=True,
    )

# Save to CSV
processing_data.to_csv("worlds_data.csv")
