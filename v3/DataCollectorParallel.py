import pandas as pd
import requests
import AwardReference as awardref
import time
import requests_cache
from multiprocessing import Pool
from tqdm import tqdm  # Progress bar library

print("Starting Data Collection")
request_session = requests_cache.CachedSession("tba_cache")
TBA_KEY = "iGhMvxiLLPojQXjY6Uvfj2mSrXQ9ZJUurqJKGQ6ZscI0THwc6ZouVkC7RF1918pr"
CENSUS_API = "15a1df667a3f481c7b25387f7002ee33016047a0"


def get_world_championship_events(data):
    return data[data["week"] == 8]


def get_einstein(data):
    return data[data["type"] == "einstein"]


def get_championship_division(data):
    return data[data["type"] == "champs_div"]


# TBA Getters
def get_awards(team_number, year):
    try:
        response = request_session.get(
            f"https://www.thebluealliance.com/api/v3/team/frc{team_number}/awards/{year}",
            headers={"X-TBA-Auth-Key": TBA_KEY},
            timeout=10,  # Added timeout for network robustness
        )
        response.raise_for_status()  # Raise an error for bad status codes
        awards = response.json()
        return awards
    except requests.RequestException as e:
        print(f"Error fetching awards for team {team_number} in {year}: {e}")
        return []  # Return empty list on error


def get_event_status(team_number, event_key):
    try:
        response = request_session.get(
            f"https://www.thebluealliance.com/api/v3/team/frc{team_number}/event/{event_key}/status",
            headers={"X-TBA-Auth-Key": TBA_KEY},
            timeout=10,  # Added timeout for network robustness
        )
        response.raise_for_status()  # Raise an error for bad status codes
        event_status = response.json()
        return event_status
    except requests.RequestException as e:
        print(f"Error fetching event status for team {team_number} at {event_key}: {e}")
        return None


def get_team_data(team_number):
    try:
        response = request_session.get(
            f"https://www.thebluealliance.com/api/v3/team/frc{team_number}",
            headers={"X-TBA-Auth-Key": TBA_KEY},
            timeout=10,  # Added timeout for network robustness
        )
        response.raise_for_status()  # Raise an error for bad status codes
        team_data = response.json()
        return team_data
    except requests.RequestException as e:
        print(f"Error fetching team data for team {team_number}: {e}")
        return None


def get_regional_median_income(team_data):
    BASE_URL = "https://api.census.gov/data/2020/acs/acs5/"

    if team_data["country"] != "USA":
        return None

    if team_data["postal_code"] == None:
        return None

    postal_code = team_data["postal_code"]

    try:
        params = {
            "get": "NAME,B19013_001E",  # 'NAME' for area name, 'B19013_001E' for median income
            "for": f"zip code tabulation area:{postal_code}",
            "key": CENSUS_API,
        }

        response = request_session.get(
            BASE_URL, params=params, timeout=10  # Added timeout for network robustness
        )
        response.raise_for_status()  # Raise an error for bad status codes
        median_income = response.json()
        return median_income
    except requests.RequestException as e:
        print(f"Error fetching median income for team {team_data['team_number']}: {e}")
        return None


def get_alliance_pick(event_status):

    if not event_status["alliance"]:
        return -1
    return event_status["alliance"]["pick"]


def count_blue_banners_excluding_current_event(awards, event_key):
    return sum(
        1
        for award in awards
        if award.get("award_type") in awardref.BLUE_BANNER_AWARDS
        and award.get("event_key") != event_key
    )


def exclude_current_event_awards(awards, event_key):
    return [award for award in awards if award.get("event_key") != event_key]


def count_winning_blue_banners(awards):
    return sum(1 for award in awards if award.get("award_type") == awardref.WINNER)


def count_finalist_awards(awards):
    return sum(1 for award in awards if award.get("award_type") == awardref.FINALIST)


def get_elim_progression(entry):
    try:
        return int(entry["count"]) - int(entry["qual_count"])
    except (ValueError, KeyError) as e:
        print(f"Error calculating elim_progression for entry {entry}: {e}")
        return 0  # Default to 0 on error


def process_entry(entry, team_year_data):
    if entry["type"] == "einstein":
        return {
            "type": "einstein",
            "team": entry["team"],
            "count": int(entry["count"]),
        }

    awards = get_awards(entry["team"], entry["year"])
    blue_banners = count_blue_banners_excluding_current_event(awards, entry["event"])

    awards_excluding_current_event = exclude_current_event_awards(
        awards, entry["event"]
    )

    # Count blue banners
    event_wins = count_winning_blue_banners(awards_excluding_current_event)
    event_finalists = count_finalist_awards(awards_excluding_current_event)

    event_status = get_event_status(entry["team"], entry["event"])

    if event_status is not None:
        elim_pick = get_alliance_pick(event_status)
    else:
        elim_pick = None

    elim_progression = get_elim_progression(entry)

    table_object = {
        "type": "regular",
        "team": entry["team"],
        "year": entry["year"],
        "norm_epa": entry["norm_epa"],
        "blue_banners": blue_banners,
        "event_wins": event_wins,
        "event_finalists": event_finalists,
        "total_awards": len(awards),
        "rank": entry["rank"],
        "elim_progression": elim_progression,
        "state": entry["state"],
        "worlds_winrate": entry["winrate"],
        "worlds_qual_winrate": entry["qual_winrate"],
        "state_epa_rank": None,
        "state_epa_percentile": None,
        "total_epa_rank": None,
        "total_epa_percentile": None,
        "elim_alliance": entry["elim_alliance"],
        "elim_pick": elim_pick,
    }

    # Team Year Entry
    team_year_entry = team_year_data[
        (team_year_data["team"] == entry["team"])
        & (team_year_data["year"] == entry["year"])
    ]
    if not team_year_entry.empty:
        team_year_entry = team_year_entry.iloc[0]
        table_object["state_epa_rank"] = team_year_entry.get("state_epa_rank")
        table_object["state_epa_percentile"] = team_year_entry.get(
            "state_epa_percentile"
        )
        table_object["total_epa_rank"] = team_year_entry.get("total_epa_rank")
        table_object["total_epa_percentile"] = team_year_entry.get(
            "total_epa_percentile"
        )

    return table_object


# Helper function to unpack arguments for multiprocessing
def process_entry_star(args):
    entry, team_year_data = args
    return process_entry(entry, team_year_data)


def main():
    data = pd.read_csv("team_events.csv")
    team_year_data = pd.read_csv("team_years.csv")

    print("Filtering Data")
    worlds = get_world_championship_events(data)

    # Filter entries with empty norm_epa, or count
    worlds = worlds.dropna(subset=["norm_epa", "count", "rank", "qual_count", "state"])

    worlds = worlds[worlds["year"] >= 2002]

    # Get random entries

    print("Filtered Length:", len(worlds))

    # Prepare lists to collect results
    regular_entries = []
    einstein_entries = []

    # Start Time
    start_time = time.time()

    # Initialize multiprocessing Pool
    with Pool(30) as pool:
        # Prepare arguments for multiprocessing
        args = ((entry, team_year_data) for _, entry in worlds.iterrows())

        # Use imap_unordered with a top-level helper function
        for result in tqdm(
            pool.imap_unordered(process_entry_star, args),
            total=len(worlds),
            desc="Processing Entries",
            unit="entry",
        ):
            if result is None:
                continue  # Skip if processing failed
            if result["type"] == "einstein":
                einstein_entries.append(result)
            else:
                regular_entries.append(result)

        pool.close()
        pool.join()

    # Create DataFrame from regular entries
    processing_data = pd.DataFrame(regular_entries)

    # Process einstein entries to update elim_progression
    if einstein_entries:
        einstein_df = pd.DataFrame(einstein_entries)
        # Group by team and sum the counts
        einstein_sums = einstein_df.groupby("team")["count"].sum().reset_index()

        # Merge with processing_data to add the counts
        processing_data = processing_data.merge(
            einstein_sums, on="team", how="left", suffixes=("", "_einstein")
        )
        processing_data["elim_progression"] += processing_data.get("count_einstein", 0)

    # Save to CSV
    processing_data.to_csv("worlds_data.csv", index=False)

    # End Time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data Collection Complete in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
    print("Data Collection Complete")
