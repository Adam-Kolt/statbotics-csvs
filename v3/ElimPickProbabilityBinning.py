import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# Optional: Enhance plot aesthetics with Seaborn
sns.set(style="whitegrid")

# Load the data
data = pd.read_csv("worlds_data.csv")

# Only years after 2011
data = data[data["year"] >= 2011]

data["Picked_Top3"] = data["elim_pick"].apply(lambda x: 1 if x <= 2 and x != -1 else 0)

dependent_variable = "norm_epa"
start = 1000
end = 2100
scale = 1

bin_width = 30

probability_bins = []

fifty_ten_2024_dep = 252


plt.figure(figsize=(8, 5))


for dep in range(start, end, bin_width):
    epa_bin = data[
        (data[dependent_variable] >= dep / scale)
        & (data[dependent_variable] < dep / scale + bin_width / scale)
    ]
    epa_bin = epa_bin[["Picked_Top3"]]

    total = len(epa_bin)
    picked = len(epa_bin[epa_bin["Picked_Top3"] == 1])
    if total == 0:
        probability = 0
    else:
        probability = picked / total

    probability_bins.append(probability)

plt.plot(
    [x / scale for x in range(start, end, bin_width)], probability_bins, marker="o"
)

fifty_ten_performances = data[data["team"] == 5010]

# Lines for 5010 dependent variable with label for year and label the value on the x-axis
for year in fifty_ten_performances["year"]:
    plt.axvline(
        x=fifty_ten_performances[fifty_ten_performances["year"] == year][
            dependent_variable
        ].values[0],
        color="red",
        linestyle="--",
    )
    plt.text(
        fifty_ten_performances[fifty_ten_performances["year"] == year][
            dependent_variable
        ].values[0],
        0.5,
        year,
        rotation=90,
    )


plt.title("Percentage Robots Picked Top 3 by " + dependent_variable)
plt.xlabel(dependent_variable)
plt.ylabel("Percentage Picked Top 3")
plt.show()
# sns.histplot(data["state_epa_percentile"], kde=True, bins=30, color="salmon")
