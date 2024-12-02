import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------
# 1. Data Loading and Filtering
# ---------------------------

# Load the worlds_data.csv
worlds_data_path = "worlds_data.csv"
worlds_data = pd.read_csv(worlds_data_path)
print(f"Loaded '{worlds_data_path}' with {len(worlds_data)} records.")

# Load the team_events.csv
team_events_path = "team_events.csv"
team_events = pd.read_csv(team_events_path)
print(f"Loaded '{team_events_path}' with {len(team_events)} records.")

# Identify team-year combinations with Einstein events
einstein_events = team_events[team_events["type"].str.lower() == "einstein"].copy()
print(f"Identified {len(einstein_events)} Einstein event records.")

# Create a unique identifier for team and year to facilitate merging
# Assuming 'team' and 'year' columns exist in both datasets
einstein_events.loc[:, "team_year"] = (
    einstein_events["team"].astype(str) + "_" + einstein_events["year"].astype(str)
)
worlds_data.loc[:, "team_year"] = (
    worlds_data["team"].astype(str) + "_" + worlds_data["year"].astype(str)
)

# Get the set of team_years that participated in Einstein events
einstein_team_years = set(einstein_events["team_year"])
print(
    f"Total unique team-year combinations in Einstein events: {len(einstein_team_years)}"
)

# Filter worlds_data to include only team-years with Einstein participation
filtered_data = worlds_data.loc[
    worlds_data["team_year"].isin(einstein_team_years)
].copy()
print(
    f"Filtered worlds_data to {len(filtered_data)} records based on Einstein participation."
)


# Drop the 'team_year' column as it's no longer needed
filtered_data.drop(columns=["team_year"], inplace=True)

# ---------------------------
# 2. Data Preprocessing
# ---------------------------

# Define dependent and independent variables
dependent_variables = [
    "rank",
    "elim_progression",
    "worlds_winrate",
    "worlds_qual_winrate",
]
independent_variables = [
    "norm_epa",
    "blue_banners",
    "state_epa_rank",
    "state_epa_percentile",
    "total_epa_rank",
    "total_epa_percentile",
]

# Specify independent variables that need to be inverted
# These are variables where lower values are better (e.g., rankings)
variables_to_invert = ["state_epa_rank", "total_epa_rank"]

# Invert specified independent variables using .loc to avoid SettingWithCopyWarning
for var in variables_to_invert:
    if var in filtered_data.columns:
        max_val = filtered_data[var].max()
        filtered_data.loc[:, f"{var}_inverted"] = (max_val + 1) - filtered_data[var]
        print(f"Inverted variable '{var}'. Created new column '{var}_inverted'.")
    else:
        print(f"Warning: Variable '{var}' not found in filtered data columns.")

# ---------------------------
# 3. Handling Ordinal Categorical Variables
# ---------------------------

# Check if 'elim_progression' is numeric
if not pd.api.types.is_numeric_dtype(filtered_data["elim_progression"]):
    print(
        "\n'elimination_progression' is not numeric. Attempting to convert ordinal categories to numeric."
    )

    # Define the ordinal mapping based on actual categories in your data
    # Update this dictionary based on your dataset's specific categories
    ordinal_mapping = {
        "Round of 16": 1,
        "Quarterfinals": 2,
        "Semifinals": 3,
        "Finals": 4,
        "Winner": 5,
    }

    # Apply the mapping using .loc
    filtered_data.loc[:, "elim_progression_numeric"] = filtered_data[
        "elim_progression"
    ].map(ordinal_mapping)

    # Check for unmapped categories
    unmapped = filtered_data["elim_progression_numeric"].isna().sum()
    if unmapped > 0:
        print(
            f"Warning: {unmapped} entries in 'elim_progression' couldn't be mapped and will be excluded from plots."
        )

    # Drop rows with unmapped 'elim_progression_numeric'
    initial_length = len(filtered_data)
    filtered_data.dropna(subset=["elim_progression_numeric"], inplace=True)
    final_length = len(filtered_data)
    print(
        f"Converted 'elim_progression' to numeric. Excluded {initial_length - final_length} records due to unmapped categories."
    )

# ---------------------------
# 4. Creating Results Directory
# ---------------------------

# Create 'results_einsteins' directory if it doesn't exist
results_dir = "results_einsteins"
os.makedirs(results_dir, exist_ok=True)
print(f"\nCreated or verified existence of '{results_dir}' directory.")

# ---------------------------
# 5. Analysis and Visualization
# ---------------------------

# Initialize a list to store R² results
r2_results = []

# Iterate over each dependent variable
for dep_var in dependent_variables:
    # Determine the correct variable name (numeric)
    if dep_var == "elim_progression":
        dep_var_plot = "elim_progression_numeric"
        if dep_var_plot not in filtered_data.columns:
            print(
                f"\nDependent variable '{dep_var}' is non-numeric and was not successfully converted. Skipping analysis for this variable."
            )
            continue
    else:
        dep_var_plot = dep_var

    # Create a subdirectory for the current dependent variable
    dep_dir = os.path.join(results_dir, dep_var)
    os.makedirs(dep_dir, exist_ok=True)
    print(
        f"\nAnalyzing dependent variable '{dep_var_plot}' and saving plots to '{dep_dir}'."
    )

    for independent_var in independent_variables:
        # Determine if the independent variable has an inverted version
        if independent_var in variables_to_invert:
            x = filtered_data[f"{independent_var}_inverted"]
            xlabel = f"{independent_var} (Inverted)"
        else:
            x = filtered_data[independent_var]
            xlabel = independent_var

        y = filtered_data[dep_var_plot]

        # Remove NaN values for plotting and regression
        valid = (~x.isna()) & (~y.isna())
        x_valid = x[valid]
        y_valid = y[valid]

        if len(x_valid) < 2:
            print(
                f"Not enough data to plot '{independent_var}' vs '{dep_var_plot}'. Skipping."
            )
            continue

        # Ensure x_valid is numeric
        if pd.api.types.is_categorical_dtype(x_valid):
            print(
                f"Independent variable '{independent_var}' is categorical. Converting to numeric codes for plotting."
            )
            x_valid_numeric = x_valid.cat.codes
        else:
            x_valid_numeric = x_valid.astype(float)

        # Create the scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(
            x_valid_numeric, y_valid, color="blue", alpha=0.6, label="Data Points"
        )

        # Calculate linear regression
        try:
            slope, intercept = np.polyfit(x_valid_numeric, y_valid, 1)
            trend_line = slope * x_valid_numeric + intercept
            plt.plot(
                x_valid_numeric,
                trend_line,
                color="red",
                linestyle="--",
                label="Trend Line",
            )
        except np.linalg.LinAlgError:
            print(
                f"Linear regression failed for '{independent_var}' vs '{dep_var_plot}'. Skipping trend line."
            )
            trend_line = None

        # Calculate R-squared
        if trend_line is not None:
            correlation_matrix = np.corrcoef(x_valid_numeric, y_valid)
            correlation_xy = correlation_matrix[0, 1]
            r_squared = correlation_xy**2 if not np.isnan(correlation_xy) else 0
        else:
            r_squared = 0
        r2_results.append(
            {
                "Dependent Variable": dep_var,
                "Independent Variable": independent_var,
                "R_squared": r_squared,
            }
        )

        # Annotate R-squared on the plot
        plt.text(
            0.05,
            0.95,
            f"$R^2$ = {r_squared:.4f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
        )

        # Set plot titles and labels
        plt.title(f"{dep_var_plot.replace('_', ' ').title()} vs {xlabel}", fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(dep_var_plot.replace("_", " ").title(), fontsize=12)
        plt.legend()
        plt.tight_layout()

        # Save the scatter plot
        plot_filename = f"{independent_var}_vs_{dep_var_plot}.png"
        plot_path = os.path.join(dep_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved scatter plot: '{plot_path}' with R² = {r_squared:.4f}")

    # ---------------------------
    # Box Plots for Blue Banners
    # ---------------------------

    # Define the independent variable for box plots
    boxplot_independent_var = "blue_banners"

    # Check if 'blue_banners' is numerical or categorical
    if pd.api.types.is_object_dtype(
        filtered_data[boxplot_independent_var]
    ) or pd.api.types.is_categorical_dtype(filtered_data[boxplot_independent_var]):
        is_categorical = True
        # Ensure it's treated as categorical
        filtered_data.loc[:, boxplot_independent_var] = filtered_data[
            boxplot_independent_var
        ].astype("category")
        group_col = boxplot_independent_var
        print(f"\n'{boxplot_independent_var}' is treated as a categorical variable.")
    else:
        is_categorical = False
        # Assuming blue_banners counts by 1, we can treat it as discrete numerical
        # Convert to integer first if necessary
        if not pd.api.types.is_integer_dtype(filtered_data[boxplot_independent_var]):
            filtered_data.loc[:, boxplot_independent_var] = filtered_data[
                boxplot_independent_var
            ].astype(int)
        # Then convert to categorical for plotting
        filtered_data.loc[:, boxplot_independent_var] = filtered_data[
            boxplot_independent_var
        ].astype("category")
        group_col = boxplot_independent_var
        print(
            f"'{boxplot_independent_var}' is treated as a discrete numerical variable and converted to categorical."
        )

    # Define the list of dependent variables for box plots
    boxplot_dependent_vars = dependent_variables  # Customize if needed

    # Create a subdirectory for box plots if it doesn't exist
    boxplot_dir = os.path.join(dep_dir, "blue_banners_box_plots")
    os.makedirs(boxplot_dir, exist_ok=True)

    for dep_var in boxplot_dependent_vars:
        # Determine the correct variable name (numeric)
        if dep_var == "elim_progression":
            dep_var_plot_box = "elim_progression_numeric"
            if dep_var_plot_box not in filtered_data.columns:
                print(
                    f"Dependent variable '{dep_var}' is non-numeric and was not successfully converted. Skipping box plot."
                )
                continue
        else:
            dep_var_plot_box = dep_var

        # Check if the dependent variable is numeric
        if not pd.api.types.is_numeric_dtype(filtered_data[dep_var_plot_box]):
            print(
                f"Dependent variable '{dep_var_plot_box}' is not numeric. Skipping box plot."
            )
            continue

        # Remove NaN values for plotting
        boxplot_data = filtered_data.loc[
            :, [boxplot_independent_var, dep_var_plot_box]
        ].dropna()

        # Check if there are enough unique groups
        if boxplot_data[boxplot_independent_var].nunique() < 2:
            print(
                f"Not enough groups in '{boxplot_independent_var}' to plot box plot for '{dep_var_plot_box}'. Skipping."
            )
            continue

        # Initialize the plot
        plt.figure(figsize=(12, 8))

        # Create the box plot
        sns.boxplot(
            x=boxplot_independent_var,
            y=dep_var_plot_box,
            data=boxplot_data,
            palette="Blues_d",
        )

        # Add titles and labels
        plt.title(
            f"Box Plot of {dep_var_plot_box.replace('_', ' ').title()} by Blue Banners",
            fontsize=16,
        )
        plt.xlabel("Blue Banners", fontsize=14)
        plt.ylabel(f"{dep_var_plot_box.replace('_', ' ').title()}", fontsize=14)

        # Rotate x-axis labels if there are many categories
        if boxplot_data[boxplot_independent_var].nunique() > 10:
            plt.xticks(rotation=45)

        plt.tight_layout()

        # Save the box plot
        boxplot_filename = (
            f"{dep_var_plot_box}_by_{boxplot_independent_var}_boxplot.png"
        )
        boxplot_path = os.path.join(boxplot_dir, boxplot_filename)
        plt.savefig(boxplot_path)
        plt.close()
        print(f"Saved box plot: '{boxplot_path}'")

    # ---------------------------
    # Distribution Plots for Blue Banners with Consistent Scales
    # ---------------------------

    # Define the list of dependent variables for distribution plots
    distribution_dependent_vars = dependent_variables  # Customize if needed

    # Create a subdirectory for distribution plots
    distribution_dir = os.path.join(dep_dir, "blue_banners_distribution_plots")
    os.makedirs(distribution_dir, exist_ok=True)

    for dep_var in distribution_dependent_vars:
        # Determine the correct variable name (numeric)
        if dep_var == "elim_progression":
            dep_var_plot_dist = "elim_progression_numeric"
            if dep_var_plot_dist not in filtered_data.columns:
                print(
                    f"Dependent variable '{dep_var}' is non-numeric and was not successfully converted. Skipping distribution plot."
                )
                continue
        else:
            dep_var_plot_dist = dep_var

        # Check if the dependent variable is numeric
        if not pd.api.types.is_numeric_dtype(filtered_data[dep_var_plot_dist]):
            print(
                f"Dependent variable '{dep_var_plot_dist}' is not numeric. Skipping distribution plot."
            )
            continue

        # Remove NaN values for plotting
        distribution_data = filtered_data.loc[
            :, [boxplot_independent_var, dep_var_plot_dist]
        ].dropna()

        # Check if there are enough unique groups
        if distribution_data[boxplot_independent_var].nunique() < 1:
            print(
                f"No groups in '{boxplot_independent_var}' to plot distribution for '{dep_var_plot_dist}'. Skipping."
            )
            continue

        # Determine global x and y limits for consistent scaling
        x_min = distribution_data[dep_var_plot_dist].min()
        x_max = distribution_data[dep_var_plot_dist].max()

        # Calculate the maximum y value across all distributions for consistent y-axis scaling
        max_count = 0
        bins = 20  # Define the number of bins

        for category in distribution_data[boxplot_independent_var].unique():
            subset = distribution_data[
                distribution_data[boxplot_independent_var] == category
            ][dep_var_plot_dist]
            counts, _ = np.histogram(subset, bins=bins, range=(x_min, x_max))
            current_max = counts.max()
            if current_max > max_count:
                max_count = current_max

        # Initialize the plot
        plt.figure(figsize=(12, 8))

        # Create a FacetGrid for distribution plots per blue_banners category
        g = sns.FacetGrid(
            distribution_data,
            col=boxplot_independent_var,
            col_wrap=4,
            sharex=True,
            sharey=True,  # Share y-axis for consistent scaling
            height=4,
        )
        g.map(
            sns.histplot,
            dep_var_plot_dist,
            kde=True,
            color="blue",
            bins=bins,
            stat="count",
            binrange=(x_min, x_max),
        )

        # Add titles and labels
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(
            f"Distribution of {dep_var_plot_dist.replace('_', ' ').title()} by Blue Banners",
            fontsize=16,
        )

        # Rotate x-axis labels if necessary
        for ax in g.axes.flatten():
            for label in ax.get_xticklabels():
                label.set_rotation(45)

        # Set consistent y-axis limits
        for ax in g.axes.flatten():
            ax.set_ylim(0, max_count + 5)  # Adding some padding

        plt.tight_layout()

        # Save the distribution plot
        distribution_filename = (
            f"{dep_var_plot_dist}_distribution_by_{boxplot_independent_var}.png"
        )
        distribution_path = os.path.join(distribution_dir, distribution_filename)
        plt.savefig(distribution_path)
        plt.close()
        print(f"Saved distribution plot: '{distribution_path}'")

    # ---------------------------
    # 6. Saving R² Overview
    # ---------------------------

    # Create a DataFrame from R² results
    r2_df_einsteins = pd.DataFrame(r2_results)

    # Pivot the table for better readability
    r2_pivot_einsteins = r2_df_einsteins.pivot(
        index="Independent Variable", columns="Dependent Variable", values="R_squared"
    )

    # Save the R² table to CSV in the 'results_einsteins' directory
    r2_csv_path_einsteins = os.path.join(
        results_dir, "r_squared_overview_einsteins.csv"
    )
    r2_pivot_einsteins.to_csv(r2_csv_path_einsteins)
    print(f"\nSaved R² overview table: '{r2_csv_path_einsteins}'")

    # ---------------------------
    # 7. Adding the Requested Distribution Plot
    # ---------------------------

    # Create a subdirectory for summary plots
    summary_dir = os.path.join(results_dir, "summary_plots")
    os.makedirs(summary_dir, exist_ok=True)

    # Create the distribution plot showing the number of teams in Einstein events relative to their blue banner counts
    plt.figure(figsize=(10, 6))
    sns.countplot(x="blue_banners", data=filtered_data, palette="Blues_d")
    plt.title("Number of Teams in Einstein Events by Blue Banners Count", fontsize=16)
    plt.xlabel("Blue Banners Count", fontsize=14)
    plt.ylabel("Number of Teams", fontsize=14)

    # Annotate counts on top of the bars
    ax = plt.gca()
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
        )

    plt.tight_layout()

    # Save the plot
    distribution_plot_filename = "teams_in_einstein_by_blue_banners.png"
    distribution_plot_path = os.path.join(summary_dir, distribution_plot_filename)
    plt.savefig(distribution_plot_path)
    plt.close()
    print(f"Saved distribution plot: '{distribution_plot_path}'")

    # ---------------------------
    # 8. Final Output Structure
    # ---------------------------

    print(
        "\nAnalysis complete. All plots and results are saved in the 'results_einsteins' directory."
    )
