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

data = data[data["elim_pick"] != -1]

# Define dependent and independent variables
dependent_variables = [
    "elim_pick",
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

# Invert specified independent variables
for var in variables_to_invert:
    if var in data.columns:
        max_val = data[var].max()
        # Inversion: Subtract each value from (max + 1) to ensure lower ranks become higher values
        data[var + "_inverted"] = (max_val + 1) - data[var]
        print(f"Variable '{var}' inverted. Created new column '{var}_inverted'.")
    else:
        print(f"Warning: Variable '{var}' not found in data columns.")

# Create 'results' directory if it doesn't exist
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Initialize a list to store R² results
r2_results = []

# Iterate over each dependent variable
for dep_var in dependent_variables:
    # Create a subdirectory for the current dependent variable
    dep_dir = os.path.join(results_dir, dep_var)
    os.makedirs(dep_dir, exist_ok=True)

    # Check if the dependent variable is numeric
    if not pd.api.types.is_numeric_dtype(data[dep_var]):
        print(
            f"Dependent variable '{dep_var}' is not numeric. Attempting to convert if possible."
        )

        # Example: If 'elim_progression' is ordinal, map it to numeric
        if dep_var == "elim_progression":
            # Define the ordinal mapping based on actual categories
            # Replace these with the actual categories in your data
            ordinal_mapping = {
                "Round of 16": 1,
                "Quarterfinals": 2,
                "Semifinals": 3,
                "Finals": 4,
                "Winner": 5,
            }
            data[dep_var + "_numeric"] = data[dep_var].map(ordinal_mapping)
            # Check for unmapped categories
            unmapped = data[dep_var + "_numeric"].isna().sum()
            if unmapped > 0:
                print(
                    f"Warning: {unmapped} entries in '{dep_var}' couldn't be mapped and will be excluded from box plots."
                )
            dep_var_numeric = dep_var + "_numeric"
        else:
            # If no mapping is defined, skip box plots and distribution plots for this variable
            print(
                f"No numeric conversion defined for '{dep_var}'. Skipping related plots."
            )
            continue
    else:
        dep_var_numeric = dep_var

    for independent_var in independent_variables:
        # Determine if the independent variable has an inverted version
        if independent_var in variables_to_invert:
            x = data[independent_var + "_inverted"]
            xlabel = f"{independent_var} (Inverted)"
        else:
            x = data[independent_var]
            xlabel = independent_var

        y = data[dep_var_numeric]

        # Remove NaN values for plotting and regression
        valid = (~np.isnan(x)) & (~np.isnan(y))
        x_valid = x[valid]
        y_valid = y[valid]

        if len(x_valid) < 2:
            print(
                f"Not enough data to plot {independent_var} vs {dep_var_numeric}. Skipping."
            )
            continue

        # Create the plot
        plt.figure(figsize=(8, 6))

        # Scatter plot
        plt.scatter(x_valid, y_valid, color="blue", alpha=0.6, label="Data Points")

        # Calculate linear regression
        slope, intercept = np.polyfit(x_valid, y_valid, 1)
        trend_line = slope * x_valid + intercept

        # Plot trend line
        plt.plot(x_valid, trend_line, color="red", linestyle="--", label="Trend Line")

        # Calculate R-squared
        correlation_matrix = np.corrcoef(x_valid, y_valid)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy**2 if not np.isnan(correlation_xy) else 0
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
        plt.title(f"{dep_var_numeric} vs {xlabel}", fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(dep_var_numeric, fontsize=12)
        plt.legend()
        plt.tight_layout()

        # Save the plot in the corresponding dependent variable folder
        plot_filename = f"{independent_var}_vs_{dep_var_numeric}.png"
        plot_path = os.path.join(dep_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")

# Create a DataFrame from R² results
r2_df = pd.DataFrame(r2_results)

# Pivot the table for better readability
r2_pivot = r2_df.pivot(
    index="Independent Variable", columns="Dependent Variable", values="R_squared"
)

# Save the R² table to CSV in the 'results' directory
r2_csv_path = os.path.join(results_dir, "r_squared_overview.csv")
r2_pivot.to_csv(r2_csv_path)
print(f"Saved R² overview table: {r2_csv_path}")

# Optional: Print the R² table
print("\nR² Values Overview:")
print(r2_pivot.round(4))

# ---------------------------
# Added Section: Box Plots for Blue Banners
# ---------------------------

# Define the independent variable for box plots
boxplot_independent_var = "blue_banners"

# Check if 'blue_banners' is numerical or categorical
if (
    data[boxplot_independent_var].dtype == "object"
    or data[boxplot_independent_var].dtype.name == "category"
):
    is_categorical = True
    # Ensure it's treated as categorical
    data[boxplot_independent_var] = data[boxplot_independent_var].astype("category")
    group_col = boxplot_independent_var
    print(f"'{boxplot_independent_var}' is treated as a categorical variable.")
else:
    is_categorical = False
    # Assuming blue_banners counts by 1, we can treat it as discrete numerical
    # Optionally, convert to string or category for plotting
    # Convert to integer first if necessary
    if not pd.api.types.is_integer_dtype(data[boxplot_independent_var]):
        data[boxplot_independent_var] = data[boxplot_independent_var].astype(int)
    data[boxplot_independent_var] = data[boxplot_independent_var].astype("category")
    group_col = boxplot_independent_var
    print(
        f"'{boxplot_independent_var}' is treated as a discrete numerical variable and converted to categorical."
    )

# Define the list of dependent variables for box plots
boxplot_dependent_vars = dependent_variables  # Customize if needed

# Create a subdirectory for box plots
boxplot_dir = os.path.join(results_dir, "blue_banners_box_plots")
os.makedirs(boxplot_dir, exist_ok=True)

for dep_var in boxplot_dependent_vars:
    # Determine the correct variable name (numeric)
    if not pd.api.types.is_numeric_dtype(data[dep_var]):
        if dep_var == "elim_progression":
            dep_var_plot = dep_var + "_numeric"
            if dep_var_plot not in data.columns:
                print(
                    f"Dependent variable '{dep_var}' is non-numeric and was not successfully converted. Skipping box plot."
                )
                continue
        else:
            print(
                f"Dependent variable '{dep_var}' is non-numeric and was not handled. Skipping box plot."
            )
            continue
    else:
        dep_var_plot = dep_var

    # Remove NaN values for plotting
    boxplot_data = data[[boxplot_independent_var, dep_var_plot]].dropna()

    # Check if there are enough unique groups
    if boxplot_data[boxplot_independent_var].nunique() < 2:
        print(
            f"Not enough groups in '{boxplot_independent_var}' to plot box plot for '{dep_var_plot}'. Skipping."
        )
        continue

    # Initialize the plot
    plt.figure(figsize=(12, 8))

    # Create the box plot
    sns.boxplot(
        x=boxplot_independent_var, y=dep_var_plot, data=boxplot_data, palette="Blues_d"
    )

    # Add titles and labels
    plt.title(
        f"Box Plot of {dep_var_plot.replace('_', ' ').title()} by Blue Banners",
        fontsize=16,
    )
    plt.xlabel("Blue Banners", fontsize=14)
    plt.ylabel(f"{dep_var_plot.replace('_', ' ').title()}", fontsize=14)

    # Rotate x-axis labels if there are many categories
    if boxplot_data[boxplot_independent_var].nunique() > 10:
        plt.xticks(rotation=45)

    plt.tight_layout()

    # Save the plot in the box plots subdirectory
    plot_filename = f"{dep_var_plot}_by_{boxplot_independent_var}_boxplot.png"
    plot_path = os.path.join(boxplot_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved box plot: {plot_path}")

# ---------------------------
# Added Section: Distribution Plots for Blue Banners
# ---------------------------

# Define the list of dependent variables for distribution plots
distribution_dependent_vars = dependent_variables  # Customize if needed

# Create a subdirectory for distribution plots
distribution_dir = os.path.join(results_dir, "blue_banners_distribution_plots")
os.makedirs(distribution_dir, exist_ok=True)

for dep_var in distribution_dependent_vars:
    # Determine the correct variable name (numeric)
    if not pd.api.types.is_numeric_dtype(data[dep_var]):
        if dep_var == "elim_progression":
            dep_var_plot = dep_var + "_numeric"
            if dep_var_plot not in data.columns:
                print(
                    f"Dependent variable '{dep_var}' is non-numeric and was not successfully converted. Skipping distribution plot."
                )
                continue
        else:
            print(
                f"Dependent variable '{dep_var}' is non-numeric and was not handled. Skipping distribution plot."
            )
            continue
    else:
        dep_var_plot = dep_var

    # Remove NaN values for plotting
    distribution_data = data[[boxplot_independent_var, dep_var_plot]].dropna()

    # Check if there are enough unique groups
    if distribution_data[boxplot_independent_var].nunique() < 1:
        print(
            f"No groups in '{boxplot_independent_var}' to plot distribution for '{dep_var_plot}'. Skipping."
        )
        continue

    # Initialize the plot
    plt.figure(figsize=(12, 8))

    # Create a FacetGrid for distribution plots per blue_banners category
    g = sns.FacetGrid(
        distribution_data,
        col=boxplot_independent_var,
        col_wrap=4,
        sharex=False,
        sharey=False,
        height=4,
    )
    g.map(sns.histplot, dep_var_plot, kde=True, color="blue", bins=20)

    # Add titles and labels
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(
        f"Distribution of {dep_var_plot.replace('_', ' ').title()} by Blue Banners",
        fontsize=16,
    )

    # Rotate x-axis labels if necessary
    for ax in g.axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    plt.tight_layout()

    # Save the plot in the distribution plots subdirectory
    plot_filename = f"{dep_var_plot}_distribution_by_{boxplot_independent_var}.png"
    plot_path = os.path.join(distribution_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved distribution plot: {plot_path}")

# for each blue banner count, print out the percentage of teams who had a elim pick below 3
for blue_banners in np.sort(data["blue_banners"].unique()):
    # Filter data for the current blue banner count
    blue_banners_data = data[data["blue_banners"] == blue_banners]
    # Calculate the percentage of teams with elim_progression below 1
    below_one_percentage = (
        (blue_banners_data["elim_pick"] < 3).sum() / len(blue_banners_data) * 100
    )
    print(
        f"For teams with {blue_banners} blue banners, {below_one_percentage:.2f}% had an elim_pick below 3."
    )


# For intervals of norm EPA between 1300 and 1900, counting by 100 print out the percentage of teams who had a elim pick below 3
for norm_epa in range(1300, 1901, 10):
    # Filter data for the current norm EPA interval
    norm_epa_data = data[
        (data["norm_epa"] >= norm_epa) & (data["norm_epa"] < norm_epa + 10)
    ]
    # Calculate the percentage of teams with elim_progression below 3
    below_one_percentage = (
        (norm_epa_data["elim_pick"] < 3).sum() / len(norm_epa_data) * 100
    )
    print(
        f"For teams with norm EPA between {norm_epa} and {norm_epa + 100}, {below_one_percentage:.2f}% had an elim_pick below 3."
    )
