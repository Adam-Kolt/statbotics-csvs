import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Optional: Enhance plot aesthetics with Seaborn
sns.set(style="whitegrid")

# Load the data
data = pd.read_csv("worlds_data.csv")

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

    for independent_var in independent_variables:
        # Determine if the independent variable has an inverted version
        if independent_var in variables_to_invert:
            x = data[independent_var + "_inverted"]
            xlabel = f"{independent_var} (Inverted)"
        else:
            x = data[independent_var]
            xlabel = independent_var

        y = data[dep_var]

        # Remove NaN values for plotting and regression
        valid = (~np.isnan(x)) & (~np.isnan(y))
        x_valid = x[valid]
        y_valid = y[valid]

        if len(x_valid) < 2:
            print(f"Not enough data to plot {independent_var} vs {dep_var}. Skipping.")
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
        plt.title(f"{dep_var} vs {xlabel}", fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(dep_var, fontsize=12)
        plt.legend()
        plt.tight_layout()

        # Save the plot in the corresponding dependent variable folder
        plot_filename = f"{independent_var}_vs_{dep_var}.png"
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
