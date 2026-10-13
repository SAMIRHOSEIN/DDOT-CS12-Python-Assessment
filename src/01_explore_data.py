# %% STEP 1 - Initial Schema Discovery (10 Unfiltered Records)

import json
from pathlib import Path
import pandas as pd

# Define project directory
ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path.cwd()
)
#%%
# Load the original unfiltered sample
file_10 = ROOT / "data" / "samples" / "sample_10_unfiltered_schema_discovery.json"

with open(file_10, "r", encoding="utf-8") as f:
    raw_10 = json.load(f)

# Extract attributes
records_10 = [feature["attributes"] for feature in raw_10["features"]]
df_10 = pd.DataFrame(records_10)

# Basic structure
print("Number of rows:", len(df_10))
print("Number of columns:", len(df_10.columns))

# Show all column names and their ArcGIS data types
print("\nCOLUMN NAMES AND DATA TYPES")
for field in raw_10["fields"]:
    print(f"{field['name']}: {field['type']}")

# Inspect important columns
important_columns = [
    "TrackingNumber",
    "PermitNumber",
    "ApplicationDate",
    "IssueDate",
    "EventTypeDescription",
    "ApplicantCompany",
    "PermitteeName",
    "LocationID",
    "WorkLocationFullAddress",
    "Longitude",
    "Latitude",
]

print("\nFIRST 5 ROWS OF IMPORTANT COLUMNS")
print(df_10[important_columns].head().to_string(index=False))

# Understand the unit of observation
print("\nNumber of unique applications:",
      df_10["TrackingNumber"].nunique())

print("Number of unique locations:",
      df_10["LocationID"].nunique())

print("\nCoordinate system:",
      raw_10["spatialReference"])


# %% STEP 2 - Data Profiling (500 Date-Filtered Records)

file_500 = (
    ROOT / "data" / "samples"
    / "sample_500_date_filtered_2026_company_activity_exploration.json"
)

with open(file_500, "r", encoding="utf-8") as f:
    raw_500 = json.load(f)

if "error" in raw_500:
    raise ValueError(raw_500["error"])

records_500 = [
    feature["attributes"]
    for feature in raw_500["features"]
]

df_500 = pd.DataFrame(records_500)

print("DATASET OVERVIEW")
print("Rows:", len(df_500))
print("Columns:", len(df_500.columns))
print("Unique applications:", df_500["TrackingNumber"].nunique())
print("Unique locations:", df_500["LocationID"].nunique())
print("Duplicate OBJECTIDs:", df_500["OBJECTID"].duplicated().sum())

locations_per_application = df_500.groupby("TrackingNumber").size()
print("Applications with multiple locations:",
      (locations_per_application > 1).sum())

print("\nPERMIT ACTIVITY TYPES")
print(df_500["EventTypeDescription"].value_counts(dropna=False).to_string())

print("\nPERMIT STATUSES")
print(df_500["StatusDescription"].value_counts(dropna=False).to_string())

print("\nAPPLICANT COMPANY NAMES")
print(
    df_500["ApplicantCompany"]
    .fillna("<NULL>")
    .value_counts()
    .to_string()
)

print("\nTOP 25 PERMITTEE NAMES")
print(
    df_500["PermitteeName"]
    .fillna("<NULL>")
    .value_counts()
    .head(25)
    .to_string()
)

print("\nTOP 25 OWNER NAMES")
print(
    df_500["OwnerName"]
    .fillna("<NULL>")
    .value_counts()
    .head(25)
    .to_string()
)

