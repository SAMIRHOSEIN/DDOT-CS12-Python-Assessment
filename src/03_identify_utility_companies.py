# %% STEP 03 - Explore utility company names (no files saved)
# Purpose:
# Read the complete 2026 date/activity dataset downloaded in Step 02.
# Explore how DC Water, PEPCO, and Washington Gas appear in
# ApplicantCompany, PermitteeName, and OwnerName.
# Print name variations, match counts, and possible additional locations.
# Use these findings to choose exact ApplicantCompany aliases in Step 04.
# %% STEP 1 - Load Complete Permit Dataset
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

file_path = (
    ROOT / "data" / "2_raw"
    / "permits_2026_applied_or_issued_staging_crane_all_companies.json"
)

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(
    [feature["attributes"] for feature in data["features"]]
)

print("Total locations:", len(df))
print("Unique applications:", df["TrackingNumber"].nunique())

# %% STEP 2 - Discover Utility Company Names
company_columns = [
    "ApplicantCompany",
    "PermitteeName",
    "OwnerName"
]

# Preliminary name patterns based on EDA findings
patterns = {
    "DC Water": r"\bDC\s*WATER\b|\bWASA\b",
    "PEPCO": r"\bPEPCO\b|\bPOTOMAC\s+ELECTRIC\b",
    "Washington Gas": r"\bWASHINGTON\s+GAS\b|\bWASH\s+GAS\b|\bWGL\b"
}

for utility, pattern in patterns.items():

    print("\n" + "=" * 55)
    print("UTILITY:", utility)
    print("=" * 55)

    matches = {}

    for column in company_columns:

        matches[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.contains(pattern, case=False, regex=True)
        )

        print(f"\n{column}: {matches[column].sum()} matches")

        print(
            df.loc[matches[column], column]
            .value_counts()
            .head(30)
            .to_string()
        )

    # A location may match in more than one column
    any_match = (
        matches["ApplicantCompany"]
        | matches["PermitteeName"]
        | matches["OwnerName"]
    )

    other_columns_only = (
        any_match & ~matches["ApplicantCompany"]
    )

    print("\nSUMMARY")
    print("Total matching permit locations:", any_match.sum())
    print("Unique tracking numbers among matches:",
          df.loc[any_match, "TrackingNumber"].nunique())
    print("Additional locations found only in PermitteeName or OwnerName",
          other_columns_only.sum())

    print("\nApplicantCompany values for those cases:")
    if other_columns_only.any():
        print(
            df.loc[other_columns_only, "ApplicantCompany"]
            .fillna("<NULL>")
            .value_counts()
            .head(15)
            .to_string()
        )
    else:
        print("None. All suggested locations also matched ApplicantCompany.")
