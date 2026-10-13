#%%
import json
from pathlib import Path
from collections import Counter

# Step 1: Locate the sample data file
project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "samples" / "permits_2026_sample.json"

# Step 2: Read the JSON file
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Step 3: Extract permit-location records
records = [item["attributes"] for item in data["features"]]

# Step 4: Count records and unique applications
print("Total locations:", len(records))

tracking_numbers = {r["TrackingNumber"] for r in records}
print("Unique applications:", len(tracking_numbers))

# Step 5: Check missing information
missing_permits = sum(r["PermitNumber"] is None for r in records)
print("Missing permit numbers:", missing_permits)

missing_coordinates = sum(
    r["Longitude"] is None or r["Latitude"] is None
    for r in records
)
print("Missing coordinates:", missing_coordinates)

missing_roadway = sum(r["RoadwaySegID"] is None for r in records)
print("Missing RoadwaySegID:", missing_roadway)

# Step 6: Examine permit types and company names
print("\nPermit types:")
for name, count in Counter(
    r["EventTypeDescription"] for r in records
).items():
    print(name, ":", count)

print("\nApplicant companies:")
for company, count in Counter(
    r["ApplicantCompany"] for r in records
).items():
    print(company, ":", count)

# %%
