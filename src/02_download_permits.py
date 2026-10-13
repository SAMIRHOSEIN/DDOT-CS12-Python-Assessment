# %% 1. Import libraries and define the API

import json
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import requests

ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path.cwd()
)

URL = (
    "https://maps2.dcgis.dc.gov/dcgis/rest/services/"
    "DCGIS_APPS/DDOT_TOPS/MapServer/24/query"
)

session = requests.Session()

def query_api(params):
    response = session.post(
        URL,
        data={"f": "json", **params},
        timeout=90
    )
    response.raise_for_status()

    result = response.json()

    if "error" in result:
        raise RuntimeError(result["error"])

    return result
# %% 2. Define selection criteria

WHERE = """
(
  (
    ApplicationDate >= TIMESTAMP '2026-01-01 00:00:00'
    AND ApplicationDate < TIMESTAMP '2027-01-01 00:00:00'
  )
  OR
  (
    IssueDate >= TIMESTAMP '2026-01-01 00:00:00'
    AND IssueDate < TIMESTAMP '2027-01-01 00:00:00'
  )
)
AND EventTypeDescription IN (
  'Construction Staging Area',
  'Mobile Crane Work Zone'
)
"""

# First retrieve ALL matching record IDs
id_result = query_api({
    "where": WHERE,
    "returnIdsOnly": "true"
})

object_ids = sorted(id_result.get("objectIds") or [])

if len(object_ids) != len(set(object_ids)):
    raise ValueError("Duplicate IDs received")

print("Total matching location records:", len(object_ids))
# %% 3. Download all matching records in batches
BATCH_SIZE = 500

all_features = []
first_page = None

for start in range(0, len(object_ids), BATCH_SIZE):

    batch_ids = object_ids[start:start + BATCH_SIZE]

    result = query_api({
        "objectIds": ",".join(map(str, batch_ids)),
        "outFields": "*",
        "returnGeometry": "true"
    })

    if first_page is None:
        first_page = result

    features = result.get("features", [])

    returned_ids = {
        feature["attributes"]["OBJECTID"]
        for feature in features
    }

    if returned_ids != set(batch_ids):
        raise ValueError(
            f"Missing or unexpected records in batch {start}"
        )

    all_features.extend(features)

    print(f"Downloaded {len(all_features)} of {len(object_ids)}")

assert len(all_features) == len(object_ids)

print("Download validation passed.")


# %% 4. Save the downloaded dataset

output_dir = ROOT / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "permits_2026_applied_or_issued_staging_crane_all_companies.json"

output_data = {
    "source": URL,
    "where": WHERE,
    "downloaded_utc": datetime.now(timezone.utc).isoformat(),
    "spatialReference": (
        first_page.get("spatialReference")
        if first_page else None
    ),
    "fields": first_page.get("fields") if first_page else [],
    "features": all_features
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("File saved:", output_path)


# %% 5. Verify activity types and permit counts

records = [
    feature["attributes"]
    for feature in all_features
]

print("\nTotal locations:", len(records))

unique_applications = {
    record["TrackingNumber"]
    for record in records
}

print("Unique applications:", len(unique_applications))

print("\nActivity types:")

activity_counts = Counter(
    record["EventTypeDescription"]
    for record in records
)

for activity, count in activity_counts.items():
    print(activity, ":", count)
