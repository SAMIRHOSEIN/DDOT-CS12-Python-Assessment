# %% STEP 04 - Filter target utilities and validate permit records
# Purpose:
# Use the company-name findings from Step 03 to select permit locations
# for DC Water, PEPCO, and Washington Gas using exact ApplicantCompany aliases.
# Step 03 only prints exploratory results, so perform the essential
# company-matching safety checks independently in this step.
# Validate identifiers, application relationships, 2026 dates, and activities.
# Print informative validation results, save the company inventory and QA
# summary, and save detailed review CSVs only when issues are found.
# Preserve source attributes and geometry in the filtered JSON for Steps 05-06.
# %% STEP 1 - Load the full, date-and-activity-filtered dataset
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent if '__file__' in globals() else Path.cwd()
INPUT = ROOT / 'data' / '2_raw' / 'permits_2026_applied_or_issued_staging_crane_all_companies.json'
OUT = ROOT / 'outputs' / '1_validation'
OUT.mkdir(parents=True, exist_ok=True)


def save_review_if_nonempty(dataframe, filename):
    # Avoid empty review files; remove an old file if a rerun has no findings.
    path = OUT / filename
    if dataframe.empty:
        path.unlink(missing_ok=True)
    else:
        dataframe.to_csv(path, index=False)


with INPUT.open('r', encoding='utf-8') as f:
    source = json.load(f)
if 'features' not in source:
    raise ValueError('The input JSON has no features; check the input file.')

features = source['features']
df = pd.DataFrame([item['attributes'] for item in features])
needed = {'OBJECTID', 'LocationID', 'TrackingNumber', 'PermitNumber',
          'ApplicationDate', 'IssueDate', 'EventTypeDescription',
          'ApplicantCompany', 'PermitteeName', 'OwnerName'}
missing_columns = needed - set(df.columns)
if missing_columns:
    raise ValueError(f'Missing required columns: {sorted(missing_columns)}')
print('Total downloaded permit-location records:', len(df))

# %% STEP 2 - Map confirmed ApplicantCompany name variants to the 3 utilities
# Exact normalized names come from the previous exploration, not a guessed substring.
def normalize_name(value):
    if pd.isna(value):
        return ''
    return ' '.join(str(value).strip().casefold().split())

company_aliases = {
    'dcwater/cip': 'DC Water',
    'dcwater/ops': 'DC Water',
    'dcwater': 'DC Water',
    'potomac electric power company': 'PEPCO',
    'pepco': 'PEPCO',
    'wash gas & light co.': 'Washington Gas',
}

df['ApplicantCompany_normalized'] = df['ApplicantCompany'].map(normalize_name)
df['UtilityCompany'] = df['ApplicantCompany_normalized'].map(company_aliases)

print('\nLocations identified by ApplicantCompany:')
print(df['UtilityCompany'].fillna('Unclassified / other company').value_counts().to_string())

# Save ALL observed applicant-company values so unrecognized spellings can be reviewed.
company_inventory = (
    df.groupby('ApplicantCompany_normalized', dropna=False)
      .agg(location_records=('OBJECTID', 'size'),
           example_original_name=('ApplicantCompany', 'first'),
           mapped_utility=('UtilityCompany', 'first'))
      .reset_index()
      .sort_values('location_records', ascending=False)
)
company_inventory.to_csv(OUT / 'applicant_company_name_inventory.csv', index=False)
print('\nTop 25 ApplicantCompany names (including unmatched):')
print(company_inventory.head(25).to_string(index=False))

# %% STEP 3 - Essential safety check for missed or conflicting utility locations
# Step 03 does the exploratory name review, but does not save results.
# Recheck only the two inclusion risks needed to validate Step 04 selection.
patterns = {
    'DC Water': r'\bDC\s*WATER\b|\bWASA\b',
    'PEPCO': r'\bPEPCO\b|\bPOTOMAC\s+ELECTRIC\b',
    'Washington Gas': r'\bWASHINGTON\s+GAS\b|\bWASH\s+GAS\b|\bWGL\b',
}
review_masks = []
aux_summary = {}
for utility, pattern in patterns.items():
    other_match = (
        df['PermitteeName'].fillna('').astype(str).str.contains(pattern, case=False, regex=True)
        | df['OwnerName'].fillna('').astype(str).str.contains(pattern, case=False, regex=True)
    )
    outside_applicant = other_match & df['UtilityCompany'].isna()
    conflicting_company = other_match & df['UtilityCompany'].notna() & df['UtilityCompany'].ne(utility)
    aux_summary[f'{utility}: other fields but not ApplicantCompany'] = int(outside_applicant.sum())
    aux_summary[f'{utility}: conflicting utility name'] = int(conflicting_company.sum())
    print(f'\n{utility}: newly suggested by PermitteeName/OwnerName:', int(outside_applicant.sum()))
    print(f'{utility}: conflicts with classified ApplicantCompany:', int(conflicting_company.sum()))
    flagged = outside_applicant | conflicting_company
    if flagged.any():
        part = df.loc[flagged, ['OBJECTID', 'LocationID', 'TrackingNumber',
                                'ApplicantCompany', 'PermitteeName', 'OwnerName',
                                'UtilityCompany']].copy()
        part['suggested_by_other_field'] = utility
        review_masks.append(part)

if review_masks:
    review = pd.concat(review_masks, ignore_index=True)
else:
    review = pd.DataFrame(columns=['OBJECTID', 'LocationID', 'TrackingNumber',
                                    'ApplicantCompany', 'PermitteeName', 'OwnerName',
                                    'UtilityCompany', 'suggested_by_other_field'])
save_review_if_nonempty(review, 'company_matches_needing_review.csv')

# %% STEP 4 - Validate identifiers and the relationship between applications and permits
permit_numbers = df['PermitNumber'].astype('string').str.strip().replace('', pd.NA)
tracking_numbers = df['TrackingNumber'].astype('string').str.strip().replace('', pd.NA)
df['PermitNumber_clean'] = permit_numbers
df['TrackingNumber_clean'] = tracking_numbers
selected = df.loc[df['UtilityCompany'].notna()].copy()

# Different utility companies associated with the same application (not just name variants)
company_per_tracking = (
    selected.dropna(subset=['TrackingNumber_clean'])
            .groupby('TrackingNumber_clean')['UtilityCompany'].nunique()
)
mixed_company_ids = company_per_tracking[company_per_tracking > 1].index
print('\nTrackingNumbers linked to more than one of the three target companies:', len(mixed_company_ids))


# Check whether any non-target company records share a TrackingNumber
# with an application identified as DC Water, PEPCO, or Washington Gas.
# This helps detect potentially excluded locations of the same application.

target_tracking_to_utility = (
    selected.dropna(subset=["TrackingNumber_clean"])
    .groupby("TrackingNumber_clean")["UtilityCompany"]
    .agg(lambda names: ", ".join(sorted(set(names))))
)

shared_tracking_records = df.loc[
    df["UtilityCompany"].isna()
    & df["TrackingNumber_clean"].isin(target_tracking_to_utility.index),
    [
        "OBJECTID",
        "LocationID",
        "TrackingNumber",
        "TrackingNumber_clean",
        "ApplicantCompany",
        "PermitteeName",
        "OwnerName",
    ]
].copy()

shared_tracking_records["TargetUtility"] = (
    shared_tracking_records["TrackingNumber_clean"]
    .map(target_tracking_to_utility)
)

print("\n--- CHECK: POSSIBLE EXCLUDED LOCATIONS ---")

print(
    "Non-target records sharing a TrackingNumber with target companies:",
    len(shared_tracking_records)
)

print(
    "Number of affected applications:",
    shared_tracking_records["TrackingNumber_clean"].nunique()
)

print("\nApplicantCompany names in these records:")
if shared_tracking_records.empty:
    print("None.")
else:
    print(
        shared_tracking_records["ApplicantCompany"]
        .fillna("<NULL>")
        .value_counts()
        .head(20)
        .to_string()
    )

# Save details only if excluded locations were found.
save_review_if_nonempty(
    shared_tracking_records.drop(columns=["TrackingNumber_clean"]),
    'non_target_locations_sharing_target_tracking.csv'
)

if not shared_tracking_records.empty:
    print("\nWARNING: Review these records before finalizing the filter.")
else:
    print("\nPASS: No non-target locations share target TrackingNumbers.")


# A tracking number may have multiple permit numbers, and vice versa.
links = df.dropna(subset=['TrackingNumber_clean', 'PermitNumber_clean'])
permits_per_tracking = links.groupby('TrackingNumber_clean')['PermitNumber_clean'].nunique()
trackings_per_permit = links.groupby('PermitNumber_clean')['TrackingNumber_clean'].nunique()
print('TrackingNumbers linked to >1 PermitNumber:', int((permits_per_tracking > 1).sum()))
print('PermitNumbers linked to >1 TrackingNumber:', int((trackings_per_permit > 1).sum()))


# Check if the same TrackingNumber has both available and
# missing PermitNumbers across its records.
# This is important for grouping all records of one application.

permit_status_by_tracking = (
    selected.dropna(subset=["TrackingNumber_clean"])
    .groupby("TrackingNumber_clean")["PermitNumber_clean"]
    .agg(
        has_permit_number=lambda x: x.notna().any(),
        has_missing_permit_number=lambda x: x.isna().any()
    )
)

# Identify TrackingNumbers with both conditions
mixed_permit_ids = permit_status_by_tracking.index[
    permit_status_by_tracking["has_permit_number"]
    & permit_status_by_tracking["has_missing_permit_number"]
]

mixed_permit_records = selected.loc[
    selected["TrackingNumber_clean"].isin(mixed_permit_ids),
    [
        "TrackingNumber",
        "PermitNumber",
        "LocationID",
        "ApplicantCompany",
        "UtilityCompany"
    ]
].copy()

print("\n--- CHECK: PERMIT NUMBER CONSISTENCY ---")

print(
    "TrackingNumbers with both missing and available PermitNumbers:",
    len(mixed_permit_ids)
)

print(
    "Records associated with these TrackingNumbers:",
    len(mixed_permit_records)
)

# Save details only if an application has mixed PermitNumber availability.
save_review_if_nonempty(
    mixed_permit_records, 'mixed_missing_and_available_permit_numbers.csv'
)

if len(mixed_permit_ids) == 0:
    print("PASS: No mixed PermitNumber availability found.")
else:
    print("REVIEW: Some applications have both missing and available PermitNumbers.")


# Export unusual relationships for manual inspection, without deleting records.
relationship_flags = selected.loc[
    selected['TrackingNumber_clean'].isin(mixed_company_ids)
    | selected['TrackingNumber_clean'].isin(permits_per_tracking[permits_per_tracking > 1].index)
    | selected['PermitNumber_clean'].isin(trackings_per_permit[trackings_per_permit > 1].index),
    ['OBJECTID', 'LocationID', 'TrackingNumber', 'PermitNumber', 'UtilityCompany']
]
save_review_if_nonempty(relationship_flags, 'identifier_relationships_needing_review.csv')

# %% STEP 5 - Verify identifiers, date criteria, and activity types
n_duplicate_objectids = int(df['OBJECTID'].duplicated().sum())
n_duplicate_locationids = int(df['LocationID'].dropna().duplicated().sum())
n_missing_locationids = int(df['LocationID'].isna().sum())
n_missing_tracking = int(selected['TrackingNumber_clean'].isna().sum())
n_missing_permit = int(selected['PermitNumber_clean'].isna().sum())

start = pd.Timestamp('2026-01-01', tz='UTC')
end = pd.Timestamp('2027-01-01', tz='UTC')
application_dates = pd.to_datetime(df['ApplicationDate'], unit='ms', utc=True, errors='coerce')
issue_dates = pd.to_datetime(df['IssueDate'], unit='ms', utc=True, errors='coerce')
date_matches = (
    application_dates.ge(start) & application_dates.lt(end)
    | issue_dates.ge(start) & issue_dates.lt(end)
)
activity_matches = df['EventTypeDescription'].isin([
    'Construction Staging Area', 'Mobile Crane Work Zone'
])
print('\nDuplicate OBJECTIDs:', n_duplicate_objectids)
print('Repeated non-null LocationIDs:', n_duplicate_locationids)
print('Missing LocationIDs:', n_missing_locationids)
print('Missing TrackingNumbers (selected):', n_missing_tracking)
print('Missing PermitNumbers (selected):', n_missing_permit)
print('Rows not matching 2026 date criterion (UTC interpretation):', int((~date_matches).sum()))
print('Rows not matching two activity types:', int((~activity_matches).sum()))

# %% STEP 6 - Save the validation summary and filtered permit records
metrics = {
    'All downloaded location records': len(df),
    'Selected DC Water locations': int(selected['UtilityCompany'].eq('DC Water').sum()),
    'Selected PEPCO locations': int(selected['UtilityCompany'].eq('PEPCO').sum()),
    'Selected Washington Gas locations': int(selected['UtilityCompany'].eq('Washington Gas').sum()),
    'Total selected location records': len(selected),
    'Unclassified location records': int(df['UtilityCompany'].isna().sum()),
    'Selected unique TrackingNumbers': int(selected['TrackingNumber_clean'].nunique()),
    'Selected unique nonmissing PermitNumbers': int(selected['PermitNumber_clean'].nunique()),
    'Duplicate OBJECTIDs': n_duplicate_objectids,
    'Repeated non-null LocationIDs': n_duplicate_locationids,
    'Missing LocationIDs': n_missing_locationids,
    'Missing TrackingNumbers in selected records': n_missing_tracking,
    'Missing PermitNumbers in selected records': n_missing_permit,
    'TrackingNumbers associated with multiple target utilities': len(mixed_company_ids),
    'TrackingNumbers with multiple PermitNumbers': int((permits_per_tracking > 1).sum()),
    'PermitNumbers with multiple TrackingNumbers': int((trackings_per_permit > 1).sum()),
    'Rows not matching date criterion under UTC': int((~date_matches).sum()),
    'Rows with unexpected permit activity': int((~activity_matches).sum()),
    'TrackingNumbers with mixed PermitNumber availability': len(mixed_permit_ids),
    **aux_summary,
}
pd.DataFrame([{'Check': k, 'Count': v} for k, v in metrics.items()]).to_csv(
    OUT / 'permit_filter_validation_summary.csv', index=False
)

# These indicate a data or classification issue requiring review before spatial analysis.
blocking_issues = (
    n_duplicate_objectids > 0
    or n_missing_locationids > 0
    or (~date_matches).any()
    or (~activity_matches).any()
    or not review.empty
    or len(mixed_company_ids) > 0
)
if blocking_issues:
    raise ValueError('Validation requires manual review. Check the CSV reports in outputs/validation/.')

selected_indices = set(selected.index)
filtered_features = []
for index, feature in enumerate(features):
    if index in selected_indices:
        copied_feature = dict(feature)
        copied_feature['attributes'] = dict(feature['attributes'])
        copied_feature['attributes']['UtilityCompany'] = str(df.at[index, 'UtilityCompany'])
        filtered_features.append(copied_feature)

fields = list(source.get('fields') or [])
if not any(field.get('name') == 'UtilityCompany' for field in fields):
    fields.append({'name': 'UtilityCompany', 'alias': 'Identified Utility Company',
                   'type': 'esriFieldTypeString', 'length': 40})

output = {
    'description': '2026 date and two activity types; 3 utilities classified by ApplicantCompany exact aliases',
    'source': source.get('source'),
    'filter': source.get('where'),
    'classification_method': 'Normalized exact ApplicantCompany name matching',
    'spatialReference': source.get('spatialReference'),
    'geometryType': source.get('geometryType'),
    'fields': fields,
    'features': filtered_features,
}
output_file = ROOT / 'data' / '3_processed' / 'permits_2026_applied_or_issued_staging_crane_three_utilities.json'
output_file.parent.mkdir(parents=True, exist_ok=True)
with output_file.open('w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print('\nFiltered permit-location dataset saved:', output_file)
print('Validation summary:', OUT / 'permit_filter_validation_summary.csv')
review_files = sorted(path.name for path in OUT.glob('*_needing_review.csv'))
review_files += sorted(path.name for path in OUT.glob('mixed_missing_and_available_permit_numbers.csv'))
review_files += sorted(path.name for path in OUT.glob('non_target_locations_sharing_target_tracking.csv'))
print('Company-name inventory:', OUT / 'applicant_company_name_inventory.csv')
print('Non-empty review files:', ', '.join(review_files) if review_files else 'None needed')
