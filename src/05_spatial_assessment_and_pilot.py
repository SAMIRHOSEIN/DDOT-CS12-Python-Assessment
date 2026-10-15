# %% STEP 1 - Load the validated permit records
# Purpose:
# Load the validated permit locations from Step 04 and assess
# the completeness and consistency of their spatial coordinates.
# Check the coordinate reference systems and compare Original
# (XCoord/YCoord) with OnSeg (OnSegX/OnSegY) coordinates.
# Identify locations with different coordinates and measure
# their separation without automatically excluding any records.
# Select a small pilot sample from the three utility companies
# and retrieve nearby roadway candidates from the DDOT GIS service.
# Use these findings to guide roadway matching in Step 06.
# Do not assign final FHWA functional classes in this step.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent if '__file__' in globals() else Path.cwd()
INPUT = (
    ROOT / 'data' / 'processed'
    / 'permits_2026_applied_or_issued_staging_crane_three_utilities.json'
)
OUT = ROOT / 'outputs' / '2_spatial'
OUT.mkdir(parents=True, exist_ok=True)

with INPUT.open('r', encoding='utf-8') as f:
    source = json.load(f)

features = source['features']
attributes = pd.DataFrame([f['attributes'] for f in features])
geometries = pd.DataFrame([f.get('geometry') or {} for f in features])
for axis in ('x', 'y'):
    if axis not in geometries:
        geometries[axis] = pd.NA

df = attributes.copy()
df['geometry_x_3857'] = pd.to_numeric(geometries['x'], errors='coerce')
df['geometry_y_3857'] = pd.to_numeric(geometries['y'], errors='coerce')

print('Number of permit records:', len(df))
print('Source geometry type:', source.get('geometryType') or 'Not recorded in saved JSON')
print('Source spatial reference:', source.get('spatialReference'))

# %% STEP 2 - Check the availability and consistency of spatial fields
# Geometry coordinates are EPSG:3857 (Web Mercator), NOT longitude/latitude.
# XCoord/OnSegX are in a different local coordinate system. Their units
# are NOT assumed to be meters until their coordinate reference is verified.
lon = pd.to_numeric(df['Longitude'], errors='coerce')
lat = pd.to_numeric(df['Latitude'], errors='coerce')
valid_lonlat = lon.between(-180, 180) & lat.between(-90, 90)
valid_shape = df['geometry_x_3857'].notna() & df['geometry_y_3857'].notna()

onseg_x = pd.to_numeric(df['OnSegX'], errors='coerce')
onseg_y = pd.to_numeric(df['OnSegY'], errors='coerce')
raw_x = pd.to_numeric(df['XCoord'], errors='coerce')
raw_y = pd.to_numeric(df['YCoord'], errors='coerce')
valid_onseg = onseg_x.notna() & onseg_y.notna() & ~((onseg_x == 0) & (onseg_y == 0))
valid_raw_xy = raw_x.notna() & raw_y.notna() & ~((raw_x == 0) & (raw_y == 0))

# Convert Web Mercator geometry back to degrees as a coordinate QA check.
# This diagnostic does not snap a point to any road.
R = 6378137.0
lon_geom = np.degrees(df['geometry_x_3857'] / R)
lat_geom = np.degrees(
    2 * np.arctan(np.exp(df['geometry_y_3857'] / R)) - np.pi / 2
)
rough_offset_m = np.sqrt(
    ((lon - lon_geom) * 111320 * np.cos(np.radians(lat))) ** 2
    + ((lat - lat_geom) * 111320) ** 2
)
rough_offset_m = rough_offset_m.where(valid_lonlat & valid_shape)

# A different on-segment coordinate does not automatically mean bad data.
# It may indicate that a point has been snapped toward a roadway.
# No displacement threshold is used. Every numerical difference is flagged.
# Therefore, this count could differ slightly from earlier runs that used 1e-6.
onseg_differs = (
    valid_onseg & valid_raw_xy
    & ((onseg_x != raw_x) | (onseg_y != raw_y))
)

metrics = {
    'Total selected permit records': len(df),
    'Records with valid Longitude and Latitude': int(valid_lonlat.sum()),
    'Records missing/invalid Longitude or Latitude': int((~valid_lonlat).sum()),
    'Records with EPSG:3857 point geometry': int(valid_shape.sum()),
    'Records missing EPSG:3857 point geometry': int((~valid_shape).sum()),
    'Records with comparable Longitude/Latitude and ArcGIS geometry': int(rough_offset_m.notna().sum()),
    'Records with OnSegX and OnSegY': int(valid_onseg.sum()),
    'Records with any numerical difference between Original and OnSeg': int(onseg_differs.sum()),
    'Records missing RoadwaySegID': int(df['RoadwaySegID'].isna().sum()),
    'Records with missing or zero StreetSegID': int(
        (pd.to_numeric(df['StreetSegID'], errors='coerce').fillna(0) == 0).sum()
    ),
}

print('\n--- SPATIAL DATA QUALITY ---')
for description, count in metrics.items():
    print(f'{description}: {count}')

# Keep individual approximate separations for QA, but avoid lengthy
# statistical tables in the console. No arbitrary threshold is applied.

pd.DataFrame(
    [{'Check': name, 'Count': count} for name, count in metrics.items()]
).to_csv(OUT / 'spatial_data_quality_summary.csv', index=False)

# %% STEP 2B - Compare ArcGIS Geometry with OnSeg (CRS consistency check)
# Test EPSG:26985 as the candidate CRS for Original and OnSeg coordinates.
# This is a working hypothesis, not a confirmed DDOT field definition.
# Both EPSG:26985 and distances calculated in this section use meters.
# ArcGIS geometry is interpreted as EPSG:3857 (Web Mercator).
# A small OnSeg-to-geometry distance supports positional consistency,
# but does NOT prove how DDOT created OnSeg or that its street is correct.
local_crs = 'EPSG:26985'  # Candidate local CRS; verify against DDOT metadata.
arcgis_crs = 'EPSG:3857'

arcgis_to_local = Transformer.from_crs(
    arcgis_crs, local_crs, always_xy=True
)

arcgis_x_local, arcgis_y_local = arcgis_to_local.transform(
    df['geometry_x_3857'].to_numpy(dtype=float),
    df['geometry_y_3857'].to_numpy(dtype=float),
)

# Copy data into explicit QA fields for inspection and later analysis.
df['ArcGIS_X_in_candidate_local_CRS'] = arcgis_x_local
df['ArcGIS_Y_in_candidate_local_CRS'] = arcgis_y_local

df['OnSeg_to_ArcGIS_m'] = np.hypot(
    onseg_x - df['ArcGIS_X_in_candidate_local_CRS'],
    onseg_y - df['ArcGIS_Y_in_candidate_local_CRS'],
).where(valid_onseg & valid_shape)

df['Original_to_ArcGIS_m'] = np.hypot(
    raw_x - df['ArcGIS_X_in_candidate_local_CRS'],
    raw_y - df['ArcGIS_Y_in_candidate_local_CRS'],
).where(valid_raw_xy & valid_shape)

print('\n--- COORDINATE REFERENCE SYSTEM CHECK ---')
print('Candidate local CRS for Original/OnSeg:', local_crs)
print('ArcGIS geometry CRS:', arcgis_crs)
print('Records compared:', int(df['OnSeg_to_ArcGIS_m'].notna().sum()))
print('Maximum OnSeg-to-ArcGIS distance (m):',
      round(df['OnSeg_to_ArcGIS_m'].max(), 3))
print('NOTE: A small distance supports coordinate consistency; it does not')
print('confirm the official CRS definition or the correct roadway match.')


# %% STEP 2C - Compare Original and OnSeg for ALL permit records
# No distance threshold is used to choose which records get reviewed.
# Every record with a numerical X or Y difference is retained for review.
df['Original_to_OnSeg_m'] = np.hypot(
    raw_x - onseg_x,
    raw_y - onseg_y,
).where(valid_raw_xy & valid_onseg)

df['Original_and_OnSeg_differ'] = onseg_differs
df['LongitudeLat_to_ArcGIS_approx_m'] = rough_offset_m

print('\n--- ORIGINAL VS ONSEG DISTANCE ANALYSIS ---')
print('Permit records:', len(df))
print('Records with both Original and OnSeg coordinates:',
      int((valid_raw_xy & valid_onseg).sum()))
print('Records with any numerical difference:', int(onseg_differs.sum()))
print('Records with identical numerical coordinates:',
      int((valid_raw_xy & valid_onseg & ~onseg_differs).sum()))
print('Maximum Original-to-OnSeg distance (m):',
      round(df['Original_to_OnSeg_m'].max(), 3))

compare_mask = valid_shape & valid_raw_xy & valid_onseg
delta = (
    df['Original_to_ArcGIS_m'] - df['OnSeg_to_ArcGIS_m']
)


# Show the exceptional record IDs without a lengthy statistics table.
original_closer_ids = df.loc[
    compare_mask & (delta < 0), 'LocationID'
].tolist()
if original_closer_ids:
    print('LocationIDs where Original is closer:', original_closer_ids[:20])
    if len(original_closer_ids) > 20:
        print('Additional IDs can be inspected in the Python DataFrame.')

# Keep only decision-relevant fields in the saved CSV reports.
# LongitudeLat_to_ArcGIS_approx_m and Original_to_ArcGIS_m are used for
# internal coordinate quality checks, but are not needed in these outputs.
# Original_to_OnSeg_m helps identify displacement; OnSeg_to_ArcGIS_m checks
# coordinate consistency before making requests to the roadway API.
report_columns = [
    'OBJECTID', 'LocationID', 'TrackingNumber', 'UtilityCompany',
    'WorkLocationFullAddress',
    'XCoord', 'YCoord', 'OnSegX', 'OnSegY',
    'geometry_x_3857', 'geometry_y_3857',
    'Original_and_OnSeg_differ', 'Original_to_OnSeg_m',
    'OnSeg_to_ArcGIS_m',
]

# Report A: every selected permit record, including identical coordinates.
full_report_path = OUT / 'all_permit_records_original_onseg_arcgis_comparison.csv'
df[report_columns].to_csv(full_report_path, index=False)

# Report B: only records where Original and OnSeg have different values.
differing_report = df.loc[onseg_differs, report_columns].copy()
differing_report = differing_report.sort_values(
    'Original_to_OnSeg_m', ascending=False
)
differing_report_path = OUT / 'records_with_different_original_onseg_coordinates.csv'
differing_report.to_csv(differing_report_path, index=False)

print('\nAll permit records coordinate report saved:', full_report_path)
print('Records with different Original/OnSeg coordinates saved:',
      differing_report_path)
print('NOTE: No roadway or functional class is assigned in this analysis.')


# %% STEP 3 - Choose a small, reproducible pilot sample from all three utilities
# Three different permit addresses per company where possible. Do not
# automatically use the closest road: intersections and alleys are ambiguous.
df['spatial_ready'] = valid_shape & valid_lonlat
pilot = (
    df.loc[df['spatial_ready']]
      .sort_values('OBJECTID')
      .drop_duplicates(subset=['UtilityCompany', 'WorkLocationFullAddress'])
      .groupby('UtilityCompany', group_keys=False)
      .head(3)
      .copy()
)

print('\n--- PILOT PERMIT RECORDS ---')
print(pilot[
    ['LocationID', 'UtilityCompany', 'WorkLocationFullAddress',
     'Longitude', 'Latitude', 'geometry_x_3857', 'geometry_y_3857']
].to_string(index=False))

# %% STEP 4 - Query nearby DDOT roadway SubBlocks (pilot only)
# DDOT Roadway SubBlock layer 162 supports spatial queries with a distance.
# A 30 m search radius is for finding CANDIDATES, not proof of road identity.
# The layer includes alleys; a missing FHWA class is not imputed.
ROAD_QUERY_URL = (
    'https://maps2.dcgis.dc.gov/dcgis/rest/services/'
    'DCGIS_DATA/Transportation_WebMercator/MapServer/162/query'
)
SEARCH_RADIUS_METERS = 30

session = requests.Session()
results = []

for _, rec in pilot.iterrows():
    params = {
        'f': 'json',
        'where': '1=1',
        'geometry': json.dumps({
            'x': float(rec['geometry_x_3857']),
            'y': float(rec['geometry_y_3857']),
            'spatialReference': {'wkid': 3857},
        }),
        'geometryType': 'esriGeometryPoint',
        'inSR': 3857,
        'spatialRel': 'esriSpatialRelIntersects',
        'distance': SEARCH_RADIUS_METERS,
        'units': 'esriSRUnit_Meter',
        'outFields': 'SUBBLOCKKEY,ROUTENAME,STREETNAME,ROADTYPE,FHWAFUNCTIONALCLASS',
        'returnGeometry': 'false',
    }

    context = {
        'LocationID': rec['LocationID'],
        'UtilityCompany': rec['UtilityCompany'],
        'WorkLocationFullAddress': rec['WorkLocationFullAddress'],
    }
    try:
        response = session.post(ROAD_QUERY_URL, data=params, timeout=45)
        response.raise_for_status()
        data = response.json()
        if 'error' in data:
            raise RuntimeError(str(data['error']))
        candidates = data.get('features', [])
        print(f"LocationID {rec['LocationID']}: {len(candidates)} nearby roadway candidates")
        if not candidates:
            results.append({**context, 'query_status': 'no_candidate'})
        for candidate in candidates:
            results.append({
                **context,
                'query_status': 'candidate_for_review',
                **candidate['attributes'],
            })
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        print(f"LocationID {rec['LocationID']}: API query error: {exc}")
        results.append({**context, 'query_status': 'api_error',
                        'error_details': str(exc)})

pilot_results = pd.DataFrame(results)
output_file = OUT / 'pilot_nearby_roadway_candidates.csv'
pilot_results.to_csv(output_file, index=False)
print('\nPilot candidate results saved to:', output_file)
print('IMPORTANT: Candidate roadways have NOT been assigned as final matches.')
