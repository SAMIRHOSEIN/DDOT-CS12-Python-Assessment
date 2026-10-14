# %% STEP 1 - Settings and libraries
"""DDOT CS-12 | Step 06 (v2): LRS-first roadway and FHWA class matching.

Run from a repository with this structure:
    src/06_match_roadways_and_assign_functional_class.py
    data/processed/permits_2026_applied_or_issued_staging_crane_three_utilities.json

Main principles
---------------
1. Identical Original / OnSeg: inspect OnSeg once.
2. Different Original / OnSeg: inspect BOTH, regardless of separation.
3. LRS proposes SubBlockKeys. Layer 162 provides official FHWA classes.
4. Validate street, quadrant, COMPLETE address ranges, roadway type and
   whether the address-matching road is the nearest reported LRS roadway.
5. Allow a verified CLASS with multiple valid SubBlockKeys, but do not invent
   a unique key. Keep every candidate and the reason for any uncertainty.
6. Special bridge addresses are handled explicitly, but are never silently
   treated as a verified street address.

Search radii (75 m and, when needed, 200 m) determine which candidates LRS
returns. They are NOT thresholds for treating Original / OnSeg as identical,
or for automatically accepting a candidate roadway.

A known source limitation: TOPS local XY is interpreted as EPSG:26985 based on
Step 05 QA. Confirm against official TOPS metadata before final submission.

Dependencies: pandas, requests, pyproj. Start with RUN_MODE='extended', then 'full'.
The standard-library SQLite API cache makes large runs resumable without rewriting
a huge JSON file; existing v2 raw responses are imported automatically.
"""

import json
import math
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests
from pyproj import Transformer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# CHOOSE ONE MODE: 'pilot' (18 rows), 'extended' (~150 rows), or 'full' (all rows).
RUN_MODE = 'full'                 # After checking extended, change to 'full'.
PILOT_PER_COMPANY_PER_GROUP = 3
EXTENDED_PER_COMPANY_PER_GROUP = 25   # 25 x 3 utilities x 2 coordinate groups ~= 150
RANDOM_SEED = 20261014                # Reproducible extended selection
LRS_SEARCH_RADII_METERS = (75, 200)   # Search windows only, NOT acceptance thresholds
REQUEST_PAUSE_SECONDS = 0.10
SAVE_EVERY = 20                       # Save result/audit CSVs every 20 new locations
MAX_CONSECUTIVE_API_ERRORS = 5        # Stop if the DDOT services appear unavailable
USE_PREVIOUS_RAW_API_CACHE = True     # Reuse raw DDOT replies, NEVER old class decisions

VERSION = 'v2'
ROOT = Path(__file__).resolve().parent.parent if '__file__' in globals() else Path.cwd()
INPUT = ROOT / 'data' / 'processed' / 'permits_2026_applied_or_issued_staging_crane_three_utilities.json'
OUT = ROOT / 'outputs' / 'roadway_matching_lrs_v2'
OUT.mkdir(parents=True, exist_ok=True)
CACHE_DB_PATH = OUT / 'api_cache_lrs_v2.sqlite3'
LEGACY_V2_CACHE_PATH = OUT / 'api_cache_lrs_v2.json'
PREVIOUS_CACHE_PATH = ROOT / 'outputs' / 'roadway_matching_lrs' / 'api_cache_lrs_first.json'

LRS_URL = ('https://rh.dcgis.dc.gov/dcgis/rest/services/DDOT/'
           'RoadwayBlocks/MapServer/exts/DdotLrsExtensions/getPointOnSubBlock')
SUBBLOCK_URL = ('https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/'
                'Transportation_WebMercator/MapServer/162/query')
LOCAL_WKID = 26985
WEB_WKID = 3857
web_to_local = Transformer.from_crs(WEB_WKID, LOCAL_WKID, always_xy=True)

session = requests.Session()
session.mount('https://', HTTPAdapter(max_retries=Retry(
    total=3, backoff_factor=0.8, status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(('GET', 'POST')))))

# DDOT documentation confirms Theodore Roosevelt Bridge carries Interstate 66.
# A route alias is not sufficient to establish the precise permit location.
SPECIAL_ROUTE_ALIASES = {'ROOSEVELT BRIDGE': 'INTERSTATE 66',
                         'THEODORE ROOSEVELT BRIDGE': 'INTERSTATE 66'}

# QA warning, NOT an automatic distance rejection threshold. A typical
# hundred-block spans at most a few hundred address numbers; a returned
# '4280 - 45099' range is suspicious and must be manually reviewed.
SUSPICIOUS_LRS_ADDRESS_SPAN = 1000


# %% STEP 2 - Load the permit-location records

def read_permit_records(path):
    if not path.exists():
        raise FileNotFoundError(f'Input JSON not found: {path}')
    with path.open(encoding='utf-8') as file:
        data = json.load(file)
    records = []
    for item in data['features']:
        row = dict(item['attributes'])
        geom = item.get('geometry') or {}
        row['ArcGIS_x_3857'] = geom.get('x')
        row['ArcGIS_y_3857'] = geom.get('y')
        records.append(row)
    table = pd.DataFrame(records)
    required = ('LocationID', 'TrackingNumber', 'UtilityCompany',
                'WorkLocationFullAddress', 'XCoord', 'YCoord', 'OnSegX', 'OnSegY')
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f'Missing permit fields: {sorted(missing)}')
    for col in ('XCoord', 'YCoord', 'OnSegX', 'OnSegY',
                'ArcGIS_x_3857', 'ArcGIS_y_3857'):
        table[col] = pd.to_numeric(table[col], errors='coerce')
    if table['LocationID'].isna().any() or table['LocationID'].duplicated().any():
        raise ValueError('LocationID values must be non-missing and unique.')
    original_ok = (table[['XCoord', 'YCoord']].notna().all(axis=1)
                   & ~((table['XCoord'] == 0) & (table['YCoord'] == 0)))
    onseg_ok = (table[['OnSegX', 'OnSegY']].notna().all(axis=1)
                & ~((table['OnSegX'] == 0) & (table['OnSegY'] == 0)))
    # No tolerance: do not discard ANY Original / OnSeg discrepancy.
    identical = table['XCoord'].eq(table['OnSegX']) & table['YCoord'].eq(table['OnSegY'])
    table['CoordinateGroup'] = 'different'
    table.loc[original_ok & onseg_ok & identical, 'CoordinateGroup'] = 'identical'
    table.loc[~(original_ok & onseg_ok), 'CoordinateGroup'] = 'missing_coordinate'
    return table


def coordinate_qa(table):
    ok = (table['OnSegX'].notna() & table['OnSegY'].notna()
          & table['ArcGIS_x_3857'].notna() & table['ArcGIS_y_3857'].notna())
    if not ok.any():
        print('OnSeg-to-ArcGIS CRS check: unavailable')
        return
    gx, gy = web_to_local.transform(table.loc[ok, 'ArcGIS_x_3857'].to_numpy(),
                                    table.loc[ok, 'ArcGIS_y_3857'].to_numpy())
    gap = ((table.loc[ok, 'OnSegX'].to_numpy() - gx)**2
           + (table.loc[ok, 'OnSegY'].to_numpy() - gy)**2)**0.5
    print('Maximum OnSeg-to-ArcGIS discrepancy (m):', round(float(gap.max()), 3))


# %% STEP 3 - Read full roadway names and address ranges

ABBREVIATIONS = {
    'STREET': 'ST', 'AVENUE': 'AVE', 'ROAD': 'RD', 'PLACE': 'PL',
    'BOULEVARD': 'BLVD', 'DRIVE': 'DR', 'LANE': 'LN', 'COURT': 'CT',
    'CIRCLE': 'CIR', 'TERRACE': 'TER', 'PARKWAY': 'PKWY',
    'HIGHWAY': 'HWY', 'NORTHWEST': 'NW', 'NORTHEAST': 'NE',
    'SOUTHWEST': 'SW', 'SOUTHEAST': 'SE', 'FIRST': '1ST',
}


def normalize_street(text):
    if not isinstance(text, str):
        return ''
    words = re.sub(r'[^A-Z0-9 ]', ' ', text.upper()).split()
    return ' '.join(ABBREVIATIONS.get(word, word) for word in words)



def route_matches_permit(route_name, permit):
    """Exact regular street matching; cautious special-bridge I-66 alias."""
    candidate = normalize_street(route_name)
    if permit['type'] == 'special' and permit['street'] == 'INTERSTATE 66':
        # DDOT uses 'INTERSTATE 66   BN' in the real Roosevelt Bridge
        # pilot response; preserve the suffix and only match the I-66 corridor.
        return candidate == 'INTERSTATE 66' or candidate.startswith('INTERSTATE 66 ')
    return bool(candidate and candidate == permit['street'])


def parse_address(text, allow_implausible=False):
    """Return a real numeric address range, not only the 'hundred block'.

    '2000 - 2099 BLOCK OF 18TH STREET NW' -> [2000, 2099], '18TH ST NW'
    '2079 - 2099 BLOCK OF 5TH STREET NW' -> [2079, 2099], '5TH ST NW'
    '1703 MINNESOTA AVENUE SE'          -> [1703, 1703], 'MINNESOTA AVE SE'
    'ROOSEVELT BRIDGE'                   -> explicitly named special route
    """
    raw = text.strip().upper() if isinstance(text, str) else ''
    raw = re.sub(r'\s+', ' ', raw)
    if raw in SPECIAL_ROUTE_ALIASES:
        return {'raw': raw, 'type': 'special', 'street': SPECIAL_ROUTE_ALIASES[raw],
                'start': None, 'end': None}
    match = re.fullmatch(r'(\d+)\s*-\s*(\d+)\s+BLOCK\s+OF\s+(.+)', raw)
    if match:
        start, end, street = int(match[1]), int(match[2]), match[3]
    else:
        match = re.fullmatch(r'(\d+)\s+BLOCK\s+OF\s+(.+)', raw)
        if match:
            start = end = int(match[1])
            street = match[2]
        else:
            match = re.fullmatch(r'(\d+)[A-Z]?\s+(.+)', raw)
            if not match:
                return {'raw': raw, 'type': 'unparsed', 'street': '',
                        'start': None, 'end': None}
            start = end = int(match[1])
            street = match[2]
    street = re.split(r'[,;]|\s+BETWEEN\s+', street)[0]
    if start > end or (end - start > 10000 and not allow_implausible):
        return {'raw': raw, 'type': 'unparsed', 'street': '',
                'start': None, 'end': None}
    return {'raw': raw, 'type': 'regular', 'street': normalize_street(street),
            'start': start, 'end': end}


def block_relation(permit, unit_block):
    """Compare the FULL permit and LRS ranges, and detect boundary-only matches.

    Return (relation, start, end). A suspicious LRS range is NOT verified.
    """
    lrs = parse_address(unit_block, allow_implausible=True)
    if lrs['type'] != 'regular':
        return 'unknown', None, None
    a, b = lrs['start'], lrs['end']
    if (b - a) > SUSPICIOUS_LRS_ADDRESS_SPAN:
        return 'suspicious_range', a, b
    if permit['type'] != 'regular':
        return 'unknown', a, b
    if lrs['street'] != permit['street']:
        return 'different_street', a, b
    p, q = permit['start'], permit['end']
    overlap_start, overlap_end = max(a, p), min(b, q)
    if overlap_start > overlap_end:
        return 'no_overlap', a, b
    if overlap_start == overlap_end and p != q:
        return 'boundary_only', a, b
    if (a, b) == (p, q):
        return 'exact', a, b
    if a <= p and q <= b:
        return 'lrs_covers_permit', a, b
    if p <= a and b <= q:
        return 'within_permit', a, b
    return 'partial_overlap', a, b


STRONG_BLOCK_RELATIONS = {'exact', 'lrs_covers_permit', 'within_permit'}
WEAK_BLOCK_RELATIONS = {'partial_overlap', 'boundary_only'}


def covers_entire_range(permit, intervals):
    """Full address-range coverage by the union of validated LRS blocks."""
    if permit['type'] != 'regular' or not intervals:
        return False
    start, end = permit['start'], permit['end']
    cursor = start
    for a, b in sorted(intervals):
        if b < cursor:
            continue
        if a > cursor:
            return False
        cursor = max(cursor, b + 1)
        if cursor > end:
            return True
    return False


# %% STEP 4 - DDOT LRS requests and exact SubBlockKey lookup

def post_json(url, parameters):
    response = session.post(url, data={'f': 'json', **parameters}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and 'error' in payload:
        raise RuntimeError(f'DDOT API error: {payload["error"]}')
    time.sleep(REQUEST_PAUSE_SECONDS)
    return payload


class SQLiteResponseCache:
    """Keep raw DDOT JSON replies across runs without large repeated JSON writes.

    No outside package needed: sqlite3 is part of standard Python.
    Old raw v1/v2 cache files are imported once when creating the database.
    """

    def __init__(self, path, seed_paths):
        fresh = not path.exists()
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('CREATE TABLE IF NOT EXISTS replies ('
                                'request_key TEXT PRIMARY KEY, payload_json TEXT NOT NULL)')
        if fresh:
            for seed_path in seed_paths:
                if not seed_path.exists():
                    continue
                with seed_path.open(encoding='utf-8') as file:
                    raw_replies = json.load(file)
                if not isinstance(raw_replies, dict):
                    raise ValueError(f'Unexpected cache content: {seed_path}')
                self.connection.executemany(
                    'INSERT OR IGNORE INTO replies(request_key, payload_json) VALUES (?, ?)',
                    ((key, json.dumps(value, ensure_ascii=False))
                     for key, value in raw_replies.items()))
                self.connection.commit()
                print(f'Imported {len(raw_replies)} raw API cache entries from {seed_path.name}')

    def __contains__(self, key):
        return self.connection.execute(
            'SELECT 1 FROM replies WHERE request_key = ?', (key,)).fetchone() is not None

    def __getitem__(self, key):
        row = self.connection.execute(
            'SELECT payload_json FROM replies WHERE request_key = ?', (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def __setitem__(self, key, payload):
        self.connection.execute(
            'INSERT OR REPLACE INTO replies(request_key, payload_json) VALUES (?, ?)',
            (key, json.dumps(payload, ensure_ascii=False)))
        self.connection.commit()  # A successful request survives an interrupted run.

    def close(self):
        self.connection.close()


def load_cache():
    sources = ([PREVIOUS_CACHE_PATH, LEGACY_V2_CACHE_PATH]
               if USE_PREVIOUS_RAW_API_CACHE else [])
    return SQLiteResponseCache(CACHE_DB_PATH, sources)


api_cache = None  # Loaded when executing main(); easy to mock in tests.


def cache_request(key, function):
    if key not in api_cache:
        api_cache[key] = function()
    return api_cache[key]


def lrs_query(x, y, radius):
    return post_json(LRS_URL, {
        'x': float(x), 'y': float(y), 'inSR': LOCAL_WKID, 'outSR': LOCAL_WKID,
        'routeType': 'Street', 'includeSectionEvents': 'false',
        'returnNearestMatchOnly': 'false', 'searchRadius': radius,
    })


def parse_lrs_candidates(payload):
    """Use the actual 'pointOnSubBlocks' records, with their own context."""
    items = payload.get('pointOnSubBlocks', []) if isinstance(payload, dict) else []
    if not isinstance(items, list):
        raise ValueError('Unexpected DDOT LRS response: pointOnSubBlocks not a list.')
    found = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = item.get('subBlockKey')
        if not key:
            continue
        key = str(key).strip()
        try:
            distance = float(item.get('offsetInMeters'))
            if not math.isfinite(distance) or distance < 0:
                distance = None
        except (ValueError, TypeError):
            distance = None
        row = {'LRS_SubBlockKey': key, 'LRS_RouteName': item.get('routeName'),
               'LRS_RouteType': item.get('routeType'),
               'LRS_UnitBlock': item.get('unitBlock'),
               'LRS_OffsetMeters': distance}
        prior = found.get(key.lower())
        if prior is None or (distance is not None and
                             (prior['LRS_OffsetMeters'] is None or
                              distance < prior['LRS_OffsetMeters'])):
            found[key.lower()] = row
    return list(found.values())


def official_subblock(key):
    safe = str(key).replace("'", "''")
    payload = post_json(SUBBLOCK_URL, {
        'where': f"SUBBLOCKKEY = '{safe}'",
        'outFields': 'SUBBLOCKKEY,ROUTENAME,ROADTYPE,FHWAFUNCTIONALCLASS',
        'returnGeometry': 'false',
    })
    if payload.get('exceededTransferLimit'):
        raise ValueError('Roadway SubBlock query exceeded transfer limit.')
    features = payload.get('features', [])
    if not isinstance(features, list):
        raise ValueError('Unexpected Roadway SubBlock response: features missing.')
    return [x.get('attributes', {}) for x in features]


def official_by_key(key):
    # Matches the existing Step 06 raw cache format.
    cache_key = 'roadway|' + str(key).lower()
    rows = cache_request(cache_key, lambda: official_subblock(key))
    exact = [row for row in rows
             if str(row.get('SUBBLOCKKEY', '')).lower() == str(key).lower()]
    return exact[0] if len(exact) == 1 else None


def lrs_candidates_for_point(rec, source, permit):
    """Progressive LRS search; NEVER stop for street-only agreement.

    The wider search is required when there is no address+block candidate,
    OR a long permit range is not fully covered by returned blocks.
    """
    if source == 'OnSeg':
        x, y = rec['OnSegX'], rec['OnSegY']
    else:
        x, y = rec['XCoord'], rec['YCoord']
    found = {}
    for radius in LRS_SEARCH_RADII_METERS:
        cache_key = f'lrs|{float(x):.7f}|{float(y):.7f}|{radius}'
        payload = cache_request(cache_key, lambda: lrs_query(x, y, radius))
        for candidate in parse_lrs_candidates(payload):
            candidate['CoordinateSource'] = source
            candidate['LRS_SearchRadius_m'] = radius
            key = candidate['LRS_SubBlockKey'].lower()
            if key not in found:
                found[key] = candidate
            elif (candidate['LRS_OffsetMeters'] is not None and
                  (found[key]['LRS_OffsetMeters'] is None or
                   candidate['LRS_OffsetMeters'] < found[key]['LRS_OffsetMeters'])):
                found[key] = candidate
        if permit['type'] == 'regular':
            relevant = [c for c in found.values()
                        if route_matches_permit(c['LRS_RouteName'], permit)
                        and block_relation(permit, c['LRS_UnitBlock'])[0]
                        in STRONG_BLOCK_RELATIONS]
            ranges = [(block_relation(permit, c['LRS_UnitBlock'])[1],
                       block_relation(permit, c['LRS_UnitBlock'])[2])
                      for c in relevant]
            # Validate official roadway and class before considering search complete.
            complete = (covers_entire_range(permit, ranges)
                        and bool(relevant)
                        and any(valid_official_address_match(c, permit) for c in relevant))
        else:
            # A bridge without an address block needs every possible search
            # result before an explicit, provisional special-address decision.
            complete = False
        if complete:
            break
    return list(found.values())


def valid_official_address_match(lrs, permit):
    """Only STREET and a valid official FHWA code can pass this check."""
    official = official_by_key(lrs['LRS_SubBlockKey'])
    if official is None:
        return False
    clazz = official.get('FHWAFUNCTIONALCLASS')
    try:
        valid_class = float(clazz) in (1, 2, 3, 4, 5, 6, 7)
    except (TypeError, ValueError):
        valid_class = False
    return (str(official.get('ROADTYPE', '')).strip() == '1'
            and route_matches_permit(official.get('ROUTENAME'), permit)
            and route_matches_permit(lrs.get('LRS_RouteName'), permit)
            and valid_class)


# %% STEP 5 - Evaluate all roadway candidates and decide review status

def as_int_class(value):
    try:
        number = float(value)
        return int(number) if number in (1, 2, 3, 4, 5, 6, 7) else None
    except (TypeError, ValueError):
        return None


def classify_location(rec):
    permit = parse_address(rec.get('WorkLocationFullAddress'))
    result = {
        'LocationID': rec['LocationID'],
        'TrackingNumber': rec.get('TrackingNumber'),
        'PermitNumber': rec.get('PermitNumber'),
        'UtilityCompany': rec.get('UtilityCompany'),
        'WorkLocationFullAddress': rec.get('WorkLocationFullAddress'),
        'CoordinateGroup': rec['CoordinateGroup'],
        'ParsedStreet': permit['street'],
        'PermitAddressStart': permit['start'],
        'PermitAddressEnd': permit['end'],
        'FunctionalClass': None,
        'ClassStatus': 'UNRESOLVED',
        'MatchStatus': 'UNRESOLVED',
        'SelectedSubBlockKey': None,
        'MatchedSubBlockKeys': '',
        'SubBlockKeyStatus': 'UNRESOLVED',
        'NearestRoadway_OnSeg': '',
        'NearestRoadway_Original': '',
        'ClassReviewNeeded': True,
        'SubBlockReviewNeeded': True,
        'NeedsManualReview': True,
        'Explanation': '',
        'OnSeg_ValidCandidates': 0,
        'Original_ValidCandidates': 0,
    }
    if rec['CoordinateGroup'] == 'missing_coordinate':
        result.update(MatchStatus='MISSING_COORDINATES',
                      Explanation='Original and/or OnSeg coordinate is missing or invalid.')
        return result, []

    sources = ['OnSeg'] if rec['CoordinateGroup'] == 'identical' else ['OnSeg', 'Original']
    by_source = {side: lrs_candidates_for_point(rec, side, permit) for side in sources}
    if not any(by_source.values()):
        result.update(MatchStatus='NO_LRS_CANDIDATES',
                      Explanation='No LRS SubBlockKey found within service search windows.')
        return result, []

    audits = []
    anchors = {}
    for side in sources:
        distances = [c['LRS_OffsetMeters'] for c in by_source[side]
                     if c['LRS_OffsetMeters'] is not None]
        minimum = min(distances) if distances else None
        anchors[side] = []
        for c in by_source[side]:
            key = c['LRS_SubBlockKey']
            official = official_by_key(key)
            official = official or {}
            relation, a, b = block_relation(permit, c['LRS_UnitBlock'])
            official_name = normalize_street(official.get('ROUTENAME'))
            lrs_name = normalize_street(c['LRS_RouteName'])
            street_ok = bool(permit['street'] and
                             route_matches_permit(official_name, permit) and
                             route_matches_permit(lrs_name, permit))
            street_type_ok = (str(official.get('ROADTYPE', '')).strip() == '1'
                              and str(c.get('LRS_RouteType', '')).lower() == 'street')
            clazz = as_int_class(official.get('FHWAFUNCTIONALCLASS'))
            base_ok = bool(street_ok and street_type_ok and clazz is not None)
            is_nearest = (minimum is not None and c['LRS_OffsetMeters'] is not None
                          and c['LRS_OffsetMeters'] == minimum)
            block_strong = relation in STRONG_BLOCK_RELATIONS
            # Do not auto accept partial/boundary-only matches or bad LRS ranges.
            address_ok = base_ok and (block_strong if permit['type'] == 'regular'
                                      else permit['type'] == 'special')
            audit = {
                'LocationID': rec['LocationID'], 'CoordinateSource': side,
                'LRS_SearchRadius_m': c['LRS_SearchRadius_m'],
                'LRS_SubBlockKey': key, 'LRS_RouteName': c['LRS_RouteName'],
                'LRS_UnitBlock': c['LRS_UnitBlock'],
                'LRS_OffsetMeters': c['LRS_OffsetMeters'],
                'LRS_NearestForPoint': is_nearest,
                'OfficialRouteName': official.get('ROUTENAME'),
                'OfficialRoadType': official.get('ROADTYPE'),
                'OfficialFHWAClass': clazz,
                'StreetMatchesPermit': street_ok,
                'BlockRelation': relation,
                'LRSBlockStart': a, 'LRSBlockEnd': b,
                'OfficialStreetAndClassValid': base_ok,
                'EligibleAddressBlockClass': address_ok,
                'SpatiallyAnchored': bool(is_nearest and address_ok),
            }
            audits.append(audit)
            if is_nearest:
                anchors[side].append(audit)
        if anchors[side]:
            result[f'NearestRoadway_{side}'] = ' | '.join(sorted({
                str(a['OfficialRouteName']) for a in anchors[side]}))

    # Important: matching a street name alone never confirms an address.
    if permit['type'] == 'unparsed':
        result.update(MatchStatus='UNPARSEABLE_ADDRESS',
                      Explanation='Special or unsupported address needs external location verification.')
        return result, audits

    eligible = [a for a in audits if a['EligibleAddressBlockClass']]
    # Deduplicate repeated key from different coordinate sources.
    by_key = {}
    for a in eligible:
        key = str(a['LRS_SubBlockKey']).lower()
        entry = by_key.setdefault(key, {
            'key': a['LRS_SubBlockKey'], 'class': a['OfficialFHWAClass'],
            'street': a['OfficialRouteName'], 'sides': set(), 'intervals': []})
        entry['sides'].add(a['CoordinateSource'])
        if a['LRSBlockStart'] is not None:
            entry['intervals'].append((a['LRSBlockStart'], a['LRSBlockEnd']))
    valid_keys = list(by_key.values())
    result['MatchedSubBlockKeys'] = '; '.join(sorted(v['key'] for v in valid_keys))
    for side in sources:
        result[f'{side}_ValidCandidates'] = len({a['LRS_SubBlockKey'] for a in eligible
                                                 if a['CoordinateSource'] == side})
    if not valid_keys:
        result.update(MatchStatus='NO_VALID_ADDRESS_BLOCK_MATCH',
                      Explanation='No official street/class candidate agrees with the permit address and block.')
        return result, audits

    classes = {v['class'] for v in valid_keys}
    if len(classes) > 1:
        result.update(MatchStatus='CONFLICTING_CLASSES',
                      Explanation='Eligible address/block candidates have different official FHWA classes.')
        return result, audits

    # One consistent class, independently of how many SubBlockKeys exist.
    result['FunctionalClass'] = classes.pop()
    # These are CANDIDATE keys until all verification gates are passed.
    result['SubBlockKeyStatus'] = ('UNIQUE_CANDIDATE' if len(valid_keys) == 1
                                   else 'MULTIPLE_CANDIDATE_KEYS')

    if permit['type'] == 'special':
        result.update(ClassStatus='PROVISIONAL', MatchStatus='SPECIAL_BRIDGE_REVIEW',
                      Explanation=('Recognized bridge route alias; official class obtained, '
                                   'but exact bridge geometry/roadway identity needs review.'))
        return result, audits

    # A full permit-address range must be supported by the candidate LRS blocks.
    intervals = [pair for v in valid_keys for pair in v['intervals']]
    if not covers_entire_range(permit, intervals):
        result.update(ClassStatus='PROVISIONAL', MatchStatus='INCOMPLETE_BLOCK_COVERAGE',
                      Explanation=('Some road candidates agree with the address, but they '
                                   'do not cover its entire stated range.'))
        return result, audits

    # Closest LRS roadway to EACH distinct coordinate must also match the
    # official permit street, full address block, and class. This RELATIVE
    # nearest-road check avoids arbitrary distance-to-road acceptance cutoffs.
    anchored_sides = []
    for side in sources:
        nearest = anchors[side]
        if not nearest or not any(a['SpatiallyAnchored'] for a in nearest):
            result.update(ClassStatus='PROVISIONAL', MatchStatus='SPATIAL_DISAGREEMENT_REVIEW',
                          Explanation=(f'{side} nearest reported LRS roadway does not '
                                       'confirm the permit street/block/class.'))
            return result, audits
        # At an exact-distance tie between intersecting streets, the location
        # may be ambiguous even though one of the tied roads matches the address.
        if any(not a['EligibleAddressBlockClass'] for a in nearest):
            result.update(ClassStatus='PROVISIONAL', MatchStatus='NEAREST_ROAD_TIE_REVIEW',
                          Explanation=f'{side} has equally near, non-matching street/block candidates.')
            return result, audits
        if any(a['OfficialFHWAClass'] != result['FunctionalClass']
               for a in nearest if a['EligibleAddressBlockClass']):
            result.update(MatchStatus='SPATIAL_CLASS_CONFLICT',
                          Explanation=f'Nearest {side} candidate has conflicting class.')
            result['FunctionalClass'] = None
            return result, audits
        anchored_sides.append(side)

    # No unresolved competing boundary/partial block with a DIFFERENT class.
    weak_competitors = [a for a in audits
                        if a['OfficialStreetAndClassValid'] and
                        a['BlockRelation'] in WEAK_BLOCK_RELATIONS and
                        a['OfficialFHWAClass'] != result['FunctionalClass']]
    if weak_competitors:
        result.update(ClassStatus='PROVISIONAL', MatchStatus='BOUNDARY_CLASS_CONFLICT',
                      Explanation='Boundary/partial block candidate has a different class.')
        return result, audits

    result.update(ClassStatus='VERIFIED', ClassReviewNeeded=False,
                  NeedsManualReview=False,
                  MatchStatus=('VERIFIED_UNIQUE_KEY' if len(valid_keys) == 1
                               else 'VERIFIED_MULTIPLE_KEYS'),
                  Explanation=('Street/quadrant, full address range, nearest-point LRS '
                               'roadway and official DDOT FHWA class agree.'))
    if len(valid_keys) == 1:
        result['SelectedSubBlockKey'] = valid_keys[0]['key']
        result['SubBlockKeyStatus'] = 'VERIFIED_UNIQUE'
        result['SubBlockReviewNeeded'] = False
    else:
        result['SubBlockKeyStatus'] = 'VERIFIED_MULTIPLE'
        result['SubBlockReviewNeeded'] = True
    return result, audits


# %% STEP 6 - Run pilot/full, resume safely, write auditable CSVs

def choose_records(df):
    """Select a reproducible pilot, larger balanced sample, or the entire input.

    The extended sample always contains the original three pilot locations in
    each utility/coordinate group plus randomly chosen additional locations.
    """
    if RUN_MODE == 'full':
        return df.sort_values('LocationID').reset_index(drop=True)
    pieces = []
    ordered = df.sort_values('LocationID')
    for _, group in ordered.groupby(['UtilityCompany', 'CoordinateGroup'], sort=True):
        first = group.head(PILOT_PER_COMPANY_PER_GROUP)
        if RUN_MODE == 'pilot':
            pieces.append(first)
        else:
            extra_count = max(0, EXTENDED_PER_COMPANY_PER_GROUP - len(first))
            remaining = group.iloc[len(first):]
            extra = remaining.sample(n=min(extra_count, len(remaining)),
                                     random_state=RANDOM_SEED)
            pieces.extend([first, extra])
    return pd.concat(pieces, ignore_index=True).sort_values('LocationID').reset_index(drop=True)


def write_csv_atomically(frame, destination):
    """Prevent a partially written CSV if execution is interrupted mid-save."""
    temporary = destination.with_name(destination.name + '.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)  # Close Excel before running on Windows.


def main():
    global api_cache
    if RUN_MODE not in ('pilot', 'extended', 'full'):
        raise ValueError("RUN_MODE must be 'pilot', 'extended', or 'full'.")
    df = read_permit_records(INPUT)
    print('\n--- PERMIT COORDINATE GROUPS ---')
    print(df['CoordinateGroup'].value_counts().to_string())
    print('Total permit locations:', len(df))
    coordinate_qa(df)
    work = choose_records(df)
    print(f'RUN_MODE: {RUN_MODE} | locations selected: {len(work)}')
    print('Selection by utility and coordinate group:')
    print(work.groupby(['UtilityCompany', 'CoordinateGroup']).size().to_string())
    api_cache = load_cache()

    results_path = OUT / f'permit_functional_class_lrs_{RUN_MODE}.csv'
    candidate_path = OUT / f'lrs_roadway_candidates_{RUN_MODE}.csv'
    review_path = OUT / f'lrs_records_needing_class_review_{RUN_MODE}.csv'
    errors_path = OUT / f'lrs_api_errors_{RUN_MODE}.csv'
    rows = []
    candidate_rows = []
    if results_path.exists():
        rows = pd.read_csv(results_path, dtype={'LocationID': 'string'}).to_dict('records')
    if candidate_path.exists():
        candidate_rows = pd.read_csv(candidate_path, dtype={'LocationID': 'string'}).to_dict('records')
    done = {str(row['LocationID']) for row in rows
            if row.get('MatchStatus') != 'API_ERROR'}
    print('Already completed from previous run:', len(done))

    def save():
        write_csv_atomically(pd.DataFrame(rows), results_path)
        write_csv_atomically(pd.DataFrame(candidate_rows), candidate_path)
        # Raw API results are already saved individually in SQLite.

    processed = 0
    errors_in_a_row = 0
    try:
        for _, rec in work.iterrows():
            lid = str(rec['LocationID'])
            if lid in done:
                continue
            try:
                row, candidates = classify_location(rec)
                errors_in_a_row = 0
            except (requests.RequestException, ValueError, TypeError, KeyError,
                    RuntimeError, json.JSONDecodeError) as exc:
                row = {'LocationID': rec['LocationID'],
                       'TrackingNumber': rec.get('TrackingNumber'),
                       'UtilityCompany': rec.get('UtilityCompany'),
                       'WorkLocationFullAddress': rec.get('WorkLocationFullAddress'),
                       'CoordinateGroup': rec['CoordinateGroup'],
                       'FunctionalClass': None, 'ClassStatus': 'UNRESOLVED',
                       'MatchStatus': 'API_ERROR', 'ClassReviewNeeded': True,
                       'NeedsManualReview': True, 'Explanation': repr(exc)[:500]}
                candidates = []
                errors_in_a_row += 1
                print(f'API/record error at LocationID {lid}: {row["Explanation"]}')
            rows = [r for r in rows if str(r['LocationID']) != lid]
            candidate_rows = [c for c in candidate_rows if str(c['LocationID']) != lid]
            rows.append(row)
            candidate_rows.extend(candidates)
            processed += 1
            if processed % SAVE_EVERY == 0:
                save()
                print(f'Processed new locations: {processed} | saved: {len(rows)} / {len(work)}')
            if errors_in_a_row >= MAX_CONSECUTIVE_API_ERRORS:
                print('STOPPED: several consecutive errors. Check the DDOT API/network, '
                      'then rerun this script to resume.')
                break
        save()
    finally:
        api_cache.close()

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        review = result_df[result_df['ClassReviewNeeded'].astype(str).str.lower() == 'true']
        write_csv_atomically(review, review_path)
        write_csv_atomically(result_df[result_df['MatchStatus'] == 'API_ERROR'], errors_path)
        print('\n--- DDOT LRS v2 MATCHING SUMMARY ---')
        print('Records saved:', len(result_df), '/', len(work))
        print(result_df['MatchStatus'].value_counts().to_string())
        print('VERIFIED class:', int((result_df['ClassStatus'] == 'VERIFIED').sum()))
        print('PROVISIONAL class:', int((result_df['ClassStatus'] == 'PROVISIONAL').sum()))
        print('NO class:', int(result_df['FunctionalClass'].isna().sum()))
        print('Class review required:', len(review))
        print('Results:', results_path)
        print('All candidate evidence:', candidate_path)
        print('Class review list:', review_path)
        if len(result_df) != len(work):
            print('INCOMPLETE RUN: rerun with the SAME mode to resume remaining locations.')
        else:
            print('COMPLETE RUN: all selected locations have an output row.')
    print('\nOnly ClassStatus=VERIFIED is a confirmed Functional Class.')


if __name__ == '__main__':
    main()
