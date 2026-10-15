# %%
# Purpose:
# Use the validated permit locations from Step 04 and the coordinate
# assessment from Step 05 to identify the correct roadway for each location.
# Query the DDOT LRS service using Original and/or OnSeg coordinates
# to retrieve nearby roadway candidates (75 m, expanding to 200 m if needed).
# Compare candidate street names, address blocks, and nearest-road distances
# with the original TOPS work-location address.
# Retrieve official FHWA Functional Class values using SubBlockKey and
# exclude alleys and invalid roadway matches.
# Assign VERIFIED, PROVISIONAL, or UNRESOLVED status based on the evidence.
# Save functional class results, candidate evidence, and review cases
# for permit-level aggregation in Step 07.

# STEP 1 - Settings and libraries
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

RUN_MODE = 'pilot'  # 'pilot' or 'full'
PILOT_PER_COMPANY_PER_GROUP = 3

LRS_SEARCH_RADII_METERS = (75, 200)  # Candidate search windows, NOT match thresholds.
REQUEST_PAUSE_SECONDS = 0.10
SAVE_EVERY = 50
MAX_CONSECUTIVE_API_ERRORS = 5
IMPORT_EXISTING_RAW_API_CACHE = True  # Only raw API replies, never old match results.

ROOT = Path(__file__).resolve().parent.parent if '__file__' in globals() else Path.cwd()
INPUT = ROOT / 'data' / '3_processed' / 'permits_2026_applied_or_issued_staging_crane_three_utilities.json'
BASE_OUT = ROOT / 'outputs' / '3_roadway_functional_class'
OUT = BASE_OUT / RUN_MODE
CACHE_DB_PATH = BASE_OUT / 'ddot_lrs_raw_api_cache.sqlite3'

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

# A recognized bridge route alias does not prove the permit's exact bridge segment.
SPECIAL_ROUTE_ALIASES = {'ROOSEVELT BRIDGE': 'INTERSTATE 66',
                         'THEODORE ROOSEVELT BRIDGE': 'INTERSTATE 66'}
SUSPICIOUS_LRS_ADDRESS_SPAN = 1000  # Flag implausible address ranges for review.


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


CAPITOL_STREETS = {'EAST CAPITOL ST', 'NORTH CAPITOL ST', 'SOUTH CAPITOL ST'}


def canonical_road_name(name):
    """Normalize DDOT route labels without globally discarding direction labels.

    DDOT labels EAST/NORTH/SOUTH CAPITOL ST 'BN' (both directions); TOPS often
    calls the SAME route EAST/NORTH/SOUTH CAPITOL STREET. Only these exact three
    route labels are allowed this alias. E.g. 'INTERSTATE 66 BN' is NOT stripped.
    """
    name = normalize_street(name)
    if name.endswith(' BN') and name[:-3] in CAPITOL_STREETS:
        return name[:-3]
    return name


def route_matches_permit(route_name, permit):
    """Conservative road-name matching, including two named intersection streets."""
    candidate = canonical_road_name(route_name)
    if permit['type'] == 'special' and permit['street'] == 'INTERSTATE 66':
        return candidate == 'INTERSTATE 66' or candidate.startswith('INTERSTATE 66 ')
    if permit['type'] == 'intersection':
        return candidate in permit['streets']
    return bool(candidate and candidate == permit['street'])


def parse_intersection_address(text):
    """Parse two plain named streets only; never mistake ramps/exits for crossings.

    Examples: 'HOLBROOK STREET NE AND MORSE STREET NE',
              'C STREET SW AND SOUTH CAPITOL STREET'.
    Exact matched names are later cross-checked against DDOT LRS and layer 162.
    """
    raw = re.sub(r'\s+', ' ', text.strip().upper()) if isinstance(text, str) else ''
    if raw.count(' AND ') != 1:
        return None
    left, right = (part.strip() for part in raw.split(' AND ', 1))
    # Ramps, directional movements, exits and 'from/to' are NOT street crossings.
    prohibited = {'RAMP', 'EXIT', 'NORTHBOUND', 'SOUTHBOUND',
                  'EASTBOUND', 'WESTBOUND', 'FROM', 'TO', 'INTERSTATE'}
    if not left or not right or any(word in prohibited for word in
                                  set(re.sub(r'[^A-Z0-9 ]', ' ', raw).split())):
        return None
    a, b = canonical_road_name(left), canonical_road_name(right)
    # Allow ordinal road names (1ST ST), reject house-number-like prefixes.
    road_ending = re.compile(r'\b(ST|AVE|RD|PL|BLVD|DR|LN|CT|CIR|TER|PKWY|HWY)\b')
    if (not a or not b or a == b or
            not road_ending.search(a) or not road_ending.search(b)):
        return None
    return {'raw': raw, 'type': 'intersection', 'street': a + ' & ' + b,
            'streets': (a, b), 'start': None, 'end': None}


def parse_address(text, allow_implausible=False):
    """Return a real numeric address range, not only the 'hundred block'.

    '2000 - 2099 BLOCK OF 18TH STREET NW' -> [2000, 2099], '18TH ST NW'
    '2079 - 2099 BLOCK OF 5TH STREET NW' -> [2079, 2099], '5TH ST NW'
    '1703 MINNESOTA AVENUE SE'          -> [1703, 1703], 'MINNESOTA AVE SE'
    'ROOSEVELT BRIDGE'                   -> explicitly named special route
    """
    raw = text.strip().upper() if isinstance(text, str) else ''
    raw = re.sub(r'\s+', ' ', raw)
    intersection = parse_intersection_address(raw)
    if intersection is not None:
        return intersection
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
    if canonical_road_name(lrs['street']) != canonical_road_name(permit['street']):
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


# %% STEP 4 - DDOT LRS and official roadway data

def post_json(url, parameters):
    response = session.post(url, data={'f': 'json', **parameters}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and 'error' in payload:
        raise RuntimeError(f'DDOT API error: {payload["error"]}')
    time.sleep(REQUEST_PAUSE_SECONDS)
    return payload


class SQLiteResponseCache:
    """Store raw DDOT API responses for reliable, resumable full-data runs.

    On a new run, optionally import compatible raw responses from other output
    folders. Never import previous location matches, classifications or CSVs.
    """

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        new_database = not path.exists()
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('CREATE TABLE IF NOT EXISTS replies ('
                                'request_key TEXT PRIMARY KEY, payload_json TEXT NOT NULL)')
        self.connection.commit()
        if new_database and IMPORT_EXISTING_RAW_API_CACHE:
            self.import_existing(path)

    def import_existing(self, destination):
        folder = ROOT / 'outputs'
        if not folder.exists():
            return
        sources = sorted((p for p in folder.rglob('*.sqlite3')
                          if 'cache' in p.name.lower() and p.resolve() != destination.resolve()),
                         key=lambda p: p.stat().st_size, reverse=True)
        imported = 0
        for source in sources:
            try:
                with sqlite3.connect(source) as old:
                    table = old.execute("SELECT name FROM sqlite_master "
                                        "WHERE type='table' AND name='replies'").fetchone()
                    if not table:
                        continue
                    cursor = old.execute('SELECT request_key, payload_json FROM replies')
                    while True:
                        batch = cursor.fetchmany(500)
                        if not batch:
                            break
                        self.connection.executemany(
                            'INSERT OR IGNORE INTO replies VALUES (?, ?)', batch)
                    self.connection.commit()
                imported += 1
            except (OSError, sqlite3.Error) as exc:
                print(f'Raw API cache import skipped for {source.name}: {exc}')
        if imported:
            count = self.connection.execute('SELECT COUNT(*) FROM replies').fetchone()[0]
            print(f'Reused {count:,} raw DDOT API responses from {imported} existing cache(s).')

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
        self.connection.commit()  # Survives a canceled run.

    def close(self):
        self.connection.close()


def cache_request(key, function):
    if key not in api_cache:
        api_cache[key] = function()
    return api_cache[key]


api_cache = None


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
            and canonical_road_name(official.get('ROUTENAME')) ==
                canonical_road_name(lrs.get('LRS_RouteName'))
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
        'AddressType': permit['type'],
        'IntersectionStreet1': (permit['streets'][0] if permit['type'] == 'intersection' else ''),
        'IntersectionStreet2': (permit['streets'][1] if permit['type'] == 'intersection' else ''),
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
                             route_matches_permit(lrs_name, permit) and
                             canonical_road_name(official_name) == canonical_road_name(lrs_name))
            street_type_ok = (str(official.get('ROADTYPE', '')).strip() == '1'
                              and str(c.get('LRS_RouteType', '')).lower() == 'street')
            clazz = as_int_class(official.get('FHWAFUNCTIONALCLASS'))
            base_ok = bool(street_ok and street_type_ok and clazz is not None)
            is_nearest = (minimum is not None and c['LRS_OffsetMeters'] is not None
                          and c['LRS_OffsetMeters'] == minimum)
            block_strong = relation in STRONG_BLOCK_RELATIONS
            # Do not auto accept partial/boundary-only matches or bad LRS ranges.
            address_ok = base_ok and (block_strong if permit['type'] == 'regular'
                                      else permit['type'] in ('special', 'intersection'))
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
                'AddressType': permit['type'],
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

    if permit['type'] == 'intersection':
        return classify_intersection(result, audits, sources, permit)

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

    # Do not auto-accept a conflicting nearby block with another class.
    weak_competitors = [a for a in audits
                        if a['OfficialStreetAndClassValid'] and
                        a['BlockRelation'] in WEAK_BLOCK_RELATIONS and
                        a['OfficialFHWAClass'] != result['FunctionalClass']]
    if weak_competitors:
        result.update(ClassStatus='PROVISIONAL', MatchStatus='BOUNDARY_CLASS_CONFLICT',
                      Explanation='Boundary/partial block candidate has a different class.')
        return result, audits

    result.update(ClassStatus='VERIFIED', ClassReviewNeeded=False,
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


def classify_intersection(result, audits, sources, permit):
    """Classify a 2-road intersection only on demonstrable LRS road agreement.

    A complete intersection verification requires BOTH named roads to be among
    the nearest LRS matches for each coordinate source, no unrelated tied
    closest road, and the SAME official class across all nearest matches.
    This avoids assigning the class of whichever street happens to be nearest.
    If both roads are seen but are not both nearest, a unanimous class is only
    PROVISIONAL because the exact crossing is not established.
    """
    named = set(permit['streets'])
    eligible = [a for a in audits if a['EligibleAddressBlockClass']]
    result['MatchedSubBlockKeys'] = '; '.join(sorted({
        str(a['LRS_SubBlockKey']) for a in eligible
    }))
    for side in sources:
        result[f'{side}_ValidCandidates'] = len({
            a['LRS_SubBlockKey'] for a in eligible if a['CoordinateSource'] == side
        })

    # Consider the closest valid SubBlock of EACH named road for each point,
    # without a distance cutoff. Preserve ALL alternatives in the audit CSV.
    nearest_named = []
    roads_found_each_side = True
    verified_geometry = True
    for side in sources:
        side_rows = [a for a in audits if a['CoordinateSource'] == side]
        finite = [a['LRS_OffsetMeters'] for a in side_rows
                  if a['LRS_OffsetMeters'] is not None]
        nearest_all = [a for a in side_rows if finite and
                       a['LRS_OffsetMeters'] == min(finite)]
        nearest_routes = {
            canonical_road_name(a['OfficialRouteName'])
            for a in nearest_all if a['EligibleAddressBlockClass']
        }
        if (nearest_routes != named or
                any(not a['EligibleAddressBlockClass'] for a in nearest_all)):
            verified_geometry = False

        for street in permit['streets']:
            matching = [a for a in side_rows if a['EligibleAddressBlockClass']
                        and canonical_road_name(a['OfficialRouteName']) == street
                        and a['LRS_OffsetMeters'] is not None]
            if not matching:
                roads_found_each_side = False
                continue
            nearest_street_offset = min(a['LRS_OffsetMeters'] for a in matching)
            nearest_named.extend(a for a in matching
                                 if a['LRS_OffsetMeters'] == nearest_street_offset)

    if not roads_found_each_side:
        result.update(MatchStatus='INTERSECTION_ROAD_MISSING_REVIEW',
                      Explanation='LRS did not return an official-class road for both named streets at every point.')
        return result, audits

    class_set = {a['OfficialFHWAClass'] for a in nearest_named}
    if len(class_set) != 1:
        result.update(MatchStatus='INTERSECTION_CLASS_CONFLICT_REVIEW',
                      Explanation='The two named intersection roads have different official FHWA classes.')
        return result, audits

    result['FunctionalClass'] = class_set.pop()
    matched_near_keys = sorted({str(a['LRS_SubBlockKey']) for a in nearest_named})
    result['MatchedSubBlockKeys'] = '; '.join(matched_near_keys)
    result['SubBlockKeyStatus'] = ('UNIQUE_CANDIDATE' if len(matched_near_keys) == 1
                                   else 'MULTIPLE_CANDIDATE_KEYS')
    if not verified_geometry:
        result.update(ClassStatus='PROVISIONAL',
                      MatchStatus='INTERSECTION_CLASS_CONSENSUS_REVIEW',
                      Explanation='Both named roads have the same class, but they are not both the nearest LRS roads at every point.')
        return result, audits

    result.update(ClassStatus='VERIFIED', ClassReviewNeeded=False,
                  MatchStatus='VERIFIED_INTERSECTION_SAME_CLASS',
                  Explanation='Both named intersecting roads are nearest LRS matches and have the same official FHWA class.')
    result['SubBlockKeyStatus'] = 'VERIFIED_MULTIPLE'
    result['SubBlockReviewNeeded'] = True  # Both streets have distinct subblocks.
    return result, audits


# %% STEP 6 - Process all permit locations and save final outputs

REVIEW_ACTIONS = {
    'SPATIAL_DISAGREEMENT_REVIEW': 'Compare TOPS coordinates, permit location details, and official roadway geometry.',
    'NO_VALID_ADDRESS_BLOCK_MATCH': 'Inspect address spelling, LRS block range, and relevant roadway candidates.',
    'INTERSECTION_CLASS_CONFLICT_REVIEW': 'Review which named intersection street contains the permitted work area.',
    'INTERSECTION_CLASS_CONSENSUS_REVIEW': 'Check the exact intersection position; class agrees but geometry is ambiguous.',
    'BOUNDARY_CLASS_CONFLICT': 'Inspect overlapping subblocks and functional class changes at the block boundary.',
    'CONFLICTING_CLASSES': 'Identify the relevant subblock using permit details and roadway geometry.',
    'UNPARSEABLE_ADDRESS': 'Inspect special ramp, exit, or unsupported address format manually.',
    'SPECIAL_BRIDGE_REVIEW': 'Verify bridge location and roadway geometry before confirming class.',
    'NEAREST_ROAD_TIE_REVIEW': 'Review equally close roadway candidates and exact permit position.',
    'INTERSECTION_ROAD_MISSING_REVIEW': 'Check whether both named roads occur in official LRS candidates.',
    'INCOMPLETE_BLOCK_COVERAGE': 'Check whether confirmed subblocks cover the complete permit address range.',
    'SPATIAL_CLASS_CONFLICT': 'Check nearest-road evidence and conflicting official classes.',
    'NO_LRS_CANDIDATES': 'Check coordinate validity and available DDOT roadway coverage.',
    'MISSING_COORDINATES': 'Check or recover location coordinates in the source TOPS record.',
    'API_ERROR': 'Retry the DDOT API request after checking service availability.',
}

RESULT_PATH = OUT / 'permit_locations_fhwa_functional_class.csv'
CANDIDATE_PATH = OUT / 'permit_locations_lrs_roadway_candidate_evidence.csv'
REVIEW_PATH = OUT / 'permit_locations_functional_class_needing_review.csv'
ERROR_PATH = OUT / 'permit_locations_api_errors.csv'
SUMMARY_PATH = OUT / 'roadway_matching_status_summary.csv'


def save_csv(frame, path):
    """Atomically replace a CSV so interruption does not corrupt saved work."""
    temporary = path.with_name(path.name + '.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)  # Close open Excel files before running on Windows.


def main():
    global api_cache
    OUT.mkdir(parents=True, exist_ok=True)
    df = read_permit_records(INPUT)

    if RUN_MODE not in ('pilot', 'full'):
        raise ValueError("RUN_MODE must be 'pilot' or 'full'.")

    if RUN_MODE == 'pilot':
        work = (
            df.sort_values('LocationID')
            .groupby(['UtilityCompany', 'CoordinateGroup'], group_keys=False)
            .head(PILOT_PER_COMPANY_PER_GROUP)
            .reset_index(drop=True)
        )
    else:
        work = df.sort_values('LocationID').reset_index(drop=True)


    print('--- DDOT FUNCTIONAL CLASS MATCHING ---')
    print('Total permit locations:', len(work))
    print('Unique permit applications:', work['TrackingNumber'].nunique())
    print('Coordinate groups:')
    print(work['CoordinateGroup'].value_counts().to_string())
    coordinate_qa(work)

    # Checkpoints in THIS final output folder only; no old class decisions reused.
    results = (pd.read_csv(RESULT_PATH, low_memory=False,
                           dtype={'LocationID': 'string', 'TrackingNumber': 'string'})
               .to_dict('records') if RESULT_PATH.exists() else [])
    candidates = (pd.read_csv(CANDIDATE_PATH, low_memory=False,
                              dtype={'LocationID': 'string'})
                  .to_dict('records') if CANDIDATE_PATH.exists() else [])
    input_ids = set(work['LocationID'].astype(str))
    if not {str(r['LocationID']) for r in results}.issubset(input_ids):
        raise ValueError('Saved results refer to another input dataset. Move the output folder '
                         'before starting a new dataset.')
    done = {str(r['LocationID']) for r in results if r.get('MatchStatus') != 'API_ERROR'}
    print('Already completed:', len(done))
    api_cache = SQLiteResponseCache(CACHE_DB_PATH)

    def save_checkpoint():
        save_csv(pd.DataFrame(results), RESULT_PATH)
        evidence_frame = pd.DataFrame(candidates) if candidates else pd.DataFrame(
            columns=['LocationID', 'CoordinateSource', 'LRS_SubBlockKey',
                     'OfficialFHWAClass', 'EligibleAddressBlockClass'])
        save_csv(evidence_frame, CANDIDATE_PATH)

    processed = 0
    errors_in_a_row = 0
    try:
        for _, rec in work.iterrows():
            lid = str(rec['LocationID'])
            if lid in done:
                continue
            try:
                outcome, evidence = classify_location(rec)
                errors_in_a_row = 0
            except (requests.RequestException, ValueError, TypeError, KeyError,
                    RuntimeError, json.JSONDecodeError) as exc:
                outcome = {'LocationID': rec['LocationID'],
                           'TrackingNumber': rec.get('TrackingNumber'),
                           'PermitNumber': rec.get('PermitNumber'),
                           'UtilityCompany': rec.get('UtilityCompany'),
                           'WorkLocationFullAddress': rec.get('WorkLocationFullAddress'),
                           'CoordinateGroup': rec.get('CoordinateGroup'),
                           'FunctionalClass': None, 'ClassStatus': 'UNRESOLVED',
                           'MatchStatus': 'API_ERROR', 'ClassReviewNeeded': True,
                           'SubBlockReviewNeeded': True, 'Explanation': str(exc)[:500]}
                evidence = []
                errors_in_a_row += 1
                print(f'Error for LocationID {lid}: {exc}')
            results = [r for r in results if str(r['LocationID']) != lid]
            candidates = [c for c in candidates if str(c['LocationID']) != lid]
            results.append(outcome)
            candidates.extend(evidence)
            processed += 1
            if processed % SAVE_EVERY == 0:
                save_checkpoint()
                print(f'Processed: {len(results):,} / {len(work):,}')
            if errors_in_a_row >= MAX_CONSECUTIVE_API_ERRORS:
                print('Stopped after repeated errors. Fix the service/network and rerun to resume.')
                break
        save_checkpoint()
    finally:
        api_cache.close()

    all_results = pd.DataFrame(results)
    if all_results.empty:
        print('No records processed.')
        return
    if all_results['LocationID'].astype(str).duplicated().any():
        raise ValueError('Duplicate LocationID detected in final results.')
    # Check that every verified class has supporting official candidate evidence.
    evidence_df = pd.DataFrame(candidates)
    if not evidence_df.empty:
        supported = evidence_df.loc[
            evidence_df['EligibleAddressBlockClass'].astype(str).str.lower().eq('true')
            & evidence_df['OfficialFHWAClass'].notna(), ['LocationID', 'OfficialFHWAClass']].copy()
        supported['OfficialFHWAClass'] = pd.to_numeric(supported['OfficialFHWAClass'], errors='coerce')
        verified = all_results.loc[all_results['ClassStatus'].eq('VERIFIED'),
                                   ['LocationID', 'FunctionalClass']].copy()
        verified['FunctionalClass'] = pd.to_numeric(verified['FunctionalClass'], errors='coerce')
        verified['LocationID'] = verified['LocationID'].astype(str)
        supported['LocationID'] = supported['LocationID'].astype(str)
        proof = verified.merge(supported, on='LocationID', how='left')
        consistent = proof.loc[proof['FunctionalClass'].eq(proof['OfficialFHWAClass']), 'LocationID']
        unsupported = sorted(set(verified['LocationID']) - set(consistent))
        if unsupported:
            raise ValueError(f'VERIFIED results without supporting candidate evidence: {unsupported[:10]}')

    review = all_results.loc[
        all_results['ClassReviewNeeded'].astype(str).str.lower().eq('true')].copy()
    review['SuggestedNextAction'] = review['MatchStatus'].map(REVIEW_ACTIONS).fillna(
        'Inspect LRS candidate evidence and original permit details.')
    errors = all_results.loc[all_results['MatchStatus'].eq('API_ERROR')]
    save_csv(review, REVIEW_PATH)
    save_csv(errors, ERROR_PATH)
    status_counts = all_results['MatchStatus'].value_counts(dropna=False).rename_axis('MatchStatus')
    status_summary = status_counts.reset_index(name='NumberOfPermitLocations')
    status_summary['PercentOfAllLocations'] = (100 * status_summary['NumberOfPermitLocations']
                                                / len(work)).round(2)
    save_csv(status_summary, SUMMARY_PATH)

    verified_count = int(all_results['ClassStatus'].eq('VERIFIED').sum())
    provisional_count = int(all_results['ClassStatus'].eq('PROVISIONAL').sum())
    unclassified_count = int(all_results['FunctionalClass'].isna().sum())
    print('\n--- FINAL ROADWAY MATCHING SUMMARY ---')
    print('Records saved:', len(all_results), '/', len(work))
    print(status_counts.to_string())
    print('Verified classes:', verified_count)
    print('Provisional classes:', provisional_count)
    print('No assigned class:', unclassified_count)
    print('Locations requiring class review:', len(review))
    print('API errors:', len(errors))
    if len(all_results) != len(work):
        print('INCOMPLETE: rerun this script to resume remaining locations.')
    else:
        print('COMPLETE: every input LocationID has one output row.')
    print('Main results:', RESULT_PATH)
    print('Candidate evidence:', CANDIDATE_PATH)
    print('Unresolved cases:', REVIEW_PATH)
    print('Status summary:', SUMMARY_PATH)
    print('Only ClassStatus=VERIFIED is a confirmed Functional Class.')


if __name__ == '__main__':
    main()
