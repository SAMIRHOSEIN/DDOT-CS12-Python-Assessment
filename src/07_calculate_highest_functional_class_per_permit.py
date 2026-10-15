# %% STEP 1 - Configuration and data conventions
# Purpose:
# Use the roadway Functional Class results from Step 06 to determine
# the highest FHWA Functional Class for each permit application.
# Group all permit locations by TrackingNumber, since one application
# may contain multiple LocationIDs and PermitNumber may be missing.
# Identify the highest roadway class using the minimum numerical FHWA
# class code (Class 1 is higher than Class 7).
# Confirm the application-level result only when all associated
# locations have VERIFIED classes; otherwise, flag it for review.
# Preserve every location and attach the application-level result
# to each location record.
# Save the final location-level and application-level CSVs,
# along with quality-control summaries and unresolved cases.
# Keep pilot and full results separate without making new API requests.

from pathlib import Path
import pandas as pd

RUN_MODE = 'full'  # 'pilot' or 'full'
ROOT = Path(__file__).resolve().parent.parent
STEP06_DIR = ROOT / 'outputs' / '3_roadway_functional_class' / RUN_MODE
INPUT_CSV = STEP06_DIR / 'permit_locations_fhwa_functional_class.csv'
STEP06_REVIEW_CSV = STEP06_DIR / 'permit_locations_functional_class_needing_review.csv'
OUTPUT_ROOT = ROOT / 'outputs' / '4_permit_highest_functional_class'
PRIMARY_OUTPUT_DIR = OUTPUT_ROOT / '01_primary_results' / RUN_MODE
DOCUMENTATION_OUTPUT_DIR = OUTPUT_ROOT / '02_quality_control_and_limitations' / RUN_MODE

VALID_FHWA_CLASSES = set(range(1, 8))
VALID_CLASS_STATUSES = {'VERIFIED', 'PROVISIONAL', 'UNRESOLVED'}


# %% STEP 2 - Load and validate Step 06 location-level classifications

def load_step06_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Step 06 output not found: {path}')

    df = pd.read_csv(
        path,
        dtype={'TrackingNumber': 'string', 'LocationID': 'string',
               'PermitNumber': 'string'},
        low_memory=False,
    )
    required = {'TrackingNumber', 'LocationID', 'UtilityCompany', 'PermitNumber',
                'WorkLocationFullAddress', 'FunctionalClass', 'ClassStatus',
                'MatchStatus', 'Explanation'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing Step 06 columns: {sorted(missing)}')
    if df.empty:
        raise ValueError('Step 06 input is empty.')

    for col in ('TrackingNumber', 'LocationID'):
        df[col] = df[col].str.strip()
        if df[col].isna().any() or df[col].eq('').any():
            raise ValueError(f'Found missing or blank {col}.')
    if df['LocationID'].duplicated().any():
        example = df.loc[df['LocationID'].duplicated(), 'LocationID'].head(5).tolist()
        raise ValueError(f'Duplicate LocationIDs in input: {example}')

    df['ClassStatus'] = df['ClassStatus'].astype('string').str.strip().str.upper()
    if df['ClassStatus'].isna().any() or not df['ClassStatus'].isin(VALID_CLASS_STATUSES).all():
        invalid = df.loc[~df['ClassStatus'].isin(VALID_CLASS_STATUSES),
                         'ClassStatus'].drop_duplicates().tolist()
        raise ValueError(f'Unrecognized ClassStatus values: {invalid}')

    df['FunctionalClass'] = pd.to_numeric(df['FunctionalClass'], errors='coerce')
    has_class = df['FunctionalClass'].notna()
    if not df.loc[has_class, 'FunctionalClass'].isin(VALID_FHWA_CLASSES).all():
        raise ValueError('FunctionalClass must be an integer from 1 through 7.')
    if df.loc[df['ClassStatus'].eq('VERIFIED'), 'FunctionalClass'].isna().any():
        raise ValueError('A VERIFIED location has no valid FunctionalClass.')
    if df.loc[df['ClassStatus'].eq('UNRESOLVED'), 'FunctionalClass'].notna().any():
        raise ValueError('An UNRESOLVED location unexpectedly has a FunctionalClass.')
    if df.loc[df['ClassStatus'].eq('PROVISIONAL'), 'FunctionalClass'].isna().any():
        raise ValueError('A PROVISIONAL location has no proposed FunctionalClass.')

    company_counts = df.groupby('TrackingNumber', dropna=False)['UtilityCompany'].nunique()
    if company_counts.gt(1).any():
        raise ValueError('One TrackingNumber belongs to multiple utility companies; '
                         'verify identifiers before aggregating.')
    number_counts = df.groupby('TrackingNumber', dropna=False)['PermitNumber'].nunique()
    if number_counts.gt(1).any():
        raise ValueError('Multiple distinct nonblank PermitNumbers share a TrackingNumber.')
    return df


def attach_step06_review_actions(df: pd.DataFrame, review_path: Path) -> pd.DataFrame:
    """Keep Step 06 suggestions for documentation; do not apply them."""
    df = df.copy()
    if not review_path.is_file():
        df['SuggestedNextAction_NotApplied'] = ''
        return df
    reviews = pd.read_csv(review_path, dtype={'LocationID': 'string'}, low_memory=False)
    if not {'LocationID', 'SuggestedNextAction'}.issubset(reviews.columns):
        df['SuggestedNextAction_NotApplied'] = ''
        return df
    reviews['LocationID'] = reviews['LocationID'].str.strip()
    if reviews['LocationID'].duplicated().any():
        raise ValueError('Duplicate LocationID in Step 06 review file.')
    actions = reviews.set_index('LocationID')['SuggestedNextAction']
    df['SuggestedNextAction_NotApplied'] = (
        df['LocationID'].map(actions).fillna('').astype(str)
    )
    return df


# %% STEP 3 - Compute one conservative result per TrackingNumber

def class_list(series: pd.Series) -> str:
    return '; '.join(str(int(n)) for n in sorted(series.dropna().unique()))


def nonblank_values(series: pd.Series) -> str:
    return '; '.join(sorted({str(x).strip() for x in series.dropna()
                             if str(x).strip()}))


def aggregate_permits(locations: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for tracking_number, group in locations.groupby('TrackingNumber', sort=True):
        verified = group.loc[group['ClassStatus'].eq('VERIFIED')]
        provisional = group.loc[group['ClassStatus'].eq('PROVISIONAL')]
        unresolved = group.loc[group['ClassStatus'].eq('UNRESOLVED')]
        unverified = group.loc[group['ClassStatus'].ne('VERIFIED')]

        total = len(group)
        verified_count = len(verified)
        if verified_count == total:
            determination = 'CONFIRMED_ALL_LOCATIONS_VERIFIED'
            explanation = 'All permit locations have verified FHWA classes.'
        elif verified_count > 0:
            determination = 'REVIEW_PARTIAL_VERIFIED_LOCATIONS'
            explanation = ('Some permit locations remain unverified; the highest '
                           'class among verified locations is NOT the confirmed '
                           'highest class for the entire permit.')
        else:
            determination = 'REVIEW_NO_VERIFIED_LOCATIONS'
            explanation = ('No permit location has a verified FHWA class; '
                           'the permit-wide highest class is unknown.')

        best_verified = (int(verified['FunctionalClass'].min())
                         if verified_count else pd.NA)
        confirmed_highest = best_verified if verified_count == total else pd.NA
        summaries.append({
            'TrackingNumber': tracking_number,
            'UtilityCompany': nonblank_values(group['UtilityCompany']),
            'PermitNumber': nonblank_values(group['PermitNumber']),
            'TotalPermitLocations': total,
            'VerifiedLocations': verified_count,
            'ProvisionalLocations': len(provisional),
            'UnresolvedLocations': len(unresolved),
            'UnverifiedLocations': len(unverified),
            'HighestFHWAFunctionalClassPerPermit': confirmed_highest,
            'HighestFHWAClassAmongVerifiedLocations': best_verified,
            'VerifiedLocationClassValues': class_list(verified['FunctionalClass']),
            'ProvisionalLocationClassValues_NotConfirmed': class_list(provisional['FunctionalClass']),
            'HighestClassDeterminationStatus': determination,
            'PermitNeedsClassReview': verified_count != total,
            'LocationIDsNeedingReview': nonblank_values(unverified['LocationID']),
            'ReviewMatchStatuses': nonblank_values(unverified['MatchStatus']),
            'HighestClassExplanation': explanation,
        })

    out = pd.DataFrame(summaries)
    out['HighestFHWAFunctionalClassPerPermit'] = (
        out['HighestFHWAFunctionalClassPerPermit'].astype('Int64')
    )
    out['HighestFHWAClassAmongVerifiedLocations'] = (
        out['HighestFHWAClassAmongVerifiedLocations'].astype('Int64')
    )
    return out


# %% STEP 4 - Merge permit-wide class back onto EACH location and build QA

def enrich_locations(locations: pd.DataFrame,
                     tracking_summary: pd.DataFrame) -> pd.DataFrame:
    permit_columns = [
        'TrackingNumber', 'TotalPermitLocations', 'VerifiedLocations',
        'ProvisionalLocations', 'UnresolvedLocations',
        'HighestFHWAFunctionalClassPerPermit',
        'HighestFHWAClassAmongVerifiedLocations',
        'HighestClassDeterminationStatus', 'PermitNeedsClassReview',
        'LocationIDsNeedingReview', 'HighestClassExplanation',
    ]
    result = locations.merge(
        tracking_summary[permit_columns], on='TrackingNumber',
        how='left', validate='many_to_one', sort=False,
    )
    # Distinguish the class of THIS location from the class of its entire permit.
    result = result.rename(columns={
        'FunctionalClass': 'LocationFHWAFunctionalClass',
        'ClassStatus': 'LocationClassStatus',
        'MatchStatus': 'LocationMatchStatus',
    })
    if len(result) != len(locations) or result['LocationID'].nunique() != len(locations):
        raise AssertionError('The location-level deliverable lost or duplicated locations.')
    if result['HighestClassDeterminationStatus'].isna().any():
        raise AssertionError('At least one location did not receive permit-level context.')
    return result


def validate_and_summarize(locations: pd.DataFrame,
                           summary: pd.DataFrame,
                           enriched: pd.DataFrame) -> pd.DataFrame:
    confirmed = summary['HighestClassDeterminationStatus'].eq(
        'CONFIRMED_ALL_LOCATIONS_VERIFIED')
    if summary['TrackingNumber'].nunique() != len(summary):
        raise AssertionError('TrackingNumber summary is not unique.')
    if int(summary['TotalPermitLocations'].sum()) != len(locations):
        raise AssertionError('Total permit locations do not reconcile.')
    if int(summary['VerifiedLocations'].sum()) != int(locations['ClassStatus'].eq('VERIFIED').sum()):
        raise AssertionError('Verified location counts do not reconcile.')
    if int(summary['UnverifiedLocations'].sum()) != int(locations['ClassStatus'].ne('VERIFIED').sum()):
        raise AssertionError('Unverified location counts do not reconcile.')
    if summary.loc[confirmed, 'HighestFHWAFunctionalClassPerPermit'].isna().any():
        raise AssertionError('A fully verified permit is missing its highest class.')
    if summary.loc[~confirmed, 'HighestFHWAFunctionalClassPerPermit'].notna().any():
        raise AssertionError('An unverified permit incorrectly received a confirmed highest class.')
    if not (summary.loc[confirmed, 'HighestFHWAFunctionalClassPerPermit'] ==
            summary.loc[confirmed, 'HighestFHWAClassAmongVerifiedLocations']).all():
        raise AssertionError('Confirmed permit class does not match its verified location classes.')
    if len(enriched) != len(locations):
        raise AssertionError('Location-level output count mismatch.')

    statuses = summary['HighestClassDeterminationStatus'].value_counts()
    metrics = [
        ('Permit locations in Step 06 input', len(locations)),
        ('Unique permit applications (TrackingNumber)', len(summary)),
        ('Verified permit locations', int(locations['ClassStatus'].eq('VERIFIED').sum())),
        ('Provisional permit locations', int(locations['ClassStatus'].eq('PROVISIONAL').sum())),
        ('Unresolved permit locations', int(locations['ClassStatus'].eq('UNRESOLVED').sum())),
        ('Permits with confirmed highest class', int(statuses.get('CONFIRMED_ALL_LOCATIONS_VERIFIED', 0))),
        ('Permits with verified and unverified locations', int(statuses.get('REVIEW_PARTIAL_VERIFIED_LOCATIONS', 0))),
        ('Permits with no verified locations', int(statuses.get('REVIEW_NO_VERIFIED_LOCATIONS', 0))),
        ('Permits needing review', int(summary['PermitNeedsClassReview'].sum())),
    ]
    return pd.DataFrame(metrics, columns=['Metric', 'Count'])


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomic replacement; close Excel before rerunning the script on Windows."""
    temporary = path.with_name(path.name + '.tmp')
    frame.to_csv(temporary, index=False, encoding='utf-8-sig')
    temporary.replace(path)


# %% STEP 5 - Run the selected sample or full data; save explainable CSVs

def main() -> None:
    if RUN_MODE not in {'pilot', 'full'}:
        raise ValueError("RUN_MODE must be 'pilot' or 'full'.")

    print(f'\nSTEP 07 | RUN_MODE={RUN_MODE}')
    locations = load_step06_results(INPUT_CSV)
    locations = attach_step06_review_actions(locations, STEP06_REVIEW_CSV)
    permits = aggregate_permits(locations)
    enriched = enrich_locations(locations, permits)
    qa = validate_and_summarize(locations, permits, enriched)

    review_permits = permits.loc[permits['PermitNeedsClassReview']].copy()
    review_locations = enriched.loc[
        enriched['LocationClassStatus'].ne('VERIFIED')
    ].copy()
    problem_summary = (
        review_locations.groupby(['LocationMatchStatus', 'LocationClassStatus'],
                                 dropna=False).size()
        .reset_index(name='NumberOfLocationsNeedingReview')
        .sort_values(['NumberOfLocationsNeedingReview', 'LocationMatchStatus'],
                     ascending=[False, True])
    )

    primary_files = {
        'permit_locations_with_highest_fhwa_functional_class.csv': enriched,
        'permit_tracking_highest_fhwa_functional_class_summary.csv': permits,
    }
    documentation_files = {
        'permit_tracking_numbers_needing_highest_class_review.csv': review_permits,
        'permit_locations_needing_fhwa_class_review_with_tracking_context.csv': review_locations,
        'permit_unverified_location_reason_summary.csv': problem_summary,
        'permit_highest_class_quality_summary.csv': qa,
    }
    for directory, files in (
        (PRIMARY_OUTPUT_DIR, primary_files),
        (DOCUMENTATION_OUTPUT_DIR, documentation_files),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        print(f'\nSaving to: {directory}')
        for filename, table in files.items():
            write_csv(table, directory / filename)
            print(f'  {filename}: {len(table):,} rows')

    print('\n--- QA SUMMARY ---')
    print(qa.to_string(index=False))
    print('\nIMPORTANT: HighestFHWAFunctionalClassPerPermit is blank when any '
          'location in that TrackingNumber is not VERIFIED.')
    print('Proposed Step 06 solutions are documented, NOT applied.')
    print('Primary results:', PRIMARY_OUTPUT_DIR)
    print('Documentation and limitations:', DOCUMENTATION_OUTPUT_DIR)


if __name__ == '__main__':
    main()
