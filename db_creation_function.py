import pandas as pd
import geopandas as gpd

### CLEANING ###
def is_mine_active(row):
    '''
    Function to define if the mine is currently active,
    based on the Changes in Status of Production
    '''

    current_year = 2022

    # Convert year values to integers, ignoring non-numeric values
    def to_int(value):
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    # Convert all year columns to integers
    open1 = to_int(row['open1'])
    close1 = to_int(row['close1'])
    open2 = to_int(row['open2'])
    close2 = to_int(row['close2'])
    open3 = to_int(row['open3'])
    close3 = to_int(row['close3'])

    # Check if any of the 'close' columns have the value 'open'
    if row['close1'] == 'open' or row['close2'] == 'open' or row['close3'] == 'open':
        return 'Active'

    # Find the latest year among open and close columns
    years = [open1, close1, open2, close2, open3, close3]
    years = [year for year in years if year is not None]

    if not years:
        return 'Unknown'

    latest_year = max(years)

    # If the latest year is a 'close' year, the mine is inactive
    if latest_year in [close1, close2, close3]:
        return 'Inactive'

    # If the latest year is an 'open' year and it's the current year or later, consider it active
    if latest_year in [open1, open2, open3] and latest_year >= current_year:
        return 'Active'

    # For all other cases, consider it inactive
    return 'Inactive'


def filter_ghg_facility_naics(df, classifications):
    # Normalize the classifications to lowercase for case-insensitive comparison
    classifications_lower = [cls.lower() for cls in classifications]

    df_copy = df.copy()
    df_copy['NAICS_Lower'] = df_copy['Industry classification'].str.lower()
    filtered_df = df_copy[df_copy['NAICS_Lower'].isin(classifications_lower)]
    filtered_df = filtered_df.drop(columns=['NAICS_Lower'])
    filtered_df = filtered_df.reset_index(drop=True)

    return filtered_df


def clean_npri(df, category_mapping):
    """
    Renames columns in a DataFrame based on a category mapping.
    Each column name is prefixed with the category.

    Args:
        df (pd.DataFrame): The DataFrame whose columns need renaming.
        category_mapping (dict): A dictionary mapping categories to their subcategories.

    Returns:
        pd.DataFrame: The DataFrame with renamed columns.
    """
    # Create a mapping of old column names to new column names
    column_renaming = {}

    for category, subcategories in category_mapping.items():
        for subcategory in subcategories:
            if subcategory in df.columns:
                # Create a new column name with the format 'category_subcategory'
                new_column_name = f"{category.lower().replace(' ', '_')}_{subcategory.lower().replace(' ', '_')}"
                column_renaming[subcategory] = new_column_name

    # Rename columns in the DataFrame
    df = df.rename(columns=column_renaming)
    return df


### GEOSPATIAL STUFF
def populate_facility_df(column_mapping, facility_df, dynamic_columns=None, source_dfs=None):
    """
    Populate a facility DataFrame based on a column mapping and optional dynamic column values.

    Parameters:
        column_mapping (dict): A dictionary where keys are DataFrame names (as strings) and values are mappings
                               of source column names to target column names.
        facility_df (pd.DataFrame): The target facility DataFrame to populate.
        dynamic_columns (dict, optional): A dictionary where keys are target column names and values are mappings
                                          of DataFrame names to specific values (e.g., facility type).
        source_dfs (dict): A dictionary where keys are DataFrame names (as strings) and values are the actual DataFrames.

    Returns:
        pd.DataFrame: The populated facility DataFrame.
    """
    # Debug: Ensure facility_df starts empty or with expected rows
    print(f"Initial facility_df rows: {len(facility_df)}")

    for source_name, mapping in column_mapping.items():
        print(f"Processing DataFrame: {source_name}")

        df = source_dfs.get(source_name)
        if df is None:
            print(f"Warning: DataFrame '{source_name}' not found.")
            continue

        # Create a temporary DataFrame for the current source
        temp_df = pd.DataFrame()

        for src_col, target_col in mapping.items():
            if target_col in facility_df.columns and src_col in df.columns:
                # Map the source column to the target column
                temp_df[target_col] = df[src_col]

        # Add dynamic columns if provided
        if dynamic_columns:
            for dynamic_col, source_values in dynamic_columns.items():
                if dynamic_col in facility_df.columns and source_name in source_values:
                    temp_df[dynamic_col] = source_values[source_name]

        # Add a 'source' column for provenance tracking
        temp_df["source"] = source_name

        # Ensure temp_df aligns with facility_df
        missing_columns = set(facility_df.columns) - set(temp_df.columns)
        for col in missing_columns:
            temp_df[col] = pd.NA

        # Debug: Print temp_df shape before appending
        print(f"Temp DF rows to append: {len(temp_df)}")

        # Append temp_df to facility_df
        facility_df = pd.concat([facility_df, temp_df], ignore_index=True)

        # Debug: Print facility_df shape after appending
        print(f"Rows in facility_df after appending {source_name}: {len(facility_df)}")

    # Final debug: Ensure the final facility_df shape is correct
    print(f"Final facility_df rows: {len(facility_df)}")
    return facility_df


def assign_ids(facility_df, id_column="facility_id"):
    """
    Assign deterministic facility IDs that include province codes and facility type prefixes.

    Parameters:
        facility_df (pd.DataFrame): The DataFrame to which the IDs will be assigned.
        id_column (str): The name of the column for the unique IDs.

    Returns:
        pd.DataFrame: The DataFrame with assigned facility IDs.
    """
    if id_column not in facility_df.columns:
        facility_df[id_column] = None  # Create the ID column if it doesn't exist

    # Dictionary mapping provinces to their codes
    province_codes = {
        "Ontario": "ON",
        "Quebec": "QC",
        "British Columbia": "BC",
        "Alberta": "AB",
        "Manitoba": "MB",
        "Saskatchewan": "SK",
        "Newfoundland and Labrador": "NL",
        "New Brunswick": "NB",
        "Nova Scotia": "NS",
        "Prince Edward Island": "PE",
        "Northwest Territories": "NT",
        "Yukon": "YT",
        "Nunavut": "NU"
    }

    def generate_id(row):
        # Get the province code
        province = row.get("province", "Unknown")
        province_code = province_codes.get(province, "ZZ")  # Use 'ZZ' for unknown provinces

        # Determine prefix based on facility type
        facility_type = row.get("facility_type", "UNKNOWN").upper()
        prefix = {
            "MINING": "MIN",
            "MANUFACTURING": "MAN",
            "PROCESSING": "PRO"
        }.get(facility_type, "OTH")  # Default prefix is 'OTH' for other types

        # Create a unique hash from facility_name, latitude, longitude, and type
        unique_hash = hash((facility_type, row["facility_name"], row["latitude"], row["longitude"])) & 0xFFFFFFFF

        # Combine province code, prefix, and hash to form the ID
        return f"{province_code}-{prefix}-{unique_hash:08d}"  # Ensures a fixed 8-digit hash

    facility_df[id_column] = facility_df.apply(generate_id, axis=1)

    return facility_df


def add_geospatial_info(facility_df, other_df, matching_columns, buffer_distance=1000, crs="EPSG:4326"):
    """
    Add information from another DataFrame to facility_df based on geospatial matching.

    Parameters:
        facility_df (pd.DataFrame): The main facility DataFrame.
        other_df (pd.DataFrame): The secondary DataFrame with additional information.
        matching_columns (dict): Columns to add from other_df. Format: {"source_column": "target_column"}.
        buffer_distance (float): Buffer distance in meters for proximity matching.
        crs (str): Coordinate Reference System, default is WGS 84 (EPSG:4326).

    Returns:
        pd.DataFrame: The updated facility_df with added information.
    """
    # Convert facility_df and other_df to GeoDataFrames
    facility_gdf = gpd.GeoDataFrame(
        facility_df,
        geometry=gpd.points_from_xy(facility_df["longitude"], facility_df["latitude"]),
        crs=crs,
    )
    other_gdf = gpd.GeoDataFrame(
        other_df,
        geometry=gpd.points_from_xy(other_df["longitude"], other_df["latitude"]),
        crs=crs,
    )

    # Reproject to a projected CRS for accurate buffering
    facility_gdf = facility_gdf.to_crs("EPSG:3857")
    other_gdf = other_gdf.to_crs("EPSG:3857")

    # Create a buffer around each facility
    facility_gdf["geometry"] = facility_gdf["geometry"].buffer(buffer_distance)

    # Perform a spatial join to find matches within the buffer
    joined_gdf = gpd.sjoin(other_gdf, facility_gdf, how="inner", predicate="within")

    # Drop duplicate matches and aggregate if necessary
    joined_gdf = joined_gdf.groupby("index_right").first()

    # Add the matching columns to facility_gdf
    for source_col, target_col in matching_columns.items():
        if source_col in other_gdf.columns:
            facility_gdf[target_col] = joined_gdf[source_col]

    # Reproject back to the original CRS
    facility_gdf = facility_gdf.to_crs(crs)

    # Drop buffer geometry for clean output
    facility_gdf = facility_gdf.drop(columns="geometry")

    return pd.DataFrame(facility_gdf)


def assign_facility_ids_by_location(facility_df, ghg_df, proximity_threshold=1000):
    """
    Assigns facility IDs from `facility_df` to `ghg_df` based on geographical proximity.

    Args:
        facility_df (pd.DataFrame): DataFrame with facility IDs, latitude, and longitude.
        ghg_df (pd.DataFrame): DataFrame with latitude and longitude, where facility IDs will be assigned.
        proximity_threshold (float): Maximum distance for matching (in meters).

    Returns:
        gpd.GeoDataFrame: Updated `ghg_df` with a new `facility_id` column as the second column.
    """
    # Step 1: Convert facility_df and ghg_df to GeoDataFrames
    if not isinstance(facility_df, gpd.GeoDataFrame):
        facility_df = gpd.GeoDataFrame(
            facility_df,
            geometry=gpd.points_from_xy(facility_df.longitude, facility_df.latitude),
            crs="EPSG:4326"  # WGS 84 Geographic Coordinate System
        )

    if not isinstance(ghg_df, gpd.GeoDataFrame):
        ghg_df = gpd.GeoDataFrame(
            ghg_df,
            geometry=gpd.points_from_xy(ghg_df.longitude, ghg_df.latitude),
            crs="EPSG:4326"  # WGS 84 Geographic Coordinate System
        )

    # Step 2: Re-project to a projected CRS for spatial operations
    facility_df_proj = facility_df.to_crs(epsg=3857)  # Web Mercator
    ghg_df_proj = ghg_df.to_crs(epsg=3857)

    # Step 3: Apply buffer to facility_df geometries based on proximity_threshold (in meters)
    facility_df_proj['geometry_buffered'] = facility_df_proj.geometry.buffer(proximity_threshold)

    # Step 4: Perform spatial join
    # Use the buffered geometry to find matches within the threshold
    joined = gpd.sjoin(
        ghg_df_proj,
        facility_df_proj[['facility_id', 'geometry_buffered']].rename(columns={'geometry_buffered': 'geometry'}),
        how="left",
        predicate="within"  # Matches points within the buffered geometries
    )

    # Step 5: Handle duplicates (if multiple facilities match the same GHG point)
    deduplicated = (
        joined[['facility_id']]
        .groupby(joined.index)  # Group by the original index of ghg_df
        .first()  # Take the first match (can also use min, max, etc.)
    )

    # Step 6: Assign facility_id to ghg_df
    ghg_df['facility_id'] = deduplicated['facility_id']

    # Step 7: Reorder columns to make facility_id the second column
    columns = ghg_df.columns.to_list()
    reordered_columns = [columns[0], 'facility_id'] + [col for col in columns if col not in ['facility_id', columns[0]]]
    ghg_df = ghg_df[reordered_columns]

    # Step 8: Calculate and print the percentage of rows with None/NaN in facility_id
    total_rows = len(ghg_df)
    unmatched_rows = ghg_df['facility_id'].isna().sum()
    unmatched_percentage = (unmatched_rows / total_rows) * 100
    print(f"Percentage of unmatched rows (facility_id = None): {unmatched_percentage:.2f}%")

    # Step 9: Return the updated GeoDataFrame in its original CRS
    return ghg_df
