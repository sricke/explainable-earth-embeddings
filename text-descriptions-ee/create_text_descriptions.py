import pandas as pd

def categorize(value, bins):
    if pd.isna(value):
        return "unknown"
    for threshold, label in bins:
        if value < threshold:
            return label
    return bins[-1][1]

def create_description(row, category_bins):
    parts = []
    location_parts = []
    feature_parts = []
    
    if pd.notna(row.get('country')):
        location_parts.append(row['country'])
    
    if pd.notna(row.get('state')):
        location_parts.append(row['state'])
    
    if 'elevation_m' in row:
        elev_cat = categorize(row['elevation_m'], category_bins['elevation'])
        feature_parts.append(f"has {elev_cat} elevation")
    
    if 'pop_density' in row:
        pop_cat = categorize(row['pop_density'], category_bins['pop_density'])
        feature_parts.append(f"represents a {pop_cat}")
    
    if 'temp_annual_C' in row:
        temp_cat = categorize(row['temp_annual_C'], category_bins['temp_annual_C'])
        feature_parts.append(f"has a {temp_cat} climate")
    
    if 'precip_annual_mm' in row:
        precip_cat = categorize(row['precip_annual_mm'], category_bins['precip_annual_mm'])
        feature_parts.append(f"receives {precip_cat} precipitation")
    
    if 'tree_cover_pct' in row:
        tree_cat = categorize(row['tree_cover_pct'], category_bins['tree_cover_pct'])
        feature_parts.append(f"is classified as {tree_cat}")
    
    if 'avg_ndvi' in row:
        ndvi_cat = categorize(row['avg_ndvi'], category_bins['ndvi'])
        feature_parts.append(f"shows {ndvi_cat}")
    
    # if 'nightlights' in row:
    #     lights_cat = categorize(row['nightlights'], category_bins['nightlights'])
    #     feature_parts.append(f"has {lights_cat}")
    
    if pd.notna(row.get('land_cover')):
        feature_parts.append(f"has land cover of {row['land_cover']}")
    
    if pd.notna(row.get('biome')):
        feature_parts.append(f"belongs to the {row['biome']} biome")
    
    if pd.notna(row.get('ecoregion')):
        feature_parts.append(f"is within the {row['ecoregion']} ecoregion")
    
    location_str = ", ".join(location_parts) if location_parts else "an unknown location"
    features_str = ", ".join(feature_parts) if feature_parts else "no additional features"
    
    return f"This point is in {location_str}, {features_str}."

def process_csv(input_csv, output_csv, category_bins):
    df = pd.read_csv(input_csv)
    df['description'] = df.apply(
        lambda row: create_description(row, category_bins), axis=1
    )

    df[['fn', 'lat', 'lon', 'description']].to_csv(output_csv, index=False)
    return df

if __name__ == "__main__":
    category_bins = {
        # Source: https://bioone.org/journals/mountain-research-and-development/volume-21/issue-1/0276-4741_2001_021_0034_ANTFMA_2.0.CO_2/A-New-Typology-for-Mountains-and-Other-Relief-Classes/10.1659/0276-4741(2001)021%5B0034:ANTFMA%5D2.0.CO;2.full
        # Lowlands: 0-200m, Platforms/Hills: 200-500m, Mountains: > 500m
        'elevation': [
            (200, "lowland"),
            (500, "platform/hill"),
            (float('inf'), "mountain")
        ],
        
        # Source: UN Degree of Urbanization (https://unhabitat.org/wcr)
        # Rural: 50-300/km², Villages/semi-dense: 300-1500/km², Towns/cities: >1500/km²
        'pop_density': [
            (300, "rural area"),
            (1500, "town or semi-dense area"),
            (float('inf'), "town or city")
        ],
        
        # Source: Köppen climate classification (https://www.noaa.gov/jetstream/global/climate-zones)
        # Polar: <10°C warmest month, Cold temperate: coldest 0-10°C, Warm temperate: coldest 0-18°C, Tropical: all months >18°C
        'temp_annual_C': [
            (0, "polar"),
            (10, "cold temperate"),
            (18, "warm temperate"),
            (float('inf'), "tropical")
        ],
        
        # Source: Aridity classification (UNEP)
        # Adjusted for average annual mm, using upper bounds of typical precipitation
        'precip_annual_mm': [
            (100, "hyper-arid"),
            (250, "arid"),
            (500, "semi-arid"),
            (700, "sub-humid"),
            (float('inf'), "humid")
        ],
        
        # Source: General vegetation classification
        # Sparse: <10%, Grassland/shrubland: 10-25%, Woodland: 25-60%, Forest: >60%
        'tree_cover_pct': [
            (10, "non-forest"),
            (25, "savanna/sparse vegetation"),
            (60, "woodland"),
            (float('inf'), "forest")
        ],
        
        # Source: NDVI interpretation (USGS, USDA)
        # Bare/water: <0.1, Sparse vegetation: 0.1-0.2, Low vegetation: 0.2-0.5, Dense vegetation: >0.5
        # Note: NDVI values are scaled by 10000 in MODIS, so these thresholds assume descaled values (-1 to 1 range)
        'ndvi': [
            (1000, "bare soil or water"),
            (2000, "very sparse vegetation"),
            (5000, "sparse to moderate vegetation"),
            (7000, "dense vegetation"),
            (float('inf'), "very dense vegetation")
        ],
        
        # Source: VIIRS nighttime lights interpretation (nW/cm²/sr)
        # Background: <0.3, Rural: 0.3-5, Semi-urban: 5-50, Urban: >50
        # Values are approximate based on World Bank analysis and VIIRS composites
        # 'nightlights': [
        #     (0.3, "no visible lights"),
        #     (5, "minimal rural lighting"),
        #     (50, "moderate lighting"),
        #     (float('inf'), "bright urban lighting")
        # ]
    }
    
    input_csv = "index_with_ee.csv"
    output_csv = "index_with_descriptions.csv"
    
    df = process_csv(input_csv, output_csv, category_bins)
    print(df['description'].head())