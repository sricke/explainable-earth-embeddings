import time
import ee
import pandas as pd
from tqdm import tqdm
import time

ee.Initialize()

def query_points_batch(points_fc, batch_index):
    """
    Query multiple points in a single batch using Earth Engine's vectorized operations.
    
    Args:
        points_fc: ee.FeatureCollection of point geometries with index property
        batch_index: int, for tracking which batch this is
    
    Returns:
        list of dictionaries with all environmental data
    """
    
    try:
        # Load all datasets
        countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
        states = ee.FeatureCollection('FAO/GAUL/2015/level1')
        ecoregions = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017')
        
        # Raster datasets
        elevation = ee.Image('USGS/SRTMGL1_003').select(['elevation'], ['elevation_m'])
        
        climate = ee.Image('WORLDCLIM/V1/BIO')
        climate_processed = (
            climate.select('bio01').divide(10).rename('temp_annual_C')
            .addBands(climate.select('bio05').divide(10).rename('temp_max_warmest_C'))
            .addBands(climate.select('bio06').divide(10).rename('temp_min_coldest_C'))
            .addBands(climate.select('bio07').divide(10).rename('temp_range_C'))
            .addBands(climate.select('bio12').rename('precip_annual_mm'))
            .addBands(climate.select('bio13').rename('precip_wettest_mm'))
            .addBands(climate.select('bio14').rename('precip_driest_mm'))
        )
        
        land_cover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
        
        soil_ph = ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02").select('b0').divide(10).rename('soil_ph')
        soil_carbon = ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02").select('b0').multiply(5).rename('soil_carbon_g_kg')
        
        # Vegetation indices (2023 average)
        evi = ee.ImageCollection('MODIS/061/MOD13A1').filterDate('2023-01-01', '2023-12-31').select('EVI').mean().rename('avg_evi')
        ndvi = ee.ImageCollection('MODIS/061/MOD13A1').filterDate('2023-01-01', '2023-12-31').select('NDVI').mean().rename('avg_ndvi')
        
        tree_cover = ee.ImageCollection("NASA/MEASURES/GFCC/TC/v3").select("tree_canopy_cover").mosaic().rename('tree_cover_pct')
        
        # Human activity
        nightlights = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').filterDate('2023-01-01', '2023-12-31').select('avg_rad').mean().rename('nightlights')
        pop_density = ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density").first().select('population_density').rename('pop_density')
        
        # Combine all raster bands into one image
        combined_image = (
            elevation
            .addBands(climate_processed)
            .addBands(land_cover)
            .addBands(soil_ph)
            .addBands(soil_carbon)
            .addBands(evi)
            .addBands(ndvi)
            .addBands(tree_cover)
            .addBands(nightlights)
            .addBands(pop_density)
        )
        
        # sample all raster data at points
        sampled = combined_image.sampleRegions(
            collection=points_fc,
            scale=30,
            geometries=True,
            tileScale=4  
        )
        
        # add country and state data via spatial join
        def add_country(feature):
            point = feature.geometry()
            country_feat = countries.filterBounds(point).first()
            return feature.set('country', country_feat.get('country_na'))
        
        sampled = sampled.map(add_country)
        
        def add_state(feature):
            point = feature.geometry()
            state_feat = states.filterBounds(point).first()
            return feature.set('state', state_feat.get('ADM1_NAME'))
        
        sampled = sampled.map(add_state)
        
        # add ecoregion data via spatial join
        def add_ecoregion(feature):
            point = feature.geometry()
            eco_feat = ecoregions.filterBounds(point).first()
            return feature.set({
                'ecoregion': eco_feat.get('ECO_NAME'),
                'biome': eco_feat.get('BIOME_NAME'),
                'realm': eco_feat.get('REALM'),
                'nnh': eco_feat.get('NNH_NAME')
            })
        
        sampled = sampled.map(add_ecoregion)
        
        result = sampled.getInfo()
        
        # process land cover codes
        lc_map = {
            10: 'Tree cover', 
            20: 'Shrubland', 
            30: 'Grassland', 
            40: 'Cropland', 
            50: 'Built-up',
            60: 'Bare/Sparse Vegetation', 
            70: 'Snow and Ice', 
            80: 'Permanent Water Bodies', 
            90: 'Herbaceous wetland', 
            95: 'Mangroves', 
            100: 'Moss and Lichen'
        }
        
        results_list = []
        for feature in result['features']:
            props = feature['properties']
            
            if 'Map' in props and props['Map'] is not None:
                props['land_cover'] = lc_map.get(int(props['Map']), None)
                del props['Map']
            else:
                props['land_cover'] = None
            
            results_list.append(props)
        
        return results_list
    
    except Exception as e:
        print(f"Error processing batch {batch_index}: {str(e)}")
        return None


def process_csv_in_batches(input_csv, output_csv, batch_size=1000):
    """
    Process the entire CSV in batches using vectorized Earth Engine queries.
    Writes results incrementally to CSV as each batch completes.
    
    Args:
        input_csv: path to input CSV with lat, lon columns
        output_csv: path to output CSV
        batch_size: number of points to process per batch (default 1000)
    """
    
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    total_points = len(df)
    print(f"Total points to process: {total_points}")
    
    num_batches = (total_points + batch_size - 1) // batch_size
    print(f"Processing in {num_batches} batches of up to {batch_size} points each")
    print(f"Results will be written incrementally to {output_csv}")
    
    output_columns = None
    points_processed = 0
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_points)
        
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        # create FeatureCollection from batch
        features = []
        for idx, row in batch_df.iterrows():
            point = ee.Geometry.Point([float(row['lon']), float(row['lat'])])
            properties = {k: v for k, v in row.items()}
            properties['original_index'] = idx
            features.append(ee.Feature(point, properties))
        
        points_fc = ee.FeatureCollection(features)
        
        # query batch with retry logic
        max_retries = 3
        retry_delay = 5
        batch_results = None
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                batch_results = query_points_batch(points_fc, batch_idx)
                end_time = time.time()
                
                if batch_results is not None:
                    print(f"\nBatch {batch_idx} completed in {end_time - start_time:.1f} seconds ({len(batch_results)} points)")
                    break
                else:
                    raise Exception("Batch returned None")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\nRetrying batch {batch_idx} (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    print(f"\nFailed to process batch {batch_idx} after {max_retries} attempts")
                    batch_results = []
                    for idx, row in batch_df.iterrows():
                        result = {k: v for k, v in row.items()}
                        result['original_index'] = idx
                        
                        for field in ['country', 'state', 'elevation_m', 'ecoregion', 'biome', 
                                     'realm', 'nnh', 'temp_annual_C', 'temp_max_warmest_C',
                                     'temp_min_coldest_C', 'temp_range_C', 'precip_annual_mm',
                                     'precip_wettest_mm', 'precip_driest_mm', 'land_cover',
                                     'soil_ph', 'soil_carbon_g_kg', 'avg_evi', 'avg_ndvi',
                                     'tree_cover_pct', 'nightlights', 'pop_density']:
                            result[field] = None
                        batch_results.append(result)
        
        batch_df_results = pd.DataFrame(batch_results)
        
        if 'original_index' in batch_df_results.columns:
            batch_df_results = batch_df_results.sort_values('original_index')
            batch_df_results = batch_df_results.drop('original_index', axis=1)
        
        # Reorder columns to put fn, lat, lon first
        cols = batch_df_results.columns.tolist()
        priority_cols = []
        if 'fn' in cols:
            priority_cols.append('fn')
        if 'lat' in cols:
            priority_cols.append('lat')
        if 'lon' in cols:
            priority_cols.append('lon')
        
        other_cols = [c for c in cols if c not in priority_cols]
        batch_df_results = batch_df_results[priority_cols + other_cols]
        
        if batch_idx == 0:
            batch_df_results.to_csv(output_csv, mode='w', index=False, header=True)
            output_columns = batch_df_results.columns.tolist()
            print(f"Created {output_csv} with {len(output_columns)} columns")
        else:
            batch_df_results = batch_df_results[output_columns]
            batch_df_results.to_csv(output_csv, mode='a', index=False, header=False)
        
        points_processed += len(batch_df_results)
        print(f"Progress: {points_processed}/{total_points} points written to file")
        
        if batch_idx < num_batches - 1:
            time.sleep(1)
    
    print(f"\n{'='*60}")
    print(f"Done! Processed {points_processed} points total.")
    print(f"Results saved to {output_csv}")
    print(f"{'='*60}")
    
    return pd.read_csv(output_csv)


if __name__ == "__main__":
    input_csv = "index.csv"
    output_csv = "index_with_ee.csv"
    
    batch_size = 100
    
    results = process_csv_in_batches(input_csv, output_csv, batch_size=batch_size)
    
    print("\nSample of results:")
    print(results.head())
    print(f"\nTotal columns: {len(results.columns)}")
    print(f"Columns: {list(results.columns)}")