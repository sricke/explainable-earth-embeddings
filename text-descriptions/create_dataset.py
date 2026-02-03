import ee
import pandas as pd

ee.Initialize()

def query_point(lat, lon):
    point = ee.Geometry.Point([lon, lat])
    data = {'lat': lat, 'lon': lon}
    
    # ADMINISTRATIVE BOUNDARIES
    try:
        #Large Scale International Boundary Polygons
        country = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filterBounds(point).first()
        data['country'] = country.get('country_na').getInfo() if country else None
    except:
        data['country'] = None
    
    try:
        #Global Administrative Unit Layers 2015, First-Level Administrative Unit
        state = ee.FeatureCollection('FAO/GAUL/2015/level1').filterBounds(point).first()
        data['state'] = state.get('ADM1_NAME').getInfo() if state else None
    except:
        data['state'] = None
    
    # TOPOGRAPHY
    try:
        #NASA SRTM Digital Elevation 30m
        # resolution: 30 m
        data['elevation_m'] = ee.Image('USGS/SRTMGL1_003').sample(point).first().get('elevation').getInfo()
    except:
        data['elevation_m'] = None
    
    # ECOREGIONS
    try:
        #RESOLVE Ecoregions 2017
        # 846 terrestrial ecoregions are grouped into 14 biomes and 8 realms
        eco = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017').filterBounds(point).first()
        data['ecoregion'] = eco.get('ECO_NAME').getInfo() if eco else None
        data['biome'] = eco.get('BIOME_NAME').getInfo() if eco else None
        data['realm'] = eco.get('REALM').getInfo() if eco else None
        data['nnh'] = eco.get('NNH_NAME').getInfo() if eco else None
    except:
        data['ecoregion'] = None
        data['biome'] = None
        data['realm'] = None
        data['nnh'] = None
    
    # CLIMATE
    try:
        #WorldClim BIO Variables V1
        # resolution: 927.67 m
        #check bands at https://developers.google.com/earth-engine/datasets/catalog/WORLDCLIM_V1_BIO#bands
        climate = ee.Image('WORLDCLIM/V1/BIO').sample(point).first()
        data['temp_annual_C'] = climate.get('bio01').getInfo() / 10
        data['temp_max_warmest_C'] = climate.get('bio05').getInfo() / 10
        data['temp_min_coldest_C'] = climate.get('bio06').getInfo() / 10
        data['temp_range_C'] = climate.get('bio07').getInfo() / 10
        data['precip_annual_mm'] = climate.get('bio12').getInfo()
        data['precip_wettest_mm'] = climate.get('bio13').getInfo()
        data['precip_driest_mm'] = climate.get('bio14').getInfo()
    except:
        data['temp_annual_C'] = None
        data['temp_max_warmest_C'] = None
        data['temp_min_coldest_C'] = None
        data['temp_range_C'] = None
        data['precip_annual_mm'] = None
        data['precip_wettest_mm'] = None
        data['precip_driest_mm'] = None
    
    # LAND COVER
    try:
        #ESA World Cover
        # resolution: 10m
        #check val map at https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200
        lc_code = ee.ImageCollection('ESA/WorldCover/v200').first().sample(point).first().get('Map').getInfo()
        lc_map = {10: 'Tree cover', 20: 'Shrubland', 30: 'Grassland', 40: 'Cropland', 50: 'Built-up',
                  60: 'Bare/Sparse Vegetation', 70: 'Snow and Ice', 80: 'Permanent Water Bodies', 90: 'Herbaceous wetland', 
                  95: 'Mangroves', 100: 'Moss and Lichen'}
        data['land_cover'] = lc_map.get(lc_code, None)
    except:
        data['land_cover'] = None
    
    # SOIL
    try:
        # OpenLandMap Soil pH in H2O
        # use top layer soil (band b0)
        # resolution: 250 m
        data['soil_ph'] = ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02").select('b0').sample(point).first().get('b0').getInfo() / 10  # b0 = topsoil layer
    except:
        data['soil_ph'] = None
    
    try:
        # OpenLandMap Soil Organic Carbon Content
        # use top soil layer (b0)
        # resolution: 250 m
        data['soil_carbon_g_kg'] = ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02").select('b0').sample(point).first().get('b0').getInfo() * 5  # b0 = topsoil layer
    except:
        data['soil_carbon_g_kg'] = None
    
    # VEGETATION
    try:
        #MOD13A1.061 Terra Vegetation Indices 16-Day Global 500m
        # resolution: 500m 
        # NDVI=normalized difference vegetation index
        # EVI=enhanced vegetation index; that minimizes canopy background variations and maintains sensitivity over dense vegetation conditions.
        evi = ee.ImageCollection('MODIS/061/MOD13A1').filterDate('2023-01-01', '2023-12-31').select('EVI').mean()  # [EVI more sensitive in high biomass areas than NDVI]
        ndvi = ee.ImageCollection('MODIS/061/MOD13A1').filterDate('2023-01-01', '2023-12-31').select('NDVI').mean()  # [EVI more sensitive in high biomass areas than NDVI]
        data['avg_evi'] = evi.sample(point).first().get('EVI').getInfo()
        data['avg_ndvi'] = ndvi.sample(point).first().get('NDVI').getInfo()
    except:
        data['avg_evi'] = None
        data['avg_ndvi'] = None
    
    try:
        # Global Forest Cover Change (GFCC) Tree Cover Multi-Year Global 30m
        # resolution: 30 m
        point = ee.Geometry.Point([lon, lat])
        data['tree_cover_pct'] = ee.ImageCollection('GLCF/GLS_TCC').mosaic().sample(point, 30).first().get('tree_canopy_cover').getInfo()
    except:
        data['tree_cover_pct'] = None
    
    # # WATER
    # try:
    #     # JRC Global Surface Water Mapping Layers, v1.4
    #     # resolution: from landsat
    #     data['water_occurrence_pct'] = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').sample(point).first().get('occurrence').getInfo()  # occurrence = % of time water present 1984-2021
    # except:
    #     from IPython import embed; embed(header="WATER OCCURENCE")
    #     data['water_occurrence_pct'] = None
    
    # HUMAN ACTIVITY
    try:
        # VIIRS Stray Light Corrected Nighttime Day/Night Band Composites Version 1
        lights = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').filterDate('2023-01-01', '2023-12-31').select('avg_rad').mean()  # avg_rad = average radiance
        data['nightlights'] = lights.sample(point).first().get('avg_rad').getInfo()
    except:
        data['nightlights'] = None
    
    return data


if __name__ == "__main__":
    locations = [
        (40.0150, -105.2705, "Boulder_CO"),
        (40.7128, -74.0060, "NYC"),
        (37.7749, -122.4194, "San_Francisco"),
        (47.6062, -122.3321, "Seattle")
    ]
    
    all_data = []
    for lat, lon, name in locations:
        data = query_point(lat, lon)
        data['location'] = name
        all_data.append(data)
    
    df = pd.DataFrame(all_data)
    cols = ['location'] + [c for c in df.columns if c != 'location']
    df = df[cols]
    
    df.to_csv('ee_data.csv', index=False)
    print(df.to_string(index=False))