# Evaluation Datasets

This directory contains scripts to generate two evaluation datasets (for SAE and SpLiCE methods):

- **Dense grid**: 100,000 points sampled uniformly at random across the world's landmasses
- **GeoYFCC subset**: 100,000 points drawn from the GeoYFCC dataset, representing a more urban-centric evaluation set. GeoYFCC is also one of the datasets used to evaluate GeoCLIP.

## Creating the dense grid

1. Download the Natural Earth Low resolution shapefile `ne_110m_admin_0_countries.shp` from [naturalearthdata.com](https://www.naturalearthdata.com)
2. Run:
   ```
   python create_dense_grid.py --shapefile /path/to/ne_110m_admin_0_countries.shp --out /out/path/dense_grid.csv
   ```

## Creating the urban-centric evaluation set

1. Download the GeoYFCC dataset following the instructions at https://github.com/abhimanyudubey/GeoYFCC (untar the file to get `Geo-YFCC.pkl`)
2. Run:
   ```
   python create_geoyfcc_subset.py --geoyfcc_path /path/to/Geo-YFCC.pkl --out /out/path/geoyfcc_subset.csv
   ```
