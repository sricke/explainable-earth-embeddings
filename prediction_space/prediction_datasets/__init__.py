from .elevation import ElevationDataset
from .temperature import TemperatureDataset
from .ndvi import NDVIDataset
from .population import PopulationDataset
from .nightlights import NightlightsDataset
from .precipitation import PrecipitationDataset
from .soil_carbon import SoilCarbonDataset
from .land_cover import LandCoverDataset
from .biome import BiomeDataset
from .ecoregion import EcoregionDataset
from .climate_zone import ClimateZoneDataset

__all__ = [
    "ElevationDataset",
    "TemperatureDataset",
    "NDVIDataset",
    "PopulationDataset",
    "NightlightsDataset",
    "PrecipitationDataset",
    "SoilCarbonDataset",
    "LandCoverDataset",
    "BiomeDataset",
    "EcoregionDataset",
    "ClimateZoneDataset",
]
