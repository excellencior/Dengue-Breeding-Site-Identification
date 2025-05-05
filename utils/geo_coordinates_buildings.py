import os
import pandas as pd
from pyproj import Proj, Transformer
from osgeo import gdal

def initialize_geotiff(tiff_path):
    """Initialize GeoTIFF file and extract transformation parameters.
    
    Args:
        tiff_path (str): Path to GeoTIFF file
    
    Returns:
        tuple: GDAL dataset and dictionary of transformation parameters
    """
    dataset = gdal.Open(os.path.abspath(tiff_path))
    geotransform = dataset.GetGeoTransform()
    return dataset, {
        'origin_x': geotransform[0],
        'origin_y': geotransform[3],
        'pixel_width': geotransform[1],
        'pixel_height': geotransform[5]
    }

def convert_to_geocoor(params, col, row):
    """Convert pixel coordinates to geographic coordinates.
    
    Args:
        params (dict): Transformation parameters
        col (int): Column number
        row (int): Row number
    
    Returns:
        tuple: Geographic x and y coordinates
    """
    x_geo = params['origin_x'] + col * params['pixel_width']
    y_geo = params['origin_y'] + row * params['pixel_height']
    return x_geo, y_geo

def reproject_coordinates(dataset, x, y):
    """Reproject coordinates from source CRS to WGS84.
    
    Args:
        dataset: GDAL dataset
        x (float): X coordinate
        y (float): Y coordinate
    
    Returns:
        tuple: Longitude and latitude
    """
    transformer = Transformer.from_proj(
        Proj(dataset.GetProjection()),
        Proj('epsg:4326')
    )
    return transformer.transform(x, y)

def calculate_risk_probability(row, accuracy):
    """Calculate risk probability for a building.
    
    Args:
        row (pd.Series): Row of building data
        accuracy (float): Accuracy parameter for risk calculation
    
    Returns:
        float: Risk probability
    """
    P_I = 1
    for class_name, det_accuracy in accuracy.items():
        n_i = row[class_name].values[0]
        P_I *= (1 - det_accuracy) ** n_i

    return 1 - P_I

def create_geolocation_risk_csv(tiff_path, input_csv_path, output_csv_path, accuracy):
    """Create CSV with geolocation and risk level for buildings.
    
    Args:
        tiff_path (str): Path to orthophoto GeoTIFF
        input_csv_path (str): Path to input CSV with building data
        output_csv_path (str): Path to save output CSV
        calculate_risk_probability (callable): Function to calculate risk
        accuracy (float): Accuracy parameter for risk calculation
    """
    tiff_path = os.path.abspath(tiff_path)
    input_csv_path = os.path.abspath(input_csv_path)
    output_csv_path = os.path.abspath(output_csv_path)
    
    dataset, params = initialize_geotiff(tiff_path)
    buildings_df = pd.read_csv(input_csv_path)[:-1]
    
    # Create geolocation DataFrame
    result_df = pd.DataFrame()
    result_df['Building ID'] = buildings_df['Building ID']
    result_df['Longitude'] = 0.0
    result_df['Latitude'] = 0.0
    result_df['Risk Level'] = 0.0
    
    for index, building_center in enumerate(buildings_df['Building ID']):
        # Calculate geolocation
        x, y, h, w, rot = map(int, building_center.split('_')[1:])
        x_geo, y_geo = convert_to_geocoor(params, x, y)
        longitude, latitude = reproject_coordinates(dataset, x_geo, y_geo)
        result_df.at[index, 'Longitude'] = longitude
        result_df.at[index, 'Latitude'] = latitude
        
        # Calculate risk level
        row = buildings_df.iloc[[index]]
        row = row.drop(columns=['Building ID', 'Object Count'])
        risk = calculate_risk_probability(row, accuracy)
        result_df.at[index, 'Risk Level'] = float(risk) * 100
    
    # Sort by risk level
    result_df = result_df.sort_values(by='Risk Level', ascending=False)
    result_df.to_csv(output_csv_path, index=False)