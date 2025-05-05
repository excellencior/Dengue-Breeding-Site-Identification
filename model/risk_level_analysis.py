import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
import cv2

def parse_building_id(building_id):
    """
    Parse building ID to extract bounding box information.
    Format: B_cx_cy_width_height_rotation
    """
    parts = building_id.split('_')
    return {
        'center_x': float(parts[1]),
        'center_y': float(parts[2]),
        'width': float(parts[3]),
        'height': float(parts[4]),
        'rotation': float(parts[5])  # in degrees, counter-clockwise
    }

def get_corner_points(center_x, center_y, width, height, angle_deg):
    """
    Calculate corner points of a rotated rectangle.
    Returns points in order: top-left, top-right, bottom-right, bottom-left
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_deg)
    
    # Calculate corner offsets from center
    half_width = width / 2
    half_height = height / 2
    
    # Create corner points before rotation (relative to center)
    corners = np.array([
        [-half_width, -half_height],  # top-left
        [half_width, -half_height],   # top-right
        [half_width, half_height],    # bottom-right
        [-half_width, half_height]    # bottom-left
    ])
    
    # Create rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # Rotate corners and add center offset
    rotated_corners = corners @ rotation_matrix.T
    rotated_corners[:, 0] += center_x
    rotated_corners[:, 1] += center_y
    
    return rotated_corners

def create_risk_heatmap(df, output_path='riskmap.jpg', kernel_bandwidth=0.1):
    """
    Create and save a heatmap visualization of building risk levels.
    
    Parameters:
    df: DataFrame with columns ['Building ID', 'Longitude', 'Latitude', 'Risk Level']
    output_path: Path where the image will be saved
    kernel_bandwidth: Bandwidth for the kernel density estimation
    """
    # Extract building information and convert to pixel coordinates
    buildings_data = []
    
    # Get coordinate ranges for normalization
    lon_min, lon_max = df['Longitude'].min(), df['Longitude'].max()
    lat_min, lat_max = df['Latitude'].min(), df['Latitude'].max()
    
    # Define image dimensions (in pixels)
    image_width = 1000
    image_height = int(image_width * ((lat_max - lat_min) / (lon_max - lon_min)))
    
    for _, row in df.iterrows():
        # Parse building ID
        bbox_info = parse_building_id(row['Building ID'])
        
        # Normalize coordinates to pixel space
        center_x = (row['Longitude'] - lon_min) / (lon_max - lon_min) * image_width
        center_y = (row['Latitude'] - lat_min) / (lat_max - lat_min) * image_height
        
        # Scale width and height to pixel space
        # Note: This scaling factor might need adjustment based on your data
        scale_factor = image_width / (lon_max - lon_min) * 0.0001
        width = bbox_info['width'] * scale_factor
        height = bbox_info['height'] * scale_factor
        
        # Get corner points
        corners = get_corner_points(center_x, center_y, width, height, bbox_info['rotation'])
        
        # Add to buildings data with risk level
        buildings_data.append(np.concatenate([corners.flatten(), [row['Risk Level']]])
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # Convert building data to numpy array
    buildings = np.array(buildings_data)
    
    # Create a grid of points for the heatmap
    grid_size = 100
    x = np.linspace(0, image_width, grid_size)
    y = np.linspace(0, image_height, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Initialize heat array
    heat = np.zeros((grid_size, grid_size))
    
    # For each building, create a polygon and contribute to the heat map
    for building in buildings:
        coords = building[:-1].reshape(-1, 2)
        risk_level = building[-1]
        
        # Create polygon
        polygon = Polygon(coords, facecolor='none', edgecolor='black', alpha=0.7)
        ax.add_patch(polygon)
        
        # Generate points within and around the building
        center_x = coords[:, 0].mean()
        center_y = coords[:, 1].mean()
        
        # Contribute to heat map based on risk level
        for i in range(grid_size):
            for j in range(grid_size):
                distance = np.sqrt((xx[i,j] - center_x)**2 + (yy[i,j] - center_y)**2)
                heat[i,j] += risk_level * np.exp(-distance**2 / (2 * (kernel_bandwidth * image_width)**2))
    
    # Normalize heat map
    heat = heat / heat.max()
    
    # Create the heatmap
    im = ax.imshow(heat, extent=[0, image_width, image_height, 0], cmap='RdYlBu_r', alpha=0.7)
    
    # Customize the plot
    ax.set_title('Building Risk Level Heatmap')
    plt.colorbar(im, label='Risk Level')
    
    # Set axis labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Add north arrow
    ax.text(0.95, 0.95, '↑N', transform=ax.transAxes, 
            fontsize=12, fontweight='bold', ha='right', va='top')
    
    # Add scale bar (approximate)
    scalebar_length = image_width * 0.1
    lon_diff = (lon_max - lon_min) * 0.1
    ax.plot([100, 100 + scalebar_length], [100, 100], 'k-', linewidth=2)
    ax.text(100 + scalebar_length/2, 150, f'{lon_diff:.4f}°',
            ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Example usage:
# df = pd.read_csv('building_data.csv')
# create_risk_heatmap(df, 'risk_heatmap.jpg')