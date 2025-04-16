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
    try:
        parts = building_id.split('_')
        return {
            'center_x': float(parts[1]),
            'center_y': float(parts[2]),
            'width': float(parts[3]),
            'height': float(parts[4]),
            'rotation': float(parts[5])  # in degrees, counter-clockwise
        }
    except (IndexError, ValueError) as e:
        print(f"Error parsing building ID {building_id}: {str(e)}")
        return None

def get_corner_points(center_x, center_y, width, height, angle_deg):
    """
    Calculate corner points of a rotated rectangle.
    Returns points in order: top-left, top-right, bottom-right, bottom-left
    """
    angle_rad = np.deg2rad(angle_deg)
    
    half_width = width / 2
    half_height = height / 2
    
    corners = np.array([
        [-half_width, -half_height],
        [half_width, -half_height],
        [half_width, half_height],
        [-half_width, half_height]
    ])
    
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    rotated_corners = corners @ rotation_matrix.T
    rotated_corners[:, 0] += center_x
    rotated_corners[:, 1] += center_y
    
    return rotated_corners

def create_risk_heatmap(csv_path, image_width, image_height, output_path='riskmap.jpg', kernel_bandwidth=0.1):
    """
    Create and save a heatmap visualization of building risk levels.
    
    Parameters:
    csv_path: Path to CSV file containing building data
    image_width: Width of the output image in pixels
    image_height: Height of the output image in pixels
    output_path: Path where the output image will be saved
    kernel_bandwidth: Bandwidth for the kernel density estimation
    """
    # Load data
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} buildings from CSV")
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        return

    # Prepare building data
    building_data = []
    
    for _, row in df.iterrows():
        bbox_info = parse_building_id(row['Building ID'])
        if bbox_info is None:
            continue
            
        # Use the original pixel coordinates from the building ID
        center_x = bbox_info['center_x']
        center_y = bbox_info['center_y']
        width = bbox_info['width']
        height = bbox_info['height']
        rotation = bbox_info['rotation']
        
        # Get corner points
        corners = get_corner_points(center_x, center_y, width, height, rotation)
        
        # Add risk level
        building_data.append(np.concatenate([corners.flatten(), [row['Risk Level']]]))
    
    if not building_data:
        print("No valid building data found")
        return
        
    # Convert to numpy array
    buildings = np.array(building_data)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # Create a grid of points for the heatmap
    grid_size = 100
    x = np.linspace(0, image_width, grid_size)
    y = np.linspace(0, image_height, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Initialize heat array
    heat = np.zeros((grid_size, grid_size))
    
    # Process each building
    for building in buildings:
        coords = building[:-1].reshape(-1, 2)
        risk_level = building[-1]
        
        # Create polygon
        polygon = Polygon(coords, facecolor='none', edgecolor='black', alpha=0.7)
        ax.add_patch(polygon)
        
        # Calculate center for heat contribution
        center_x = coords[:, 0].mean()
        center_y = coords[:, 1].mean()
        
        # Contribute to heat map
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
    
    # Set axis labels and limits
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xlim(0, image_width)
    ax.set_ylim(0, image_height)
    
    ax.text(0.95, 0.95, '↑N', transform=ax.transAxes, 
            fontsize=12, fontweight='bold', ha='right', va='top')
    
    # Add scale bar
    scalebar_length = image_width * 0.03  # Reduced from 0.1 to 0.03
    ax.plot([100, 100 + scalebar_length], [60, 60], 'k-', linewidth=1)  # Reduced linewidth from 2 to 1
    ax.text(100 + scalebar_length/2, 80, f'{int(scalebar_length)}px',  # Reduced gap from 150 to 80
            ha='center', va='bottom', fontsize=8)  # Added smaller fontsize
    
    plt.tight_layout()
    
    # Save the figure
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {output_path}")
    except Exception as e:
        print(f"Error saving heatmap: {str(e)}")
    finally:
        plt.close()

# Example usage:
# create_risk_heatmap('path/to/your/csv', image_width=16460, image_height=14590, output_path='riskmap.jpg')