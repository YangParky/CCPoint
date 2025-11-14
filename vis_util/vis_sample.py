import numpy as np
import open3d as o3d
import os

# Set the base path
base_dir = r'./vis_util/samples/'

# Get a list of all .ply files in the directory
ply_files = [f for f in os.listdir(base_dir) if f.endswith('.ply')]

# Loop through each .ply file
for ply_file in ply_files:
    # Construct the full file path
    base_path = os.path.join(base_dir, ply_file)

    # Construct the save path
    save_path = os.path.join(base_dir, ply_file.replace('.ply', '.png'))

    # Read the point cloud data
    pc = o3d.io.read_point_cloud(base_path)

    # Calculate distances from each point to the reference point (centroid)
    centroid = np.mean(np.asarray(pc.points), axis=0)
    distances = np.linalg.norm(np.asarray(pc.points) - centroid, axis=1)

    # Define thresholds for different parts
    threshold1 = np.min(distances) + (np.max(distances) - np.min(distances)) / 3
    threshold2 = np.min(distances) + 2 * (np.max(distances) - np.min(distances)) / 3

    # Assign colors based on distance thresholds
    colors = np.zeros((len(pc.points), 3))
    colors[distances <= threshold1, :] = [0.8, 0.0, 0.0]  # Deep red for part 1
    colors[distances > threshold1 & distances <= threshold2, :] = [0.0, 0.8, 0.0]  # Green for part 2
    colors[distances > threshold2, :] = [0.0, 0.0, 1.0]  # Blue for part 3

    # Assign colors to the point cloud
    pc.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the colored point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=800)
    vis.add_geometry(pc)
    vis.run()
    vis.capture_screen_image(save_path)
    vis.destroy_window()