import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import math
import random
import csv
import pandas as pd

class FSTrackGenerator:
    def __init__(self, 
                 track_width=3.5,         # FSG: Min 3m
                 cone_spacing=4.5,        # FSG: Max 5m
                 num_control_points=15, 
                 randomness=0.3,
                 min_turn_radius=3.0):    # FSG: Min 3m radius
        """
        track_width: Distance between blue and yellow cones (meters)
        cone_spacing: Distance between cones along the track (meters)
        num_control_points: Complexity of the track shape
        randomness: How 'jagged' the track is (0.0 to 1.0)
        min_turn_radius: Minimum physical turning radius allowed
        """
        self.track_width = track_width
        self.cone_spacing = cone_spacing
        self.num_control_points = num_control_points
        self.randomness = randomness
        self.min_turn_radius = min_turn_radius
        
        # Lists now store [x, y, id]
        self.blue_cones = []   
        self.yellow_cones = [] 
        self.big_orange_cones = [] 
        self.center_line = []

    def _calculate_curvature_radius(self, tck, u_values):
        """Calculates radius of curvature at sampled points"""
        dx, dy = splev(u_values, tck, der=1)
        ddx, ddy = splev(u_values, tck, der=2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = np.power(dx**2 + dy**2, 1.5)
        curvature = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        radius = np.divide(1.0, curvature, out=np.full_like(curvature, np.inf), where=curvature!=0)
        return radius

    def generate(self, map_size=100, max_retries=100000):
        """Generates a valid FSG track using rejection sampling"""
        
        for attempt in range(max_retries):
            angles = np.linspace(0, 2 * np.pi, self.num_control_points, endpoint=False)
            radius = map_size * 0.45
            x_pts = []
            y_pts = []
            
            for angle in angles:
                r = radius + np.random.uniform(-radius*self.randomness, radius*self.randomness)
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                x_pts.append(x)
                y_pts.append(y)

            x_pts.append(x_pts[0])
            y_pts.append(y_pts[0])

            try:
                tck, u = splprep([x_pts, y_pts], s=0, per=True)
            except Exception:
                continue 

            u_check = np.linspace(0, 1, 1000)
            radii = self._calculate_curvature_radius(tck, u_check)
            
            if np.min(radii) < self.min_turn_radius:
                continue

            self._place_cones(tck)
            return

        print("Error: Could not generate a valid track within constraints.")

    def _place_cones(self, tck):
        """Places cones along the valid spline with IDs"""
        u_fine = np.linspace(0, 1, 2000)
        x_fine, y_fine = splev(u_fine, tck)
        dx, dy = splev(u_fine, tck, der=1)
        
        self.blue_cones = []
        self.yellow_cones = []
        
        # Centerline now stores [x, y, id]
        # 'id' for centerline corresponds to the sample index
        self.center_line = [[x, y, i] for i, (x, y) in enumerate(zip(x_fine, y_fine))]

        current_dist = 0
        cone_id_counter = 0
        
        for i in range(len(x_fine)):
            if i > 0:
                dist = np.sqrt((x_fine[i] - x_fine[i-1])**2 + (y_fine[i] - y_fine[i-1])**2)
                current_dist += dist
            
            if i == 0 or current_dist >= self.cone_spacing:
                current_dist = 0
                
                norm = np.sqrt(dx[i]**2 + dy[i]**2)
                nx = -dy[i] / norm
                ny = dx[i] / norm
                
                bx = x_fine[i] + nx * (self.track_width / 2)
                by = y_fine[i] + ny * (self.track_width / 2)
                
                yx = x_fine[i] - nx * (self.track_width / 2)
                yy = y_fine[i] - ny * (self.track_width / 2)

                # Append [x, y, id]
                self.blue_cones.append([bx, by, cone_id_counter])
                self.yellow_cones.append([yx, yy, cone_id_counter])
                cone_id_counter += 1

        if len(self.blue_cones) > 4:
            self.big_orange_cones = []
            # They inherit the ID from the blue/yellow cones they replaced
            self.big_orange_cones.append(self.blue_cones[0])
            self.big_orange_cones.append(self.yellow_cones[0])
            self.big_orange_cones.append(self.blue_cones[1])
            self.big_orange_cones.append(self.yellow_cones[1])
            
            self.blue_cones = self.blue_cones[2:]
            self.yellow_cones = self.yellow_cones[2:]

    def get_cones_in_fov(self, car_pos, car_heading, fov_angle_deg=100, max_range=15):
        """
        Returns: List of cones: [[x, y, id, type_string], ...]
        """
        visible_cones = []
        fov_rad_half = np.radians(fov_angle_deg) / 2.0
        
        all_cones = []
        for c in self.blue_cones: all_cones.append(c + ['blue'])
        for c in self.yellow_cones: all_cones.append(c + ['yellow'])
        for c in self.big_orange_cones: all_cones.append(c + ['big_orange'])
        
        for cone in all_cones:
            cx, cy, c_id, c_type = cone
            dx = cx - car_pos[0]
            dy = cy - car_pos[1]
            
            dist = np.sqrt(dx**2 + dy**2)
            if dist > max_range:
                continue
                
            angle_to_cone = np.arctan2(dy, dx)
            angle_diff = angle_to_cone - car_heading
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            
            if np.abs(angle_diff) <= fov_rad_half:
                visible_cones.append([cx, cy, c_id, c_type])
                
        return pd.DataFrame(visible_cones, columns=["x", "y", "id", "color"])

    def save_to_csv(self, filename="track_layout.csv"):
        """Saves track to CSV including the ID column"""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "id", "color"])
            for c in self.blue_cones: writer.writerow([c[0], c[1], c[2], "blue"])
            for c in self.yellow_cones: writer.writerow([c[0], c[1], c[2], "yellow"])
            for c in self.big_orange_cones: writer.writerow([c[0], c[1], c[2], "big_orange"])
            for c in self.center_line: writer.writerow([c[0], c[1], c[2], "centerline"])

    def load_from_dataframe(self, df):
        """Loads track layout from a pandas DataFrame, preserving IDs"""
        self.blue_cones = []
        self.yellow_cones = []
        self.big_orange_cones = []
        self.center_line = []
        
        # Ensure ID is integer if it exists, otherwise auto-generate not supported here
        # Assumes columns: x, y, id, color
        
        for _, row in df.iterrows():
            x = row['x']
            y = row['y']
            color = row['color']
            
            # Handle case where 'id' might not be in older CSVs
            c_id = int(row['id']) if 'id' in row else -1
            
            if color == 'blue':
                self.blue_cones.append([x, y, c_id])
            elif color == 'yellow':
                self.yellow_cones.append([x, y, c_id])
            elif color == 'big_orange':
                self.big_orange_cones.append([x, y, c_id])
            elif color == 'centerline':
                self.center_line.append([x, y, c_id])

    def plot(self, car_state=None, visible_cones=None):
        plt.figure(figsize=(12, 12))
        plt.axis('equal')
        
        # Slice [:, :2] to get just X and Y for plotting, ignoring ID
        b = np.array(self.blue_cones)
        y = np.array(self.yellow_cones)
        o = np.array(self.big_orange_cones)
        c = np.array(self.center_line)
        
        alpha_base = 0.3 if visible_cones else 1.0
        
        if len(c) > 0: plt.plot(c[:,0], c[:,1], 'k--', alpha=0.2, label='Centerline')
        if len(b) > 0: plt.scatter(b[:,0], b[:,1], c='blue', s=20, alpha=alpha_base, label='Blue')
        if len(y) > 0: plt.scatter(y[:,0], y[:,1], c='gold', s=20, alpha=alpha_base, label='Yellow')
        if len(o) > 0: plt.scatter(o[:,0], o[:,1], c='darkorange', s=80, marker='^', alpha=alpha_base, label='Gate')

        if car_state and visible_cones is not None:
            cx, cy, yaw = car_state['x'], car_state['y'], car_state['yaw']
            plt.arrow(cx, cy, np.cos(yaw)*2, np.sin(yaw)*2, head_width=1, color='red', label='Car')
            
            # visible_cones is now [x, y, id, type]
            # We need to parse it carefully for plotting
            vis_b = np.array([c for c in visible_cones if c[3] == 'blue'])
            vis_y = np.array([c for c in visible_cones if c[3] == 'yellow'])
            vis_o = np.array([c for c in visible_cones if c[3] == 'big_orange'])
            
            # Plot using first two columns (x,y)
            if len(vis_b) > 0: plt.scatter(vis_b[:,0], vis_b[:,1], c='blue', s=80, edgecolors='black')
            if len(vis_y) > 0: plt.scatter(vis_y[:,0], vis_y[:,1], c='gold', s=80, edgecolors='black')
            if len(vis_o) > 0: plt.scatter(vis_o[:,0], vis_o[:,1], c='darkorange', s=100, marker='^', edgecolors='black')

        plt.title(f"Track with IDs")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    gen = FSTrackGenerator(min_turn_radius=3.5)
    gen.generate()
    
    # Demo: Print IDs of first few cones
    print("First 3 Blue Cones (x, y, id):")
    for c in gen.blue_cones[:3]:
        print(c)
        
    # Demo FOV with IDs
    idx = len(gen.center_line) // 4
    car_x, car_y, _ = gen.center_line[idx] # unpack id too
    yaw = 1.57
    visible = gen.get_cones_in_fov((car_x, car_y), yaw)
    
    print(f"\nVisible Cones (x, y, id, type):")
    for v in visible[:3]: # Print first 3
        print(v)
        
    gen.plot(car_state={'x': car_x, 'y': car_y, 'yaw': yaw}, visible_cones=visible)