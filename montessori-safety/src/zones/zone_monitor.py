"""
Module 3: Zone-Based Boundary Monitoring.
Detects when tracked persons cross exit boundaries or enter danger zones.
Uses polygon intersection — no ML training needed.
"""

import cv2
import numpy as np
import json
from collections import defaultdict


class ZoneMonitor:
    """
    Monitors predefined zones in the camera view.
    Triggers alerts when tracked persons enter danger zones or cross exit boundaries.
    """
    
    def __init__(self, config_path=None):
        """
        Args:
            config_path: Path to JSON config with zone definitions.
                        If None, zones must be set manually or via the setup tool.
        """
        self.exit_boundaries = []     # List of polygons (exit lines/areas)
        self.danger_zones = []        # List of polygons (restricted areas)
        self.zone_names = {}          # {zone_id: name}
        
        # Track crossing state per person: {track_id: {'in_exit': bool, 'in_zones': set}}
        self.person_states = defaultdict(lambda: {'in_exit': False, 'in_zones': set()})
        
        # Temporal buffer: person must be in zone for N frames before alert
        self.TEMPORAL_BUFFER = 5
        self.zone_frame_counts = defaultdict(lambda: defaultdict(int))
        
        if config_path:
            self.load_config(config_path)
    
    def add_exit_boundary(self, polygon, name="Exit"):
        """
        Add an exit boundary polygon.
        
        Args:
            polygon: List of [x, y] points defining the boundary area
            name: Name for this exit
        """
        zone_id = f"exit_{len(self.exit_boundaries)}"
        self.exit_boundaries.append(np.array(polygon, dtype=np.int32))
        self.zone_names[zone_id] = name
    
    def add_danger_zone(self, polygon, name="Danger Zone"):
        """
        Add a restricted danger zone polygon.
        
        Args:
            polygon: List of [x, y] points defining the zone
            name: Name for this zone (e.g., "Kitchen", "Storage")
        """
        zone_id = f"danger_{len(self.danger_zones)}"
        self.danger_zones.append(np.array(polygon, dtype=np.int32))
        self.zone_names[zone_id] = name
    
    def check_person(self, track_id, bbox):
        """
        Check if a person is in any zone.
        
        Args:
            track_id: Person's tracking ID
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            list of alert dicts: [{'type': 'leaving'|'danger_zone', 'zone_name': str, 'confidence': float}]
        """
        # Compute centroid of bounding box (bottom-center is more accurate for "feet")
        x1, y1, x2, y2 = bbox
        centroid_x = (x1 + x2) // 2
        centroid_y = y2  # Bottom of bounding box (roughly feet position)
        point = (centroid_x, centroid_y)
        
        alerts = []
        
        # Check exit boundaries
        for i, boundary in enumerate(self.exit_boundaries):
            zone_id = f"exit_{i}"
            is_inside = cv2.pointPolygonTest(boundary, point, False) >= 0
            
            if is_inside:
                self.zone_frame_counts[track_id][zone_id] += 1
                
                if self.zone_frame_counts[track_id][zone_id] >= self.TEMPORAL_BUFFER:
                    if not self.person_states[track_id]['in_exit']:
                        self.person_states[track_id]['in_exit'] = True
                        alerts.append({
                            'type': 'leaving',
                            'activity': 'Leaving Classroom',
                            'zone_name': self.zone_names.get(zone_id, "Exit"),
                            'confidence': 0.95,
                            'person_id': track_id,
                            'location': point
                        })
            else:
                self.zone_frame_counts[track_id][zone_id] = 0
                if zone_id == f"exit_{i}":
                    self.person_states[track_id]['in_exit'] = False
        
        # Check danger zones
        for i, zone in enumerate(self.danger_zones):
            zone_id = f"danger_{i}"
            is_inside = cv2.pointPolygonTest(zone, point, False) >= 0
            
            if is_inside:
                self.zone_frame_counts[track_id][zone_id] += 1
                
                if self.zone_frame_counts[track_id][zone_id] >= self.TEMPORAL_BUFFER:
                    if zone_id not in self.person_states[track_id]['in_zones']:
                        self.person_states[track_id]['in_zones'].add(zone_id)
                        alerts.append({
                            'type': 'danger_zone',
                            'activity': 'Entering Danger Zone',
                            'zone_name': self.zone_names.get(zone_id, "Danger Zone"),
                            'confidence': 0.95,
                            'person_id': track_id,
                            'location': point
                        })
            else:
                self.zone_frame_counts[track_id][zone_id] = 0
                self.person_states[track_id]['in_zones'].discard(zone_id)
        
        return alerts
    
    def draw_zones(self, frame):
        """Draw all zones on frame for visualization."""
        # Draw exit boundaries in blue
        for boundary in self.exit_boundaries:
            cv2.polylines(frame, [boundary], True, (255, 100, 0), 2)
            # Fill with transparent overlay
            overlay = frame.copy()
            cv2.fillPoly(overlay, [boundary], (255, 100, 0))
            frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        
        # Draw danger zones in red
        for zone in self.danger_zones:
            cv2.polylines(frame, [zone], True, (0, 0, 255), 2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone], (0, 0, 255))
            frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        
        return frame
    
    def save_config(self, path):
        """Save zone configuration to JSON."""
        config = {
            'exit_boundaries': [b.tolist() for b in self.exit_boundaries],
            'danger_zones': [z.tolist() for z in self.danger_zones],
            'zone_names': self.zone_names
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, path):
        """Load zone configuration from JSON."""
        with open(path, 'r') as f:
            config = json.load(f)
        
        self.exit_boundaries = [np.array(b, dtype=np.int32) for b in config['exit_boundaries']]
        self.danger_zones = [np.array(z, dtype=np.int32) for z in config['danger_zones']]
        self.zone_names = config.get('zone_names', {})
    
    def cleanup(self, active_track_ids):
        """Remove state for persons no longer tracked."""
        old_ids = [tid for tid in self.person_states if tid not in active_track_ids]
        for tid in old_ids:
            del self.person_states[tid]
            if tid in self.zone_frame_counts:
                del self.zone_frame_counts[tid]