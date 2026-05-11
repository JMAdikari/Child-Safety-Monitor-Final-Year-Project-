"""
Interactive Zone Setup Tool.
Opens camera and lets you click to define exit boundaries and danger zones.

Usage:
    python src/zones/zone_setup.py --camera 0 --output configs/zones.json
"""

import cv2
import numpy as np
import argparse
import json


class ZoneSetupTool:
    def __init__(self, camera_source=0):
        self.cap = cv2.VideoCapture(camera_source)
        self.current_polygon = []
        self.exit_boundaries = []
        self.danger_zones = []
        self.mode = 'exit'  # 'exit' or 'danger'
        self.frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_polygon.append([x, y])
            print(f"  Point added: ({x}, {y})")
    
    def run(self, output_path):
        """Run interactive zone setup."""
        cv2.namedWindow("Zone Setup")
        cv2.setMouseCallback("Zone Setup", self.mouse_callback)
        
        # Capture a frame
        ret, self.frame = self.cap.read()
        if not ret:
            print("ERROR: Cannot read from camera")
            return
        
        print("\n" + "=" * 50)
        print("ZONE SETUP TOOL")
        print("=" * 50)
        print("\nInstructions:")
        print("  - Click to add polygon points")
        print("  - Press 'e' to switch to Exit boundary mode")
        print("  - Press 'd' to switch to Danger zone mode")
        print("  - Press 'n' to finish current polygon and start a new one")
        print("  - Press 'r' to reset current polygon")
        print("  - Press 's' to save and quit")
        print("  - Press 'q' to quit without saving")
        print(f"\nCurrent mode: {self.mode.upper()}")
        
        while True:
            display = self.frame.copy()
            
            # Draw existing zones
            for boundary in self.exit_boundaries:
                cv2.polylines(display, [np.array(boundary)], True, (255, 100, 0), 2)
                cv2.putText(display, "EXIT", tuple(boundary[0]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            
            for zone in self.danger_zones:
                cv2.polylines(display, [np.array(zone)], True, (0, 0, 255), 2)
                cv2.putText(display, "DANGER", tuple(zone[0]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw current polygon being defined
            if len(self.current_polygon) > 0:
                color = (255, 100, 0) if self.mode == 'exit' else (0, 0, 255)
                pts = np.array(self.current_polygon)
                for pt in pts:
                    cv2.circle(display, tuple(pt), 5, color, -1)
                if len(pts) > 1:
                    cv2.polylines(display, [pts], False, color, 2)
            
            # Show mode
            mode_text = f"Mode: {self.mode.upper()} | Points: {len(self.current_polygon)}"
            cv2.putText(display, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow("Zone Setup", display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('e'):
                self.mode = 'exit'
                print(f"Switched to EXIT boundary mode")
            elif key == ord('d'):
                self.mode = 'danger'
                print(f"Switched to DANGER zone mode")
            elif key == ord('n'):
                # Save current polygon
                if len(self.current_polygon) >= 3:
                    if self.mode == 'exit':
                        self.exit_boundaries.append(self.current_polygon.copy())
                        print(f"Exit boundary saved ({len(self.current_polygon)} points)")
                    else:
                        self.danger_zones.append(self.current_polygon.copy())
                        print(f"Danger zone saved ({len(self.current_polygon)} points)")
                    self.current_polygon = []
                else:
                    print("Need at least 3 points for a polygon")
            elif key == ord('r'):
                self.current_polygon = []
                print("Current polygon reset")
            elif key == ord('s'):
                # Save current polygon if active
                if len(self.current_polygon) >= 3:
                    if self.mode == 'exit':
                        self.exit_boundaries.append(self.current_polygon.copy())
                    else:
                        self.danger_zones.append(self.current_polygon.copy())
                
                # Save to file
                config = {
                    'exit_boundaries': self.exit_boundaries,
                    'danger_zones': self.danger_zones
                }
                with open(output_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"\nSaved {len(self.exit_boundaries)} exit boundaries and "
                      f"{len(self.danger_zones)} danger zones to {output_path}")
                break
            elif key == ord('q'):
                print("Quit without saving")
                break
        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', default=0, help='Camera index or video file')
    parser.add_argument('--output', default='configs/zones.json')
    args = parser.parse_args()
    
    tool = ZoneSetupTool(int(args.camera) if args.camera.isdigit() else args.camera)
    tool.run(args.output)