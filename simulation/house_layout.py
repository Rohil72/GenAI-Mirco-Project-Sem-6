"""
Define the 2D grid layout of rooms and device positions for Pygame simulation.

Grid: 20x15 tiles, TILE_SIZE = 48px, Window = 960x720.
"""

import os
import sys
import yaml

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

TILE_SIZE = 48
GRID_COLS = 20
GRID_ROWS = 15
WINDOW_WIDTH = GRID_COLS * TILE_SIZE   # 960
WINDOW_HEIGHT = GRID_ROWS * TILE_SIZE  # 720


class HouseLayout:
    """2D grid layout of rooms and device positions.
    
    Room rectangles are defined in tile coordinates (x, y, w, h).
    Device positions are pixel coordinates within their rooms.
    """
    
    # Room definitions: name -> (x, y, w, h) in tiles
    ROOMS = {
        "living_room": (0, 0, 9, 8),
        "kitchen":     (10, 0, 9, 6),
        "bathroom":    (10, 7, 4, 8),
        "bedroom":     (0, 9, 9, 6),
        "hallway":     (14, 7, 5, 4),
        "utility":     (14, 11, 5, 4),
    }
    
    # Room display colors (muted, low-saturation)
    ROOM_COLORS = {
        "living_room": (235, 222, 200),   # warm beige
        "kitchen":     (220, 235, 210),   # soft green
        "bathroom":    (200, 218, 235),   # cool blue
        "bedroom":     (230, 215, 235),   # lavender
        "hallway":     (225, 225, 215),   # light grey-beige
        "utility":     (215, 220, 225),   # steel grey
    }
    
    def __init__(self):
        """Load devices from config and assign positions."""
        devices_path = os.path.join(_BASE_DIR, "config", "devices.yaml")
        with open(devices_path) as f:
            config = yaml.safe_load(f)
        
        self.devices = config["devices"]
        self.device_map = {d["id"]: d for d in self.devices}
        
        # Assign pixel positions for each device within its room
        # Positions are absolute pixel coords on the window
        self._device_positions = self._compute_device_positions()
    
    def _compute_device_positions(self) -> dict:
        """Compute sensible pixel positions for devices within their rooms."""
        positions = {}
        
        # Group devices by room
        room_devices = {}
        for dev in self.devices:
            room = dev["room"]
            if room not in room_devices:
                room_devices[room] = []
            room_devices[room].append(dev["id"])
        
        for room_name, device_ids in room_devices.items():
            if room_name not in self.ROOMS:
                continue
            
            rx, ry, rw, rh = self.ROOMS[room_name]
            # Convert to pixels
            px = rx * TILE_SIZE
            py = ry * TILE_SIZE
            pw = rw * TILE_SIZE
            ph = rh * TILE_SIZE
            
            # Distribute devices evenly within the room
            n = len(device_ids)
            for i, dev_id in enumerate(device_ids):
                # Place devices in a row, centered vertically
                margin_x = pw * 0.15
                margin_y = ph * 0.35
                usable_w = pw - 2 * margin_x
                
                if n == 1:
                    x = px + pw // 2
                else:
                    x = int(px + margin_x + (usable_w * i / (n - 1)))
                
                y = int(py + ph - margin_y)
                positions[dev_id] = (x, y)
        
        return positions
    
    def get_device_position(self, device_id: str) -> tuple[int, int]:
        """Return the (x, y) pixel position of a device."""
        return self._device_positions.get(device_id, (0, 0))
    
    def get_device_rect(self, device_id: str):
        """Return a pygame.Rect for the device's bounding box.
        
        Returns a tuple (x, y, w, h) to avoid importing pygame at module level.
        The actual pygame.Rect is created by the caller.
        """
        x, y = self.get_device_position(device_id)
        size = 20  # device icon diameter
        return (x - size // 2, y - size // 2, size, size)
    
    def get_room_rect_pixels(self, room_name: str) -> tuple[int, int, int, int]:
        """Return (x, y, w, h) in pixels for a room."""
        rx, ry, rw, rh = self.ROOMS[room_name]
        return (rx * TILE_SIZE, ry * TILE_SIZE, rw * TILE_SIZE, rh * TILE_SIZE)
    
    def get_room_center(self, room_name: str) -> tuple[int, int]:
        """Return the (x, y) pixel center of a room."""
        rx, ry, rw, rh = self.ROOMS[room_name]
        cx = (rx + rw / 2) * TILE_SIZE
        cy = (ry + rh / 2) * TILE_SIZE
        return (int(cx), int(cy))
    
    def get_room_for_device(self, device_id: str) -> str:
        """Return the room name for a device."""
        dev = self.device_map.get(device_id)
        return dev["room"] if dev else None
    
    def get_all_device_ids(self) -> list[str]:
        """Return all device IDs."""
        return [d["id"] for d in self.devices]
    
    def get_room_color(self, room_name: str) -> tuple[int, int, int]:
        """Return the RGB fill color for a room."""
        return self.ROOM_COLORS.get(room_name, (220, 220, 220))
    
    def get_device_name(self, device_id: str) -> str:
        """Return the human-readable name for a device."""
        dev = self.device_map.get(device_id)
        return dev["name"] if dev else device_id
