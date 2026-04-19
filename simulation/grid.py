"""
Render the grid, room outlines, room labels, and device icons using Pygame.

All drawing uses pygame primitives (rects, circles, text) — no image assets.
"""

import pygame
from simulation.house_layout import HouseLayout, TILE_SIZE


# Colors
COLOR_BG = (40, 40, 45)
COLOR_GRID_LINE = (60, 60, 65)
COLOR_ROOM_BORDER = (80, 80, 90)
COLOR_DEVICE_ON = (255, 220, 50)      # bright yellow
COLOR_DEVICE_OFF = (90, 90, 100)      # dark grey
COLOR_DEVICE_GLOW = (255, 240, 120, 80)  # glow around ON devices
COLOR_LABEL_ROOM = (180, 180, 190)
COLOR_LABEL_DEVICE = (160, 160, 170)
COLOR_HUD_BG = (30, 30, 35, 200)
COLOR_HUD_TEXT = (220, 220, 230)


class GridRenderer:
    """Renders the house grid, rooms, and device states onto a pygame surface."""
    
    def __init__(self, surface: pygame.Surface, layout: HouseLayout):
        self.surface = surface
        self.layout = layout
        
        # Initialize fonts
        pygame.font.init()
        self.font_room = pygame.font.SysFont("segoeui", 20, bold=True)
        self.font_device = pygame.font.SysFont("segoeui", 12)
        self.font_hud = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_legend = pygame.font.SysFont("segoeui", 14)
    
    def draw_background(self):
        """Draw room rectangles with muted fill colours and grid lines."""
        self.surface.fill(COLOR_BG)
        
        # Draw rooms
        for room_name in self.layout.ROOMS:
            x, y, w, h = self.layout.get_room_rect_pixels(room_name)
            color = self.layout.get_room_color(room_name)
            
            # Filled room rectangle
            pygame.draw.rect(self.surface, color, (x, y, w, h))
            # Room border
            pygame.draw.rect(self.surface, COLOR_ROOM_BORDER, (x, y, w, h), 2)
        
        # Draw subtle grid lines within rooms
        for room_name in self.layout.ROOMS:
            x, y, w, h = self.layout.get_room_rect_pixels(room_name)
            # Vertical grid lines
            for gx in range(x + TILE_SIZE, x + w, TILE_SIZE):
                pygame.draw.line(self.surface, COLOR_GRID_LINE,
                                 (gx, y), (gx, y + h), 1)
            # Horizontal grid lines
            for gy in range(y + TILE_SIZE, y + h, TILE_SIZE):
                pygame.draw.line(self.surface, COLOR_GRID_LINE,
                                 (x, gy), (x + w, gy), 1)
    
    def draw_room_labels(self):
        """Render room names in the centre of each room."""
        for room_name in self.layout.ROOMS:
            cx, cy = self.layout.get_room_center(room_name)
            # Offset label upward to make room for devices
            label_y = cy - 20
            
            # Format room name nicely
            display_name = room_name.replace("_", " ").title()
            text_surface = self.font_room.render(display_name, True, COLOR_LABEL_ROOM)
            text_rect = text_surface.get_rect(center=(cx, label_y))
            self.surface.blit(text_surface, text_rect)
    
    def draw_device(self, device_id: str, is_on: bool):
        """Draw a device icon (circle) at its position.
        
        Bright yellow if on, dark grey if off.
        Device name label drawn beneath.
        """
        x, y = self.layout.get_device_position(device_id)
        radius = 10
        
        if is_on:
            # Draw glow effect (larger, semi-transparent circle)
            glow_surface = pygame.Surface((radius * 6, radius * 6), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (255, 240, 120, 60),
                               (radius * 3, radius * 3), radius * 3)
            self.surface.blit(glow_surface,
                              (x - radius * 3, y - radius * 3))
            
            # Main circle
            pygame.draw.circle(self.surface, COLOR_DEVICE_ON, (x, y), radius)
            # White border for emphasis
            pygame.draw.circle(self.surface, (255, 255, 255), (x, y), radius, 2)
        else:
            # Off state
            pygame.draw.circle(self.surface, COLOR_DEVICE_OFF, (x, y), radius)
            pygame.draw.circle(self.surface, (70, 70, 80), (x, y), radius, 1)
        
        # Device name label
        name = self.layout.get_device_name(device_id)
        # Shorten long names
        short_name = name.replace(" Light", "💡").replace("Television", "📺")
        short_name = short_name.replace("Coffee Maker", "☕").replace("Thermostat", "🌡️")
        short_name = short_name.replace("Washing Machine", "🧺")
        
        label = self.font_device.render(short_name, True, COLOR_LABEL_DEVICE)
        label_rect = label.get_rect(center=(x, y + radius + 12))
        self.surface.blit(label, label_rect)
    
    def draw_hud(self, sim_time_str: str, speed: float):
        """Draw a HUD bar at the top showing current simulation time and speed."""
        hud_height = 36
        hud_rect = pygame.Rect(0, 0, self.surface.get_width(), hud_height)
        
        # Semi-transparent background
        hud_surface = pygame.Surface((hud_rect.width, hud_rect.height), pygame.SRCALPHA)
        hud_surface.fill((30, 30, 35, 220))
        self.surface.blit(hud_surface, (0, 0))
        
        # Time text
        time_text = self.font_hud.render(
            f"🕐 {sim_time_str}  |  Speed: {speed:.0f}x  |  +/- to adjust  |  ESC to quit",
            True, COLOR_HUD_TEXT
        )
        self.surface.blit(time_text, (12, 8))
    
    def draw_legend(self):
        """Draw ON/OFF legend in the bottom-right corner."""
        x_start = self.surface.get_width() - 160
        y_start = self.surface.get_height() - 50
        
        # Background
        legend_surface = pygame.Surface((150, 45), pygame.SRCALPHA)
        legend_surface.fill((30, 30, 35, 180))
        self.surface.blit(legend_surface, (x_start - 5, y_start - 5))
        
        # ON indicator
        pygame.draw.circle(self.surface, COLOR_DEVICE_ON,
                           (x_start + 10, y_start + 8), 7)
        on_label = self.font_legend.render("= ON", True, COLOR_HUD_TEXT)
        self.surface.blit(on_label, (x_start + 22, y_start))
        
        # OFF indicator
        pygame.draw.circle(self.surface, COLOR_DEVICE_OFF,
                           (x_start + 80, y_start + 8), 7)
        off_label = self.font_legend.render("= OFF", True, COLOR_HUD_TEXT)
        self.surface.blit(off_label, (x_start + 92, y_start))
