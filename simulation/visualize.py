"""
Pygame main loop — ties together GridRenderer and Simulator.

Replays smart-home logs visually at configurable speed.

Usage:
    python -m simulation.visualize --log data/raw/synthetic_logs.jsonl
"""

import os
import sys
import argparse

import pygame

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from simulation.house_layout import HouseLayout, WINDOW_WIDTH, WINDOW_HEIGHT
from simulation.grid import GridRenderer
from simulation.simulator import Simulator
from utils.logging_utils import get_logger

logger = get_logger("visualize")

FPS = 30


def run_simulation(log_path: str, speed: float = 300.0):
    """Run the Pygame smart-home simulation.
    
    Args:
        log_path: Path to the .jsonl event log file.
        speed: Speed multiplier. Default 300 = 5 min sim per 1 sec real.
    """
    pygame.init()
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Smart Home Simulator")
    clock = pygame.time.Clock()
    
    # Initialize components
    layout = HouseLayout()
    renderer = GridRenderer(screen, layout)
    sim = Simulator(log_path)
    
    start_time, end_time = sim.get_time_range()
    logger.info(f"Simulation range: {start_time} -> {end_time}")
    logger.info(f"Total events: {len(sim.events)}")
    logger.info(f"Speed: {speed}x (press +/- to adjust)")
    
    running = True
    
    while running:
        delta_time = clock.tick(FPS) / 1000.0  # seconds since last frame
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS,
                                    pygame.K_KP_PLUS):
                    speed *= 2.0
                    speed = min(speed, 50000.0)
                    logger.info(f"Speed: {speed}x")
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed /= 2.0
                    speed = max(speed, 1.0)
                    logger.info(f"Speed: {speed}x")
                elif event.key == pygame.K_r:
                    # Reset simulation
                    sim = Simulator(log_path)
                    logger.info("Simulation reset")
        
        # Advance simulation
        if not sim.is_finished():
            sim.tick(delta_time, speed)
        
        # Get current state
        state = sim.get_state()
        current_time = state["current_time"]
        device_states = state["device_states"]
        
        # Draw everything
        renderer.draw_background()
        renderer.draw_room_labels()
        
        # Draw all devices
        for device_id in layout.get_all_device_ids():
            is_on = device_states.get(device_id, False)
            renderer.draw_device(device_id, is_on)
        
        # Draw HUD
        time_str = current_time.strftime("%A %Y-%m-%d %H:%M:%S")
        renderer.draw_hud(time_str, speed)
        
        # Draw legend
        renderer.draw_legend()
        
        # Draw progress bar at the bottom
        progress = sim.get_progress()
        bar_height = 4
        bar_y = WINDOW_HEIGHT - bar_height
        bar_width = int(WINDOW_WIDTH * progress)
        pygame.draw.rect(screen, (60, 60, 70),
                         (0, bar_y, WINDOW_WIDTH, bar_height))
        pygame.draw.rect(screen, (100, 180, 255),
                         (0, bar_y, bar_width, bar_height))
        
        # Show "FINISHED" overlay if done
        if sim.is_finished():
            font_big = pygame.font.SysFont("segoeui", 28, bold=True)
            done_text = font_big.render(
                "✅ Simulation Complete — Press R to restart, ESC to quit",
                True, (200, 255, 200))
            rect = done_text.get_rect(
                center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30))
            screen.blit(done_text, rect)
        
        pygame.display.flip()
    
    pygame.quit()
    logger.info("Simulation ended.")


def main(args=None):
    parser = argparse.ArgumentParser(description="Run smart home simulation")
    parser.add_argument("--log", type=str, required=True,
                        help="Path to .jsonl event log")
    parser.add_argument("--speed", type=float, default=300.0,
                        help="Speed multiplier (default: 300)")
    parsed = parser.parse_args(args)
    
    run_simulation(log_path=parsed.log, speed=parsed.speed)


if __name__ == "__main__":
    main()
