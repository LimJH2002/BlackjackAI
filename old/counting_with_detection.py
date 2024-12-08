import math
import cv2
from ultralytics import YOLO
import numpy as np
from mss import mss
import win32gui
import win32con
import win32ui
import win32api
from collections import defaultdict

# YOLO model
model = YOLO("./runs/detect/train3/weights/last.pt")

# Card counting values using Hi-Lo system (ideal for single-deck games)
CARD_VALUES = {
    '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,  # Low cards (positive count)
    '7': 0, '8': 0, '9': 0,                   # Neutral cards
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1  # High cards (negative count)
}

class BlackjackCounter:
    def __init__(self):
        self.reset_count()
    
    def reset_count(self):
        self.running_count = 0
        self.cards_seen = set()
        self.cards_remaining = 52
        print("Counter reset - New deck started")
    
    def get_card_value(self, card_name):
        rank = card_name[:-1]
        return CARD_VALUES.get(rank, 0)
    
    def update_count(self, card_name):
        """Returns True if card was counted, False if it was a duplicate"""
        if card_name in self.cards_seen:
            return False
            
        self.cards_seen.add(card_name)
        count_value = self.get_card_value(card_name)
        self.running_count += count_value
        self.cards_remaining = 52 - len(self.cards_seen)
        return True
    
    def get_count_info(self):
        # Count remaining high and low cards
        high_cards_seen = sum(1 for card in self.cards_seen 
                            if card[:-1] in ['10', 'J', 'Q', 'K', 'A'])
        low_cards_seen = sum(1 for card in self.cards_seen 
                           if card[:-1] in ['2', '3', '4', '5', '6'])
        
        return {
            'running_count': self.running_count,
            'cards_remaining': self.cards_remaining,
            'high_cards_remaining': 20 - high_cards_seen,
            'low_cards_remaining': 20 - low_cards_seen,
            'cards_seen': sorted(list(self.cards_seen)) 
        }

def create_count_overlay(count_info):
    # Constants for layout
    WIDTH = 400
    BASE_HEIGHT = 280
    CARD_WIDTH = 60  # Width allocated for each card text
    CARDS_PER_ROW = WIDTH // CARD_WIDTH  # Number of cards that fit in one row
    
    # Calculate number of rows needed for cards
    num_cards = len(count_info['cards_seen'])
    num_rows = (num_cards + CARDS_PER_ROW - 1) // CARDS_PER_ROW  # Ceiling division
    
    # Calculate total height needed
    total_height = BASE_HEIGHT + (num_rows * 25)
    
    overlay = np.zeros((total_height, WIDTH, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Main stats
    cv2.putText(overlay, f"Running Count: {count_info['running_count']}", 
                (10, 40), font, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"Cards Remaining: {count_info['cards_remaining']}", 
                (10, 80), font, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"High Cards Left: {count_info['high_cards_remaining']}", 
                (10, 120), font, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"Low Cards Left: {count_info['low_cards_remaining']}", 
                (10, 160), font, 0.7, (255, 255, 255), 2)
    
    # Advantage indicator
    if count_info['running_count'] >= 3 and count_info['cards_remaining'] < 40:
        advantage = "Player Advantage"
        color = (0, 255, 0) 
    elif count_info['running_count'] <= -3:
        advantage = "House Advantage"
        color = (0, 0, 255)  
    else:
        advantage = "Neutral"
        color = (255, 255, 255)  
    
    cv2.putText(overlay, f"Advantage: {advantage}", 
                (10, 200), font, 0.7, color, 2)
    
    if num_cards > 0:
        cv2.putText(overlay, "Cards Seen:", 
                    (10, 240), font, 0.7, (255, 255, 255), 2)
        
        cards = sorted(count_info['cards_seen'])
        for i, card in enumerate(cards):
            row = i // CARDS_PER_ROW
            col = i % CARDS_PER_ROW
            x_pos = 10 + (col * CARD_WIDTH)
            y_pos = 270 + (row * 25)
            cv2.putText(overlay, card, 
                       (x_pos, y_pos), font, 0.6, (200, 200, 200), 1)
    
    return overlay

def main():
    # Screen capture setup
    sct = mss()
    monitor = sct.monitors[1]
    
    # Window setup for drawing
    hwnd = win32gui.GetDesktopWindow()
    hdc = win32gui.GetWindowDC(hwnd)
    hdc_draw = win32ui.CreateDCFromHandle(hdc)
    pen = win32ui.CreatePen(win32con.PS_SOLID, 3, win32api.RGB(255, 0, 255))
    hdc_draw.SelectObject(pen)
    
    # Initialize counter
    counter = BlackjackCounter()
    
    # Create named window for overlay
    cv2.namedWindow('Card Count Overlay', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Card Count Overlay', monitor['left'], monitor['top'])
    
    # Track cards in current frame
    previous_frame_cards = set()
    
    try:
        while True:
            # Capture screen
            screenshot = np.array(sct.grab(monitor))
            img = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            # Run detection
            results = model(img, stream=True)
            
            # Current frame's cards
            current_frame_cards = set()
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get detection info
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    
                    # Only process high-confidence detections
                    if confidence >= 0.7:
                        cls = int(box.cls[0])
                        card_name = classNames[cls]
                        
                        # Add to current frame's cards
                        current_frame_cards.add(card_name)
                        
                        # Draw bounding box
                        hdc_draw.MoveTo(x1, y1)
                        hdc_draw.LineTo(x1, y2)
                        hdc_draw.LineTo(x2, y2)
                        hdc_draw.LineTo(x2, y1)
                        hdc_draw.LineTo(x1, y1)
                        
                        # Set color based on whether card has been seen
                        if card_name in counter.cards_seen:
                            # Red for duplicates
                            label = f"{card_name} (COUNTED)"
                            hdc_draw.SetTextColor(win32api.RGB(255, 0, 0))
                        else:
                            # Green for new cards
                            label = f"{card_name} NEW"
                            hdc_draw.SetTextColor(win32api.RGB(0, 255, 0))
                        
                        hdc_draw.TextOut(x1 + 5, y1 - 20, label)
            
            # Update count for new cards only
            new_cards = current_frame_cards - previous_frame_cards
            for card in new_cards:
                counter.update_count(card)
            
            # Update previous frame cards
            previous_frame_cards = current_frame_cards
            
            # Create and show overlay
            count_info = counter.get_count_info()
            overlay = create_count_overlay(count_info)
            cv2.imshow('Card Count Overlay', overlay)
            
            # Check for keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                counter.reset_count()
                previous_frame_cards.clear()
                
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        win32gui.ReleaseDC(hwnd, hdc)
        hdc_draw.DeleteDC()

if __name__ == "__main__":
    # Card classes from your original code
    classNames = [
        "10c", "10d", "10h", "10s", "2c", "2d", "2h", "2s", "3c", "3d", "3h", "3s",
        "4c", "4d", "4h", "4s", "5c", "5d", "5h", "5s", "6c", "6d", "6h", "6s",
        "7c", "7d", "7h", "7s", "8c", "8d", "8h", "8s", "9c", "9d", "9h", "9s",
        "Ac", "Ad", "Ah", "As", "Jc", "Jd", "Jh", "Js", "Kc", "Kd", "Kh", "Ks",
        "Qc", "Qd", "Qh", "Qs"
    ]
    main()