import math
import sys
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

# YOLO model
model = YOLO("./models/yolo8s-batchsize(-1)-epoch150/runs/detect/train/weights/last.pt")

# Card counting values using Hi-Lo system
CARD_VALUES = {
    "2": 1,
    "3": 1,
    "4": 1,
    "5": 1,
    "6": 1,  # Low cards (positive count)
    "7": 0,
    "8": 0,
    "9": 0,  # Neutral cards
    "10": -1,
    "J": -1,
    "Q": -1,
    "K": -1,
    "A": -1,  # High cards (negative count)
}


class BlackjackCounter:
    def __init__(self):
        self.reset_count()

    def reset_count(self):
        self.running_count = 0
        self.cards_seen = set()
        self.cards_remaining = 52
        self.suit_counts = {suit: 0 for suit in ["c", "d", "h", "s"]}
        self.rank_counts = {
            rank: 0
            for rank in [
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "J",
                "Q",
                "K",
                "A",
            ]
        }
        print("Counter reset - New deck started")

    def get_card_value(self, card_name):
        rank = card_name[:-1]
        return CARD_VALUES.get(rank, 0)

    def is_valid_card(self, card_name):
        """Check if adding this card would violate single deck constraints"""
        rank = card_name[:-1]
        suit = card_name[-1]

        # Check if we've already seen this exact card
        if card_name in self.cards_seen:
            return False

        # Check if we've seen too many of this rank or suit
        if self.suit_counts[suit] >= 13:  # Max 13 cards per suit
            return False
        if self.rank_counts[rank] >= 4:  # Max 4 cards per rank
            return False

        return True

    def update_count(self, card_name):
        """Returns True if card was counted, False if it was a duplicate or invalid"""
        if not self.is_valid_card(card_name):
            return False

        self.cards_seen.add(card_name)
        count_value = self.get_card_value(card_name)
        self.running_count += count_value
        self.cards_remaining = 52 - len(self.cards_seen)

        # Update rank and suit counts
        rank = card_name[:-1]
        suit = card_name[-1]
        self.suit_counts[suit] += 1
        self.rank_counts[rank] += 1
        return True


def add_stats_to_frame(frame, counter):
    # Create a semi-transparent black rectangle for better text visibility
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Calculate needed height for all cards (assuming 2 rows of cards)
    num_cards = len(counter.cards_seen)
    box_height = 190 if num_cards > 0 else 150

    # Make background rectangle
    cv2.rectangle(overlay, (10, 10), (w - 10, box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Add text with stats
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)

    # Main stats
    cv2.putText(
        frame, f"Running Count: {counter.running_count}", (20, 40), font, 0.7, white, 2
    )
    cv2.putText(
        frame,
        f"Cards Remaining: {counter.cards_remaining}",
        (20, 70),
        font,
        0.7,
        white,
        2,
    )

    # Determine advantage status and color
    if counter.running_count >= 3:
        advantage_msg = "PLAYER ADVANTAGE"
        advantage_color = (0, 255, 0)  # Green
    elif counter.running_count <= -3:
        advantage_msg = "HOUSE ADVANTAGE"
        advantage_color = (0, 0, 255)  # Red
    else:
        advantage_msg = "NEUTRAL"
        advantage_color = white

    # Add advantage status with larger font and position
    cv2.putText(frame, advantage_msg, (w - 400, 50), font, 1.0, advantage_color, 3)

    # Show cards in array
    if counter.cards_seen:
        cv2.putText(frame, "Cards Seen:", (20, 100), font, 0.7, white, 2)
        cards = sorted(counter.cards_seen)
        cards_per_row = min(10, len(cards))  # Maximum 10 cards per row

        for i, card in enumerate(cards):
            row = i // cards_per_row
            col = i % cards_per_row
            x_pos = 20 + (col * 60)  # 60 pixels per card horizontally
            y_pos = 130 + (row * 30)  # 30 pixels between rows
            cv2.putText(frame, card, (x_pos, y_pos), font, 0.6, white, 2)

    return frame


def main():
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    # Set camera properties to match training resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer delay

    # Verify settings
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")

    counter = BlackjackCounter()
    previous_frame_cards = set()
    detection_cooldown = 0

    # Timing variables
    frame_count = 0
    start_time = cv2.getTickCount()
    processing_time = 0  # Track YOLO inference time

    print("Starting card detection. Press 'q' to quit or 'r' to reset count.")

    try:
        while True:
            loop_start = cv2.getTickCount()

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Resize frame if it's not 416x416
            if frame.shape[0] != 416 or frame.shape[1] != 416:
                frame = cv2.resize(frame, (416, 416))

            # Calculate FPS every second
            frame_count += 1
            if frame_count % 2 == 0:  # Update every 2 frames since we're at ~2 FPS
                current_time = cv2.getTickCount()
                elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
                fps = frame_count / elapsed_time

                # Reset counters if more than a second has passed
                if elapsed_time > 1.0:
                    frame_count = 0
                    start_time = current_time

            # Track inference start time
            inference_start = cv2.getTickCount()

            results = model(frame, stream=True)
            current_frame_cards = set()

            # Calculate inference time
            inference_end = cv2.getTickCount()
            processing_time = (inference_end - inference_start) / cv2.getTickFrequency()

            # Decrement cooldown
            if detection_cooldown > 0:
                detection_cooldown -= 1

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    if confidence >= 0.7:
                        cls = int(box.cls[0])
                        card_name = classNames[cls]
                        current_frame_cards.add(card_name)

            # Only process new cards if cooldown is 0
            if detection_cooldown == 0:
                new_cards = current_frame_cards - previous_frame_cards
                for card in new_cards:
                    if counter.update_count(card):
                        print(
                            f"New card counted: {card}, Running Count: {counter.running_count}"
                        )
                        detection_cooldown = (
                            1  # Reduced since we're already running at ~2 FPS
                        )

            previous_frame_cards = current_frame_cards

            # Add stats and timing info to frame
            frame_with_stats = add_stats_to_frame(frame, counter)
            cv2.putText(
                frame_with_stats,
                f"FPS: {fps:.1f} | Inference: {processing_time*1000:.0f}ms",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Blackjack Counter", frame_with_stats)

            # Calculate remaining time needed to maintain ~2 FPS
            loop_end = cv2.getTickCount()
            loop_time = (loop_end - loop_start) / cv2.getTickFrequency()
            wait_time = max(500, int((500 - loop_time) * 1000))

            key = cv2.waitKey(500) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            elif key == ord("r"):
                print("Resetting count...")
                counter.reset_count()
                previous_frame_cards.clear()
                detection_cooldown = 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print(f"Final processing time: {processing_time*1000:.0f}ms")
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()


def main():
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    # Try to set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Verify settings
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")

    counter = BlackjackCounter()
    previous_frame_cards = set()
    detection_cooldown = 0

    # Initialize timing variables
    fps = 0.0
    processing_time = 0.0
    frame_count = 0
    last_fps_update = cv2.getTickCount()

    print("Starting card detection. Press 'q' to quit or 'r' to reset count.")

    try:
        while True:
            frame_start = cv2.getTickCount()

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Pad the frame to square for YOLO (640x480 -> 640x640)
            h, w = frame.shape[:2]
            pad_size = abs(h - w) // 2
            if h < w:
                frame = cv2.copyMakeBorder(
                    frame,
                    pad_size,
                    pad_size,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            else:
                frame = cv2.copyMakeBorder(
                    frame,
                    0,
                    0,
                    pad_size,
                    pad_size,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )

            # Calculate FPS
            frame_count += 1
            current_time = cv2.getTickCount()
            elapsed = (current_time - last_fps_update) / cv2.getTickFrequency()

            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                last_fps_update = current_time

            # Run YOLO detection
            inference_start = cv2.getTickCount()
            results = model(frame, stream=True)
            inference_end = cv2.getTickCount()
            processing_time = (inference_end - inference_start) / cv2.getTickFrequency()

            current_frame_cards = set()

            # Process detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    if confidence >= 0.7:
                        cls = int(box.cls[0])
                        card_name = classNames[cls]
                        current_frame_cards.add(card_name)

            # Update card count if cooldown is finished
            if detection_cooldown == 0:
                new_cards = current_frame_cards - previous_frame_cards
                for card in new_cards:
                    if counter.update_count(card):
                        print(
                            f"New card counted: {card}, Running Count: {counter.running_count}"
                        )
                        detection_cooldown = (
                            1  # Set to 1 since we're already running slowly
                        )

            if detection_cooldown > 0:
                detection_cooldown -= 1

            previous_frame_cards = current_frame_cards

            # Add stats to frame
            frame_with_stats = add_stats_to_frame(frame, counter)

            # Add timing information
            timing_text = f"FPS: {fps:.1f} | Inference: {processing_time*1000:.0f}ms"
            cv2.putText(
                frame_with_stats,
                timing_text,
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Blackjack Counter", frame_with_stats)

            # Calculate appropriate wait time
            frame_end = cv2.getTickCount()
            frame_time = (frame_end - frame_start) / cv2.getTickFrequency()
            wait_time = max(1, int((0.5 - frame_time) * 1000))

            key = cv2.waitKey(2000) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            elif key == ord("r"):
                print("Resetting count...")
                counter.reset_count()
                previous_frame_cards.clear()
                detection_cooldown = 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        print(f"Final processing time: {processing_time*1000:.0f}ms")
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()


def main():
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    # Try to set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Verify settings
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")

    counter = BlackjackCounter()
    previous_frame_cards = set()

    print("Starting card detection.")
    print("Controls:")
    print("  'c' - Capture and detect cards")
    print("  'r' - Reset count")
    print("  'q' - Quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Pad the frame to square for YOLO
            h, w = frame.shape[:2]
            pad_size = abs(h - w) // 2
            if h < w:
                frame = cv2.copyMakeBorder(
                    frame,
                    pad_size,
                    pad_size,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            else:
                frame = cv2.copyMakeBorder(
                    frame,
                    0,
                    0,
                    pad_size,
                    pad_size,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )

            # Show live preview with stats but no detection
            frame_with_stats = add_stats_to_frame(frame, counter)
            cv2.putText(
                frame_with_stats,
                "Press 'c' to capture",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Blackjack Counter", frame_with_stats)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                # Capture and process frame
                print("\nCapturing frame...")

                # Run YOLO detection
                inference_start = cv2.getTickCount()
                results = model(frame, stream=True)
                inference_end = cv2.getTickCount()
                processing_time = (
                    inference_end - inference_start
                ) / cv2.getTickFrequency()

                print(f"Processing time: {processing_time*1000:.0f}ms")

                current_frame_cards = set()

                # Process detections
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        confidence = math.ceil((box.conf[0] * 100)) / 100
                        if confidence >= 0.7:
                            cls = int(box.cls[0])
                            card_name = classNames[cls]
                            current_frame_cards.add(card_name)

                # Update card count
                new_cards = current_frame_cards - previous_frame_cards
                cards_added = False
                for card in new_cards:
                    if counter.update_count(card):
                        print(
                            f"New card counted: {card}, Running Count: {counter.running_count}"
                        )
                        cards_added = True

                if not cards_added and len(current_frame_cards) > 0:
                    print("No new cards to add (already counted or invalid)")
                elif len(current_frame_cards) == 0:
                    print("No cards detected in frame")

                previous_frame_cards = current_frame_cards

            elif key == ord("r"):
                print("\nResetting count...")
                counter.reset_count()
                previous_frame_cards.clear()

            elif key == ord("q"):
                print("\nQuitting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Card classes
    classNames = [
        "10c",
        "10d",
        "10h",
        "10s",
        "2c",
        "2d",
        "2h",
        "2s",
        "3c",
        "3d",
        "3h",
        "3s",
        "4c",
        "4d",
        "4h",
        "4s",
        "5c",
        "5d",
        "5h",
        "5s",
        "6c",
        "6d",
        "6h",
        "6s",
        "7c",
        "7d",
        "7h",
        "7s",
        "8c",
        "8d",
        "8h",
        "8s",
        "9c",
        "9d",
        "9h",
        "9s",
        "Ac",
        "Ad",
        "Ah",
        "As",
        "Jc",
        "Jd",
        "Jh",
        "Js",
        "Kc",
        "Kd",
        "Kh",
        "Ks",
        "Qc",
        "Qd",
        "Qh",
        "Qs",
    ]
    main()
