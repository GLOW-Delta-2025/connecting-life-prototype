import cv2
import numpy as np
from collections import defaultdict


def track_heads_dual_view(video_path, threshold=200, min_area=50, proximity_threshold=20, window_scale=0.7):
    """
    Track heads in a heat camera video with improved clustering and temporal smoothing
    to eliminate multiple detections and create smoother tracking. Shows in a smaller window.

    Args:
        video_path (str): Path to the heat camera video file
        threshold (int): Brightness threshold for hot spot detection (0-255)
        min_area (int): Minimum pixel area to consider as a head
        proximity_threshold (int): Maximum distance between points to be considered the same head
        window_scale (float): Scale factor for the display window (0.0-1.0)

    Returns:
        None
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a larger canvas for the side-by-side view
    dual_width = frame_width * 2
    dual_height = frame_height

    # Calculate display size
    display_width = int(dual_width * window_scale)
    display_height = int(dual_height * window_scale)

    # Dictionary to store tracking history for each detected head
    # Maps track_id -> list of recent positions [(x, y), (x, y), ...]
    head_tracks = {}
    # Dictionary to keep track of how many frames a track has been missing
    track_missing_count = {}
    next_track_id = 0
    max_track_history = 5  # Number of frames to keep for smoothing

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert to grayscale if it's not already
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Apply thresholding to identify hot spots (heads)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank map for the dots
        dot_map = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # List to store all detected head positions
        all_positions = []

        # Process each contour
        for contour in contours:
            # Filter out small areas that might be noise
            if cv2.contourArea(contour) < min_area:
                continue

            # Get the center of the contour (head)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Store center and heat value (using the average pixel value in the contour)
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                mean_val = cv2.mean(gray, mask=mask)[0]
                all_positions.append((cx, cy, mean_val))

        # Group nearby points using a clustering approach
        grouped_points = defaultdict(list)
        assigned = set()

        # Sort points by heat value (descending) to prioritize hotter spots
        all_positions.sort(key=lambda x: x[2], reverse=True)

        for i, (x, y, val) in enumerate(all_positions):
            if i in assigned:
                continue

            # Create a new group starting with this point
            group_id = len(grouped_points)
            grouped_points[group_id].append((x, y, val))
            assigned.add(i)

            # Find all points within proximity_threshold
            for j, (x2, y2, val2) in enumerate(all_positions):
                if j in assigned:
                    continue

                # Calculate distance between points
                dist = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
                if dist <= proximity_threshold:
                    grouped_points[group_id].append((x2, y2, val2))
                    assigned.add(j)

        # Calculate the current frame's positions (before applying smoothing)
        current_positions = []
        for group in grouped_points.values():
            if not group:
                continue

            # If only one point in group, use it
            if len(group) == 1:
                current_positions.append((group[0][0], group[0][1]))
                continue

            # Find the center of mass of the group
            avg_x = sum(p[0] for p in group) / len(group)
            avg_y = sum(p[1] for p in group) / len(group)

            # Find the point closest to the center of mass
            closest_point = min(group, key=lambda p: (p[0] - avg_x) ** 2 + (p[1] - avg_y) ** 2)
            current_positions.append((closest_point[0], closest_point[1]))

        # Match current detections with existing tracks
        matched_tracks = set()  # Set of track IDs that were matched
        unmatched_detections = list(current_positions)  # List of positions that weren't matched

        # For each existing track, find the closest current detection
        if head_tracks:  # Only if we have existing tracks
            for track_id, positions in head_tracks.items():
                if not positions:  # Skip empty tracks
                    continue

                last_pos = positions[-1]
                best_match_idx = -1
                min_dist = float('inf')

                # Find the closest detection to this track
                for i, pos in enumerate(current_positions):
                    dist = np.sqrt((last_pos[0] - pos[0]) ** 2 + (last_pos[1] - pos[1]) ** 2)
                    # Use a reasonable maximum distance for matching
                    max_match_dist = proximity_threshold * 1.5  # Allow more distance for matching
                    if dist < min_dist and dist < max_match_dist:
                        min_dist = dist
                        best_match_idx = i

                # If we found a match, update the track
                if best_match_idx >= 0:
                    matched_position = current_positions[best_match_idx]
                    head_tracks[track_id].append(matched_position)
                    # Keep only the most recent positions for smoothing
                    if len(head_tracks[track_id]) > max_track_history:
                        head_tracks[track_id] = head_tracks[track_id][-max_track_history:]
                    matched_tracks.add(track_id)
                    track_missing_count[track_id] = 0  # Reset missing count

                    # Remove this detection from unmatched list
                    if matched_position in unmatched_detections:
                        unmatched_detections.remove(matched_position)

        # Create new tracks for unmatched detections
        for pos in unmatched_detections:
            track_id = next_track_id
            head_tracks[track_id] = [pos]
            track_missing_count[track_id] = 0
            next_track_id += 1

        # For tracks that weren't matched, increment their missing count
        for track_id in list(head_tracks.keys()):
            if track_id not in matched_tracks:
                track_missing_count[track_id] = track_missing_count.get(track_id, 0) + 1

        # Remove tracks that have been missing for too many frames
        max_missing_frames = 15  # Increased from original to be more lenient
        for track_id in list(track_missing_count.keys()):
            if track_missing_count[track_id] > max_missing_frames:
                if track_id in head_tracks:
                    del head_tracks[track_id]
                del track_missing_count[track_id]

        # Draw the smoothed positions on the map and original frame
        active_tracks = 0
        for track_id, positions in head_tracks.items():
            if not positions:
                continue

            active_tracks += 1

            # Apply smoothing by averaging the last N positions
            smoothed_x = sum(p[0] for p in positions) / len(positions)
            smoothed_y = sum(p[1] for p in positions) / len(positions)

            # Round to integers for drawing
            cx, cy = int(smoothed_x), int(smoothed_y)

            # Draw a filled circle (dot) on the map
            cv2.circle(dot_map, (cx, cy), 10, (0, 0, 255), -1)

            # Draw a small indicator on the original frame too
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            # Draw track ID for debugging
            cv2.putText(dot_map, str(track_id), (cx + 12, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Create the dual view
        dual_view = np.zeros((dual_height, dual_width, 3), dtype=np.uint8)
        dual_view[:, :frame_width] = frame
        dual_view[:, frame_width:] = dot_map

        # Add labels
        cv2.putText(dual_view, "Original Video", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(dual_view, f"Head Position Map ({active_tracks} tracks)", (frame_width + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Resize for display
        display_view = cv2.resize(dual_view, (display_width, display_height))

        # Display the dual view
        cv2.imshow('Dual View', display_view)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    print("Processing complete")


if __name__ == "__main__":
    video_file = "../infrared.mkv"

    # You may need to adjust these parameters based on your specific video
    track_heads_dual_view(
        video_path=video_file,
        threshold=100,  # Adjust based on how hot the heads appear in your video
        min_area=50,  # Adjust based on how big the head signatures are in your video
        proximity_threshold=150,  # Maximum distance between points to be considered the same head
        window_scale=0.7  # Adjust this value to change window size (0.5 = half size, 1.0 = full size)
    )