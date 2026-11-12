"""
ScaleCalibrator - Manages scale calibration workflow for accurate measurements.
"""

import numpy as np


class ScaleCalibrator:
    """
    Handles scale calibration for converting pixels to real-world measurements.

    Scale calibration allows users to click two points on a known distance
    (e.g., a ruler, grid line) to establish precise conversion factors
    instead of relying on DPI metadata.
    """

    def __init__(self):
        """Initialize the scale calibrator"""
        self.mode = None  # None, "original", or "result"
        self.points = []  # Two points defining the scale line
        self.length = None  # Real-world length in current units
        self.factor = 1.0  # Pixels per unit

    def is_active(self):
        """Check if calibration is currently in progress"""
        return self.mode is not None

    def get_mode(self):
        """Get current calibration mode ('original', 'result', or None)"""
        return self.mode

    def get_points(self):
        """Get the two calibration points"""
        return self.points

    def get_point_count(self):
        """Get number of points placed"""
        return len(self.points)

    def is_calibrated(self):
        """Check if scale has been successfully calibrated"""
        return self.length is not None and self.factor != 1.0

    def get_scale_factor(self):
        """Get the calibrated scale factor (pixels per unit)"""
        return self.factor

    def get_scale_length(self):
        """Get the calibrated real-world length"""
        return self.length

    def start_calibration(self, mode):
        """
        Start scale calibration process.

        Args:
            mode: Either "original" (calibrate on original image) or
                  "result" (calibrate on transformed result)
        """
        if mode not in ["original", "result"]:
            raise ValueError("Mode must be 'original' or 'result'")

        self.mode = mode
        self.points = []

    def add_point(self, x, y):
        """
        Add a calibration point.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            bool: True if this completes the calibration (2 points), False otherwise
        """
        if self.mode is None:
            return False

        if len(self.points) < 2:
            self.points.append((float(x), float(y)))

        return len(self.points) == 2

    def calculate_pixel_distance(self):
        """
        Calculate the pixel distance between the two calibration points.

        Returns:
            float: Euclidean distance in pixels, or None if not enough points
        """
        if len(self.points) != 2:
            return None

        pt1, pt2 = self.points
        distance = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        return float(distance)

    def set_real_world_length(self, length):
        """
        Set the real-world length and calculate scale factor.

        Args:
            length: Real-world length in current units

        Returns:
            float: The calculated scale factor (pixels per unit)

        Raises:
            ValueError: If length is invalid or not enough points
        """
        if len(self.points) != 2:
            raise ValueError("Need exactly 2 points to set scale")

        if length <= 0:
            raise ValueError("Length must be positive")

        pixel_distance = self.calculate_pixel_distance()

        # Calculate scale factor (pixels per unit)
        self.factor = pixel_distance / length
        self.length = length

        return self.factor

    def cancel(self):
        """Cancel the current calibration"""
        self.mode = None
        self.points = []

    def reset(self):
        """Reset all calibration data"""
        self.mode = None
        self.points = []
        self.length = None
        self.factor = 1.0

    def get_point_near(self, x, y, threshold=10):
        """
        Find if there's a calibration point near the given coordinates.

        Args:
            x: X coordinate to check
            y: Y coordinate to check
            threshold: Maximum distance in pixels

        Returns:
            int: Index of point (0 or 1), or None if no point nearby
        """
        for i, (px, py) in enumerate(self.points):
            distance = np.sqrt((x - px)**2 + (y - py)**2)
            if distance <= threshold:
                return i
        return None

    def update_point(self, index, x, y):
        """
        Update the position of a calibration point.

        Args:
            index: Point index (0 or 1)
            x: New X coordinate
            y: New Y coordinate

        Returns:
            bool: True if update successful, False otherwise
        """
        if 0 <= index < len(self.points):
            self.points[index] = (float(x), float(y))
            return True
        return False

    def get_status_message(self):
        """
        Get a status message describing current calibration state.

        Returns:
            str: Human-readable status message
        """
        if not self.is_active():
            if self.is_calibrated():
                return f"Scale calibrated: {self.length:.1f} units"
            return "No scale calibration"

        point_count = len(self.points)
        if point_count == 0:
            return "Scale calibration: Click first point on known distance"
        elif point_count == 1:
            return "Scale calibration: Click second point"
        else:
            return "Scale calibration: Enter real-world distance"
