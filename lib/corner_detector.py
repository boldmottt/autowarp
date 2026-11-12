"""
CornerDetector - Automatic document corner detection using edge and contour analysis.
"""

import logging
import numpy as np
import cv2


class CornerDetector:
    """
    Detects document corners in images using edge detection and contour analysis.

    Supports:
    - Light documents on dark backgrounds
    - Dark documents on light backgrounds
    - Documents touching or extending beyond image edges
    - Handles rounded corners and imperfect quadrilaterals
    """

    def __init__(self):
        """Initialize the corner detector"""
        pass

    def detect(self, image):
        """
        Automatically detect document corners in an image.

        Args:
            image: numpy array in RGB format

        Returns:
            List of 4 corner points [(x, y), ...] ordered as:
            [top-left, top-right, bottom-right, bottom-left]
            Returns None if detection fails
        """
        if image is None:
            return None

        logging.info("Auto-detecting corners")

        try:
            # Image dimensions
            h, w = image.shape[:2]
            image_area = h * w

            # Convert RGB to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply Canny edge detection with improved adaptive thresholds
            # Use median-based thresholds, but with better handling for light/dark documents
            median = np.median(blurred)

            # For dark backgrounds (median < 100), use fixed thresholds that work better
            # For light backgrounds, use adaptive thresholds
            if median < 100:
                # Dark background (like book on table) - use higher fixed thresholds
                lower = 50
                upper = 150
            else:
                # Light background - use adaptive thresholds
                lower = int(max(0, 0.66 * median))
                upper = int(min(255, 1.33 * median))

            edges = cv2.Canny(blurred, lower, upper, apertureSize=3)

            # Dilate edges slightly to close small gaps
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # If very few edge pixels, try with more aggressive edge detection
            edge_pixel_ratio = np.count_nonzero(edges) / image_area
            if edge_pixel_ratio < 0.05 and len(contours) < 10:
                # Try again with different parameters
                edges2 = cv2.Canny(blurred, 30, 100, apertureSize=5)
                edges2 = cv2.dilate(edges2, kernel, iterations=2)
                contours2, _ = cv2.findContours(edges2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours2) > len(contours):
                    contours = contours2
                    edges = edges2

            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Find the largest quadrilateral contour
            # Simple approach: find biggest 4-sided shape that isn't the image boundary
            document_contour = None
            border_threshold = min(h, w) * 0.02  # 2% of smaller dimension

            for contour in contours[:20]:  # Check top 20 largest contours
                peri = cv2.arcLength(contour, True)

                # Try different epsilon values to get 4 vertices
                for epsilon_factor in [0.02, 0.03, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)

                    if len(approx) == 4:
                        area = cv2.contourArea(approx)

                        # Must be substantial (>5%) but not the whole image (<95%)
                        if 0.05 * image_area < area < 0.95 * image_area:
                            # Check if this is just the image boundary
                            corners = approx.reshape(-1, 2)
                            corners_at_boundary = 0

                            for x, y in corners:
                                # A corner is "at boundary" if it's very close to an edge
                                at_left = x < border_threshold
                                at_right = x > (w - border_threshold)
                                at_top = y < border_threshold
                                at_bottom = y > (h - border_threshold)

                                # Must be near a corner (two edges), not just one edge
                                if (at_left or at_right) and (at_top or at_bottom):
                                    corners_at_boundary += 1

                            # Skip if all 4 corners are at image corners (it's the boundary)
                            if corners_at_boundary >= 4:
                                continue

                            document_contour = approx
                            break

                if document_contour is not None:
                    break

            # If no 4-sided contour found, try to detect document touching edges
            if document_contour is None:
                # Look for largest contour that might be partially off-screen
                for contour in contours[:10]:
                    area = cv2.contourArea(contour)
                    if area > 0.20 * image_area:  # Must be substantial
                        peri = cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                        # Accept contours with 3-6 vertices (partial documents)
                        if 3 <= len(approx) <= 6:
                            # Check if any points are near image boundaries
                            points = approx.reshape(-1, 2)
                            near_boundary = False
                            margin = 5  # pixels

                            for pt in points:
                                x, y = pt
                                if (x < margin or x > w - margin or
                                    y < margin or y > h - margin):
                                    near_boundary = True
                                    break

                            if near_boundary:
                                # Supplement with corner points if needed
                                corner_points = self._supplement_with_corners(approx, w, h)
                                if corner_points is not None and len(corner_points) == 4:
                                    document_contour = corner_points
                                    break

            if document_contour is not None:
                # Extract corner points
                if len(document_contour) == 4:
                    corners = document_contour.reshape(4, 2).astype(float)
                else:
                    corners = document_contour

                # Convert to list of tuples
                points = [(float(x), float(y)) for x, y in corners]

                # Order points properly
                ordered = self.order_points(points)
                points = [(float(x), float(y)) for x, y in ordered]

                return points
            else:
                return None

        except Exception as e:
            logging.error(f"Auto-detection failed: {str(e)}")
            return None

    def _supplement_with_corners(self, contour, width, height):
        """
        Supplement partial contour with image corner points.

        For documents that extend beyond the image edges, this method
        fills in the missing corners with the image corner coordinates.

        Args:
            contour: Partial contour with 3-6 vertices
            width: Image width
            height: Image height

        Returns:
            numpy array with 4 corner points, or None if supplementing fails
        """
        points = contour.reshape(-1, 2)
        margin = 10

        # Determine which corners are missing
        corners_present = {
            'tl': False, 'tr': False, 'bl': False, 'br': False
        }

        for pt in points:
            x, y = pt
            if x < margin and y < margin:
                corners_present['tl'] = True
            elif x > width - margin and y < margin:
                corners_present['tr'] = True
            elif x < margin and y > height - margin:
                corners_present['bl'] = True
            elif x > width - margin and y > height - margin:
                corners_present['br'] = True

        # Add missing corners
        result_points = list(points)

        if not corners_present['tl']:
            result_points.append([0, 0])
        if not corners_present['tr']:
            result_points.append([width - 1, 0])
        if not corners_present['bl']:
            result_points.append([0, height - 1])
        if not corners_present['br']:
            result_points.append([width - 1, height - 1])

        # If we have exactly 4 points now, return them
        if len(result_points) == 4:
            return np.array(result_points).reshape(4, 1, 2).astype(np.int32)

        return None

    def order_points(self, pts):
        """
        Order points in consistent clockwise order.

        Args:
            pts: List of (x, y) tuples or numpy array

        Returns:
            numpy array ordered as: [top-left, top-right, bottom-right, bottom-left]
        """
        pts = np.array(pts, dtype="float32")

        # Sort by y-coordinate (top to bottom)
        sorted_by_y = pts[np.argsort(pts[:, 1])]

        # Top two points
        top_pts = sorted_by_y[:2]
        top_pts = top_pts[np.argsort(top_pts[:, 0])]  # Sort by x
        tl, tr = top_pts

        # Bottom two points
        bottom_pts = sorted_by_y[2:]
        bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]  # Sort by x
        bl, br = bottom_pts

        return np.array([tl, tr, br, bl], dtype="float32")

    def create_debug_visualization(self, image):
        """
        Create a debug visualization showing detected edges and contours.

        Args:
            image: numpy array in RGB format

        Returns:
            tuple: (debug_image, info_dict) where debug_image is a numpy array
                   and info_dict contains detection metadata
        """
        h, w = image.shape[:2]
        image_area = h * w

        # Convert RGB to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        median = np.median(blurred)
        if median < 100:
            lower, upper = 50, 150
        else:
            lower = int(max(0, 0.66 * median))
            upper = int(min(255, 1.33 * median))

        edges = cv2.Canny(blurred, lower, upper, apertureSize=3)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Convert edges to RGB for display
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # Create contour visualization on original image
        contour_vis = image.copy()

        # Draw top 5 contours in different colors
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, contour in enumerate(contours[:5]):
            cv2.drawContours(contour_vis, [contour], 0, colors[i], 2)
            # Label with contour number and area
            area = cv2.contourArea(contour)
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                pct = 100 * area / image_area
                cv2.putText(contour_vis, f"#{i+1} {pct:.0f}%", (cx-30, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

        # Scale images to fit in debug window
        max_debug_size = 400
        scale = min(max_debug_size / h, max_debug_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        edges_small = cv2.resize(edges_rgb, (new_w, new_h))
        contour_small = cv2.resize(contour_vis, (new_w, new_h))

        # Combine side by side
        combined = np.hstack([edges_small, contour_small])

        # Add info text
        info_text = f"Median: {median:.0f} | Canny: {lower}-{upper} | Contours: {len(contours)}"
        cv2.putText(combined, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        info_dict = {
            'median': median,
            'canny_lower': lower,
            'canny_upper': upper,
            'num_contours': len(contours),
            'width': new_w * 2,
            'height': new_h
        }

        return combined, info_dict
