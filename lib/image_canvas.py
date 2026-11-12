"""
ImageCanvas - Helper class for managing canvas zoom, pan, and display.
"""

import tkinter as tk
import numpy as np
import cv3
from PIL import Image, ImageTk


class ImageCanvas:
    """Helper class to manage zoom, pan, and display for a canvas"""

    def __init__(self, canvas, canvas_width, canvas_height):
        self.canvas = canvas
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        # Zoom and pan state
        self.zoom_level = 1.0
        self.pan_offset = [0.0, 0.0]
        self.base_scale_factor = 1.0
        self.needs_initial_center = False

        # Drag state for panning
        self.panning = False
        self.drag_start = None

        # PhotoImage reference (must keep reference to prevent garbage collection)
        self.photo = None

    def update_canvas_size(self, width, height):
        """Update canvas dimensions"""
        self.canvas_width = width
        self.canvas_height = height

    def reset_view(self):
        """Reset zoom and pan to initial state"""
        self.zoom_level = 1.0
        self.pan_offset = [0.0, 0.0]
        self.needs_initial_center = True

    def zoom_in(self, center_x=None, center_y=None):
        """Zoom in by 20%, centered on given point"""
        if center_x is None:
            center_x = self.canvas_width / 2
        if center_y is None:
            center_y = self.canvas_height / 2

        old_zoom = self.zoom_level
        self.zoom_level = min(self.zoom_level * 1.2, 10.0)

        # Adjust pan to keep the cursor position fixed
        zoom_ratio = self.zoom_level / old_zoom
        self.pan_offset[0] = center_x - (center_x - self.pan_offset[0]) * zoom_ratio
        self.pan_offset[1] = center_y - (center_y - self.pan_offset[1]) * zoom_ratio

    def zoom_out(self, center_x=None, center_y=None):
        """Zoom out by 20%, centered on given point"""
        if center_x is None:
            center_x = self.canvas_width / 2
        if center_y is None:
            center_y = self.canvas_height / 2

        old_zoom = self.zoom_level
        self.zoom_level = max(self.zoom_level / 1.2, 0.1)

        # Adjust pan to keep the cursor position fixed
        zoom_ratio = self.zoom_level / old_zoom
        self.pan_offset[0] = center_x - (center_x - self.pan_offset[0]) * zoom_ratio
        self.pan_offset[1] = center_y - (center_y - self.pan_offset[1]) * zoom_ratio

    def zoom_fit(self):
        """Reset zoom to fit image in canvas"""
        self.zoom_level = 1.0
        self.pan_offset = [0.0, 0.0]
        self.needs_initial_center = True

    def get_zoom_percentage(self):
        """Get current zoom level as percentage"""
        return int(self.zoom_level * self.base_scale_factor * 100)

    def clear(self):
        """Clear the canvas"""
        self.canvas.delete("all")
        self.photo = None

    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to image coordinates"""
        effective_scale = self.base_scale_factor * self.zoom_level
        img_x = (canvas_x - self.pan_offset[0]) / effective_scale
        img_y = (canvas_y - self.pan_offset[1]) / effective_scale
        return img_x, img_y

    def image_to_canvas_coords(self, img_x, img_y):
        """Convert image coordinates to canvas coordinates"""
        effective_scale = self.base_scale_factor * self.zoom_level
        canvas_x = img_x * effective_scale + self.pan_offset[0]
        canvas_y = img_y * effective_scale + self.pan_offset[1]
        return canvas_x, canvas_y

    def display_image(self, image_rgb, overlay_callback=None):
        """
        Display an image on the canvas with current zoom/pan settings

        Args:
            image_rgb: numpy array in RGB format
            overlay_callback: optional function(canvas_image, effective_scale, pan_offset)
                            to draw overlays on the canvas image
        """
        if image_rgb is None:
            return

        height, width = image_rgb.shape[:2]

        # Calculate base scale to fit canvas (with small margin)
        scale_w = self.canvas_width / width
        scale_h = self.canvas_height / height
        self.base_scale_factor = min(scale_w, scale_h) * 0.98  # 98% to ensure it fits

        # Apply zoom
        effective_scale = self.base_scale_factor * self.zoom_level

        new_width = int(width * effective_scale)
        new_height = int(height * effective_scale)

        # Resize for display
        display_image = cv3.resize(image_rgb, new_width, new_height)

        # Create canvas-sized image with background
        canvas_image = np.full((self.canvas_height, self.canvas_width, 3), 64, dtype=np.uint8)

        # Center the image on initial load or fit
        if self.needs_initial_center:
            center_x = (self.canvas_width - new_width) / 2.0
            center_y = (self.canvas_height - new_height) / 2.0
            self.pan_offset = [center_x, center_y]
            self.needs_initial_center = False

        # Calculate positions for placing the image on canvas
        x_offset = int(max(0, self.pan_offset[0]))
        y_offset = int(max(0, self.pan_offset[1]))

        # Calculate which part of the display image to show
        img_x_start = int(max(0, -self.pan_offset[0]))
        img_y_start = int(max(0, -self.pan_offset[1]))

        # Calculate how much of the image can fit on canvas
        img_x_end = int(min(new_width, img_x_start + self.canvas_width - x_offset))
        img_y_end = int(min(new_height, img_y_start + self.canvas_height - y_offset))

        # Place the visible portion of the image on canvas
        if img_y_end > img_y_start and img_x_end > img_x_start:
            visible_portion = display_image[img_y_start:img_y_end, img_x_start:img_x_end]
            h, w = visible_portion.shape[:2]
            canvas_image[y_offset:y_offset+h, x_offset:x_offset+w] = visible_portion

        # Call overlay callback if provided
        if overlay_callback:
            overlay_callback(canvas_image, effective_scale, self.pan_offset)

        # Convert to PhotoImage
        img_pil = Image.fromarray(canvas_image)
        self.photo = ImageTk.PhotoImage(image=img_pil)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def start_pan(self, x, y):
        """Start panning operation"""
        self.panning = True
        self.drag_start = (x, y)

    def update_pan(self, x, y):
        """Update pan offset during drag"""
        if self.panning and self.drag_start:
            dx = x - self.drag_start[0]
            dy = y - self.drag_start[1]
            self.pan_offset[0] += dx
            self.pan_offset[1] += dy
            self.drag_start = (x, y)
            return True
        return False

    def end_pan(self):
        """End panning operation"""
        self.panning = False
        self.drag_start = None
