"""
AutoWarp - 시편 사진 원근 보정 GUI
사진 위에서 네 귀퉁이를 잡으면 실물 규격(mm×dpi)에 맞춰 펴서 저장한다.
dewarp(MIT, (c) 2025 Robert Marchese)를 기반으로 재작업.

핵심 기본값: 규격 170×70mm · 300dpi · 자동 검출 ON
"""

import argparse
import logging
from datetime import datetime
import cv2  # For perspective transforms, color conversion, and text rendering
import cv3  # For basic I/O and drawing operations
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk  # Convenience imports for dialogs and themed widgets
from PIL import Image, ImageTk
from pillow_heif import register_heif_opener  # For HEIC file support

# Import our modules
from lib import UnitConverter, ImageCanvas, CornerDetector, ScaleCalibrator

# Register HEIF opener with Pillow to enable HEIC support
register_heif_opener()


class AutoWarpGUI:
    # Standard page sizes in mm (width x height)
    # Page sizes stored as (width_mm, height_mm) in portrait orientation
    # The display name will be generated dynamically with current units
    PAGE_SIZES_DATA = {
        "Custom": None,
        "시편 170x70": (170.0, 70.0),
        "시편 170x70 (반)": (85.0, 35.0),
        "US Letter Portrait": (215.9, 279.4),
        "US Letter Landscape": (279.4, 215.9),
        "US Legal Portrait": (215.9, 355.6),
        "US Legal Landscape": (355.6, 215.9),
        "Tabloid Portrait": (279.4, 431.8),
        "Tabloid Landscape": (431.8, 279.4),
        "A4 Portrait": (210.0, 297.0),
        "A4 Landscape": (297.0, 210.0),
        "A5 Portrait": (148.0, 210.0),
        "A5 Landscape": (210.0, 148.0),
        "Post-it Note (3x3)": (76.2, 76.2),
        "Index Card": (127.0, 76.2),
    }

    def get_page_size_display_names(self):
        """Generate display names for page sizes (without dimensions)"""
        display_names = []
        for name in self.PAGE_SIZES_DATA.keys():
            if name == "Custom":
                display_names.append("Custom")
            else:
                display_names.append(name)
        return display_names

    def get_page_size_key_from_display(self, display_name):
        """Extract the page size key from a display name"""
        # Display name is now the same as the key
        return display_name if display_name in self.PAGE_SIZES_DATA else "Custom"

    def update_page_size_dropdown(self):
        """Update the page size dropdown values"""
        display_names = self.get_page_size_display_names()

        # Update combobox values for both layouts
        if hasattr(self, 'page_size_combo'):
            self.page_size_combo['values'] = display_names
        if hasattr(self, 'tab_page_size_combo'):
            self.tab_page_size_combo['values'] = display_names

        # Keep current selection (display name is same as key now)
        current = self.page_size_var.get()
        if current not in display_names:
            self.page_size_var.set("Custom")

    def __init__(self, root, dpi=300, units="mm", crop=False, auto_detect=False):
        self.root = root
        self.root.title("AutoWarp")

        # Set window icon
        try:
            import os
            import sys

            # Get the correct base path for both dev and PyInstaller bundled app
            if getattr(sys, 'frozen', False):
                # Running as PyInstaller bundle
                base_path = sys._MEIPASS
            else:
                # Running in normal Python environment
                base_path = os.path.dirname(os.path.abspath(__file__))

            icon_path = os.path.join(base_path, "assets", "autowarp.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except Exception:
            pass  # Use default icon if unable to load

        # State variables
        self.image = None
        self.display_image = None
        self.original_image = None
        self.original_file_path = None
        self.points = []
        self.transformed_image = None

        # Canvas dimensions (will be set in setup_ui)
        self.canvas_width = 500  # Initial placeholder
        self.canvas_height = 400  # Initial placeholder

        # ImageCanvas instances (will be initialized in setup_ui)
        self.left_canvas = None
        self.right_canvas = None

        # Drag state for point manipulation
        self.dragging_point = None
        self.drag_start = None

        # Scale calibration
        self.scale_calibrator = ScaleCalibrator()
        self.dragging_scale_point = None  # Index of scale point being dragged

        # Unit converter (manages conversions between pixels, mm, inches)
        self.unit_converter = UnitConverter(units=units, dpi=dpi, scale_factor=1.0)

        # Corner detector (automatic document corner detection)
        self.corner_detector = CornerDetector()

        # Layout mode tracking
        self.layout_mode = "side-by-side"  # or "tabbed"
        self.layout_threshold_width = 800  # Switch to tabbed mode below this width
        self.layout_hysteresis = 50  # Hysteresis range to prevent rapid switching
        self.notebook = None  # Will hold the notebook widget when in tabbed mode

        # DPI for mm conversion (from command line or default 300 DPI)
        self.dpi = dpi
        self.dpi_var = tk.StringVar(value=str(dpi))
        self.dpi_var.trace_add('write', self.on_dpi_changed)

        # Input image DPI (detected from file metadata, defaults to output DPI if not found)
        self.input_dpi = dpi

        # Units preference (from command line or default mm)
        self.units = tk.StringVar(value=units)
        self._previous_units = units  # Track previous units for conversion
        self.units.trace_add('write', self.on_units_changed)

        # Crop mode (from command line or default False = transform entire image)
        self.crop_image = tk.BooleanVar(value=crop)

        # Auto-detect corners on load (from command line or default False)
        self.auto_detect_on_load = tk.BooleanVar(value=auto_detect)

        # Track if user has manually set dimensions
        self.dimensions_manually_set = False
        self._updating_dimensions = False  # Flag to prevent callback during auto-update

        # Setup UI
        self.setup_ui()

        # Track which side the context menu is for
        self.context_menu_side = None  # "left" or "right"

        # Track which canvas currently has focus/cursor for keyboard shortcuts
        self.focused_canvas = None  # "left" or "right"

        # Create unified context menu
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Save Result...", command=self.save_image)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Use as Original", command=self.use_result_as_original)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Set Scale...", command=self.context_menu_set_scale)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Rotate 90 deg CW", command=self.context_menu_rotate_cw)
        self.context_menu.add_command(label="Rotate 90 deg CCW", command=self.context_menu_rotate_ccw)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Flip Horizontal", command=self.context_menu_flip_horizontal)
        self.context_menu.add_command(label="Flip Vertical", command=self.context_menu_flip_vertical)
        self.context_menu.add_separator()
        self.context_menu.add_checkbutton(label="Crop Mode", variable=self.crop_image, command=self.on_crop_mode_changed)

    # Property accessors for backward compatibility with scale calibrator
    @property
    def scale_mode(self):
        return self.scale_calibrator.get_mode()

    @property
    def scale_points(self):
        return self.scale_calibrator.get_points()

    @property
    def scale_length(self):
        return self.scale_calibrator.get_scale_length()

    @property
    def scale_factor(self):
        return self.scale_calibrator.get_scale_factor()

    def setup_ui(self):
        # Calculate canvas dimensions based on screen size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Use 45% of screen width for each canvas, 75% of screen height
        self.canvas_width = int(screen_width * 0.45)
        self.canvas_height = int(screen_height * 0.75)

        # Set minimum sizes
        self.canvas_width = max(400, self.canvas_width)
        self.canvas_height = max(300, self.canvas_height)

        # Menu Bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File Menu
        self.file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Load Image...", command=self.load_image, accelerator="Ctrl+O")
        self.file_menu.add_command(label="Save Result...", command=self.save_image, accelerator="Ctrl+S", state=tk.DISABLED)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Preferences...", command=self.show_preferences)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Alt+F4")

        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.load_image())
        self.root.bind('<Control-s>', lambda e: self.save_image() if self.transformed_image is not None else None)
        self.root.bind('r', self.on_key_rotate_right)
        self.root.bind('R', self.on_key_rotate_right)
        self.root.bind('l', self.on_key_rotate_left)
        self.root.bind('L', self.on_key_rotate_left)

        # Canvas Frame - this will hold either side-by-side or tabbed layout
        self.canvas_container = ttk.Frame(self.root)
        self.canvas_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create SIDE-BY-SIDE container
        self.sidebyside_container = ttk.Frame(self.canvas_container)

        # Main Canvas for original image (side-by-side)
        self.left_frame = ttk.Frame(self.sidebyside_container)
        self.left_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.canvas = tk.Canvas(self.left_frame, bg='gray', cursor="cross",
                               width=self.canvas_width, height=self.canvas_height,
                               highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Initialize left ImageCanvas helper
        self.left_canvas = ImageCanvas(self.canvas, self.canvas_width, self.canvas_height)

        # Action buttons overlay (upper left corner of left canvas)
        action_overlay = ttk.Frame(self.left_frame, relief=tk.RAISED, borderwidth=1)
        action_overlay.place(relx=0.0, rely=0.0, x=5, y=5, anchor=tk.NW)

        ttk.Button(action_overlay, text="Reset", command=self.reset_points, width=6).pack(side=tk.LEFT, padx=1, pady=3)
        ttk.Button(action_overlay, text="Auto", command=self.auto_detect_corners, width=6).pack(side=tk.LEFT, padx=1, pady=3)

        # Zoom controls overlay (bottom center of left canvas)
        zoom_overlay = ttk.Frame(self.left_frame, relief=tk.RAISED, borderwidth=1)
        zoom_overlay.place(relx=0.5, rely=1.0, x=0, y=-5, anchor=tk.S)

        ttk.Button(zoom_overlay, text="+", command=self.zoom_in, width=3).pack(side=tk.LEFT, padx=1, pady=3)
        ttk.Button(zoom_overlay, text="-", command=self.zoom_out, width=3).pack(side=tk.LEFT, padx=1, pady=3)
        ttk.Button(zoom_overlay, text="Fit", command=self.zoom_fit, width=4).pack(side=tk.LEFT, padx=1, pady=3)
        self.zoom_label = ttk.Label(zoom_overlay, text="100%", width=5)
        self.zoom_label.pack(side=tk.LEFT, padx=3, pady=3)

        # Bind events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_canvas_context_menu)  # Right-click for context menu
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Enter>", lambda e: setattr(self, 'focused_canvas', 'left'))
        self.canvas.bind("<Leave>", lambda e: setattr(self, 'focused_canvas', None))

        # Result Canvas (side-by-side)
        self.right_frame = ttk.Frame(self.sidebyside_container)
        self.right_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.result_canvas = tk.Canvas(self.right_frame, bg='gray',
                                       width=self.canvas_width, height=self.canvas_height,
                                       highlightthickness=0)
        self.result_canvas.pack(fill=tk.BOTH, expand=True)

        # Initialize right ImageCanvas helper
        self.right_canvas = ImageCanvas(self.result_canvas, self.canvas_width, self.canvas_height)

        self.result_canvas.bind("<Configure>", self.on_result_canvas_resize)
        self.result_canvas.bind("<Button-1>", self.on_result_canvas_click)
        self.result_canvas.bind("<B1-Motion>", self.on_result_canvas_drag)
        self.result_canvas.bind("<ButtonRelease-1>", self.on_result_canvas_release)
        self.result_canvas.bind("<Button-3>", self.on_result_canvas_right_click)
        self.result_canvas.bind("<MouseWheel>", self.on_result_mouse_wheel)
        self.result_canvas.bind("<Enter>", lambda e: setattr(self, 'focused_canvas', 'right'))
        self.result_canvas.bind("<Leave>", lambda e: setattr(self, 'focused_canvas', None))

        # Dimension controls overlay (upper left corner of right canvas)
        dimensions_overlay = ttk.Frame(self.right_frame, relief=tk.RAISED, borderwidth=1)
        dimensions_overlay.place(relx=0.0, rely=0.0, x=5, y=5, anchor=tk.NW)

        # Zoom controls overlay for result canvas (bottom center)
        result_zoom_overlay = ttk.Frame(self.right_frame, relief=tk.RAISED, borderwidth=1)
        result_zoom_overlay.place(relx=0.5, rely=1.0, x=0, y=-5, anchor=tk.S)

        ttk.Button(result_zoom_overlay, text="+", command=self.result_zoom_in, width=3).pack(side=tk.LEFT, padx=1, pady=3)
        ttk.Button(result_zoom_overlay, text="-", command=self.result_zoom_out, width=3).pack(side=tk.LEFT, padx=1, pady=3)
        ttk.Button(result_zoom_overlay, text="Fit", command=self.result_zoom_fit, width=4).pack(side=tk.LEFT, padx=1, pady=3)
        self.result_zoom_label = ttk.Label(result_zoom_overlay, text="100%", width=5)
        self.result_zoom_label.pack(side=tk.LEFT, padx=3, pady=3)

        # Consolidated size selector - all on one line
        self.size_label = ttk.Label(dimensions_overlay, text="Size (mm):", font=('TkDefaultFont', 8))
        self.size_label.grid(row=0, column=0, sticky=tk.E, padx=(3, 3), pady=3)

        self.page_size_var = tk.StringVar(value="Custom")
        self.page_size_combo = ttk.Combobox(dimensions_overlay, textvariable=self.page_size_var,
                                             values=self.get_page_size_display_names(),
                                             state='readonly', width=18)
        self.page_size_combo.grid(row=0, column=1, sticky=tk.W, padx=2, pady=3)
        self.page_size_combo.bind('<<ComboboxSelected>>', self.on_page_size_changed)

        self.width_var = tk.StringVar(value="")  # Empty until points selected
        self.width_var.trace_add('write', self.on_dimension_changed)
        self.width_spinbox = ttk.Spinbox(dimensions_overlay, textvariable=self.width_var, from_=1, to=9999, increment=0.5, width=8)
        self.width_spinbox.grid(row=0, column=2, padx=2, pady=3)

        ttk.Label(dimensions_overlay, text="x", font=('TkDefaultFont', 8)).grid(row=0, column=3, padx=2, pady=3)

        self.height_var = tk.StringVar(value="")  # Empty until points selected
        self.height_var.trace_add('write', self.on_dimension_changed)
        self.height_spinbox = ttk.Spinbox(dimensions_overlay, textvariable=self.height_var, from_=1, to=9999, increment=0.5, width=8)
        self.height_spinbox.grid(row=0, column=4, padx=2, pady=3)

        self.transform_btn = ttk.Button(dimensions_overlay, text="Apply", command=self.apply_transform, state=tk.DISABLED, width=6)
        self.transform_btn.grid(row=0, column=5, padx=(5, 3), pady=3)

        # Status Bar at bottom
        status_frame = ttk.Frame(self.root, relief=tk.SUNKEN, padding="2")
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.status_label = ttk.Label(status_frame, text="Load an image to begin", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # DPI display on the right side of status bar
        self.dpi_display_label = ttk.Label(status_frame, text="DPI: 300", anchor=tk.E)
        self.dpi_display_label.pack(side=tk.RIGHT, padx=10)

        # Configure grid weights for side-by-side container
        self.sidebyside_container.columnconfigure(0, weight=1)
        self.sidebyside_container.columnconfigure(1, weight=1)
        self.sidebyside_container.rowconfigure(0, weight=1)

        # Create TABBED container
        self.notebook = ttk.Notebook(self.canvas_container)

        # Create tab frames with their own canvases
        self.tab_left_frame = ttk.Frame(self.notebook)
        self.tab_right_frame = ttk.Frame(self.notebook)

        # Create canvases for tabs
        self.tab_canvas = tk.Canvas(self.tab_left_frame, bg='gray', cursor="cross",
                                   width=self.canvas_width, height=self.canvas_height,
                                   highlightthickness=0)
        self.tab_canvas.pack(fill=tk.BOTH, expand=True)

        self.tab_result_canvas = tk.Canvas(self.tab_right_frame, bg='gray',
                                          width=self.canvas_width, height=self.canvas_height,
                                          highlightthickness=0)
        self.tab_result_canvas.pack(fill=tk.BOTH, expand=True)

        # Initialize ImageCanvas helpers for tab canvases
        self.tab_left_canvas = ImageCanvas(self.tab_canvas, self.canvas_width, self.canvas_height)
        self.tab_right_canvas = ImageCanvas(self.tab_result_canvas, self.canvas_width, self.canvas_height)

        # Bind events for tab canvases (same as main canvases)
        self.tab_canvas.bind("<Button-1>", self.on_canvas_click)
        self.tab_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.tab_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.tab_canvas.bind("<Button-3>", self.on_canvas_context_menu)  # Right-click for context menu
        self.tab_canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.tab_canvas.bind("<Configure>", self.on_tab_canvas_resize)
        self.tab_canvas.bind("<Enter>", lambda e: setattr(self, 'focused_canvas', 'left'))
        self.tab_canvas.bind("<Leave>", lambda e: setattr(self, 'focused_canvas', None))

        self.tab_result_canvas.bind("<Configure>", self.on_tab_result_canvas_resize)
        self.tab_result_canvas.bind("<Button-1>", self.on_result_canvas_click)
        self.tab_result_canvas.bind("<B1-Motion>", self.on_result_canvas_drag)
        self.tab_result_canvas.bind("<ButtonRelease-1>", self.on_result_canvas_release)
        self.tab_result_canvas.bind("<Button-3>", self.on_result_canvas_right_click)
        self.tab_result_canvas.bind("<MouseWheel>", self.on_result_mouse_wheel)
        self.tab_result_canvas.bind("<Enter>", lambda e: setattr(self, 'focused_canvas', 'right'))
        self.tab_result_canvas.bind("<Leave>", lambda e: setattr(self, 'focused_canvas', None))

        # Create overlay controls for tab canvases
        tab_action_overlay = ttk.Frame(self.tab_left_frame, relief=tk.RAISED, borderwidth=1)
        tab_action_overlay.place(relx=0.0, rely=0.0, x=5, y=5, anchor=tk.NW)
        ttk.Button(tab_action_overlay, text="Reset", command=self.reset_points, width=6).pack(side=tk.LEFT, padx=1, pady=3)
        ttk.Button(tab_action_overlay, text="Auto", command=self.auto_detect_corners, width=6).pack(side=tk.LEFT, padx=1, pady=3)

        tab_zoom_overlay = ttk.Frame(self.tab_left_frame, relief=tk.RAISED, borderwidth=1)
        tab_zoom_overlay.place(relx=0.5, rely=1.0, x=0, y=-5, anchor=tk.S)
        ttk.Button(tab_zoom_overlay, text="+", command=self.zoom_in, width=3).pack(side=tk.LEFT, padx=1, pady=3)
        ttk.Button(tab_zoom_overlay, text="-", command=self.zoom_out, width=3).pack(side=tk.LEFT, padx=1, pady=3)
        ttk.Button(tab_zoom_overlay, text="Fit", command=self.zoom_fit, width=4).pack(side=tk.LEFT, padx=1, pady=3)
        self.tab_zoom_label = ttk.Label(tab_zoom_overlay, text="100%", width=5)
        self.tab_zoom_label.pack(side=tk.LEFT, padx=3, pady=3)

        tab_dimensions_overlay = ttk.Frame(self.tab_right_frame, relief=tk.RAISED, borderwidth=1)
        tab_dimensions_overlay.place(relx=0.0, rely=0.0, x=5, y=5, anchor=tk.NW)

        tab_result_zoom_overlay = ttk.Frame(self.tab_right_frame, relief=tk.RAISED, borderwidth=1)
        tab_result_zoom_overlay.place(relx=0.5, rely=1.0, x=0, y=-5, anchor=tk.S)
        ttk.Button(tab_result_zoom_overlay, text="+", command=self.result_zoom_in, width=3).pack(side=tk.LEFT, padx=1, pady=3)
        ttk.Button(tab_result_zoom_overlay, text="-", command=self.result_zoom_out, width=3).pack(side=tk.LEFT, padx=1, pady=3)
        ttk.Button(tab_result_zoom_overlay, text="Fit", command=self.result_zoom_fit, width=4).pack(side=tk.LEFT, padx=1, pady=3)
        self.tab_result_zoom_label = ttk.Label(tab_result_zoom_overlay, text="100%", width=5)
        self.tab_result_zoom_label.pack(side=tk.LEFT, padx=3, pady=3)

        # Share dimension controls between both layouts (consolidated single line)
        ttk.Label(tab_dimensions_overlay, text="Size (mm):", font=('TkDefaultFont', 8)).grid(row=0, column=0, sticky=tk.E, padx=(3, 3), pady=3)
        self.tab_page_size_combo = ttk.Combobox(tab_dimensions_overlay, textvariable=self.page_size_var,
                     values=self.get_page_size_display_names(),
                     state='readonly', width=18)
        self.tab_page_size_combo.grid(row=0, column=1, sticky=tk.W, padx=2, pady=3)
        self.tab_width_spinbox = ttk.Spinbox(tab_dimensions_overlay, textvariable=self.width_var, from_=1, to=9999, increment=0.5, width=8)
        self.tab_width_spinbox.grid(row=0, column=2, padx=2, pady=3)
        ttk.Label(tab_dimensions_overlay, text="x", font=('TkDefaultFont', 8)).grid(row=0, column=3, padx=2, pady=3)
        self.tab_height_spinbox = ttk.Spinbox(tab_dimensions_overlay, textvariable=self.height_var, from_=1, to=9999, increment=0.5, width=8)
        self.tab_height_spinbox.grid(row=0, column=4, padx=2, pady=3)
        self.tab_transform_btn = ttk.Button(tab_dimensions_overlay, text="Apply", command=self.apply_transform, state=tk.DISABLED, width=6)
        self.tab_transform_btn.grid(row=0, column=5, padx=(5, 3), pady=3)

        # Add tabs to notebook
        self.notebook.add(self.tab_left_frame, text="Original")
        self.notebook.add(self.tab_right_frame, text="Result")

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.canvas_container.columnconfigure(0, weight=1)
        self.canvas_container.rowconfigure(0, weight=1)

        # Start in side-by-side mode
        self.sidebyside_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Bind window resize to check for layout mode change
        self.root.bind("<Configure>", self.on_window_resize)

    def units_to_pixels(self, value):
        """Convert value in current units to pixels based on DPI"""
        try:
            dpi = int(self.dpi_var.get())
        except ValueError:
            dpi = 300

        self.unit_converter.set_dpi(dpi)
        self.unit_converter.set_units(self.units.get())
        return self.unit_converter.units_to_pixels(value)

    def pixels_to_units(self, pixels, dpi=None):
        """Convert pixels to current units based on DPI

        Args:
            pixels: pixel value to convert
            dpi: DPI to use for conversion (defaults to output DPI)
        """
        if dpi is None:
            try:
                dpi = int(self.dpi_var.get())
            except ValueError:
                dpi = 300

        self.unit_converter.set_dpi(dpi)
        self.unit_converter.set_units(self.units.get())
        return self.unit_converter.pixels_to_units(pixels)

    def on_dimension_changed(self, *args):
        """Called when width or height field is modified"""
        # Only mark as manually set if we're not programmatically updating
        if not self._updating_dimensions:
            logging.info(f"Dimensions manually changed: width={self.width_var.get()}, height={self.height_var.get()}")
            self.dimensions_manually_set = True
            # User manually changed dimensions, reset to Custom
            self.page_size_var.set("Custom")
            # Auto-apply transform if we have 4 corners
            if len(self.points) == 4 and self.image is not None:
                self.apply_transform()

    def on_page_size_changed(self, event=None):
        """Called when page size is selected from dropdown"""
        selected_display = self.page_size_var.get()
        selected_key = self.get_page_size_key_from_display(selected_display)
        logging.info(f"Page size changed to: {selected_key}")
        page_dims = self.PAGE_SIZES_DATA.get(selected_key)

        if page_dims is not None:
            # Standard page size selected - update dimensions
            width_mm, height_mm = page_dims

            # Convert to current units
            units = self.units.get()
            if units == "mm":
                width_val = width_mm
                height_val = height_mm
            elif units == "inches":
                width_val = width_mm / 25.4
                height_val = height_mm / 25.4
            elif units == "pixels":
                try:
                    dpi = int(self.dpi_var.get())
                except ValueError:
                    dpi = 300
                width_val = (width_mm / 25.4) * dpi
                height_val = (height_mm / 25.4) * dpi

            # Update the dimension fields
            self._updating_dimensions = True
            if units == "pixels":
                self.width_var.set(str(int(round(width_val))))
                self.height_var.set(str(int(round(height_val))))
            else:
                self.width_var.set(f"{width_val:.1f}")
                self.height_var.set(f"{height_val:.1f}")
            self._updating_dimensions = False

            # Mark as manually set since user chose specific dimensions
            self.dimensions_manually_set = True

            # If we have 4 corners placed, automatically re-apply transform with new dimensions
            if len(self.points) == 4 and self.image is not None:
                self.apply_transform()

    def on_dpi_changed(self, *args):
        """Called when DPI is modified"""
        try:
            dpi_value = int(self.dpi_var.get())
            self.dpi_display_label.config(text=f"DPI: {dpi_value}")
        except (ValueError, tk.TclError, AttributeError):
            # Ignore errors during initialization or invalid values
            pass

    def on_units_changed(self, *args):
        """Called when units preference is modified"""
        try:
            new_units = self.units.get()
            old_units = getattr(self, '_previous_units', new_units)
            if old_units != new_units:
                logging.info(f"Units changed from {old_units} to {new_units}")

            # Convert existing dimension values to new units
            if old_units != new_units and hasattr(self, 'width_var') and hasattr(self, 'height_var'):
                try:
                    width_str = self.width_var.get()
                    height_str = self.height_var.get()

                    if width_str and height_str:
                        width_val = float(width_str)
                        height_val = float(height_str)
                        dpi = int(self.dpi_var.get()) if hasattr(self, 'dpi_var') else 300

                        # Use converter to convert from old units to new units
                        new_width = self.unit_converter.convert_units(width_val, old_units, new_units, dpi)
                        new_height = self.unit_converter.convert_units(height_val, old_units, new_units, dpi)

                        # Update the dimension fields
                        self._updating_dimensions = True
                        if new_units == "pixels":
                            self.width_var.set(str(int(round(new_width))))
                            self.height_var.set(str(int(round(new_height))))
                        else:
                            self.width_var.set(f"{new_width:.1f}")
                            self.height_var.set(f"{new_height:.1f}")
                        self._updating_dimensions = False
                except (ValueError, TypeError):
                    # Ignore conversion errors for empty or invalid values
                    pass

            # Store current units for next change
            self._previous_units = new_units

            # Update size label and spinbox increments using converter
            label = self.unit_converter.get_unit_label(new_units)
            increment = self.unit_converter.get_spinbox_increment(new_units)

            self.size_label.config(text=f"Size ({label}):")
            self.width_spinbox.config(increment=increment)
            self.height_spinbox.config(increment=increment)
            if hasattr(self, 'tab_width_spinbox'):
                self.tab_width_spinbox.config(increment=increment)
                self.tab_height_spinbox.config(increment=increment)
            # Update page size dropdown to show dimensions in new units
            self.update_page_size_dropdown()
        except (tk.TclError, AttributeError):
            # Ignore errors during initialization
            pass

    def on_crop_mode_changed(self):
        """Called when crop mode checkbox is toggled"""
        # If we have a transformed image, re-apply the transform with new crop setting
        if len(self.points) == 4 and self.image is not None:
            self.apply_transform()

    def use_result_as_original(self):
        """Move the result image to the original pane for further editing"""
        if self.transformed_image is None:
            return

        # Save the result as the new original
        self.original_image = self.transformed_image.copy()
        self.image = self.transformed_image.copy()

        # Reset points and result
        self.points = []
        self.transformed_image = None
        self.transform_btn.config(state=tk.DISABLED)
        self.tab_transform_btn.config(state=tk.DISABLED)
        self.file_menu.entryconfig("Save Result...", state=tk.DISABLED)

        # Clear dimension fields
        self._updating_dimensions = True
        self.width_var.set("")
        self.height_var.set("")
        self._updating_dimensions = False
        self.dimensions_manually_set = False

        # Reset view and display
        self.left_canvas.reset_view()
        self.tab_left_canvas.reset_view()
        self.display_on_canvas()
        self.display_on_tab_canvas()

        # Clear result canvases
        self.result_canvas.delete("all")
        self.tab_result_canvas.delete("all")

        self.status_label.config(text="Result moved to original. Click 4 corners to continue editing.")

    def context_menu_set_scale(self):
        """Set scale - dispatches to left or right based on context_menu_side"""
        if self.context_menu_side == "left":
            self.start_scale_calibration_original()
        elif self.context_menu_side == "right":
            self.start_scale_calibration_result()

    def context_menu_rotate_cw(self):
        """Rotate clockwise - dispatches to left or right based on context_menu_side"""
        if self.context_menu_side == "left":
            self.rotate_original(clockwise=True)
        elif self.context_menu_side == "right":
            self.rotate_result(clockwise=True)

    def context_menu_rotate_ccw(self):
        """Rotate counter-clockwise - dispatches to left or right based on context_menu_side"""
        if self.context_menu_side == "left":
            self.rotate_original(clockwise=False)
        elif self.context_menu_side == "right":
            self.rotate_result(clockwise=False)

    def context_menu_flip_horizontal(self):
        """Flip horizontal - dispatches to left or right based on context_menu_side"""
        if self.context_menu_side == "left":
            self.flip_original(horizontal=True)
        elif self.context_menu_side == "right":
            self.flip_result(horizontal=True)

    def context_menu_flip_vertical(self):
        """Flip vertical - dispatches to left or right based on context_menu_side"""
        if self.context_menu_side == "left":
            self.flip_original(horizontal=False)
        elif self.context_menu_side == "right":
            self.flip_result(horizontal=False)

    def on_key_rotate_right(self, event):
        """Handle 'R' key press - rotate clockwise based on focused canvas"""
        if self.focused_canvas == "left":
            self.rotate_original(clockwise=True)
        elif self.focused_canvas == "right":
            self.rotate_result(clockwise=True)

    def on_key_rotate_left(self, event):
        """Handle 'L' key press - rotate counter-clockwise based on focused canvas"""
        if self.focused_canvas == "left":
            self.rotate_original(clockwise=False)
        elif self.focused_canvas == "right":
            self.rotate_result(clockwise=False)

    def rotate_original(self, clockwise=True):
        """Rotate the original image 90 degrees"""
        if self.image is None:
            return

        direction = "clockwise" if clockwise else "counter-clockwise"
        logging.info(f"Rotating original image {direction}")

        # Rotate the image (clockwise = -90, counter-clockwise = 90)
        rotation_code = cv2.ROTATE_90_CLOCKWISE if clockwise else cv2.ROTATE_90_COUNTERCLOCKWISE
        self.original_image = cv2.rotate(self.original_image, rotation_code)
        self.image = self.original_image.copy()

        # Clear points and result since rotation invalidates them
        self.points = []
        self.transformed_image = None
        self.transform_btn.config(state=tk.DISABLED)
        self.tab_transform_btn.config(state=tk.DISABLED)
        self.file_menu.entryconfig("Save Result...", state=tk.DISABLED)

        # Clear dimension fields
        self._updating_dimensions = True
        self.width_var.set("")
        self.height_var.set("")
        self._updating_dimensions = False
        self.dimensions_manually_set = False

        # Reset view and display
        self.left_canvas.reset_view()
        self.tab_left_canvas.reset_view()
        self.display_on_canvas()
        self.display_on_tab_canvas()

        # Clear result canvases
        self.result_canvas.delete("all")
        self.tab_result_canvas.delete("all")

        # Auto-detect corners if preference is enabled
        if self.auto_detect_on_load.get():
            self.auto_detect_corners(show_debug=False)
            direction = "clockwise" if clockwise else "counter-clockwise"
            self.status_label.config(text=f"Image rotated 90 deg {direction}. Corners auto-detected.")
        else:
            direction = "clockwise" if clockwise else "counter-clockwise"
            self.status_label.config(text=f"Image rotated 90 deg {direction}. Click 4 corners to transform.")

    def flip_original(self, horizontal=True):
        """Flip the original image horizontally or vertically"""
        if self.image is None:
            return

        # Flip the image (1 = horizontal, 0 = vertical)
        flip_code = 1 if horizontal else 0
        self.original_image = cv2.flip(self.original_image, flip_code)
        self.image = self.original_image.copy()

        # Clear points and result since flip invalidates them
        self.points = []
        self.transformed_image = None
        self.transform_btn.config(state=tk.DISABLED)
        self.tab_transform_btn.config(state=tk.DISABLED)
        self.file_menu.entryconfig("Save Result...", state=tk.DISABLED)

        # Clear dimension fields
        self._updating_dimensions = True
        self.width_var.set("")
        self.height_var.set("")
        self._updating_dimensions = False
        self.dimensions_manually_set = False

        # Reset view and display
        self.left_canvas.reset_view()
        self.tab_left_canvas.reset_view()
        self.display_on_canvas()
        self.display_on_tab_canvas()

        # Clear result canvases
        self.result_canvas.delete("all")
        self.tab_result_canvas.delete("all")

        direction = "horizontal" if horizontal else "vertical"
        self.status_label.config(text=f"Image flipped {direction}. Click 4 corners to transform.")

    def rotate_result(self, clockwise=True):
        """Rotate the result image 90 degrees"""
        if self.transformed_image is None:
            return

        # Rotate the result image
        rotation_code = cv2.ROTATE_90_CLOCKWISE if clockwise else cv2.ROTATE_90_COUNTERCLOCKWISE
        self.transformed_image = cv2.rotate(self.transformed_image, rotation_code)

        # Reset view and display result
        self.right_canvas.reset_view()
        self.tab_right_canvas.reset_view()
        self.display_result()
        self.display_on_tab_result()

        direction = "clockwise" if clockwise else "counter-clockwise"
        self.status_label.config(text=f"Result rotated 90 deg {direction}.")

    def flip_result(self, horizontal=True):
        """Flip the result image horizontally or vertically"""
        if self.transformed_image is None:
            return

        # Flip the result image (1 = horizontal, 0 = vertical)
        flip_code = 1 if horizontal else 0
        self.transformed_image = cv2.flip(self.transformed_image, flip_code)

        # Reset view and display result
        self.right_canvas.reset_view()
        self.tab_right_canvas.reset_view()
        self.display_result()
        self.display_on_tab_result()

        direction = "horizontal" if horizontal else "vertical"
        self.status_label.config(text=f"Result flipped {direction}.")

    def start_scale_calibration_original(self):
        """Start scale calibration mode on original image"""
        if self.image is None:
            return

        self.scale_calibrator.start_calibration("original")
        self.status_label.config(text=self.scale_calibrator.get_status_message())

        # Change cursor
        if self.layout_mode == "side-by-side":
            self.canvas.config(cursor="crosshair")
        else:
            self.tab_canvas.config(cursor="crosshair")

    def start_scale_calibration_result(self):
        """Start scale calibration mode on result image"""
        if self.transformed_image is None:
            return

        self.scale_calibrator.start_calibration("result")
        self.status_label.config(text=self.scale_calibrator.get_status_message())

        # Change cursor
        if self.layout_mode == "side-by-side":
            self.result_canvas.config(cursor="crosshair")
        else:
            self.tab_result_canvas.config(cursor="crosshair")

    def cancel_scale_calibration(self):
        """Cancel scale calibration mode"""
        self.scale_calibrator.cancel()

        # Reset cursors
        if self.layout_mode == "side-by-side":
            self.canvas.config(cursor="cross")
            self.result_canvas.config(cursor="arrow")
        else:
            self.tab_canvas.config(cursor="cross")
            self.tab_result_canvas.config(cursor="arrow")

        self.display_on_canvas()
        self.display_on_tab_canvas()
        if self.transformed_image is not None:
            self.display_result()
            self.display_on_tab_result()

    def finish_scale_calibration(self):
        """Show dialog to enter the real-world length"""
        if self.scale_calibrator.get_point_count() != 2:
            return

        # Calculate pixel distance
        pixel_distance = self.scale_calibrator.calculate_pixel_distance()

        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Set Scale")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center dialog
        dialog_width = 300
        dialog_height = 150
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (dialog_width // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dialog_height // 2)
        dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
        dialog.resizable(False, False)

        # Content
        content = ttk.Frame(dialog, padding="20")
        content.pack(fill=tk.BOTH, expand=True)

        units_text = self.units.get()
        if units_text == "pixels":
            units_text = "px"
        elif units_text == "inches":
            units_text = "in"

        ttk.Label(content, text=f"Enter the real-world length of the line:").pack(pady=(0, 10))

        length_frame = ttk.Frame(content)
        length_frame.pack(pady=10)

        length_var = tk.StringVar()
        length_entry = ttk.Entry(length_frame, textvariable=length_var, width=15)
        length_entry.pack(side=tk.LEFT, padx=(0, 5))
        length_entry.focus()

        ttk.Label(length_frame, text=units_text).pack(side=tk.LEFT)

        button_frame = ttk.Frame(content)
        button_frame.pack(pady=(10, 0))

        def on_ok():
            try:
                real_length = float(length_var.get())
                # Use calibrator to set scale (validates length > 0)
                scale_factor = self.scale_calibrator.set_real_world_length(real_length)

                # Update unit converter with new scale factor
                self.unit_converter.set_scale_factor(scale_factor)

                # If we have 4 points already, recalculate dimensions
                if len(self.points) == 4:
                    self.calculate_output_dimensions()

                dialog.destroy()
                self.cancel_scale_calibration()

                self.status_label.config(text=f"Scale set: {pixel_distance:.1f} pixels = {real_length:.1f} {units_text}")

            except ValueError as e:
                self.status_label.config(text=f"Error: {str(e)}")
                dialog.destroy()
                self.cancel_scale_calibration()

        def on_cancel():
            dialog.destroy()
            self.cancel_scale_calibration()

        ttk.Button(button_frame, text="OK", command=on_ok, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side=tk.LEFT, padx=5)

        # Bind Enter key
        dialog.bind('<Return>', lambda e: on_ok())
        dialog.bind('<Escape>', lambda e: on_cancel())

    def show_preferences(self):
        """Show the preferences dialog"""
        # Create modal dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Preferences")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog_width = 350
        dialog_height = 280
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (dialog_width // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dialog_height // 2)
        dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
        dialog.resizable(False, False)

        # Content frame
        content_frame = ttk.Frame(dialog, padding="20")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # DPI Setting
        ttk.Label(content_frame, text="DPI (Dots Per Inch):").grid(row=0, column=0, sticky=tk.W, pady=10)
        dpi_entry = ttk.Entry(content_frame, textvariable=self.dpi_var, width=10)
        dpi_entry.grid(row=0, column=1, padx=10, pady=10)
        ttk.Label(content_frame, text="(Default: 300)").grid(row=1, column=0, columnspan=2, sticky=tk.W)

        # Units Setting
        ttk.Label(content_frame, text="Measurement Units:").grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
        units_frame = ttk.Frame(content_frame)
        units_frame.grid(row=2, column=1, sticky=tk.W, pady=(10, 5))
        ttk.Radiobutton(units_frame, text="mm", variable=self.units, value="mm").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(units_frame, text="inches", variable=self.units, value="inches").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(units_frame, text="pixels", variable=self.units, value="pixels").pack(side=tk.LEFT, padx=5)

        # Crop Mode Setting
        crop_check = ttk.Checkbutton(
            content_frame,
            text="Crop Image",
            variable=self.crop_image
        )
        crop_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(15, 5))

        # Auto-Detect Corners Setting
        auto_detect_check = ttk.Checkbutton(
            content_frame,
            text="Detect corners when loading",
            variable=self.auto_detect_on_load
        )
        auto_detect_check.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(5, 5))

        # Button frame
        button_frame = ttk.Frame(content_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=(20, 0))

        def on_ok():
            # Validate DPI value
            try:
                dpi_value = int(self.dpi_var.get())
                if dpi_value < 1 or dpi_value > 9999:
                    raise ValueError("DPI must be between 1 and 9999")
            except ValueError:
                self.dpi_var.set("300")
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        ttk.Button(button_frame, text="OK", command=on_ok, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side=tk.LEFT, padx=5)

        # Wait for the dialog to close
        self.root.wait_window(dialog)

    def zoom_in(self, center_x=None, center_y=None):
        """Zoom in by 20%, centered on given point (left canvas)"""
        if self.image is None:
            return
        if self.layout_mode == "side-by-side":
            self.left_canvas.zoom_in(center_x, center_y)
            self.update_zoom_display()
            self.display_on_canvas()
        else:
            self.tab_left_canvas.zoom_in(center_x, center_y)
            zoom_pct = self.tab_left_canvas.get_zoom_percentage()
            self.tab_zoom_label.config(text=f"{zoom_pct}%")
            self.display_on_tab_canvas()

    def zoom_out(self, center_x=None, center_y=None):
        """Zoom out by 20%, centered on given point (left canvas)"""
        if self.image is None:
            return
        if self.layout_mode == "side-by-side":
            self.left_canvas.zoom_out(center_x, center_y)
            self.update_zoom_display()
            self.display_on_canvas()
        else:
            self.tab_left_canvas.zoom_out(center_x, center_y)
            zoom_pct = self.tab_left_canvas.get_zoom_percentage()
            self.tab_zoom_label.config(text=f"{zoom_pct}%")
            self.display_on_tab_canvas()

    def zoom_fit(self):
        """Reset zoom to fit image in canvas (left canvas)"""
        if self.image is None:
            return
        if self.layout_mode == "side-by-side":
            self.left_canvas.zoom_fit()
            self.update_zoom_display()
            self.display_on_canvas()
        else:
            self.tab_left_canvas.zoom_fit()
            zoom_pct = self.tab_left_canvas.get_zoom_percentage()
            self.tab_zoom_label.config(text=f"{zoom_pct}%")
            self.display_on_tab_canvas()

    def update_zoom_display(self):
        """Update zoom percentage label for left canvas"""
        zoom_pct = self.left_canvas.get_zoom_percentage()
        self.zoom_label.config(text=f"{zoom_pct}%")

    def on_mouse_wheel(self, event):
        """Handle mouse wheel zoom centered on cursor (left canvas)"""
        if self.image is None:
            return

        # zoom_in/zoom_out already handle layout_mode switching
        if event.delta > 0:
            self.zoom_in(event.x, event.y)
        else:
            self.zoom_out(event.x, event.y)

    def on_window_resize(self, event):
        """Handle window resize to switch between side-by-side and tabbed layouts"""
        # Only handle window resize events (not widget resize events)
        if event.widget != self.root:
            return

        window_width = event.width

        # Implement hysteresis to prevent rapid switching
        # When in side-by-side mode, need to go below (threshold - hysteresis) to switch to tabbed
        # When in tabbed mode, need to go above (threshold + hysteresis) to switch to side-by-side

        if self.layout_mode == "side-by-side":
            # Currently in side-by-side, switch to tabbed if width drops below lower threshold
            if window_width < (self.layout_threshold_width - self.layout_hysteresis):
                self.switch_layout_mode("tabbed")
        else:
            # Currently in tabbed, switch to side-by-side if width rises above upper threshold
            if window_width >= (self.layout_threshold_width + self.layout_hysteresis):
                self.switch_layout_mode("side-by-side")

    def switch_layout_mode(self, mode):
        """Switch between side-by-side and tabbed layout modes"""
        if mode == self.layout_mode:
            return

        self.layout_mode = mode

        if mode == "tabbed":
            # Switch to tabbed mode
            self.setup_tabbed_layout()
        else:
            # Switch to side-by-side mode
            self.setup_sidebyside_layout()

        # Force UI update
        self.root.update_idletasks()

        # Refresh displays after layout change
        if self.image is not None:
            self.display_on_canvas()
        if self.transformed_image is not None:
            self.display_result()

    def setup_tabbed_layout(self):
        """Switch to tabbed layout"""
        # Hide side-by-side container
        self.sidebyside_container.grid_forget()

        # Show notebook
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Force update and redraw
        self.root.update_idletasks()

        if self.image is not None:
            self.display_on_tab_canvas()
        if self.transformed_image is not None:
            self.display_on_tab_result()

    def setup_sidebyside_layout(self):
        """Switch to side-by-side layout"""
        # Hide notebook
        self.notebook.grid_forget()

        # Show side-by-side container
        self.sidebyside_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Force update and redraw
        self.root.update_idletasks()

        if self.image is not None:
            self.display_on_canvas()
        if self.transformed_image is not None:
            self.display_result()

    def on_tab_canvas_resize(self, event):
        """Handle tab canvas resize events"""
        new_width = event.width
        new_height = event.height

        if new_width > 1 and new_height > 1:
            if new_width != self.tab_left_canvas.canvas_width or new_height != self.tab_left_canvas.canvas_height:
                self.tab_left_canvas.update_canvas_size(new_width, new_height)
                if self.image is not None:
                    self.tab_left_canvas.reset_view()
                    self.display_on_tab_canvas()

    def on_tab_result_canvas_resize(self, event):
        """Handle tab result canvas resize events"""
        new_width = event.width
        new_height = event.height

        if new_width > 1 and new_height > 1:
            if new_width != self.tab_right_canvas.canvas_width or new_height != self.tab_right_canvas.canvas_height:
                self.tab_right_canvas.update_canvas_size(new_width, new_height)
                if self.transformed_image is not None:
                    self.tab_right_canvas.reset_view()
                    self.display_on_tab_result()

    def on_canvas_resize(self, event):
        """Handle canvas resize events"""
        # Update canvas dimensions when window is resized
        new_width = event.width
        new_height = event.height

        # Only update if dimensions actually changed and are valid
        if new_width > 1 and new_height > 1:
            # In tabbed mode, update both canvas dimensions since they share the space
            if self.layout_mode == "tabbed":
                self.canvas_width = new_width
                self.canvas_height = new_height
                self.left_canvas.update_canvas_size(new_width, new_height)
                self.right_canvas.update_canvas_size(new_width, new_height)

                # Redisplay both images if they exist
                if self.image is not None:
                    self.left_canvas.reset_view()
                    self.display_on_canvas()
                if self.transformed_image is not None:
                    self.right_canvas.reset_view()
                    self.display_result()
            elif new_width != self.left_canvas.canvas_width or new_height != self.left_canvas.canvas_height:
                # In side-by-side mode, only update left canvas
                self.left_canvas.update_canvas_size(new_width, new_height)

                # Redisplay the image with new dimensions
                if self.image is not None:
                    self.left_canvas.reset_view()
                    self.display_on_canvas()

    def on_result_canvas_resize(self, event):
        """Handle result canvas resize events"""
        new_width = event.width
        new_height = event.height

        # Only handle in side-by-side mode (tabbed mode is handled in on_canvas_resize)
        if self.layout_mode == "side-by-side" and new_width > 1 and new_height > 1:
            if new_width != self.right_canvas.canvas_width or new_height != self.right_canvas.canvas_height:
                self.right_canvas.update_canvas_size(new_width, new_height)

                # Redisplay the result image if it exists
                if self.transformed_image is not None:
                    self.right_canvas.reset_view()
                    self.display_result()

    def result_zoom_in(self, center_x=None, center_y=None):
        """Zoom in on result canvas by 20%, centered on given point"""
        if self.transformed_image is None:
            return
        if self.layout_mode == "side-by-side":
            self.right_canvas.zoom_in(center_x, center_y)
            self.display_result()
        else:
            self.tab_right_canvas.zoom_in(center_x, center_y)
            self.display_on_tab_result()

    def result_zoom_out(self, center_x=None, center_y=None):
        """Zoom out on result canvas by 20%, centered on given point"""
        if self.transformed_image is None:
            return
        if self.layout_mode == "side-by-side":
            self.right_canvas.zoom_out(center_x, center_y)
            self.display_result()
        else:
            self.tab_right_canvas.zoom_out(center_x, center_y)
            self.display_on_tab_result()

    def result_zoom_fit(self):
        """Reset result canvas zoom to fit image"""
        if self.transformed_image is None:
            return
        if self.layout_mode == "side-by-side":
            self.right_canvas.zoom_fit()
            self.display_result()
        else:
            self.tab_right_canvas.zoom_fit()
            self.display_on_tab_result()

    def result_update_zoom_display(self):
        """Update result zoom percentage label"""
        zoom_pct = self.right_canvas.get_zoom_percentage()
        self.result_zoom_label.config(text=f"{zoom_pct}%")

    def on_canvas_right_click(self, event):
        """Start panning with right mouse button"""
        if self.image is not None:
            canvas_helper = self.left_canvas if self.layout_mode == "side-by-side" else self.tab_left_canvas
            canvas_helper.start_pan(event.x, event.y)

    def on_canvas_context_menu(self, event):
        """Show context menu on canvas (Shift+Right-Click or platform context menu key)"""
        if self.image is None:
            return

        # Set context menu side to left
        self.context_menu_side = "left"

        # Configure menu items based on which side we're on
        # Left side: disable "Save Result" and "Use as Original", enable others
        self.context_menu.entryconfig("Save Result...", state=tk.DISABLED)
        self.context_menu.entryconfig("Use as Original", state=tk.DISABLED)
        self.context_menu.entryconfig("Crop Mode", state=tk.DISABLED)

        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def on_canvas_pan(self, event):
        """Pan the image with right mouse button"""
        if self.image is not None:
            canvas_helper = self.left_canvas if self.layout_mode == "side-by-side" else self.tab_left_canvas
            if canvas_helper.update_pan(event.x, event.y):
                # Only update the active canvas for pan
                if self.layout_mode == "side-by-side":
                    self.display_on_canvas()
                else:
                    self.display_on_tab_canvas()

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.heic *.heif"), ("All files", "*.*")]
        )

        if file_path:
            self.load_image_from_path(file_path)

    def load_image_from_path(self, file_path):
        """Load an image from the given file path"""
        logging.info(f"Loading image: {file_path}")
        # Check if file is HEIC format (needs special handling)
        is_heic = file_path.lower().endswith(('.heic', '.heif'))

        if is_heic:
            # Load HEIC with PIL/pillow-heif, then convert to numpy array for OpenCV
            try:
                pil_image = Image.open(file_path)
                # Convert PIL image to RGB numpy array
                self.original_image = np.array(pil_image.convert('RGB'))
            except Exception as e:
                self.status_label.config(text=f"Error: Could not load HEIC image - {e}")
                return
        else:
            # Load image with OpenCV for standard formats
            self.original_image = cv3.imread(file_path)
            if self.original_image is None:
                self.status_label.config(text="Error: Could not load image")
                return

        # Store the original file path for save dialog
        self.original_file_path = file_path

        # Read DPI from image metadata using PIL
        try:
            pil_image = Image.open(file_path)
            dpi_info = pil_image.info.get('dpi')
            if dpi_info:
                # DPI info is a tuple (x_dpi, y_dpi), use x_dpi
                self.input_dpi = int(dpi_info[0])
            else:
                # No DPI metadata found, use output DPI as default
                self.input_dpi = int(self.dpi_var.get())
        except Exception:
            # If we can't read DPI, use output DPI as default
            self.input_dpi = int(self.dpi_var.get())

        # cv3 loads images in RGB by default (no conversion needed)
        self.image = self.original_image

        # Reset state
        self.points = []
        self.transformed_image = None
        self.left_canvas.reset_view()

        # Clear the result canvas
        if self.right_canvas:
            self.right_canvas.clear()
        if hasattr(self, 'tab_right_canvas') and self.tab_right_canvas:
            self.tab_right_canvas.clear()

        # Clear dimension fields and reset manual flag
        self._updating_dimensions = True
        self.width_var.set("")
        self.height_var.set("")
        self._updating_dimensions = False
        self.dimensions_manually_set = False
        # Reset page size selector to Custom
        self.page_size_var.set("Custom")

        # Display the image on both canvases
        self.display_on_canvas()
        self.display_on_tab_canvas()
        self.status_label.config(text=f"Image loaded (Input DPI: {self.input_dpi}). Click 4 corners (drag to adjust). Drag to pan, scroll to zoom.")

        # Auto-detect corners if preference is enabled
        if self.auto_detect_on_load.get():
            self.auto_detect_corners(show_debug=False)

    def display_on_canvas(self):
        if self.image is None:
            return

        # Define overlay callback to draw points and lines
        def draw_points_overlay(canvas_image, effective_scale, pan_offset):
            # Draw scale calibration line if in scale mode for original
            if self.scale_mode == "original" and len(self.scale_points) > 0:
                # Draw line first if we have 2 points
                if len(self.scale_points) == 2:
                    pt1_x, pt1_y = self.left_canvas.image_to_canvas_coords(
                        self.scale_points[0][0], self.scale_points[0][1])
                    pt2_x, pt2_y = self.left_canvas.image_to_canvas_coords(
                        self.scale_points[1][0], self.scale_points[1][1])
                    # Draw thin cyan line
                    cv3.line(canvas_image, int(pt1_x), int(pt1_y),
                            int(pt2_x), int(pt2_y), color=(0, 255, 255), t=1)

                # Draw endpoints on top with smaller circles
                for i, pt in enumerate(self.scale_points):
                    pt_x, pt_y = self.left_canvas.image_to_canvas_coords(pt[0], pt[1])
                    pt_x, pt_y = int(pt_x), int(pt_y)
                    # Outer circle (cyan)
                    cv3.circle(canvas_image, pt_x, pt_y, 5, color=(0, 255, 255), t=1)
                    # Inner filled circle (cyan)
                    cv3.circle(canvas_image, pt_x, pt_y, 3, color=(0, 255, 255), fill=True)

            # Draw lines connecting points
            # If we have 4 points, reorder them to form a proper quadrilateral
            if len(self.points) == 4:
                ordered_rect = self.order_points(self.points)
                points_to_draw = ordered_rect
            else:
                points_to_draw = self.points

            for i in range(len(points_to_draw)):
                if i < len(points_to_draw) - 1 or len(points_to_draw) == 4:
                    next_i = (i + 1) % len(points_to_draw)
                    pt1_x, pt1_y = self.left_canvas.image_to_canvas_coords(
                        points_to_draw[i][0], points_to_draw[i][1])
                    pt2_x, pt2_y = self.left_canvas.image_to_canvas_coords(
                        points_to_draw[next_i][0], points_to_draw[next_i][1])

                    # Draw line (OpenCV automatically clips to image bounds)
                    cv3.line(canvas_image, int(pt1_x), int(pt1_y),
                            int(pt2_x), int(pt2_y), color=(0, 255, 0), t=2)

            # Draw points on top
            for i, pt in enumerate(self.points):
                pt_x, pt_y = self.left_canvas.image_to_canvas_coords(pt[0], pt[1])
                pt_x, pt_y = int(pt_x), int(pt_y)

                # Only draw if within canvas bounds (with some margin for visibility)
                if -20 <= pt_x < self.left_canvas.canvas_width + 20 and \
                   -20 <= pt_y < self.left_canvas.canvas_height + 20:
                    # Outer circle (same size as scale points)
                    cv3.circle(canvas_image, pt_x, pt_y, 5, color=(0, 0, 255), t=1)
                    # Inner filled circle (same size as scale points)
                    cv3.circle(canvas_image, pt_x, pt_y, 3, color=(255, 0, 0), fill=True)

        # Display image with overlay
        self.left_canvas.display_image(self.image, overlay_callback=draw_points_overlay)
        self.update_zoom_display()

    def display_on_tab_canvas(self):
        """Display image on tab canvas (same as display_on_canvas but for tab)"""
        if self.image is None:
            return

        # Define overlay callback to draw points and lines
        def draw_points_overlay(canvas_image, effective_scale, pan_offset):
            # Draw scale calibration line if in scale mode for original
            if self.scale_mode == "original" and len(self.scale_points) > 0:
                # Draw line first if we have 2 points
                if len(self.scale_points) == 2:
                    pt1_x, pt1_y = self.tab_left_canvas.image_to_canvas_coords(
                        self.scale_points[0][0], self.scale_points[0][1])
                    pt2_x, pt2_y = self.tab_left_canvas.image_to_canvas_coords(
                        self.scale_points[1][0], self.scale_points[1][1])
                    # Draw thin cyan line
                    cv3.line(canvas_image, int(pt1_x), int(pt1_y),
                            int(pt2_x), int(pt2_y), color=(0, 255, 255), t=1)

                # Draw endpoints on top with smaller circles
                for i, pt in enumerate(self.scale_points):
                    pt_x, pt_y = self.tab_left_canvas.image_to_canvas_coords(pt[0], pt[1])
                    pt_x, pt_y = int(pt_x), int(pt_y)
                    # Outer circle (cyan)
                    cv3.circle(canvas_image, pt_x, pt_y, 5, color=(0, 255, 255), t=1)
                    # Inner filled circle (cyan)
                    cv3.circle(canvas_image, pt_x, pt_y, 3, color=(0, 255, 255), fill=True)

            # Draw lines connecting points
            # If we have 4 points, reorder them to form a proper quadrilateral
            if len(self.points) == 4:
                ordered_rect = self.order_points(self.points)
                points_to_draw = ordered_rect
            else:
                points_to_draw = self.points

            for i in range(len(points_to_draw)):
                if i < len(points_to_draw) - 1 or len(points_to_draw) == 4:
                    next_i = (i + 1) % len(points_to_draw)
                    pt1_x, pt1_y = self.tab_left_canvas.image_to_canvas_coords(
                        points_to_draw[i][0], points_to_draw[i][1])
                    pt2_x, pt2_y = self.tab_left_canvas.image_to_canvas_coords(
                        points_to_draw[next_i][0], points_to_draw[next_i][1])

                    # Draw line
                    cv3.line(canvas_image, int(pt1_x), int(pt1_y),
                            int(pt2_x), int(pt2_y), color=(0, 255, 0), t=2)

            # Draw points on top
            for i, pt in enumerate(self.points):
                pt_x, pt_y = self.tab_left_canvas.image_to_canvas_coords(pt[0], pt[1])
                pt_x, pt_y = int(pt_x), int(pt_y)

                # Only draw if within canvas bounds
                if -20 <= pt_x < self.tab_left_canvas.canvas_width + 20 and \
                   -20 <= pt_y < self.tab_left_canvas.canvas_height + 20:
                    # Outer circle (same size as scale points)
                    cv3.circle(canvas_image, pt_x, pt_y, 5, color=(0, 0, 255), t=1)
                    # Inner filled circle (same size as scale points)
                    cv3.circle(canvas_image, pt_x, pt_y, 3, color=(255, 0, 0), fill=True)

        # Display image with overlay
        self.tab_left_canvas.display_image(self.image, overlay_callback=draw_points_overlay)
        # Update zoom label
        zoom_pct = self.tab_left_canvas.get_zoom_percentage()
        self.tab_zoom_label.config(text=f"{zoom_pct}%")

    def display_on_tab_result(self):
        """Display result image on tab result canvas"""
        if self.transformed_image is None:
            return

        # Define overlay callback to draw scale line if in result scale mode
        def draw_scale_overlay(canvas_image, effective_scale, pan_offset):
            if self.scale_mode == "result" and len(self.scale_points) > 0:
                # Draw line first if we have 2 points
                if len(self.scale_points) == 2:
                    pt1_x, pt1_y = self.tab_right_canvas.image_to_canvas_coords(
                        self.scale_points[0][0], self.scale_points[0][1])
                    pt2_x, pt2_y = self.tab_right_canvas.image_to_canvas_coords(
                        self.scale_points[1][0], self.scale_points[1][1])
                    # Draw thin cyan line
                    cv3.line(canvas_image, int(pt1_x), int(pt1_y),
                            int(pt2_x), int(pt2_y), color=(0, 255, 255), t=1)

                # Draw endpoints on top with smaller circles
                for i, pt in enumerate(self.scale_points):
                    pt_x, pt_y = self.tab_right_canvas.image_to_canvas_coords(pt[0], pt[1])
                    pt_x, pt_y = int(pt_x), int(pt_y)
                    # Outer circle (cyan)
                    cv3.circle(canvas_image, pt_x, pt_y, 5, color=(0, 255, 255), t=1)
                    # Inner filled circle (cyan)
                    cv3.circle(canvas_image, pt_x, pt_y, 3, color=(0, 255, 255), fill=True)

        overlay_callback = draw_scale_overlay if self.scale_mode == "result" else None
        self.tab_right_canvas.display_image(self.transformed_image, overlay_callback=overlay_callback)
        zoom_pct = self.tab_right_canvas.get_zoom_percentage()
        self.tab_result_zoom_label.config(text=f"{zoom_pct}%")

    def get_point_at_position(self, x, y, threshold=15):
        """Find if there's a point near the given position"""
        # Use appropriate canvas based on layout mode
        canvas_helper = self.left_canvas if self.layout_mode == "side-by-side" else self.tab_left_canvas

        for i, pt in enumerate(self.points):
            scaled_x, scaled_y = canvas_helper.image_to_canvas_coords(pt[0], pt[1])

            distance = np.sqrt((x - scaled_x)**2 + (y - scaled_y)**2)
            if distance < threshold:
                return i
        return None

    def get_scale_point_at_position(self, x, y, threshold=10):
        """Find if there's a scale point near the given position"""
        if len(self.scale_points) == 0:
            return None

        # Use appropriate canvas based on layout mode and scale mode
        if self.scale_mode == "original":
            canvas_helper = self.left_canvas if self.layout_mode == "side-by-side" else self.tab_left_canvas
        elif self.scale_mode == "result":
            canvas_helper = self.right_canvas if self.layout_mode == "side-by-side" else self.tab_right_canvas
        else:
            return None

        for i, pt in enumerate(self.scale_points):
            scaled_x, scaled_y = canvas_helper.image_to_canvas_coords(pt[0], pt[1])

            distance = np.sqrt((x - scaled_x)**2 + (y - scaled_y)**2)
            if distance < threshold:
                return i
        return None

    def on_canvas_click(self, event):
        if self.image is None:
            return

        # Use appropriate canvas based on layout mode
        canvas_helper = self.left_canvas if self.layout_mode == "side-by-side" else self.tab_left_canvas
        canvas_widget = self.canvas if self.layout_mode == "side-by-side" else self.tab_canvas

        # Handle scale calibration mode on original
        if self.scale_mode == "original":
            # Check if clicking near an existing scale point to drag it
            scale_point_idx = self.get_scale_point_at_position(event.x, event.y)
            if scale_point_idx is not None:
                self.dragging_scale_point = scale_point_idx
                self.drag_start = (event.x, event.y)
                canvas_widget.config(cursor="hand2")
                return

            # Add new point if we don't have 2 yet
            if len(self.scale_points) < 2:
                x, y = canvas_helper.canvas_to_image_coords(event.x, event.y)
                self.scale_points.append((x, y))

                # Always display to show the new point
                self.display_on_canvas()
                self.display_on_tab_canvas()

                if len(self.scale_points) == 2:
                    # Schedule the dialog to appear after canvas updates
                    self.root.after(10, self.finish_scale_calibration)
            return

        # Check if clicking on existing point
        point_idx = self.get_point_at_position(event.x, event.y)

        if point_idx is not None:
            # Start dragging this point
            self.dragging_point = point_idx
            self.drag_start = (event.x, event.y)
            canvas_widget.config(cursor="hand2")
        elif len(self.points) < 4:
            # Add new point - convert canvas coordinates to image coordinates
            x, y = canvas_helper.canvas_to_image_coords(event.x, event.y)

            # Allow points beyond image boundaries (no clamping)
            self.points.append((x, y))
            logging.info(f"Point {len(self.points)} added at ({x:.1f}, {y:.1f})")

            # Display on both canvases to keep them in sync
            self.display_on_canvas()
            self.display_on_tab_canvas()

            if len(self.points) == 4:
                self.calculate_output_dimensions()
                self.transform_btn.config(state=tk.NORMAL)
                if self.layout_mode == "tabbed":
                    self.tab_transform_btn.config(state=tk.NORMAL)
                # Auto-apply transform when 4th point is placed
                self.apply_transform()
                self.status_label.config(text="Transform applied. Adjust points if needed.")
            else:
                labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
                self.status_label.config(text=f"Point {len(self.points)}/4 added. Suggest: {labels[len(self.points)]}")
        else:
            # All 4 points already placed, start panning
            canvas_helper.start_pan(event.x, event.y)
            canvas_widget.config(cursor="fleur")

    def on_canvas_drag(self, event):
        if self.image is None:
            return

        # Use appropriate canvas based on layout mode
        canvas_helper = self.left_canvas if self.layout_mode == "side-by-side" else self.tab_left_canvas

        if self.dragging_point is not None:
            # Update point position - convert canvas coordinates to image coordinates
            x, y = canvas_helper.canvas_to_image_coords(event.x, event.y)

            # Allow points beyond image boundaries (no clamping)
            self.points[self.dragging_point] = (x, y)

            # Display on both canvases to keep them in sync
            self.display_on_canvas()
            self.display_on_tab_canvas()
        elif self.dragging_scale_point is not None and self.scale_mode == "original":
            # Update scale point position
            x, y = canvas_helper.canvas_to_image_coords(event.x, event.y)
            self.scale_points[self.dragging_scale_point] = (x, y)

            # Display on both canvases to keep them in sync
            self.display_on_canvas()
            self.display_on_tab_canvas()
        elif canvas_helper.update_pan(event.x, event.y):
            # Pan the image on the active canvas only
            if self.layout_mode == "side-by-side":
                self.display_on_canvas()
            else:
                self.display_on_tab_canvas()

    def on_canvas_release(self, event):
        # Use appropriate canvas based on layout mode
        canvas_helper = self.left_canvas if self.layout_mode == "side-by-side" else self.tab_left_canvas
        canvas_widget = self.canvas if self.layout_mode == "side-by-side" else self.tab_canvas

        if self.dragging_point is not None:
            self.dragging_point = None
            self.drag_start = None
            canvas_widget.config(cursor="cross")
            # Recalculate dimensions and auto-apply after dragging a point
            if len(self.points) == 4:
                self.calculate_output_dimensions()
                self.apply_transform()
        elif self.dragging_scale_point is not None:
            self.dragging_scale_point = None
            canvas_widget.config(cursor="crosshair")
        elif canvas_helper.panning:
            canvas_helper.end_pan()
            canvas_widget.config(cursor="cross")

    def reset_points(self):
        self.points = []
        self.transformed_image = None
        self.transform_btn.config(state=tk.DISABLED)
        self.file_menu.entryconfig("Save Result...", state=tk.DISABLED)
        self.result_canvas.delete("all")

        # Clear dimension fields and reset manual flag
        self._updating_dimensions = True
        self.width_var.set("")
        self.height_var.set("")
        self._updating_dimensions = False
        self.dimensions_manually_set = False

        if self.image is not None:
            self.display_on_canvas()
            self.status_label.config(text="Points reset. Click 4 corner points to begin.")

    def auto_detect_corners(self, show_debug=False):
        """Automatically detect document corners using edge and contour detection"""
        if self.image is None:
            self.status_label.config(text="Please load an image first")
            return

        # Use CornerDetector to find corners
        detected_points = self.corner_detector.detect(self.image)

        if detected_points is not None:
            # Set the detected points
            self.points = detected_points

            # Update display
            self.display_on_canvas()
            self.display_on_tab_canvas()

            # Calculate dimensions and enable transform
            self.calculate_output_dimensions()
            self.transform_btn.config(state=tk.NORMAL)
            self.tab_transform_btn.config(state=tk.NORMAL)

            # Auto-apply transform after auto-detection
            self.apply_transform()
            self.status_label.config(text="Auto-detected 4 corners and applied transform. Adjust points if needed.")

            # Show debug window if enabled
            if show_debug:
                self._show_detection_debug()
        else:
            self.status_label.config(text="Could not detect document corners. Try manual selection.")

            # Show debug window to help diagnose
            if show_debug:
                self._show_detection_debug()

    def _show_detection_debug(self):
        """Show debug window with edge detection and contour analysis"""
        if self.image is None:
            return

        # Use CornerDetector to create debug visualization
        combined, info_dict = self.corner_detector.create_debug_visualization(self.image)

        # Create popup window
        debug_window = tk.Toplevel(self.root)
        debug_window.title("Corner Detection Debug")
        debug_window.geometry(f"{info_dict['width'] + 20}x{info_dict['height'] + 60}")

        # Convert to PIL and display
        pil_debug = Image.fromarray(combined)
        photo = ImageTk.PhotoImage(pil_debug)

        label = tk.Label(debug_window, image=photo)
        label.image = photo  # Keep reference
        label.pack(pady=5)

        # Add close button
        close_btn = tk.Button(debug_window, text="Close", command=debug_window.destroy)
        close_btn.pack(pady=5)

    def order_points(self, pts):
        """Order points as: top-left, top-right, bottom-right, bottom-left"""
        return self.corner_detector.order_points(pts)

    def calculate_output_dimensions(self):
        """Calculate output dimensions based on the distances between selected points"""
        if len(self.points) != 4:
            return

        # Don't recalculate if user has manually set dimensions
        if self.dimensions_manually_set:
            return

        # Order the points
        rect = self.order_points(self.points)
        (tl, tr, br, bl) = rect

        # Calculate distances between points (in pixels)
        top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        bottom_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        left_height = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
        right_height = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))

        # Average the opposing sides
        avg_width_pixels = (top_width + bottom_width) / 2.0
        avg_height_pixels = (left_height + right_height) / 2.0

        # Convert pixels to current units
        # If scale has been calibrated, use scale_factor; otherwise use input image DPI
        if self.scale_length is not None and self.scale_factor != 1.0:
            # Scale factor is pixels per unit, so divide pixels by scale_factor to get units
            width_value = avg_width_pixels / self.scale_factor
            height_value = avg_height_pixels / self.scale_factor
        else:
            # Use input image DPI for conversion
            width_value = self.pixels_to_units(avg_width_pixels, dpi=self.input_dpi)
            height_value = self.pixels_to_units(avg_height_pixels, dpi=self.input_dpi)

        # Update the dimension fields
        # Set flag to prevent triggering the manual edit callback
        self._updating_dimensions = True
        if self.units.get() == "pixels":
            # Pixels - show as integer
            self.width_var.set(str(int(round(width_value))))
            self.height_var.set(str(int(round(height_value))))
        else:
            # mm or inches - show with 1 decimal place
            self.width_var.set(f"{width_value:.1f}")
            self.height_var.set(f"{height_value:.1f}")
        self._updating_dimensions = False

    def apply_transform(self):
        if len(self.points) != 4:
            self.status_label.config(text="Please select exactly 4 points")
            return

        # Order the points
        rect = self.order_points(self.points)
        (tl, tr, br, bl) = rect

        # Get DPI
        try:
            dpi = int(self.dpi_var.get())
        except ValueError:
            dpi = 300

        # Check crop mode
        crop_enabled = self.crop_image.get()

        # Log transform parameters
        width_val = self.width_var.get()
        height_val = self.height_var.get()
        units = self.units.get()
        logging.info(f"Transform: points={self.points}, width={width_val}, height={height_val}, units={units}, dpi={dpi}, crop={crop_enabled}")
        if not crop_enabled:
            # Transform entire image mode (crop unchecked)

            # Get dimensions from spinbox (user's real-world measurements)
            try:
                width_value = float(self.width_var.get())
                height_value = float(self.height_var.get())
            except ValueError:
                self.status_label.config(text="Invalid dimensions")
                return

            # Convert to pixels for the quadrilateral
            quad_width = self.units_to_pixels(width_value)
            quad_height = self.units_to_pixels(height_value)

            # Get original image dimensions
            img_height, img_width = self.original_image.shape[:2]

            # First, create a temporary transform to see where the image corners will end up
            # Map the selected quad to origin with user's specified dimensions
            temp_dst = np.array([
                [0, 0],
                [quad_width - 1, 0],
                [quad_width - 1, quad_height - 1],
                [0, quad_height - 1]], dtype="float32")

            M_temp = cv2.getPerspectiveTransform(rect, temp_dst)

            # Transform the four corners of the original image to see the bounding box
            img_corners = np.array([
                [0, 0],
                [img_width - 1, 0],
                [img_width - 1, img_height - 1],
                [0, img_height - 1]], dtype="float32")

            # Apply transform to each corner
            transformed_corners = cv2.perspectiveTransform(img_corners.reshape(-1, 1, 2), M_temp).reshape(-1, 2)

            # Find bounding box of transformed image
            min_x = int(np.floor(transformed_corners[:, 0].min()))
            max_x = int(np.ceil(transformed_corners[:, 0].max()))
            min_y = int(np.floor(transformed_corners[:, 1].min()))
            max_y = int(np.ceil(transformed_corners[:, 1].max()))

            # Calculate output canvas size
            output_width = max_x - min_x
            output_height = max_y - min_y

            # Adjust destination points to account for negative offsets
            # This ensures the entire transformed image fits in the output
            offset_x = -min_x
            offset_y = -min_y

            dst = np.array([
                [offset_x, offset_y],
                [offset_x + quad_width - 1, offset_y],
                [offset_x + quad_width - 1, offset_y + quad_height - 1],
                [offset_x, offset_y + quad_height - 1]], dtype="float32")

            # Compute final perspective transform
            M = cv2.getPerspectiveTransform(rect, dst)

            # Convert to BGR for warpPerspective, then back to RGB
            original_bgr = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
            transformed_bgr = cv2.warpPerspective(original_bgr, M, (output_width, output_height))
            self.transformed_image = cv2.cvtColor(transformed_bgr, cv2.COLOR_BGR2RGB)

            # Calculate output dimensions in mm
            output_width_mm = (output_width / dpi) * 25.4
            output_height_mm = (output_height / dpi) * 25.4

            # Get units abbreviation for display
            units_abbr = "px" if self.units.get() == "pixels" else ("in" if self.units.get() == "inches" else "mm")
            status_msg = f"Transform applied! Output: {output_width_mm:.0f}x{output_height_mm:.0f}mm @ {dpi}DPI ({output_width}x{output_height}px) [Full image, quad={width_value:.1f}x{height_value:.1f}{units_abbr}]"
        else:
            # Crop to selected region mode (original behavior)
            # Get dimensions in current units and convert to pixels
            try:
                width_value = float(self.width_var.get())
                height_value = float(self.height_var.get())
            except ValueError:
                self.status_label.config(text="Invalid dimensions")
                return

            # Convert to pixels
            output_width = self.units_to_pixels(width_value)
            output_height = self.units_to_pixels(height_value)

            # Destination points
            dst = np.array([
                [0, 0],
                [output_width - 1, 0],
                [output_width - 1, output_height - 1],
                [0, output_height - 1]], dtype="float32")

            # Compute perspective transform
            M = cv2.getPerspectiveTransform(rect, dst)

            # Convert to BGR for warpPerspective, then back to RGB
            original_bgr = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
            transformed_bgr = cv2.warpPerspective(original_bgr, M, (output_width, output_height))
            self.transformed_image = cv2.cvtColor(transformed_bgr, cv2.COLOR_BGR2RGB)

            # Get units abbreviation for display
            units_abbr = "px" if self.units.get() == "pixels" else ("in" if self.units.get() == "inches" else "mm")
            status_msg = f"Transform applied! Output: {width_value:.1f}x{height_value:.1f}{units_abbr} @ {dpi}DPI ({output_width}x{output_height}px)"

        # Reset zoom and pan for new result (both canvases)
        self.right_canvas.reset_view()
        self.tab_right_canvas.reset_view()

        # Display result on both canvases
        self.display_result()
        self.display_on_tab_result()
        self.status_label.config(text=status_msg)
        self.file_menu.entryconfig("Save Result...", state=tk.NORMAL)

    def display_result(self):
        if self.transformed_image is None:
            return

        # Define overlay callback to draw scale line if in result scale mode
        def draw_scale_overlay(canvas_image, effective_scale, pan_offset):
            if self.scale_mode == "result" and len(self.scale_points) > 0:
                # Draw line first if we have 2 points
                if len(self.scale_points) == 2:
                    pt1_x, pt1_y = self.right_canvas.image_to_canvas_coords(
                        self.scale_points[0][0], self.scale_points[0][1])
                    pt2_x, pt2_y = self.right_canvas.image_to_canvas_coords(
                        self.scale_points[1][0], self.scale_points[1][1])
                    # Draw thin cyan line
                    cv3.line(canvas_image, int(pt1_x), int(pt1_y),
                            int(pt2_x), int(pt2_y), color=(0, 255, 255), t=1)

                # Draw endpoints on top with smaller circles
                for i, pt in enumerate(self.scale_points):
                    pt_x, pt_y = self.right_canvas.image_to_canvas_coords(pt[0], pt[1])
                    pt_x, pt_y = int(pt_x), int(pt_y)
                    # Outer circle (cyan)
                    cv3.circle(canvas_image, pt_x, pt_y, 5, color=(0, 255, 255), t=1)
                    # Inner filled circle (cyan)
                    cv3.circle(canvas_image, pt_x, pt_y, 3, color=(0, 255, 255), fill=True)

        # Display result image using right_canvas ImageCanvas helper
        overlay_callback = draw_scale_overlay if self.scale_mode == "result" else None
        self.right_canvas.display_image(self.transformed_image, overlay_callback=overlay_callback)
        self.result_update_zoom_display()

    def on_result_mouse_wheel(self, event):
        """Handle mouse wheel zoom on result canvas centered on cursor"""
        if self.transformed_image is None:
            return

        if event.delta > 0:
            self.result_zoom_in(event.x, event.y)
        else:
            self.result_zoom_out(event.x, event.y)

    def on_result_canvas_click(self, event):
        """Start panning on result canvas with left mouse button"""
        if self.transformed_image is None:
            return

        # Use appropriate canvas based on layout mode
        canvas_helper = self.right_canvas if self.layout_mode == "side-by-side" else self.tab_right_canvas
        canvas_widget = self.result_canvas if self.layout_mode == "side-by-side" else self.tab_result_canvas

        # Handle scale calibration mode on result
        if self.scale_mode == "result":
            # Check if clicking near an existing scale point to drag it
            scale_point_idx = self.get_scale_point_at_position(event.x, event.y)
            if scale_point_idx is not None:
                self.dragging_scale_point = scale_point_idx
                self.drag_start = (event.x, event.y)
                canvas_widget.config(cursor="hand2")
                return

            # Add new point if we don't have 2 yet
            if len(self.scale_points) < 2:
                x, y = canvas_helper.canvas_to_image_coords(event.x, event.y)
                self.scale_points.append((x, y))

                # Always display to show the new point
                self.display_result()
                self.display_on_tab_result()

                if len(self.scale_points) == 2:
                    # Schedule the dialog to appear after canvas updates
                    self.root.after(10, self.finish_scale_calibration)
            return

        canvas_helper.start_pan(event.x, event.y)
        canvas_widget.config(cursor="fleur")

    def on_result_canvas_drag(self, event):
        """Pan the result image with left mouse button"""
        if self.transformed_image is None:
            return

        # Use appropriate canvas based on layout mode
        canvas_helper = self.right_canvas if self.layout_mode == "side-by-side" else self.tab_right_canvas

        if self.dragging_scale_point is not None and self.scale_mode == "result":
            # Update scale point position
            x, y = canvas_helper.canvas_to_image_coords(event.x, event.y)
            self.scale_points[self.dragging_scale_point] = (x, y)

            # Display on both canvases to keep them in sync
            self.display_result()
            self.display_on_tab_result()
        elif canvas_helper.update_pan(event.x, event.y):
            # Pan the image on the active canvas only
            if self.layout_mode == "side-by-side":
                self.display_result()
            else:
                self.display_on_tab_result()

    def on_result_canvas_release(self, event):
        """End panning on result canvas"""
        # Use appropriate canvas based on layout mode
        canvas_helper = self.right_canvas if self.layout_mode == "side-by-side" else self.tab_right_canvas
        canvas_widget = self.result_canvas if self.layout_mode == "side-by-side" else self.tab_result_canvas

        if self.dragging_scale_point is not None:
            self.dragging_scale_point = None
            canvas_widget.config(cursor="crosshair")
        elif canvas_helper.panning:
            canvas_helper.end_pan()
            canvas_widget.config(cursor="arrow")

    def on_result_canvas_right_click(self, event):
        """Show context menu on right click in result canvas"""
        if self.transformed_image is None:
            return

        # Set context menu side to right
        self.context_menu_side = "right"

        # Configure menu items based on which side we're on
        # Right side: enable "Save Result", "Use as Original", and "Crop Mode"
        self.context_menu.entryconfig("Save Result...", state=tk.NORMAL)
        self.context_menu.entryconfig("Use as Original", state=tk.NORMAL)
        self.context_menu.entryconfig("Crop Mode", state=tk.NORMAL)

        # Show context menu at cursor position
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def save_image(self):
        if self.transformed_image is None:
            return

        # Determine default extension and filename from original file
        import os
        default_ext = ".jpg"
        initial_file = None

        if self.original_file_path:
            directory = os.path.dirname(self.original_file_path)
            basename = os.path.basename(self.original_file_path)
            name_without_ext, ext = os.path.splitext(basename)

            # Determine extension (convert HEIC to PNG for output)
            if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                default_ext = ext.lower()
                if default_ext == '.jpeg':
                    default_ext = '.jpg'
            elif ext.lower() in ['.heic', '.heif']:
                # HEIC files will be saved as PNG
                default_ext = '.png'

            # 파일명 제안: <원본>_autowarp.<ext>
            suggested_name = f"{name_without_ext}_autowarp{default_ext}"
            initial_file = os.path.join(directory, suggested_name)

        file_path = filedialog.asksaveasfilename(
            defaultextension=default_ext,
            initialfile=initial_file if initial_file else None,
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("BMP", "*.bmp"), ("All files", "*.*")]
        )

        if file_path:
            # Get output DPI - if scale calibration is active, calculate actual DPI
            try:
                dpi = int(self.dpi_var.get())
            except ValueError:
                dpi = 300

            # If scale was calibrated, calculate the actual DPI to write to file
            if self.scale_calibrator.is_calibrated():
                # Get actual image dimensions in pixels
                h, w = self.transformed_image.shape[:2]

                # Get the dimensions we displayed to user
                try:
                    width_value = float(self.width_var.get())
                    height_value = float(self.height_var.get())
                    current_units = self.units.get()

                    # Convert dimensions to inches for DPI calculation
                    if current_units == "inches":
                        width_inches = width_value
                        height_inches = height_value
                    elif current_units == "mm":
                        width_inches = width_value / 25.4
                        height_inches = height_value / 25.4
                    elif current_units == "pixels":
                        # If units are pixels, don't override DPI
                        width_inches = None
                        height_inches = None

                    if width_inches and height_inches:
                        # Calculate actual DPI from pixels and inches
                        # Use average of width and height DPI
                        dpi_w = w / width_inches
                        dpi_h = h / height_inches
                        dpi = int((dpi_w + dpi_h) / 2)
                except (ValueError, ZeroDivisionError):
                    pass  # Fall back to default DPI

            # Convert numpy array (RGB) to PIL Image
            pil_image = Image.fromarray(self.transformed_image)

            # Save with DPI metadata
            pil_image.save(file_path, dpi=(dpi, dpi))
            self.status_label.config(text=f"Image saved to {file_path} @ {dpi} DPI")


def main():
    parser = argparse.ArgumentParser(description='AutoWarp - 시편 사진 원근 보정')
    parser.add_argument('image', nargs='?', help='Image file to load on startup')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for dimension conversion (default: 300)')
    parser.add_argument('--units', choices=['mm', 'inches', 'pixels'], default='mm',
                        help='Measurement units for dimensions (default: mm)')
    parser.add_argument('--no-crop', dest='crop', action='store_false',
                        help='자동 크롭 끄기 (기본: 켜짐)')
    parser.add_argument('--no-auto-detect', dest='auto_detect', action='store_false',
                        help='자동 귀퉁이 검출 끄기 (기본: 켜짐)')
    parser.add_argument('--log', action='store_true',
                        help='사용 로그를 autowarp_*.log 로 남긴다')
    parser.set_defaults(crop=True, auto_detect=True)
    args = parser.parse_args()

    # Setup logging if requested
    if args.log:
        log_filename = f'autowarp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("=== AutoWarp session started ===")
        logging.info(f"Command line args: dpi={args.dpi}, units={args.units}, crop={args.crop}, auto_detect={args.auto_detect}")
    else:
        logging.disable(logging.CRITICAL)

    root = tk.Tk()
    app = AutoWarpGUI(root, dpi=args.dpi, units=args.units, crop=args.crop, auto_detect=args.auto_detect)

    # Load image if provided via command line
    if args.image:
        # Ensure UI is fully initialized before loading image
        root.update_idletasks()
        app.load_image_from_path(args.image)

    root.mainloop()


if __name__ == "__main__":
    main()
