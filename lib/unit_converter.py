"""
Unit conversion between pixels, millimeters, and inches.
Handles DPI-based conversions and scale factor calibration.
"""


class UnitConverter:
    """
    Handles conversion between different measurement units.

    Supports three unit types:
    - pixels: Direct pixel measurements
    - mm: Millimeters (converted via DPI)
    - inches: Inches (converted via DPI)

    Can use either DPI or a calibrated scale factor for conversions.
    """

    # Conversion constant
    MM_PER_INCH = 25.4

    def __init__(self, units="mm", dpi=300, scale_factor=1.0):
        """
        Initialize the unit converter.

        Args:
            units: Current unit type ("mm", "inches", or "pixels")
            dpi: Dots per inch for conversion (default: 300)
            scale_factor: Calibrated pixels per unit (default: 1.0 = not calibrated)
        """
        self.units = units
        self.dpi = dpi
        self.scale_factor = scale_factor

    def set_units(self, units):
        """Change the current unit type"""
        self.units = units

    def set_dpi(self, dpi):
        """Change the DPI setting"""
        self.dpi = dpi

    def set_scale_factor(self, scale_factor):
        """Set calibrated scale factor (pixels per unit)"""
        self.scale_factor = scale_factor

    def is_calibrated(self):
        """Check if scale has been calibrated"""
        return self.scale_factor != 1.0

    def units_to_pixels(self, value, units=None, dpi=None):
        """
        Convert value in current units to pixels.

        Args:
            value: Value in current units
            units: Override current units (optional)
            dpi: Override current DPI (optional)

        Returns:
            Integer pixel value
        """
        if units is None:
            units = self.units
        if dpi is None:
            dpi = self.dpi

        if units == "pixels":
            return int(value)
        elif units == "inches":
            return int(value * dpi)
        else:  # mm
            return int((value / self.MM_PER_INCH) * dpi)

    def pixels_to_units(self, pixels, units=None, dpi=None, use_scale=False):
        """
        Convert pixels to current units.

        Args:
            pixels: Pixel value to convert
            units: Override current units (optional)
            dpi: Override current DPI (optional)
            use_scale: Use calibrated scale_factor instead of DPI (default: False)

        Returns:
            Float value in target units
        """
        if units is None:
            units = self.units
        if dpi is None:
            dpi = self.dpi

        # If scale is calibrated and we're asked to use it
        if use_scale and self.is_calibrated():
            return pixels / self.scale_factor

        # Otherwise use DPI conversion
        if units == "pixels":
            return float(pixels)
        elif units == "inches":
            return pixels / dpi
        else:  # mm
            return (pixels / dpi) * self.MM_PER_INCH

    def convert_units(self, value, from_units, to_units, dpi=None):
        """
        Convert a value from one unit system to another.

        Args:
            value: Value to convert
            from_units: Source unit type ("mm", "inches", or "pixels")
            to_units: Target unit type ("mm", "inches", or "pixels")
            dpi: DPI to use for conversion (optional, uses instance DPI if not specified)

        Returns:
            Float value in target units
        """
        if dpi is None:
            dpi = self.dpi

        # Convert to pixels first (common intermediate)
        pixels = self.units_to_pixels(value, units=from_units, dpi=dpi)

        # Convert from pixels to target units
        return self.pixels_to_units(pixels, units=to_units, dpi=dpi, use_scale=False)

    def get_unit_label(self, units=None):
        """
        Get display label for unit type.

        Args:
            units: Unit type to get label for (uses current if not specified)

        Returns:
            String label ("mm", "in", or "px")
        """
        if units is None:
            units = self.units

        if units == "pixels":
            return "px"
        elif units == "inches":
            return "in"
        else:
            return "mm"

    def get_spinbox_increment(self, units=None):
        """
        Get recommended spinbox increment for unit type.

        Args:
            units: Unit type (uses current if not specified)

        Returns:
            Float increment value
        """
        if units is None:
            units = self.units

        if units == "pixels":
            return 1
        elif units == "inches":
            return 0.1
        else:  # mm
            return 0.5
