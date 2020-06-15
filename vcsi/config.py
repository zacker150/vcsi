import configparser
import os

from .media.util import TimestampPosition

DEFAULT_CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".config/vcsi.conf")
DEFAULT_CONFIG_SECTION = "vcsi"

DEFAULT_METADATA_FONT_SIZE = 16
DEFAULT_TIMESTAMP_FONT_SIZE = 12

# Defaults
DEFAULT_METADATA_FONT = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
DEFAULT_TIMESTAMP_FONT = "/usr/share/fonts/TTF/DejaVuSans.ttf"
FALLBACK_FONTS = ["/Library/Fonts/Arial Unicode.ttf"]

# Replace defaults on Windows to support unicode/CJK and multiple fallbacks
if os.name == 'nt':
    DEFAULT_METADATA_FONT = "C:/Windows/Fonts/msgothic.ttc"
    DEFAULT_TIMESTAMP_FONT = "C:/Windows/Fonts/msgothic.ttc"
    FALLBACK_FONTS = [
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/Everson Mono.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/arial.ttf"
    ]

DEFAULT_CONTACT_SHEET_WIDTH = 1500
DEFAULT_DELAY_PERCENT = None
DEFAULT_START_DELAY_PERCENT = 7
DEFAULT_END_DELAY_PERCENT = DEFAULT_START_DELAY_PERCENT
DEFAULT_GRID_SPACING = None
DEFAULT_GRID_HORIZONTAL_SPACING = 5
DEFAULT_GRID_VERTICAL_SPACING = DEFAULT_GRID_HORIZONTAL_SPACING
DEFAULT_METADATA_POSITION = "top"
DEFAULT_METADATA_FONT_COLOR = "ffffff"
DEFAULT_BACKGROUND_COLOR = "000000"
DEFAULT_TIMESTAMP_FONT_COLOR = "ffffff"
DEFAULT_TIMESTAMP_BACKGROUND_COLOR = "000000aa"
DEFAULT_TIMESTAMP_BORDER_COLOR = "000000"
DEFAULT_TIMESTAMP_BORDER_SIZE = 1
DEFAULT_ACCURATE_DELAY_SECONDS = 1
DEFAULT_METADATA_MARGIN = 10
DEFAULT_METADATA_HORIZONTAL_MARGIN = DEFAULT_METADATA_MARGIN
DEFAULT_METADATA_VERTICAL_MARGIN = DEFAULT_METADATA_MARGIN
DEFAULT_CAPTURE_ALPHA = 255
DEFAULT_TIMESTAMP_HORIZONTAL_PADDING = 3
DEFAULT_TIMESTAMP_VERTICAL_PADDING = 1
DEFAULT_TIMESTAMP_HORIZONTAL_MARGIN = 5
DEFAULT_TIMESTAMP_VERTICAL_MARGIN = 5
DEFAULT_IMAGE_QUALITY = 100
DEFAULT_IMAGE_FORMAT = "jpg"
DEFAULT_TIMESTAMP_POSITION = TimestampPosition.se
DEFAULT_FRAME_TYPE = None
DEFAULT_INTERVAL = None

class Config:
    metadata_font_size = DEFAULT_METADATA_FONT_SIZE
    metadata_font = DEFAULT_METADATA_FONT
    timestamp_font_size = DEFAULT_TIMESTAMP_FONT_SIZE
    timestamp_font = DEFAULT_TIMESTAMP_FONT
    fallback_fonts = FALLBACK_FONTS
    contact_sheet_width = DEFAULT_CONTACT_SHEET_WIDTH
    delay_percent = DEFAULT_DELAY_PERCENT
    start_delay_percent = DEFAULT_START_DELAY_PERCENT
    end_delay_percent = DEFAULT_END_DELAY_PERCENT
    grid_spacing = DEFAULT_GRID_SPACING
    grid_horizontal_spacing = DEFAULT_GRID_HORIZONTAL_SPACING
    grid_vertical_spacing = DEFAULT_GRID_VERTICAL_SPACING
    metadata_position = DEFAULT_METADATA_POSITION
    metadata_font_color = DEFAULT_METADATA_FONT_COLOR
    background_color = DEFAULT_BACKGROUND_COLOR
    timestamp_font_color = DEFAULT_TIMESTAMP_FONT_COLOR
    timestamp_background_color = DEFAULT_TIMESTAMP_BACKGROUND_COLOR
    timestamp_border_color = DEFAULT_TIMESTAMP_BORDER_COLOR
    timestamp_border_size = DEFAULT_TIMESTAMP_BORDER_SIZE
    accurate_delay_seconds = DEFAULT_ACCURATE_DELAY_SECONDS
    metadata_margin = DEFAULT_METADATA_MARGIN
    metadata_horizontal_margin = DEFAULT_METADATA_HORIZONTAL_MARGIN
    metadata_vertical_margin = DEFAULT_METADATA_VERTICAL_MARGIN
    capture_alpha = DEFAULT_CAPTURE_ALPHA
    grid_size = None
    timestamp_horizontal_padding = DEFAULT_TIMESTAMP_HORIZONTAL_PADDING
    timestamp_vertical_padding = DEFAULT_TIMESTAMP_VERTICAL_PADDING
    timestamp_horizontal_margin = DEFAULT_TIMESTAMP_HORIZONTAL_MARGIN
    timestamp_vertical_margin = DEFAULT_TIMESTAMP_VERTICAL_MARGIN
    quality = DEFAULT_IMAGE_QUALITY
    format = DEFAULT_IMAGE_FORMAT
    timestamp_position = DEFAULT_TIMESTAMP_POSITION
    frame_type = DEFAULT_FRAME_TYPE
    interval = DEFAULT_INTERVAL

    @classmethod
    def load_configuration(cls, filename=DEFAULT_CONFIG_FILE):
        config = configparser.ConfigParser(default_section=DEFAULT_CONFIG_SECTION)
        config.read(filename)

        for config_entry in cls.__dict__.keys():
            # skip magic attributes
            if config_entry.startswith('__'):
                continue
            setattr(cls, config_entry, config.get(
                DEFAULT_CONFIG_SECTION,
                config_entry,
                fallback=getattr(cls, config_entry)
            ))
        # special cases
        # fallback_fonts is an array, it's reflected as comma separated list in config file
        fallback_fonts = config.get(DEFAULT_CONFIG_SECTION, 'fallback_fonts', fallback=None)
        if fallback_fonts:
            cls.fallback_fonts = comma_separated_string_type(fallback_fonts)