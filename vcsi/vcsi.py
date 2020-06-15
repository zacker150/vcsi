#!/usr/bin/env python3

"""Create a video contact sheet.
"""

from __future__ import print_function

import datetime
import math
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from PIL import Image

from vcsi.media.mediainfo import MediaInfo
from vcsi.media.util import *
from vcsi.media.mediacapture import MediaCapture

from .config import FALLBACK_FONTS, DEFAULT_FRAME_TYPE, DEFAULT_METADATA_MARGIN, DEFAULT_CONTACT_SHEET_WIDTH, \
    DEFAULT_CONFIG_FILE, Config
from vcsi.error import error_exit

import argparse
import configparser
import tempfile
import textwrap
from glob import glob
from glob import escape

from PIL import ImageDraw, ImageFont
from jinja2 import Template
import texttable
import parsedatetime
from tqdm import tqdm

from .media import *

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "VERSION")) as f:
    VERSION = f.readline().strip()
__version__ = VERSION
__author__ = "Nils Amiet"


VALID_TIMESTAMP_POSITIONS = [x.name for x in TimestampPosition]


def grid_desired_size(
        grid,
        media_info,
        width=Config.contact_sheet_width,
        horizontal_margin=Config.grid_horizontal_spacing):
    """Computes the size of the images placed on a mxn grid with given fixed width.
    Returns (width, height)
    """
    desired_width = (width - (grid.x - 1) * horizontal_margin) / grid.x
    desired_width = int(math.floor(desired_width))

    return media_info.desired_size(width=desired_width)


def total_delay_seconds(media_info, args):
    """Computes the total seconds to skip (beginning + ending).
    """
    start_delay_seconds = math.floor(media_info.duration_seconds * args.start_delay_percent / 100)
    end_delay_seconds = math.floor(media_info.duration_seconds * args.end_delay_percent / 100)
    delay = start_delay_seconds + end_delay_seconds
    return delay


def timestamp_generator(media_info, args):
    """Generates `num_samples` uniformly distributed timestamps over time.
    Timestamps will be selected in the range specified by start_delay_percent and end_delay percent.
    For example, `end_delay_percent` can be used to avoid making captures during the ending credits.
    """
    delay = total_delay_seconds(media_info, args)
    capture_interval = (media_info.duration_seconds - delay) / (args.num_samples + 1)

    if args.interval is not None:
        capture_interval = int(args.interval.total_seconds())
    start_delay_seconds = math.floor(media_info.duration_seconds * args.start_delay_percent / 100)
    time = start_delay_seconds + capture_interval

    for i in range(args.num_samples):
        yield (time, MediaInfo.pretty_duration(time, show_millis=True))
        time += capture_interval


def select_sharpest_images(
        media_info,
        media_capture,
        args):
    """Make `num_samples` captures and select `num_selected` captures out of these
    based on blurriness and color variety.
    """

    desired_size = grid_desired_size(
        args.grid,
        media_info,
        width=args.vcs_width,
        horizontal_margin=args.grid_horizontal_spacing)

    if args.manual_timestamps is None:
        timestamps = timestamp_generator(media_info, args)
    else:
        timestamps = [(MediaInfo.pretty_to_seconds(x), x) for x in args.manual_timestamps]

    def do_capture(ts_tuple, width, height, suffix, args):
        fd, filename = tempfile.mkstemp(suffix=suffix)

        media_capture.make_capture(ts_tuple[1], width, height, filename)

        blurriness = 1
        avg_color = 0

        if not args.fast:
            blurriness = media_capture.compute_blurriness(filename)
            avg_color = media_capture.compute_avg_color(filename)

        os.close(fd)
        frm = Frame(
            filename=filename,
            blurriness=blurriness,
            timestamp=ts_tuple[0],
            avg_color=avg_color
        )
        return frm

    if args.fast:
        suffix = ".jpg"  # faster processing time
    else:
        suffix = ".png"  # arguably higher image quality
        
    # use multiple threads
    with ThreadPoolExecutor() as executor:
        print("Starting tasks")
        futures = [executor.submit(do_capture, timestamp_tuple, desired_size[0], desired_size[1], suffix, args)
                   for timestamp_tuple in tqdm(timestamps, total=args.num_samples)]
        print()

        print("Sampling")
        blurs = [future.result() for future in tqdm(futures)]

    time_sorted = sorted(blurs, key=lambda x: x.timestamp)

    # group into num_selected groups
    if args.num_groups > 1:
        group_size = max(1, int(math.floor(len(time_sorted) / args.num_groups)))
        groups = chunks(time_sorted, group_size)

        # find top sharpest for each group
        selected_items = [best(x) for x in groups]
    else:
        selected_items = time_sorted

    selected_items = select_color_variety(selected_items, args.num_selected)

    return selected_items, time_sorted


def select_color_variety(frames, num_selected):
    """Select captures so that they are not too similar to each other.
    """
    avg_color_sorted = sorted(frames, key=lambda x: x.avg_color)
    min_color = avg_color_sorted[0].avg_color
    max_color = avg_color_sorted[-1].avg_color
    color_span = max_color - min_color
    min_color_distance = int(color_span * 0.05)

    blurriness_sorted = sorted(frames, key=lambda x: x.blurriness, reverse=True)

    selected_items = []
    unselected_items = []
    while blurriness_sorted:
        frame = blurriness_sorted.pop()

        if not selected_items:
            selected_items += [frame]
        else:
            color_distance = min([abs(frame.avg_color - x.avg_color) for x in selected_items])
            if color_distance < min_color_distance:
                # too close to existing selected frame
                # don't select unless we run out of frames
                unselected_items += [(frame, color_distance)]
            else:
                selected_items += [frame]

    missing_items_count = num_selected - len(selected_items)
    if missing_items_count > 0:
        remaining_items = sorted(unselected_items, key=lambda x: x[0].blurriness)
        selected_items += [x[0] for x in remaining_items[:missing_items_count]]

    return selected_items


def best(captures):
    """Returns the least blurry capture
    """
    return sorted(captures, key=lambda x: x.blurriness)[0]


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def draw_metadata(
        draw,
        args,
        header_line_height=None,
        header_lines=None,
        header_font=None,
        header_font_color=None,
        start_height=None):
    """Draw metadata header
    """
    h = start_height
    h += args.metadata_vertical_margin

    for line in header_lines:
        draw.text((args.metadata_horizontal_margin, h), line, font=header_font, fill=header_font_color)
        h += header_line_height

    h += args.metadata_vertical_margin

    return h


def max_line_length(
        media_info,
        metadata_font,
        header_margin,
        width=Config.contact_sheet_width,
        text=None):
    """Find the number of characters that fit in width with given font.
    """
    if text is None:
        text = media_info.filename

    max_width = width - 2 * header_margin

    max_length = 0
    for i in range(len(text) + 1):
        text_chunk = text[:i]
        text_width = 0 if len(text_chunk) == 0 else metadata_font.getsize(text_chunk)[0]

        max_length = i
        if text_width > max_width:
            break

    return max_length


def prepare_metadata_text_lines(media_info, header_font, header_margin, width, template_path=None):
    """Prepare the metadata header text and return a list containing each line.
    """
    template = ""
    if template_path is None:
        template = """{{filename}}
        File size: {{size}}
        Duration: {{duration}}
        Dimensions: {{sample_width}}x{{sample_height}}"""
    else:
        with open(template_path) as f:
            template = f.read()

    params = media_info.template_attributes()
    template = Template(template).render(params)
    template_lines = template.split("\n")
    template_lines = [x.strip() for x in template_lines if len(x) > 0]

    header_lines = []
    for line in template_lines:
        remaining_chars = line
        while len(remaining_chars) > 0:
            max_metadata_line_length = max_line_length(
                media_info,
                header_font,
                header_margin,
                width=width,
                text=remaining_chars)
            wraps = textwrap.wrap(remaining_chars, max_metadata_line_length)
            header_lines.append(wraps[0])
            remaining_chars = remaining_chars[len(wraps[0]):].strip()

    return header_lines


def compute_timestamp_position(args, w, h, text_size, desired_size, rectangle_hpadding, rectangle_vpadding):
    """Compute the (x,y) position of the upper left and bottom right points of the rectangle surrounding timestamp text.
    """
    position = args.timestamp_position

    x_offset = 0
    if position in [TimestampPosition.west, TimestampPosition.nw, TimestampPosition.sw]:
        x_offset = args.timestamp_horizontal_margin
    elif position in [TimestampPosition.north, TimestampPosition.center, TimestampPosition.south]:
        x_offset = (desired_size[0] / 2) - (text_size[0] / 2) - rectangle_hpadding
    else:
        x_offset = desired_size[0] - text_size[0] - args.timestamp_horizontal_margin - 2 * rectangle_hpadding

    y_offset = 0
    if position in [TimestampPosition.nw, TimestampPosition.north, TimestampPosition.ne]:
        y_offset = args.timestamp_vertical_margin
    elif position in [TimestampPosition.west, TimestampPosition.center, TimestampPosition.east]:
        y_offset = (desired_size[1] / 2) - (text_size[1] / 2) - rectangle_vpadding
    else:
        y_offset = desired_size[1] - text_size[1] - args.timestamp_vertical_margin - 2 * rectangle_vpadding

    upper_left = (
        w + x_offset,
        h + y_offset
    )

    bottom_right = (
        upper_left[0] + text_size[0] + 2 * rectangle_hpadding,
        upper_left[1] + text_size[1] + 2 * rectangle_vpadding
    )

    return upper_left, bottom_right


def load_font(args, font_path, font_size, default_font_path):
    """Loads given font and defaults to fallback fonts if that fails."""
    if args.is_verbose:
        print("Loading font...")

    fonts = [font_path] + FALLBACK_FONTS
    if font_path == default_font_path:
        for font in fonts:
            if args.is_verbose:
                print("Trying to load font:", font)
            if os.path.exists(font):
                try:
                    return ImageFont.truetype(font, font_size)
                except OSError:
                    pass
        print("Falling back to default font.")
        return ImageFont.load_default()
    else:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            error_exit("Cannot load font: {}".format(font_path))


def compose_contact_sheet(
        media_info,
        frames,
        args):
    """Creates a video contact sheet with the media information in a header
    and the selected frames arranged on a mxn grid with optional timestamps
    """
    desired_size = grid_desired_size(
        args.grid,
        media_info,
        width=args.vcs_width,
        horizontal_margin=args.grid_horizontal_spacing)
    width = args.grid.x * (desired_size[0] + args.grid_horizontal_spacing) - args.grid_horizontal_spacing
    height = args.grid.y * (desired_size[1] + args.grid_vertical_spacing) - args.grid_vertical_spacing

    header_font = load_font(args, args.metadata_font, args.metadata_font_size, Config.metadata_font)
    timestamp_font = load_font(args, args.timestamp_font, args.timestamp_font_size, Config.timestamp_font)

    header_lines = prepare_metadata_text_lines(
        media_info,
        header_font,
        args.metadata_horizontal_margin,
        width,
        template_path=args.metadata_template_path)

    line_spacing_coefficient = 1.2
    header_line_height = int(args.metadata_font_size * line_spacing_coefficient)
    header_height = 2 * args.metadata_margin + len(header_lines) * header_line_height

    if args.metadata_position == "hidden":
        header_height = 0

    final_image_width = width
    final_image_height = height + header_height
    transparent = (255, 255, 255, 0)
    image = Image.new("RGBA", (final_image_width, final_image_height), args.background_color)
    image_capture_layer = Image.new("RGBA", (final_image_width, final_image_height), transparent)
    image_header_text_layer = Image.new("RGBA", (final_image_width, final_image_height), transparent)
    image_timestamp_layer = Image.new("RGBA", (final_image_width, final_image_height), transparent)
    image_timestamp_text_layer = Image.new("RGBA", (final_image_width, final_image_height), transparent)

    draw_header_text_layer = ImageDraw.Draw(image_header_text_layer)
    draw_timestamp_layer = ImageDraw.Draw(image_timestamp_layer)
    draw_timestamp_text_layer = ImageDraw.Draw(image_timestamp_text_layer)
    h = 0

    def draw_metadata_helper():
        """Draw metadata with fixed arguments
        """
        return draw_metadata(
            draw_header_text_layer,
            args,
            header_line_height=header_line_height,
            header_lines=header_lines,
            header_font=header_font,
            header_font_color=args.metadata_font_color,
            start_height=h)

    # draw metadata
    if args.metadata_position == "top":
        h = draw_metadata_helper()

    # draw capture grid
    w = 0
    frames = sorted(frames, key=lambda x: x.timestamp)
    for i, frame in enumerate(frames):
        f = Image.open(frame.filename)
        f.putalpha(args.capture_alpha)
        image_capture_layer.paste(f, (w, h))

        # show timestamp
        if args.show_timestamp:
            timestamp_time = MediaInfo.pretty_duration(frame.timestamp, show_centis=True)
            timestamp_duration = MediaInfo.pretty_duration(media_info.duration_seconds, show_centis=True)
            parsed_time = MediaInfo.parse_duration(frame.timestamp)
            parsed_duration = MediaInfo.parse_duration(media_info.duration_seconds)
            timestamp_args = {
                "TIME": timestamp_time,
                "DURATION": timestamp_duration,
                "THUMBNAIL_NUMBER": i + 1,
                "H": str(parsed_time["hours"]).zfill(2),
                "M": str(parsed_time["minutes"]).zfill(2),
                "S": str(parsed_time["seconds"]).zfill(2),
                "c": str(parsed_time["centis"]).zfill(2),
                "m": str(parsed_time["millis"]).zfill(3),
                "dH": str(parsed_duration["hours"]).zfill(2),
                "dM": str(parsed_duration["minutes"]).zfill(2),
                "dS": str(parsed_duration["seconds"]).zfill(2),
                "dc": str(parsed_duration["centis"]).zfill(2),
                "dm": str(parsed_duration["millis"]).zfill(3)
            }
            timestamp_text = args.timestamp_format.format(**timestamp_args)
            text_size = timestamp_font.getsize(timestamp_text)

            # draw rectangle
            rectangle_hpadding = args.timestamp_horizontal_padding
            rectangle_vpadding = args.timestamp_vertical_padding

            upper_left, bottom_right = compute_timestamp_position(args, w, h, text_size, desired_size,
                                                                  rectangle_hpadding, rectangle_vpadding)

            if not args.timestamp_border_mode:
                draw_timestamp_layer.rectangle(
                    [upper_left, bottom_right],
                    fill=args.timestamp_background_color
                )
            else:
                offset_factor = args.timestamp_border_size
                offsets = [
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1)
                ]

                final_offsets = []
                for offset_counter in range(1, offset_factor + 1):
                    final_offsets += [(x[0] * offset_counter, x[1] * offset_counter) for x in offsets]

                for offset in final_offsets:
                    # draw border first
                    draw_timestamp_text_layer.text(
                        (
                            upper_left[0] + rectangle_hpadding + offset[0],
                            upper_left[1] + rectangle_vpadding + offset[1]
                        ),
                        timestamp_text,
                        font=timestamp_font,
                        fill=args.timestamp_border_color
                    )

            # draw timestamp
            draw_timestamp_text_layer.text(
                (
                    upper_left[0] + rectangle_hpadding,
                    upper_left[1] + rectangle_vpadding
                ),
                timestamp_text,
                font=timestamp_font,
                fill=args.timestamp_font_color
            )

        # update x position for next frame
        w += desired_size[0] + args.grid_horizontal_spacing

        # update y position
        if (i + 1) % args.grid.x == 0:
            h += desired_size[1] + args.grid_vertical_spacing

        # update x position
        if (i + 1) % args.grid.x == 0:
            w = 0

    # draw metadata
    if args.metadata_position == "bottom":
        h -= args.grid_vertical_spacing
        h = draw_metadata_helper()

    # alpha blend
    out_image = Image.alpha_composite(image, image_capture_layer)
    out_image = Image.alpha_composite(out_image, image_header_text_layer)
    out_image = Image.alpha_composite(out_image, image_timestamp_layer)
    out_image = Image.alpha_composite(out_image, image_timestamp_text_layer)

    return out_image


def save_image(args, image, media_info, output_path):
    """Save the image to `output_path`
    """
    image = image.convert("RGB")
    try:
        image.save(output_path, optimize=True, quality=args.image_quality)
        return True
    except KeyError:
        return False


def cleanup(frames, args):
    """Delete temporary captures
    """
    if args.is_verbose:
        print("Deleting {} temporary frames...".format(len(frames)))
    for frame in frames:
        try:
            if args.is_verbose:
                print("Deleting {} ...".format(frame.filename))
            os.unlink(frame.filename)
        except Exception as e:
            if args.is_verbose:
                print("[Error] Failed to delete {}".format(frame.filename))
                print(e)


def print_template_attributes():
    """Display all the available template attributes in a tabular format
    """
    table = MediaInfo.list_template_attributes()

    tab = texttable.Texttable()
    tab.set_cols_dtype(["t", "t", "t"])
    rows = [[x["name"], x["description"], x["example"]] for x in table]
    tab.add_rows(rows, header=False)
    tab.header(["Attribute name", "Description", "Example"])
    print(tab.draw())


def mxn_type(string):
    """Type parser for argparse. Argument of type "mxn" will be converted to Grid(m, n).
    An exception will be thrown if the argument is not of the required form
    """
    try:
        split = string.split("x")
        assert (len(split) == 2)
        m = int(split[0])
        assert (m >= 0)
        n = int(split[1])
        assert (n >= 0)
        return Grid(m, n)
    except (IndexError, ValueError, AssertionError):
        error = "Grid must be of the form mxn, where m is the number of columns and n is the number of rows."
        raise argparse.ArgumentTypeError(error)


def metadata_position_type(string):
    """Type parser for argparse. Argument of type string must be one of ["top", "bottom", "hidden"].
    An exception will be thrown if the argument is not one of these.
    """
    valid_metadata_positions = ["top", "bottom", "hidden"]

    lowercase_position = string.lower()
    if lowercase_position in valid_metadata_positions:
        return lowercase_position
    else:
        error = 'Metadata header position must be one of %s' % (str(valid_metadata_positions, ))
        raise argparse.ArgumentTypeError(error)


def hex_color_type(string):
    """Type parser for argparse. Argument must be an hexadecimal number representing a color.
    For example 'AABBCC' (RGB) or 'AABBCCFF' (RGBA). An exception will be raised if the argument
    is not of that form.
    """
    try:
        components = tuple(bytearray.fromhex(string))
        if len(components) == 3:
            components += (255,)
        c = Color(*components)
        return c
    except:
        error = "Color must be an hexadecimal number, for example 'AABBCC'"
        raise argparse.ArgumentTypeError(error)


def manual_timestamps(string):
    """Type parser for argparse. Argument must be a comma-separated list of frame timestamps.
    For example 1:11:11.111,2:22:22.222
    """
    try:
        timestamps = string.split(",")
        timestamps = [x.strip() for x in timestamps if x]

        # check whether timestamps are valid
        for t in timestamps:
            MediaInfo.pretty_to_seconds(t)

        return timestamps
    except Exception as e:
        print(e)
        error = "Manual frame timestamps must be comma-separated and of the form h:mm:ss.mmmm"
        raise argparse.ArgumentTypeError(error)


def timestamp_position_type(string):
    """Type parser for argparse. Argument must be a valid timestamp position"""
    try:
        return getattr(TimestampPosition, string)
    except AttributeError:
        error = "Invalid timestamp position: %s. Valid positions are: %s" % (string, VALID_TIMESTAMP_POSITIONS)
        raise argparse.ArgumentTypeError(error)


def interval_type(string):
    """Type parser for argparse. Argument must be a valid interval format.
    Supports any format supported by `parsedatetime`, including:
        * "30sec" (every 30 seconds)
        * "5 minutes" (every 5 minutes)
        * "1h" (every hour)
        * "2 hours 1 min and 30 seconds"
    """
    m = datetime.datetime.min
    cal = parsedatetime.Calendar()
    interval = cal.parseDT(string, sourceTime=m)[0] - m
    if interval == m:
        error = "Invalid interval format: {}".format(string)
        raise argparse.ArgumentTypeError(error)

    return interval


def comma_separated_string_type(string):
    """Type parser for argparse. Argument must be a comma-separated list of strings."""
    splits = string.split(",")
    splits = [x.strip() for x in splits]
    splits = [x for x in splits if len(x) > 0]
    return splits





def main():
    """Program entry point
    """
    # Argument parser before actual argument parser to let the user overwrite the config path
    preargparser = argparse.ArgumentParser(add_help=False)
    preargparser.add_argument("-c", "--config", dest="configfile", default=None)
    preargs, _ = preargparser.parse_known_args()
    try:
        if preargs.configfile:
            # check if the given config file exists
            # abort if not, because the user wants to use a specific file and not the default config
            if os.path.exists(preargs.configfile):
                Config.load_configuration(preargs.configfile)
            else:
                error_exit("Could find config file")
        else:
            # check if the config file exists and load it
            if os.path.exists(DEFAULT_CONFIG_FILE):
                Config.load_configuration(DEFAULT_CONFIG_FILE)
    except configparser.MissingSectionHeaderError as e:
        error_exit(e.message)

    parser = argparse.ArgumentParser(description="Create a video contact sheet",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filenames", nargs="+")
    parser.add_argument(
        "-o", "--output",
        help="save to output file",
        dest="output_path")
    # adding --config to the main parser to display it when the user asks for help
    # the value is not important anymore
    parser.add_argument(
        "-c", "--config",
        help="Config file to load defaults from",
        default=DEFAULT_CONFIG_FILE
    )
    parser.add_argument(
        "--start-delay-percent",
        help="do not capture frames in the first n percent of total time",
        dest="start_delay_percent",
        type=int,
        default=Config.start_delay_percent)
    parser.add_argument(
        "--end-delay-percent",
        help="do not capture frames in the last n percent of total time",
        dest="end_delay_percent",
        type=int,
        default=Config.end_delay_percent)
    parser.add_argument(
        "--delay-percent",
        help="do not capture frames in the first and last n percent of total time",
        dest="delay_percent",
        type=int,
        default=Config.delay_percent)
    parser.add_argument(
        "--grid-spacing",
        help="number of pixels spacing captures both vertically and horizontally",
        dest="grid_spacing",
        type=int,
        default=Config.grid_spacing)
    parser.add_argument(
        "--grid-horizontal-spacing",
        help="number of pixels spacing captures horizontally",
        dest="grid_horizontal_spacing",
        type=int,
        default=Config.grid_horizontal_spacing)
    parser.add_argument(
        "--grid-vertical-spacing",
        help="number of pixels spacing captures vertically",
        dest="grid_vertical_spacing",
        type=int,
        default=Config.grid_vertical_spacing)
    parser.add_argument(
        "-w", "--width",
        help="width of the generated contact sheet",
        dest="vcs_width",
        type=int,
        default=Config.contact_sheet_width)
    parser.add_argument(
        "-g", "--grid",
        help="display frames on a mxn grid (for example 4x5). The special value zero (as in 2x0 or 0x5 or 0x0) is only allowed when combined with --interval or with --manual. Zero means that the component should be automatically deduced based on other arguments passed.",
        dest="grid",
        type=mxn_type,
        default=Config.grid_size)
    parser.add_argument(
        "-s", "--num-samples",
        help="number of samples",
        dest="num_samples",
        type=int,
        default=None)
    parser.add_argument(
        "-t", "--show-timestamp",
        action="store_true",
        help="display timestamp for each frame",
        dest="show_timestamp")
    parser.add_argument(
        "--metadata-font-size",
        help="size of the font used for metadata",
        dest="metadata_font_size",
        type=int,
        default=Config.metadata_font_size)
    parser.add_argument(
        "--metadata-font",
        help="TTF font used for metadata",
        dest="metadata_font",
        default=Config.metadata_font)
    parser.add_argument(
        "--timestamp-font-size",
        help="size of the font used for timestamps",
        dest="timestamp_font_size",
        type=int,
        default=Config.timestamp_font_size)
    parser.add_argument(
        "--timestamp-font",
        help="TTF font used for timestamps",
        dest="timestamp_font",
        default=Config.timestamp_font)
    parser.add_argument(
        "--metadata-position",
        help="Position of the metadata header. Must be one of ['top', 'bottom', 'hidden']",
        dest="metadata_position",
        type=metadata_position_type,
        default=Config.metadata_position)
    parser.add_argument(
        "--background-color",
        help="Color of the background in hexadecimal, for example AABBCC",
        dest="background_color",
        type=hex_color_type,
        default=hex_color_type(Config.background_color))
    parser.add_argument(
        "--metadata-font-color",
        help="Color of the metadata font in hexadecimal, for example AABBCC",
        dest="metadata_font_color",
        type=hex_color_type,
        default=hex_color_type(Config.metadata_font_color))
    parser.add_argument(
        "--timestamp-font-color",
        help="Color of the timestamp font in hexadecimal, for example AABBCC",
        dest="timestamp_font_color",
        type=hex_color_type,
        default=hex_color_type(Config.timestamp_font_color))
    parser.add_argument(
        "--timestamp-background-color",
        help="Color of the timestamp background rectangle in hexadecimal, for example AABBCC",
        dest="timestamp_background_color",
        type=hex_color_type,
        default=hex_color_type(Config.timestamp_background_color))
    parser.add_argument(
        "--timestamp-border-color",
        help="Color of the timestamp border in hexadecimal, for example AABBCC",
        dest="timestamp_border_color",
        type=hex_color_type,
        default=hex_color_type(Config.timestamp_border_color))
    parser.add_argument(
        "--template",
        help="Path to metadata template file",
        dest="metadata_template_path",
        default=None)
    parser.add_argument(
        "-m", "--manual",
        help="Comma-separated list of frame timestamps to use, for example 1:11:11.111,2:22:22.222",
        dest="manual_timestamps",
        type=manual_timestamps,
        default=None)
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="display verbose messages",
        dest="is_verbose")
    parser.add_argument(
        "-a", "--accurate",
        action="store_true",
        help="""Make accurate captures. This capture mode is way slower than the default one
        but it helps when capturing frames from HEVC videos.""",
        dest="is_accurate")
    parser.add_argument(
        "-A", "--accurate-delay-seconds",
        type=int,
        default=Config.accurate_delay_seconds,
        help="""Fast skip to N seconds before capture time, then do accurate capture
        (decodes N seconds of video before each capture). This is used with accurate capture mode only.""",
        dest="accurate_delay_seconds")
    parser.add_argument(
        "--metadata-margin",
        type=int,
        default=Config.metadata_margin,
        help="Margin (in pixels) in the metadata header.",
        dest="metadata_margin")
    parser.add_argument(
        "--metadata-horizontal-margin",
        type=int,
        default=Config.metadata_horizontal_margin,
        help="Horizontal margin (in pixels) in the metadata header.",
        dest="metadata_horizontal_margin")
    parser.add_argument(
        "--metadata-vertical-margin",
        type=int,
        default=Config.metadata_vertical_margin,
        help="Vertical margin (in pixels) in the metadata header.",
        dest="metadata_vertical_margin")
    parser.add_argument(
        "--timestamp-horizontal-padding",
        type=int,
        default=Config.timestamp_horizontal_padding,
        help="Horizontal padding (in pixels) for timestamps.",
        dest="timestamp_horizontal_padding")
    parser.add_argument(
        "--timestamp-vertical-padding",
        type=int,
        default=Config.timestamp_vertical_padding,
        help="Vertical padding (in pixels) for timestamps.",
        dest="timestamp_vertical_padding")
    parser.add_argument(
        "--timestamp-horizontal-margin",
        type=int,
        default=Config.timestamp_horizontal_margin,
        help="Horizontal margin (in pixels) for timestamps.",
        dest="timestamp_horizontal_margin")
    parser.add_argument(
        "--timestamp-vertical-margin",
        type=int,
        default=Config.timestamp_vertical_margin,
        help="Vertical margin (in pixels) for timestamps.",
        dest="timestamp_vertical_margin")
    parser.add_argument(
        "--quality",
        type=int,
        default=Config.quality,
        help="Output image quality. Must be an integer in the range 0-100. 100 = best quality.",
        dest="image_quality")
    parser.add_argument(
        "-f", "--format",
        type=str,
        default=Config.format,
        help="Output image format. Can be any format supported by pillow. For example 'png' or 'jpg'.",
        dest="image_format")
    parser.add_argument(
        "-T", "--timestamp-position",
        type=timestamp_position_type,
        default=Config.timestamp_position,
        help="Timestamp position. Must be one of %s." % (VALID_TIMESTAMP_POSITIONS,),
        dest="timestamp_position")
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process every file in the specified directory recursively.",
        dest="recursive")
    parser.add_argument(
        "--timestamp-border-mode",
        action="store_true",
        help="Draw timestamp text with a border instead of the default rectangle.",
        dest="timestamp_border_mode")
    parser.add_argument(
        "--timestamp-border-size",
        type=int,
        default=Config.timestamp_border_size,
        help="Size of the timestamp border in pixels (used only with --timestamp-border-mode).",
        dest="timestamp_border_size")
    parser.add_argument(
        "--capture-alpha",
        type=int,
        default=Config.capture_alpha,
        help="Alpha channel value for the captures (transparency in range [0, 255]). Defaults to 255 (opaque)",
        dest="capture_alpha")
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s version {version}".format(version=__version__))
    parser.add_argument(
        "--list-template-attributes",
        action="store_true",
        dest="list_template_attributes")
    parser.add_argument(
        "--frame-type",
        type=str,
        default=DEFAULT_FRAME_TYPE,
        help="Frame type passed to ffmpeg 'select=eq(pict_type,FRAME_TYPE)' filter. Should be one of ('I', 'B', 'P') or the special type 'key' which will use the 'select=key' filter instead.",
        dest="frame_type")
    parser.add_argument(
        "--interval",
        type=interval_type,
        default=Config.interval,
        help="Capture frames at specified interval. Interval format is any string supported by `parsedatetime`. For example '5m', '3 minutes 5 seconds', '1 hour 15 min and 20 sec' etc.",
        dest="interval")
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Ignore any error encountered while processing files recursively and continue to the next file.",
        dest="ignore_errors")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite output file if it already exists, simply ignore this file and continue processing other unprocessed files.",
        dest="no_overwrite"
    )
    parser.add_argument(
        "--exclude-extensions",
        type=comma_separated_string_type,
        default=[],
        help="Do not process files that end with the given extensions.",
        dest="exclude_extensions"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode. Just make a contact sheet as fast as possible, regardless of output image quality. May mess up the terminal.",
        dest="fast")
    parser.add_argument(
        "-O", "--thumbnail-output",
        help="Save thumbnail files to the specified output directory. If set, the thumbnail files will not be deleted after successful creation of the contact sheet.",
        default=None,
        dest="thumbnail_output_path"
    )
    parser.add_argument(
        "-S", "--actual-size",
        help="Make thumbnails of actual size. In other words, thumbnails will have the actual 1:1 size of the video resolution.",
        action="store_true",
        dest="actual_size"
    )
    parser.add_argument(
        "--timestamp-format",
        help="Use specified timestamp format. Replaced values include: {TIME}, {DURATION}, {THUMBNAIL_NUMBER}, {H} (hours), {M} (minutes), {S} (seconds), {c} (centiseconds), {m} (milliseconds), {dH}, {dM}, {dS}, {dc} and {dm} (same as previous values but for the total duration). Example format: '{TIME} / {DURATION}'. Another example: '{THUMBNAIL_NUMBER}'. Yet another example: '{H}:{M}:{S}.{m} / {dH}:{dM}:{dS}.{dm}'.",
        default="{TIME}",
        dest="timestamp_format"
    )

    args = parser.parse_args()

    if args.list_template_attributes:
        print_template_attributes()
        sys.exit(0)

    def process_file_or_ignore(filepath, args):
        try:
            process_file(filepath, args)
        except Exception:
            if not args.ignore_errors:
                raise
            else:
                print("[WARN]: failed to process {} ... skipping.".format(filepath), file=sys.stderr)

    if args.recursive:
        for path in args.filenames:
            for root, subdirs, files in os.walk(path):
                for f in files:
                    filepath = os.path.join(root, f)
                    process_file_or_ignore(filepath, args)
    else:
        for path in args.filenames:
            if os.path.isdir(path):
                for filepath in os.listdir(path):
                    abs_filepath = os.path.join(path, filepath)
                    if not os.path.isdir(abs_filepath):
                        process_file_or_ignore(abs_filepath, args)

            else:
                files_to_process = glob(escape(path))
                if len(files_to_process) == 0:
                    files_to_process = [path]
                for filename in files_to_process:
                    process_file_or_ignore(filename, args)


def process_file(path, args):
    """Generate a video contact sheet for the file at given path
    """
    if args.is_verbose:
        print("Considering {}...".format(path))

    args = deepcopy(args)

    if not os.path.exists(path):
        if args.ignore_errors:
            print("File does not exist, skipping: {}".format(path))
            return
        else:
            error_message = "File does not exist: {}".format(path)
            error_exit(error_message)

    file_extension = path.lower().split(".")[-1]
    if file_extension in args.exclude_extensions:
        print("[WARN] Excluded extension {}. Skipping.".format(file_extension))
        return

    output_path = args.output_path
    if not output_path:
        output_path = path + "." + args.image_format
    elif os.path.isdir(output_path):
        output_path = os.path.join(output_path, os.path.basename(path) + "." + args.image_format)

    if args.no_overwrite:
        if os.path.exists(output_path):
            print("[INFO] contact-sheet already exists, skipping: {}".format(output_path))
            return

    print("Processing {}...".format(path))

    if args.interval is not None and args.manual_timestamps is not None:
        error_exit("Cannot use --interval and --manual at the same time.")

    if args.vcs_width != DEFAULT_CONTACT_SHEET_WIDTH and args.actual_size:
        error_exit("Cannot use --width and --actual-size at the same time.")

    if args.delay_percent is not None:
        args.start_delay_percent = args.delay_percent
        args.end_delay_percent = args.delay_percent

    args.num_groups = 5

    media_info = MediaInfo(
        path,
        verbose=args.is_verbose)
    media_capture = MediaCapture(
        path,
        accurate=args.is_accurate,
        skip_delay_seconds=args.accurate_delay_seconds,
        frame_type=args.frame_type
    )

    # metadata margins
    if not args.metadata_margin == DEFAULT_METADATA_MARGIN:
        args.metadata_horizontal_margin = args.metadata_margin
        args.metadata_vertical_margin = args.metadata_margin

    if args.interval is None and args.manual_timestamps is None and (args.grid.x == 0 or args.grid.y == 0):
        error = "Row or column of size zero is only supported with --interval or --manual."
        error_exit(error)

    if args.interval is not None:
        total_delay = total_delay_seconds(media_info, args)
        selected_duration = media_info.duration_seconds - total_delay
        args.num_samples = math.floor((selected_duration - 1) / args.interval.total_seconds())
        args.num_selected = args.num_samples
        args.num_groups = args.num_samples

    # manual frame selection
    if args.manual_timestamps is not None:
        mframes_size = len(args.manual_timestamps)

        args.num_selected = mframes_size
        args.num_samples = mframes_size
        args.num_groups = mframes_size

    if args.interval is not None or args.manual_timestamps is not None:
        square_side = math.ceil(math.sqrt(args.num_samples))

        if args.grid is None:
            args.grid = Grid(square_side, square_side)
        elif args.grid.x == 0 and args.grid.y == 0:
            args.grid = Grid(square_side, square_side)
        elif args.grid.x == 0:
            # y is fixed
            x = math.ceil(args.num_samples / args.grid.y)
            args.grid = Grid(x, args.grid.y)
        elif args.grid.y == 0:
            # x is fixed
            y = math.ceil(args.num_samples / args.grid.x)
            args.grid = Grid(args.grid.x, y)

    args.num_selected = args.grid.x * args.grid.y
    if args.num_samples is None:
        args.num_samples = args.num_selected

    if args.num_groups is None:
        args.num_groups = args.num_selected

    # make sure num_selected is not too large
    if args.interval is None and args.manual_timestamps is None:
        if args.num_selected > args.num_groups:
            args.num_groups = args.num_selected

        if args.num_selected > args.num_samples:
            args.num_samples = args.num_selected

        # make sure num_samples is large enough
        if args.num_samples < args.num_selected or args.num_samples < args.num_groups:
            args.num_samples = args.num_selected
            args.num_groups = args.num_selected

    if args.grid_spacing is not None:
        args.grid_horizontal_spacing = args.grid_spacing
        args.grid_vertical_spacing = args.grid_spacing

    if args.actual_size:
        x = args.grid.x
        width = media_info.display_width
        args.vcs_width = x * width + (x - 1) * args.grid_horizontal_spacing

    selected_frames, temp_frames = select_sharpest_images(media_info, media_capture, args)

    print("Composing contact sheet...")
    image = compose_contact_sheet(media_info, selected_frames, args)

    is_save_successful = save_image(args, image, media_info, output_path)

    # save selected frames of the contact sheet to the predefined location in thumbnail_output_path
    thumbnail_output_path = args.thumbnail_output_path
    if thumbnail_output_path is not None:
        os.makedirs(thumbnail_output_path, exist_ok=True)
        print("Copying thumbnails to {} ...".format(thumbnail_output_path))
        for i, frame in enumerate(selected_frames):
            print(frame.filename)
            thumbnail_file_extension = frame.filename.lower().split(".")[-1]
            thumbnail_filename = "{filename}.{number}.{extension}".format(filename=os.path.basename(path),
                                                                          number=str(i).zfill(4),
                                                                          extension=thumbnail_file_extension)
            thumbnail_destination = os.path.join(thumbnail_output_path, thumbnail_filename)
            shutil.copyfile(frame.filename, thumbnail_destination)

    print("Cleaning up temporary files...")
    cleanup(temp_frames, args)

    if not is_save_successful:
        error_exit("Unsupported image format: %s." % (args.image_format,))


if __name__ == "__main__":
    main()
