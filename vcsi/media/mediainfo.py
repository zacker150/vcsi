import json
import math
import os
import subprocess

from vcsi.config import Config
from vcsi.error import error_exit


class MediaInfo(object):
    """Collect information about a video file
    """

    def __init__(self, path, verbose=False):
        self.probe_media(path)
        self.find_video_stream()
        self.find_audio_stream()
        self.compute_display_resolution()
        self.compute_format()
        self.parse_attributes()

        if verbose:
            print(self.filename)
            print("%sx%s" % (self.sample_width, self.sample_height))
            print("%sx%s" % (self.display_width, self.display_height))
            print(self.duration)
            print(self.size)

    def probe_media(self, path):
        """Probe video file using ffprobe
        """
        ffprobe_command = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "--",
            path
        ]

        try:
            output = subprocess.check_output(ffprobe_command)
            self.ffprobe_dict = json.loads(output.decode("utf-8"))
        except FileNotFoundError:
            error = "Could not find 'ffprobe' executable. Please make sure ffmpeg/ffprobe is installed and is in your PATH."
            error_exit(error)

    def human_readable_size(self, num, suffix='B'):
        """Converts a number of bytes to a human readable format
        """
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix)

    def find_video_stream(self):
        """Find the first stream which is a video stream
        """
        for stream in self.ffprobe_dict["streams"]:
            try:
                if stream["codec_type"] == "video":
                    self.video_stream = stream
                    break
            except:
                pass

    def find_audio_stream(self):
        """Find the first stream which is an audio stream
        """
        for stream in self.ffprobe_dict["streams"]:
            try:
                if stream["codec_type"] == "audio":
                    self.audio_stream = stream
                    break
            except:
                pass

    def compute_display_resolution(self):
        """Computes the display resolution.
        Some videos have a sample resolution that differs from the display resolution
        (non-square pixels), thus the proper display resolution has to be computed.
        """
        self.sample_width = int(self.video_stream["width"])
        self.sample_height = int(self.video_stream["height"])

        # videos recorded with a smartphone may have a "rotate" flag
        try:
            rotation = int(self.video_stream["tags"]["rotate"])
        except KeyError:
            rotation = None

        if rotation in [90, 270]:
            # swap width and height
            self.sample_width, self.sample_height = self.sample_height, self.sample_width

        sample_aspect_ratio = "1:1"
        try:
            sample_aspect_ratio = self.video_stream["sample_aspect_ratio"]
        except KeyError:
            pass

        if sample_aspect_ratio == "1:1":
            self.display_width = self.sample_width
            self.display_height = self.sample_height
        else:
            sample_split = sample_aspect_ratio.split(":")
            sw = int(sample_split[0])
            sh = int(sample_split[1])

            self.display_width = int(self.sample_width * sw / sh)
            self.display_height = int(self.sample_height)

        if self.display_width == 0:
            self.display_width = self.sample_width

        if self.display_height == 0:
            self.display_height = self.sample_height

    def compute_format(self):
        """Compute duration, size and retrieve filename
        """
        format_dict = self.ffprobe_dict["format"]

        try:
            # try getting video stream duration first
            self.duration_seconds = float(self.video_stream["duration"])
        except (KeyError, AttributeError):
            # otherwise fallback to format duration
            self.duration_seconds = float(format_dict["duration"])

        self.duration = MediaInfo.pretty_duration(self.duration_seconds)

        self.filename = os.path.basename(format_dict["filename"])

        self.size_bytes = int(format_dict["size"])
        self.size = self.human_readable_size(self.size_bytes)

    @staticmethod
    def pretty_to_seconds(
            pretty_duration):
        """Converts pretty printed timestamp to seconds
        """
        millis_split = pretty_duration.split(".")
        millis = 0
        if len(millis_split) == 2:
            millis = int(millis_split[1])
            left = millis_split[0]
        else:
            left = pretty_duration

        left_split = left.split(":")
        if len(left_split) < 3:
            hours = 0
            minutes = int(left_split[0])
            seconds = int(left_split[1])
        else:
            hours = int(left_split[0])
            minutes = int(left_split[1])
            seconds = int(left_split[2])

        result = (millis / 1000.0) + seconds + minutes * 60 + hours * 3600
        return result

    @staticmethod
    def pretty_duration(
            seconds,
            show_centis=False,
            show_millis=False):
        """Converts seconds to a human readable time format
        """
        hours = int(math.floor(seconds / 3600))
        remaining_seconds = seconds - 3600 * hours

        minutes = math.floor(remaining_seconds / 60)
        remaining_seconds = remaining_seconds - 60 * minutes

        duration = ""

        if hours > 0:
            duration += "%s:" % (int(hours),)

        duration += "%s:%s" % (str(int(minutes)).zfill(2), str(int(math.floor(remaining_seconds))).zfill(2))

        if show_centis or show_millis:
            coeff = 1000 if show_millis else 100
            digits = 3 if show_millis else 2
            centis = math.floor((remaining_seconds - math.floor(remaining_seconds)) * coeff)
            duration += ".%s" % (str(int(centis)).zfill(digits))

        return duration

    @staticmethod
    def parse_duration(seconds):
        hours = int(math.floor(seconds / 3600))
        remaining_seconds = seconds - 3600 * hours

        minutes = math.floor(remaining_seconds / 60)
        remaining_seconds = remaining_seconds - 60 * minutes
        seconds = math.floor(remaining_seconds)

        millis = math.floor((remaining_seconds - math.floor(remaining_seconds)) * 1000)
        centis = math.floor((remaining_seconds - math.floor(remaining_seconds)) * 100)

        return {
            "hours": hours,
            "minutes": minutes,
            "seconds": seconds,
            "centis": centis,
            "millis": millis
        }

    def desired_size(self, width=Config.contact_sheet_width):
        """Computes the height based on a given width and fixed aspect ratio.
        Returns (width, height)
        """
        ratio = width / float(self.display_width)
        desired_height = int(math.floor(self.display_height * ratio))
        return (width, desired_height)

    def parse_attributes(self):
        """Parse multiple media attributes
        """
        # video
        try:
            self.video_codec = self.video_stream["codec_name"]
        except KeyError:
            self.video_codec = None

        try:
            self.video_codec_long = self.video_stream["codec_long_name"]
        except KeyError:
            self.video_codec_long = None

        try:
            self.sample_aspect_ratio = self.video_stream["sample_aspect_ratio"]
        except KeyError:
            self.sample_aspect_ratio = None

        try:
            self.display_aspect_ratio = self.video_stream["display_aspect_ratio"]
        except KeyError:
            self.display_aspect_ratio = None

        try:
            self.frame_rate = self.video_stream["avg_frame_rate"]
            splits = self.frame_rate.split("/")

            if len(splits) == 2:
                self.frame_rate = int(splits[0]) / int(splits[1])
            else:
                self.frame_rate = int(self.frame_rate)

            self.frame_rate = round(self.frame_rate, 3)
        except KeyError:
            self.frame_rate = None
        except ZeroDivisionError:
            self.frame_rate = None

        # audio
        try:
            self.audio_codec = self.audio_stream["codec_name"]
        except (KeyError, AttributeError):
            self.audio_codec = None

        try:
            self.audio_codec_long = self.audio_stream["codec_long_name"]
        except (KeyError, AttributeError):
            self.audio_codec_long = None

        try:
            self.audio_sample_rate = int(self.audio_stream["sample_rate"])
        except (KeyError, AttributeError):
            self.audio_sample_rate = None

        try:
            self.audio_bit_rate = int(self.audio_stream["bit_rate"])
        except (KeyError, AttributeError):
            self.audio_bit_rate = None

    def template_attributes(self):
        """Returns the template attributes and values ready for use in the metadata header
        """
        return dict((x["name"], getattr(self, x["name"])) for x in MediaInfo.list_template_attributes())

    @staticmethod
    def list_template_attributes():
        """Returns a list a of all supported template attributes with their description and example
        """
        table = []
        table.append({"name": "size", "description": "File size (pretty format)", "example": "128.3 MiB"})
        table.append({"name": "size_bytes", "description": "File size (bytes)", "example": "4662788373"})
        table.append({"name": "filename", "description": "File name", "example": "video.mkv"})
        table.append({"name": "duration", "description": "Duration (pretty format)", "example": "03:07"})
        table.append({"name": "sample_width", "description": "Sample width (pixels)", "example": "1920"})
        table.append({"name": "sample_height", "description": "Sample height (pixels)", "example": "1080"})
        table.append({"name": "display_width", "description": "Display width (pixels)", "example": "1920"})
        table.append({"name": "display_height", "description": "Display height (pixels)", "example": "1080"})
        table.append({"name": "video_codec", "description": "Video codec", "example": "h264"})
        table.append({"name": "video_codec_long", "description": "Video codec (long name)",
                      "example": "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10"})
        table.append({"name": "display_aspect_ratio", "description": "Display aspect ratio", "example": "16:9"})
        table.append({"name": "sample_aspect_ratio", "description": "Sample aspect ratio", "example": "1:1"})
        table.append({"name": "audio_codec", "description": "Audio codec", "example": "aac"})
        table.append({"name": "audio_codec_long", "description": "Audio codec (long name)",
                      "example": "AAC (Advanced Audio Coding)"})
        table.append({"name": "audio_sample_rate", "description": "Audio sample rate (Hz)", "example": "44100"})
        table.append({"name": "audio_bit_rate", "description": "Audio bit rate (bits/s)", "example": "192000"})
        table.append({"name": "frame_rate", "description": "Frame rate (frames/s)", "example": "23.974"})
        return table
