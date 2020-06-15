import math
import os
import subprocess

import numpy as np
from PIL import Image

from .mediainfo import MediaInfo
from ..config import Config
from ..error import error_exit

try:
    from subprocess import DEVNULL
except ImportError:
    DEVNULL = open(os.devnull, 'wb')


class MediaCapture(object):
    """Capture frames of a video
    """

    def __init__(self, path, accurate=False, skip_delay_seconds=Config.accurate_delay_seconds,
                 frame_type=Config.frame_type):
        self.path = path
        self.accurate = accurate
        self.skip_delay_seconds = skip_delay_seconds
        self.frame_type = frame_type

    def make_capture(self, time, width, height, out_path="out.png"):
        """Capture a frame at given time with given width and height using ffmpeg
        """
        skip_delay = MediaInfo.pretty_duration(self.skip_delay_seconds, show_millis=True)

        ffmpeg_command = [
            "ffmpeg",
            "-ss", time,
            "-i", self.path,
            "-vframes", "1",
            "-s", "%sx%s" % (width, height),
        ]

        if self.frame_type is not None:
            select_args = [
                "-vf", "select='eq(frame_type\\,{})'".format(self.frame_type)
            ]

        if self.frame_type == "key":
            select_args = [
                "-vf", "select=key"
            ]

        if self.frame_type is not None:
            ffmpeg_command += select_args

        ffmpeg_command += [
            "-y",
            out_path
        ]

        if self.accurate:
            time_seconds = MediaInfo.pretty_to_seconds(time)
            skip_time_seconds = time_seconds - self.skip_delay_seconds

            if skip_time_seconds < 0:
                ffmpeg_command = [
                    "ffmpeg",
                    "-i", self.path,
                    "-ss", time,
                    "-vframes", "1",
                    "-s", "%sx%s" % (width, height),
                ]

                if self.frame_type is not None:
                    ffmpeg_command += select_args

                ffmpeg_command += [
                    "-y",
                    out_path
                ]
            else:
                skip_time = MediaInfo.pretty_duration(skip_time_seconds, show_millis=True)
                ffmpeg_command = [
                    "ffmpeg",
                    "-ss", skip_time,
                    "-i", self.path,
                    "-ss", skip_delay,
                    "-vframes", "1",
                    "-s", "%sx%s" % (width, height),
                ]

                if self.frame_type is not None:
                    ffmpeg_command += select_args

                ffmpeg_command += [
                    "-y",
                    out_path
                ]

        try:
            subprocess.call(ffmpeg_command, stderr=DEVNULL, stdout=DEVNULL)
        except FileNotFoundError:
            error = "Could not find 'ffmpeg' executable. Please make sure ffmpeg/ffprobe is installed and is in your PATH."
            error_exit(error)

    def compute_avg_color(self, image_path):
        """Computes the average color of an image
        """
        i = Image.open(image_path)
        i = i.convert('P')
        p = i.getcolors()

        # compute avg color
        total_count = 0
        avg_color = 0
        for count, color in p:
            total_count += count
            avg_color += count * color

        avg_color /= total_count

        return avg_color

    def compute_blurriness(self, image_path):
        """Computes the blurriness of an image. Small value means less blurry.
        """
        i = Image.open(image_path)
        i = i.convert('L')  # convert to grayscale

        a = np.asarray(i)
        b = abs(np.fft.rfft2(a))
        max_freq = self.avg9x(b)

        if max_freq != 0:
            return 1 / max_freq
        else:
            return 1

    def avg9x(self, matrix, percentage=0.05):
        """Computes the median of the top n% highest values.
        By default, takes the top 5%
        """
        xs = matrix.flatten()
        srt = sorted(xs, reverse=True)
        length = int(math.floor(percentage * len(srt)))

        matrix_subset = srt[:length]
        return np.median(matrix_subset)

    def max_freq(self, matrix):
        """Returns the maximum value in the matrix
        """
        m = 0
        for row in matrix:
            mx = max(row)
            if mx > m:
                m = mx

        return m
