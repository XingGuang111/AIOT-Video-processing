"""Race-tolerant scans while downstream deletes consumed directories."""
import os
from pathlib import Path


def second_directories(root, min_ts=0):
    result = []
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                try:
                    if entry.name.isdigit() and int(entry.name) >= min_ts and entry.is_dir(follow_symlinks=False):
                        result.append(Path(entry.path))
                except OSError:
                    continue
    except FileNotFoundError:
        pass
    return result


def ready_seconds(root, min_ts=0):
    result = []
    for directory in second_directories(root, min_ts):
        try:
            if (directory / '_ready.json').is_file():
                result.append(directory)
        except OSError:
            continue
    return result


def buffer_usage(root):
    frames = size = 0
    for directory in second_directories(root):
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.name.endswith('.jpg'):
                        continue
                    try:
                        size += entry.stat(follow_symlinks=False).st_size
                        frames += 1
                    except OSError:
                        continue
        except (FileNotFoundError, PermissionError):
            continue
    return frames, size
