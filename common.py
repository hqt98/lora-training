import os
import shutil
import mimetypes
import re
from zipfile import ZipFile


def clean_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def clean_directories(paths):
    for path in paths:
        clean_directory(path)


def random_seed():
    return int.from_bytes(os.urandom(2), "big")

