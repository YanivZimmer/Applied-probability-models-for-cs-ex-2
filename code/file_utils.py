import os.path
from pathlib import Path

def is_file_exists(directory, file_name):
    file_location = os.path.join(directory, file_name)
    if Path(file_location).is_file():
        return True
    return False
