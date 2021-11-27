import re

def events_in_file(filename):
    events = []
    with open(filename) as f:
        lines = f.read().split('\n')[::2]
        content_lines = lines[1::2]
        for content in content_lines:
            events = events + content.strip().split(" ")
    return events