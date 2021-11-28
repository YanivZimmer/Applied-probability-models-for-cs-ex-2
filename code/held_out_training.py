from math import floor


def split_held_outtrain_validation(events):
    index = floor(len(events) / 2)
    train = events[:index]
    validation = events[index:]
    return train, validation


