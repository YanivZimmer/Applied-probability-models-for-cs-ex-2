from math import floor
from collections import Counter
from .consts import *


def split_held_outtrain_validation(events):
    index = floor(len(events) / 2)
    train = events[:index]
    validation = events[index:]
    return train, validation


def counter_to_dictionary(counter_data):
    occurrences_word_dict = {}
    max_occourence = 0
    for item, occourence in counter_data.items():
        if occourence in occurrences_word_dict:
            occurrences_word_dict[occourence].append(item)
        else:
            occurrences_word_dict[occourence] = [item]
        max_occourence = max(max_occourence, occourence)
    return occurrences_word_dict, max_occourence


def calculate_t_r(occurrences_word_dict_train, counter_train, test_data, r):
    cnt = 0
    if r not in occurrences_word_dict_train:
        if r == 0:
            for word in test_data:
                if word not in counter_train:
                    cnt = cnt + 1
            return cnt
        return 0
    words_appeared_r_times_in_train = occurrences_word_dict_train[r]
    for word in test_data:
        if word in words_appeared_r_times_in_train:
            cnt = cnt + 1
    return cnt


def calc_n_r(occurrences_word_dict_train, train_data, r):
    if r in occurrences_word_dict_train:
        return len(occurrences_word_dict_train[r])
    return VOCABULARY_SIZE - len(set(train_data))


def calc_held_out(train_data, test_data, word):
    counter_train = Counter(train_data)
    if word in counter_train.keys():
        r = counter_train[word]
    else:
        r = 0
    occurrences_word_dict_train, max_occourence_train = counter_to_dictionary(counter_train)

    numenator = calculate_t_r(occurrences_word_dict_train=occurrences_word_dict_train, counter_train=counter_train,
                              test_data=test_data, r=r)
    dec = calc_n_r(occurrences_word_dict_train=occurrences_word_dict_train, train_data=train_data, r=r) * len(test_data)
    return numenator / dec
