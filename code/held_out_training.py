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


def calculate_t_r(occurrences_word_dict_train, counter_train, counter_test, test_data, r):
    cnt = 0
    if r not in occurrences_word_dict_train:
        if r == 0:
            for word in test_data:
                if word not in counter_train:
                    cnt = cnt + 1
            return cnt
        return 0
    words_appeared_r_times_in_train = occurrences_word_dict_train[r]
    for word in words_appeared_r_times_in_train:
        if word in counter_test:
            cnt = cnt + counter_test[word]
    return cnt


def calc_n_r(occurrences_word_dict_train, train_data, r):
    if r in occurrences_word_dict_train:
        return len(occurrences_word_dict_train[r])
    return VOCABULARY_SIZE - len(set(train_data))


def calculation_held_out(occurrences_word_dict_train, counter_train, counter_test, test_data, train_data, word):
    if word in counter_train.keys():
        r = counter_train[word]
    else:
        r = 0
    # occurrences_word_dict_train, max_occourence_train = counter_to_dictionary(counter_train)

    numenator = calculate_t_r(occurrences_word_dict_train=occurrences_word_dict_train, counter_train=counter_train,
                              counter_test=counter_test, test_data=test_data, r=r)
    dec = calc_n_r(occurrences_word_dict_train=occurrences_word_dict_train, train_data=train_data, r=r) * len(test_data)
    return numenator / dec


def calc_held_out(train_data, test_data, word):
    counter_train = Counter(train_data)
    counter_test = Counter(test_data)
    occurrences_word_dict_train, max_occourence_train = counter_to_dictionary(counter_train)
    return calculation_held_out(occurrences_word_dict_train, counter_train, counter_test, test_data, train_data, word)


def validation_held_out(test_data, train_data):
    counter_train = Counter(test_data)
    counter_test = Counter(test_data)
    occurrences_word_dict_train, max_occourence_train = counter_to_dictionary(counter_train)
    set_test_data = set(test_data)
    total = 0
    for word in set_test_data:
        total = total + calculation_held_out(occurrences_word_dict_train=occurrences_word_dict_train,
                                             counter_train=counter_train, counter_test=counter_test,
                                             test_data=test_data, train_data=train_data, word=word)
    total = total + (calculation_held_out(occurrences_word_dict_train=occurrences_word_dict_train,
                                          counter_train=counter_train, counter_test=counter_test,
                                          test_data=test_data, train_data=train_data, word=UNSEEN_WORD)) * calc_n_r(
        occurrences_word_dict_train=counter_train,
        train_data=train_data, r=0)
    return total
