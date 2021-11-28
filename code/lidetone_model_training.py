from math import floor, log2, pow
from collections import Counter
from code.consts import *


def split_train_validation(events, split_rate):
    index = floor(split_rate * len(events))
    train = events[:index]
    validation = events[index:]
    return train, validation


def count_unique_events(events):
    vocab = set(events)
    return len(vocab)


def count_event_in_events(event, events):
    return events.count(event)


# lindMLE=(occurrences of w in events+lamda)/(number of event+lamda*|vocab|)
def lind_mle(lamda, w_in_events, num_of_events, vocab_size):
    numerator = w_in_events + lamda
    denominator = num_of_events + lamda * vocab_size
    return numerator / denominator


def calc_prob_lind(data, lamda, vocab_size):
    prob = {}
    data_len = len(data)
    counter = Counter(data)
    for item in counter.items():
        lind = lind_mle(lamda=lamda, w_in_events=item[1], num_of_events=data_len, vocab_size=vocab_size)
        prob[item[0]] = lind
    return prob


def calc_log_event(event, prob_dict, prob_unseen):
    if event not in prob_dict.keys():
        return log2(prob_unseen)
    return log2(prob_dict[event])


def calc_log_sum(data, prob_dict, prob_unseen):
    sum_ = 0
    for event in data:
        sum_ = sum_ + calc_log_event(event=event, prob_dict=prob_dict, prob_unseen=prob_unseen)
    return sum_


def calc_preplexity(data, prob_dict, prob_unseen):
    power = (-1 * calc_log_sum(data=data, prob_dict=prob_dict, prob_unseen=prob_unseen)) / (len(data))
    base = 2
    result = pow(base, power)
    return result


# calc preplexity for validation set
def preplexity(valid_data, train_data, lamda):
    prob_dict = calc_prob_lind(data=train_data, lamda=lamda, vocab_size=VOCABULARY_SIZE)
    prob_unseen = lind_mle(lamda=lamda, w_in_events=0, num_of_events=len(train_data)
                           , vocab_size=VOCABULARY_SIZE)
    prepl = calc_preplexity(data=valid_data, prob_dict=prob_dict, prob_unseen=prob_unseen)
    return prepl


def find_optimal_lamda(valid_data, train_data):
    min_prep = preplexity(valid_data=valid_data, train_data=train_data, lamda=0.01)
    min_lamda = 0.01
    for lamda in range(2, 101):
        noramlized_lamda = lamda / 100
        prep_ = preplexity(valid_data=valid_data, train_data=train_data, lamda=noramlized_lamda)
        if prep_ < min_prep:
            min_prep = prep_
            min_lamda = noramlized_lamda
    return min_lamda, min_prep
