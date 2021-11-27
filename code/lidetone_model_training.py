from math import floor, log2, pow
from collections import Counter


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


#
def calc_prob_lind(data, lamda):
    prob = {}
    data_len = len(data)
    counter = Counter(data)
    vocab_size = count_unique_events(data)
    for item in counter.items():
        lind = lind_mle(lamda=lamda, w_in_events=item[1], num_of_events=data_len, vocab_size=vocab_size)
        prob[item[0]] = lind
    return prob


# # TODO- ask what log base should be used?
def calc_log_event(event, prob_dict, prob_unseen):
    if event not in prob_dict.keys():
        return prob_unseen
    return log2(prob_dict[event])


def calc_log_sum(data, prob_dict, prob_unseen):
    sum = 0
    for event in data:
        sum = sum + calc_log_event(event=event, prob_dict=prob_dict, prob_unseen=prob_unseen)
    return sum


def calc_preplexity(data, prob_dict, prob_unseen):
    power = -1 * calc_log_sum(data=data, prob_dict=prob_dict, prob_unseen=prob_unseen) / (len(data))
    base = 2
    result = pow(base, power)
    return result


# calc preplexity for validation set
def preplexity(valid_data, train_data, lamda):
    prob_dict = calc_prob_lind(data=train_data, lamda=lamda)
    prob_unseen = lind_mle(lamda=lamda, w_in_events=0, num_of_events=len(train_data)
                           , vocab_size=count_unique_events(train_data))
    prepl = calc_preplexity(data=valid_data, prob_dict=prob_dict, prob_unseen=prob_unseen)
    return prepl
