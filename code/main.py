import sys, os
from math import floor
from collections import Counter
import os.path
import os.path
from pathlib import Path
from math import floor, log2, pow
from collections import Counter

DIRECTORY_PATH = "../dataset"
VOCABULARY_SIZE = 300000
LIDESTONE_SPLIT_RATE = 0.9
UNSEEN_WORD = "unseen-word"


### TODO change main.py file name to ex2.py

def is_file_exists(directory, file_name):
    file_location = os.path.join(directory, file_name)
    if Path(file_location).is_file():
        return True
    return False


class Unigram(object):
    def __init__(self, develop_file, test_file, input_word, output_file, vocabulary_size, directory_path):
        self.develop_file = develop_file
        self.test_file = test_file
        self.input_word = input_word
        self.output_file = output_file
        self.vocabulary_size = vocabulary_size
        self.directory_path = directory_path

    def p_uniform(self):
        return 1 / self.vocabulary_size

    def __str__(self):
        return "dev:{}, test:{}, input:{}, output:{}".format(self.develop_file, self.test_file, self.input_word,
                                                             self.output_file)

    def validate_files(self):
        if is_file_exists(self.directory_path, self.develop_file) and is_file_exists(
                self.directory_path, self.test_file) and not is_file_exists(self.directory_path,
                                                                            self.output_file):
            return True
        return False

    def develop_path(self):
        return os.path.join(self.directory_path, self.develop_file)

    def test_path(self):
        return os.path.join(self.directory_path, self.test_file)

    def output_path(self):
        return os.path.join(self.directory_path, self.output_file)


def initator(arguments, output_list):
    unigram_model = Unigram(arguments[0], arguments[1], arguments[2], arguments[3], VOCABULARY_SIZE, DIRECTORY_PATH)
    if not unigram_model.validate_files():
        return None
    output_list.append(unigram_model.develop_file)
    output_list.append(unigram_model.test_file)
    output_list.append(unigram_model.input_word)
    output_list.append(unigram_model.output_file)
    output_list.append(unigram_model.vocabulary_size)
    output_list.append(unigram_model.p_uniform())
    return unigram_model


### preprocessing
def events_in_file(filename):
    events = []
    with open(filename) as f:
        lines = f.read().split('\n')[::2]
        content_lines = lines[1::2]
        for content in content_lines:
            events = events + content.strip().split(" ")
    return events


def preprocessing(path):
    events = events_in_file(path)
    return events


### lidestone


def split_lidetone_train_validation(events, split_rate):
    index = round(split_rate * len(events))
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
    for lamda in range(2, 201):
        noramlized_lamda = lamda / 100
        prep_ = preplexity(valid_data=valid_data, train_data=train_data, lamda=noramlized_lamda)
        if prep_ < min_prep:
            min_prep = prep_
            min_lamda = noramlized_lamda
    return min_lamda, min_prep


def validate_lidestone_model_training(test_data):
    len_test_data = len(test_data)
    set_test_data = set(test_data)
    total = 0
    for word in set_test_data:
        number_of_occurences_for_input = count_event_in_events(word, test_data)
        total = total + lind_mle(0.06, w_in_events=number_of_occurences_for_input, num_of_events=len_test_data,
                                 vocab_size=VOCABULARY_SIZE)
    total = total + (VOCABULARY_SIZE - len(set_test_data)) * lind_mle(0.06, w_in_events=0, num_of_events=len_test_data,
                                                                      vocab_size=VOCABULARY_SIZE)

    return total


def lidetone(unigram_model, events, output_list):
    train_dev, validation_dev = split_lidetone_train_validation(events, LIDESTONE_SPLIT_RATE)
    validation_dev_len = len(validation_dev)
    train_dev_len = len(train_dev)

    output_list.append(validation_dev_len)
    output_list.append(train_dev_len)

    observed_vocab_size = count_unique_events(train_dev)
    output_list.append(observed_vocab_size)

    number_of_occurences_for_input = count_event_in_events(unigram_model.input_word, train_dev)

    output_list.append(number_of_occurences_for_input)
    output_list.append(number_of_occurences_for_input / train_dev_len)

    number_of_occurences_for_unseen_input = count_event_in_events(UNSEEN_WORD, train_dev)
    output_list.append(number_of_occurences_for_unseen_input / train_dev_len)

    output_list.append(lind_mle(0.1, number_of_occurences_for_input, train_dev_len, VOCABULARY_SIZE))

    output_list.append(lind_mle(0.1, number_of_occurences_for_unseen_input, train_dev_len, VOCABULARY_SIZE))

    output_list.append(preplexity(valid_data=validation_dev, train_data=train_dev, lamda=0.01))

    output_list.append(preplexity(valid_data=validation_dev, train_data=train_dev, lamda=0.1))

    output_list.append(preplexity(valid_data=validation_dev, train_data=train_dev, lamda=1))

    optimal_lamda, min_prep = find_optimal_lamda(valid_data=validation_dev, train_data=train_dev)
    output_list.append(optimal_lamda)
    output_list.append(min_prep)

    # validation
    print(validate_lidestone_model_training(test_data=validation_dev))


#### end lidestone

### heldout


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


def held_out(unigram_model, events, output_list):
    train_data, test_data = split_held_outtrain_validation(events=events)

    train_data_len = len(train_data)
    test_data_len = len(test_data)
    output_list.append(train_data_len)
    output_list.append(test_data_len)
    output_list.append(calc_held_out(train_data, test_data, unigram_model.input_word))
    output_list.append(calc_held_out(train_data, test_data, UNSEEN_WORD))

    # validation
    print(validation_held_out(test_data=test_data, train_data=train_data))


### end heldout

def total_event_test_set(unigram_model):
    return preprocessing(unigram_model.test_path())


def model_evaluation_test(unigram_model, output_list):
    print(unigram_model.test_path())
    events = total_event_test_set(unigram_model)
    output_list.append(len(events))
    best_lamda = 0.06  # output_list[18]

    train_test, validation_test = split_lidetone_train_validation(events, LIDESTONE_SPLIT_RATE)
    prep = preplexity(valid_data=validation_test, train_data=train_test, lamda=best_lamda)
    print(prep)

    # print("events={0},train-len={1},valid-len={2},prep={3}".format(len(events), len(train_test), len(validation_test),
    #                                                                prep))


def output_list_to_string(output_list):
    # TODO: insert ID
    str = "#Students	Ben Nageris	Yaniv Zimmer <ID1> <ID2>\n"
    for idx, item in enumerate(output_list):
        str = str + "#Output{idx}     {item}\n".format(idx=idx + 1, item=item)
    return str


def run(arguments):
    output_list = []
    unigram_model = initator(arguments, output_list)
    if unigram_model is None:
        print("Input us incorrect!")
        return None
    events = preprocessing(unigram_model.develop_path())
    output_list.append(len(events))
    lidetone(unigram_model, events, output_list)
    held_out(unigram_model, events, output_list)
    model_evaluation_test(unigram_model=unigram_model, output_list=output_list)

    print(output_list_to_string(output_list))
    # TODO implement function that iterates output and writes to file in requested format OutputX: Y


if __name__ == "__main__":
    run(sys.argv[1:])
