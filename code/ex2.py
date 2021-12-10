# Ben Nageris And Yaniv Zimmer <ID1> <ID2>
import sys, os
import os.path
from pathlib import Path
from math import floor, log2, pow
from collections import Counter

EPSILON = 0.00000001
LAMBDA_DEFUALT_VALUE = -1
DIRECTORY_PATH = "../dataset"
VOCABULARY_SIZE = 300000
LIDESTONE_SPLIT_RATE = 0.9
HEDLOUT_SPLIT_RATE = 0.5
UNSEEN_WORD = "unseen-word"


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
                self.directory_path, self.test_file):
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


def split_data(events, split_rate):
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
def lind_mle(lamda, occourences_w_in_events, total_num_of_events, vocabulary_size):
    numerator = occourences_w_in_events + lamda
    denominator = total_num_of_events + lamda * vocabulary_size
    return numerator / denominator


def calc_prob_lind(data_counter, data_len, lamda, vocab_size):
    prob = {}
    for name, occourences in data_counter.items():
        lind = lind_mle(lamda=lamda, occourences_w_in_events=occourences, total_num_of_events=data_len,
                        vocabulary_size=vocab_size)
        prob[name] = lind
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
def lidetone_preplexity(valid_data, counter_train, train_data_len, lamda):
    prob_dict = calc_prob_lind(data_counter=counter_train, data_len=train_data_len, lamda=lamda,
                               vocab_size=VOCABULARY_SIZE)
    prob_unseen = lind_mle(lamda=lamda, occourences_w_in_events=0, total_num_of_events=train_data_len
                           , vocabulary_size=VOCABULARY_SIZE)
    prepl = calc_preplexity(data=valid_data, prob_dict=prob_dict, prob_unseen=prob_unseen)
    return prepl


def counter_to_word_occurrence_dict(counter):
    dict_ = {}
    for item, value in counter.items():
        dict_[item] = value
    return dict_


def initiate_heldout_params(train_data, test_data):
    counter_train = Counter(train_data)
    word_to_occurrence_dict = counter_to_word_occurrence_dict(counter_train)
    counter_test = Counter(test_data)
    occurrences_word_dict_train, max_occourence_train = counter_to_occourenct_dictionary(counter_train)
    len_dictionary = dictionary_to_len_dictionary(occurrences_word_dict_train)
    set_test_data = set(test_data)
    number_of_unique_word_in_train = len(train_data)
    number_of_unique_word_in_test = len(set_test_data)
    frequency_prob_dict = {}
    unique_events_in_train_cnt = len(set(train_data))
    t_r_cache = {}
    for word in set_test_data:
        if word in word_to_occurrence_dict:
            freq = word_to_occurrence_dict[word]
        else:
            freq = 0
        if freq not in frequency_prob_dict:
            frequency_prob_dict[freq] = calculation_held_out(occurrences_word_dict_train=occurrences_word_dict_train,
                                                             counter_train=counter_train, counter_test=counter_test,
                                                             test_data=test_data, word=word,
                                                             len_train_data=number_of_unique_word_in_train,
                                                             unique_events_in_train_cnt=unique_events_in_train_cnt,
                                                             t_r_cache=t_r_cache,
                                                             len_dictionary=len_dictionary)
    q = len(set(train_data)) + len(set(test_data))
    return counter_train, len(train_data), counter_test, len(
        test_data), occurrences_word_dict_train, len_dictionary, word_to_occurrence_dict, number_of_unique_word_in_train, \
           number_of_unique_word_in_test, occurrences_word_dict_train, max_occourence_train, frequency_prob_dict, t_r_cache, q


def word_to_occourence_to_word_probability_dict(word_to_occurrence_dict, frequency_prob_dict):
    dict_ = {}
    for word in word_to_occurrence_dict:
        occurrence = word_to_occurrence_dict[word]

        val = frequency_prob_dict[occurrence]
        dict_[word] = val
    return dict_


def heldout_preplexity(test_data, word_to_occurrence_dict, frequency_prob_dict, output_list):
    prob_dict = word_to_occourence_to_word_probability_dict(word_to_occurrence_dict=word_to_occurrence_dict,
                                                            frequency_prob_dict=frequency_prob_dict)
    prob_unseen = output_list[23]
    return calc_preplexity(test_data, prob_dict, prob_unseen)


def find_optimal_lamda(valid_data, counter_train, train_data_len):  # valid_data, train_data):
    min_prep = lidetone_preplexity(valid_data=valid_data, counter_train=counter_train, train_data_len=train_data_len
                                   , lamda=0.01)
    min_lamda = 0.01
    for lamda in range(2, 201):
        noramlized_lamda = lamda / 100
        prep_ = lidetone_preplexity(valid_data=valid_data, counter_train=counter_train, train_data_len=train_data_len,
                                    lamda=noramlized_lamda)
        if prep_ < min_prep:
            min_prep = prep_
            min_lamda = noramlized_lamda
    return min_lamda, min_prep


def validate_lidestone_model_training(test_data, lamda=0.06):
    len_test_data = len(test_data)
    set_test_data = set(test_data)
    total = 0
    for word in set_test_data:
        number_of_occurences_for_input = count_event_in_events(word, test_data)
        total = total + lind_mle(lamda, occourences_w_in_events=number_of_occurences_for_input,
                                 total_num_of_events=len_test_data,
                                 vocabulary_size=VOCABULARY_SIZE)
    total = total + (VOCABULARY_SIZE - len(set_test_data)) * lind_mle(lamda, occourences_w_in_events=0,
                                                                      total_num_of_events=len_test_data,
                                                                      vocabulary_size=VOCABULARY_SIZE)

    return total


def lidetone(unigram_model, events, output_list):
    train_data, develop_data = split_data(events, LIDESTONE_SPLIT_RATE)
    counter_train = Counter(train_data)
    train_data_len = len(train_data)
    develop_len = len(develop_data)

    output_list.append(develop_len)
    output_list.append(train_data_len)

    observed_vocab_size = count_unique_events(train_data)
    output_list.append(observed_vocab_size)

    number_of_occurences_for_input = count_event_in_events(unigram_model.input_word, train_data)

    output_list.append(number_of_occurences_for_input)
    output_list.append(number_of_occurences_for_input / train_data_len)

    number_of_occurences_for_unseen_input = count_event_in_events(UNSEEN_WORD, train_data)
    output_list.append(number_of_occurences_for_unseen_input / train_data_len)

    output_list.append(lind_mle(0.1, number_of_occurences_for_input, train_data_len, VOCABULARY_SIZE))

    output_list.append(lind_mle(0.1, number_of_occurences_for_unseen_input, train_data_len, VOCABULARY_SIZE))

    output_list.append(
        lidetone_preplexity(valid_data=develop_data, counter_train=counter_train, train_data_len=train_data_len,
                            lamda=0.01))
    output_list.append(
        lidetone_preplexity(valid_data=develop_data, counter_train=counter_train, train_data_len=train_data_len,
                            lamda=0.1))
    output_list.append(
        lidetone_preplexity(valid_data=develop_data, counter_train=counter_train, train_data_len=train_data_len,
                            lamda=1))
    optimal_lamda, min_prep = find_optimal_lamda(valid_data=develop_data, counter_train=counter_train,
                                                 train_data_len=train_data_len)
    output_list.append(optimal_lamda)
    output_list.append(min_prep)

    # validation
    return train_data, train_data_len, develop_data, develop_len, counter_train


#### end lidestone

### heldout


# def split_heldout_train_validation(events):
#     index = floor(len(events) / 2)
#     train = events[:index]
#     validation = events[index:]
#     return train, validation


def counter_to_occourenct_dictionary(counter_data):
    occurrences_word_dict = {}
    max_occourence = 0
    for item, occourence in counter_data.items():
        if occourence in occurrences_word_dict:
            occurrences_word_dict[occourence].append(item)
        else:
            occurrences_word_dict[occourence] = [item]
        max_occourence = max(max_occourence, occourence)
    return occurrences_word_dict, max_occourence


def dictionary_to_len_dictionary(occurrences_word_dict):
    len_dictionary = {}
    for item in occurrences_word_dict:
        len_dictionary[item] = len(occurrences_word_dict[item])
    return len_dictionary


def calculate_t_r(occurrences_word_dict_train, counter_train, counter_test, test_data, r):
    cnt = 0
    if r not in occurrences_word_dict_train:
        if r == 0:
            for word in test_data:
                if word not in counter_train.keys():
                    cnt = cnt + 1
            return cnt
        return 0
    words_appeared_r_times_in_train = occurrences_word_dict_train[r]
    for word in words_appeared_r_times_in_train:
        if word in counter_test:
            cnt = cnt + counter_test[word]
    return cnt


# calculate the number of words in test that we observed r rimes
def calc_n_r(occurrences_word_dict_train, number_of_unique_word_in_train, r, len_dictionary):
    if r in occurrences_word_dict_train:
        return len_dictionary[r]
    return VOCABULARY_SIZE - number_of_unique_word_in_train


def calculation_held_out(occurrences_word_dict_train, counter_train, counter_test, test_data, word,
                         len_train_data, unique_events_in_train_cnt, t_r_cache, len_dictionary):
    if word in counter_train.keys():
        r = counter_train[word]
    else:
        r = 0
    if r not in t_r_cache:
        t_r_cache[r] = calculate_t_r(occurrences_word_dict_train=occurrences_word_dict_train,
                                     counter_train=counter_train,
                                     counter_test=counter_test, test_data=test_data, r=r)
    numenator = t_r_cache[r]
    dec = calc_n_r(occurrences_word_dict_train=occurrences_word_dict_train,
                   number_of_unique_word_in_train=unique_events_in_train_cnt, r=r,
                   len_dictionary=len_dictionary) * len_train_data
    return numenator / dec


def validation_heldout(word_to_occurrence_dict, frequency_prob_dict, train_data, len_dictionary,
                       occurrences_word_dict_train):
    total = 0
    unique_events_in_train_cnt = len(set(train_data))
    for word in set(train_data):
        total = total + heldout_lookup_for_probability(word_to_occurrence_dict=word_to_occurrence_dict,
                                                       frequency_prob_dict=frequency_prob_dict, word=word)
    total = total + (heldout_lookup_for_probability(word_to_occurrence_dict=word_to_occurrence_dict,
                                                    frequency_prob_dict=frequency_prob_dict,
                                                    word=UNSEEN_WORD) *
                     calc_n_r(len_dictionary=len_dictionary, r=0,
                              occurrences_word_dict_train=occurrences_word_dict_train,
                              number_of_unique_word_in_train=unique_events_in_train_cnt))
    return total


def heldout_lookup_for_probability(word_to_occurrence_dict, frequency_prob_dict, word):
    if word in word_to_occurrence_dict:
        freq = word_to_occurrence_dict[word]
    else:
        freq = 0
    return frequency_prob_dict[freq]


def heldout(unigram_model, events, output_list):
    train_data, test_data = split_data(events=events, split_rate=HEDLOUT_SPLIT_RATE)
    counter_train, len_train_data, counter_test, len_test_data, occurrences_word_dict_train, len_dictionary, word_to_occurrence_dict, number_of_unique_word_in_train, \
    number_of_unique_word_in_test, occurrences_word_dict_train, max_occourence_train, frequency_prob_dict, t_r_cache, unique_events_in_train_cnt = initiate_heldout_params(
        train_data=train_data, test_data=test_data)
    output_list.append(len_train_data)
    output_list.append(len_test_data)
    output_list.append(heldout_lookup_for_probability(word_to_occurrence_dict=word_to_occurrence_dict,
                                                      frequency_prob_dict=frequency_prob_dict,
                                                      word=unigram_model.input_word))
    output_list.append(heldout_lookup_for_probability(word_to_occurrence_dict=word_to_occurrence_dict,
                                                      frequency_prob_dict=frequency_prob_dict,
                                                      word=UNSEEN_WORD))
    return word_to_occurrence_dict, frequency_prob_dict, train_data, len_dictionary, occurrences_word_dict_train, t_r_cache, unique_events_in_train_cnt


def validate_score(score, required_score=1, epsilon=EPSILON):
    return True if required_score - epsilon < score < required_score + epsilon else False


def validation(word_to_occurrence_dict, frequency_prob_dict, heldout_train_data, len_dictionary,
               occurrences_word_dict_train, lidetone_train_data, lamda):
    heldout_validation_score = validation_heldout(word_to_occurrence_dict=word_to_occurrence_dict,
                                                  frequency_prob_dict=frequency_prob_dict,
                                                  train_data=heldout_train_data, len_dictionary=len_dictionary,
                                                  occurrences_word_dict_train=occurrences_word_dict_train)
    lidetone_validation_score = validate_lidestone_model_training(test_data=lidetone_train_data, lamda=lamda)
    print(heldout_validation_score)
    print(lidetone_validation_score)
    if not validate_score(score=heldout_validation_score):
        print("Heldout Model Validation Failed")
        return False
    if not validate_score(score=lidetone_validation_score):
        print("Lidetone Model Validation Failed")
        return False
    return True


### end heldout

def total_event_test_set(unigram_model):
    return preprocessing(unigram_model.test_path())


def model_evaluation_test(counter_train, train_data_len, unigram_model, frequency_prob_dict, word_to_occurrence_dict,
                          output_list):
    validation_events = total_event_test_set(unigram_model)
    len_validation_events = len(validation_events)
    output_list.append(len_validation_events)

    best_lamda = output_list[18]
    lidetone_prep = lidetone_preplexity(valid_data=validation_events, counter_train=counter_train,
                                        train_data_len=train_data_len,
                                        lamda=best_lamda)

    heldout_prep = heldout_preplexity(test_data=validation_events, word_to_occurrence_dict=word_to_occurrence_dict,
                                      frequency_prob_dict=frequency_prob_dict, output_list=output_list)

    output_list.append(lidetone_prep)
    output_list.append(heldout_prep)
    output_list.append("L" if lidetone_prep < heldout_prep else "H")


def output_list_to_string(output_list):
    # TODO: insert ID
    str = "#Students	Ben Nageris	Yaniv Zimmer <ID1> <ID2>\n"
    for idx, item in enumerate(output_list):
        str = str + "#Output{idx}\t{item}\n".format(idx=idx + 1, item=item)
    return str


def get_t_r(t_r_cache, r):
    if r in t_r_cache:
        return t_r_cache[r]
    return -1


def run(arguments):
    output_list = []
    unigram_model = initator(arguments, output_list)
    if unigram_model is None:
        print("Input us incorrect!")
        return None
    events = preprocessing(unigram_model.develop_path())
    output_list.append(len(events))
    lidestone_train_data, lidestone_train_data_len, lidestone_develop_data, lidestone_develop_len, listestone_counter_train = lidetone(
        unigram_model, events,
        output_list)

    word_to_occurrence_dict, frequency_prob_dict, heldout_train_data, len_dictionary, occurrences_word_dict_train, t_r_cache, unique_events_in_train_cnt = heldout(
        unigram_model, events, output_list)

    validation(word_to_occurrence_dict=word_to_occurrence_dict, frequency_prob_dict=frequency_prob_dict,
               heldout_train_data=heldout_train_data, len_dictionary=len_dictionary,
               occurrences_word_dict_train=occurrences_word_dict_train, lidetone_train_data=lidestone_train_data,
               lamda=output_list[18])

    model_evaluation_test(listestone_counter_train, lidestone_train_data_len, unigram_model, frequency_prob_dict,
                          word_to_occurrence_dict, output_list)

    count_unique_train_data = len(set(heldout_train_data))
    table_output = "\n"
    for r in range(0, 10):
        table_output = table_output + "{}\t\t".format(r)
        t_r = get_t_r(t_r_cache, r)
        n_r = calc_n_r(occurrences_word_dict_train, count_unique_train_data, r, len_dictionary)
        table_output = table_output + "{}\t{}\t{}\t{}".format(
            round(lind_mle(output_list[18], r, lidestone_train_data_len, VOCABULARY_SIZE) * lidestone_train_data_len, 5),
            round(t_r / n_r, 5),
            n_r,
            t_r)
        if r != 10:
            table_output = table_output + "\n"
    output_list.append(table_output)

    output_string = output_list_to_string(output_list)
    print(output_string)
    with open(unigram_model.output_path(), "w+") as f:
        f.write(output_string)


if __name__ == "__main__":
    run(sys.argv[1:])
