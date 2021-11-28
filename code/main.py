import sys
from code.unigram import Unigram
from code.consts import *
import code.preprocessing as preprocessing_utils
import code.lidetone_model_training as lidetone_model_training

### TODO change main.py file name to ex2.py
# # calc prod and lind for dataset
# from collections import Counter
# #
# #
# def calc_prob_unigram(data):
#     prob = {}
#     data_len = len(data)
#     counter = Counter(data)
#     for item in counter.items:
#         prob[item[0]] = item[1] / data_len
#     return prob
#
# #
# def calc_prob_lind(data, lamda):
#     prob = {}
#     data_len = len(data)
#     counter = Counter(data)
#     vocab_size = count_unique_events(data)
#     for item in counter.items():
#         lind = lind_mle(lamda=lamda, w_in_events=item[1], num_of_events=data_len, vocab_size=vocab_size)
#         prob[item[0]] = lind
#     return prob
#
#
# # calc preplexity and helper methods
# import math
#
#
# # TODO- ask what log base should be used?
# def calc_log_event(event, prob_dict, prob_unseen):
#     if event not in prob_dict.keys():
#         return prob_unseen
#     return math.log2(prob_dict[event])
#
#
# def calc_log_sum(data, prob_dict, prob_unseen):
#     sum = 0
#     for event in data:
#         sum = sum + calc_log_event(event=event, prob_dict=prob_dict, prob_unseen=prob_unseen)
#     return sum
#
#
# def calc_preplexity(data, prob_dict, prob_unseen):
#     power = -1 * calc_log_sum(data=data, prob_dict=prob_dict, prob_unseen=prob_unseen) / (len(data))
#     base = 2
#     result = math.pow(base, power)
#     return result
#
#
# # calc preplexity for validation set
# def preplexity(valid_data, train_data, lamda):
#     prob_dict = calc_prob_lind(data=train_data, lamda=lamda)
#     prob_unseen = lind_mle(lamda=lamda, w_in_events=0, num_of_events=len(train_data)
#                            , vocab_size=count_unique_events(train_data))
#     prepl = calc_preplexity(data=valid_data, prob_dict=prob_dict, prob_unseen=prob_unseen)
#     return prepl
#
#
# # NOT TESTED YET!
# Output16 = preplexity(valid_data=val_dev, train_data=train_dev, lamda=0.01)
# Output17 = preplexity(valid_data=val_dev, train_data=train_dev, lamda=0.1)
# Output18 = preplexity(valid_data=val_dev, train_data=train_dev, lamda=1.0)
# # print(Output16, Output17, Output18)

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


def preprocessing(unigram_model, output_list):
    events = preprocessing_utils.events_in_file(unigram_model.develop_path())
    output_list.append(len(events))
    return events


def lidetone(unigram_model, events, output_list):
    train_dev, validation_dev = lidetone_model_training.split_train_validation(events, LIDESTONE_SPLIT_RATE)
    validation_dev_len = len(validation_dev)
    train_dev_len = len(train_dev)

    output_list.append(validation_dev_len)
    output_list.append(train_dev_len)

    observed_vocab_size = lidetone_model_training.count_unique_events(train_dev)
    output_list.append(observed_vocab_size)

    number_of_occurences_for_input = lidetone_model_training.count_event_in_events(unigram_model.input_word, train_dev)

    output_list.append(number_of_occurences_for_input)
    output_list.append(number_of_occurences_for_input / train_dev_len)

    number_of_occurences_for_unseen_input = 0
    output_list.append(number_of_occurences_for_unseen_input / train_dev_len)

    # TODO: check if vocabulary size is needed to be the obsereved or the global
    output_list.append(
        lidetone_model_training.lind_mle(0.1, number_of_occurences_for_input, train_dev_len, VOCABULARY_SIZE))

    output_list.append(
        lidetone_model_training.lind_mle(0.1, number_of_occurences_for_unseen_input, train_dev_len, VOCABULARY_SIZE))

    output_list.append(lidetone_model_training.preplexity(valid_data=validation_dev, train_data=train_dev, lamda=0.01))

    output_list.append(lidetone_model_training.preplexity(valid_data=validation_dev, train_data=train_dev, lamda=0.1))

    output_list.append(lidetone_model_training.preplexity(valid_data=validation_dev, train_data=train_dev, lamda=1))



def run(arguments):
    output_list = []
    unigram_model = initator(arguments, output_list)
    if unigram_model is None:
        print("Input us incorrect!")
        return None
    events = preprocessing(unigram_model, output_list)
    lidetone(unigram_model, events, output_list)
    print(output_list)
    # TODO imeplemnt function that iterates output and writes to file in requested format OutputX: Y


if __name__ == "__main__":
    run(sys.argv[1:])
