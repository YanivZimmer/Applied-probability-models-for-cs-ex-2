import sys
from code.unigram import Unigram
from code.consts import *
import code.preprocessing as preprocessing_utils
import code.lidetone_model_training as lidetone_model_training
import code.held_out_training as held_out_training


### TODO change main.py file name to ex2.py

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

    number_of_occurences_for_unseen_input = lidetone_model_training.count_event_in_events(UNSEEN_WORD, train_dev)
    output_list.append(number_of_occurences_for_unseen_input / train_dev_len)

    output_list.append(
        lidetone_model_training.lind_mle(0.1, number_of_occurences_for_input, train_dev_len, VOCABULARY_SIZE))

    output_list.append(
        lidetone_model_training.lind_mle(0.1, number_of_occurences_for_unseen_input, train_dev_len, VOCABULARY_SIZE))

    output_list.append(lidetone_model_training.preplexity(valid_data=validation_dev, train_data=train_dev, lamda=0.01))

    output_list.append(lidetone_model_training.preplexity(valid_data=validation_dev, train_data=train_dev, lamda=0.1))

    output_list.append(lidetone_model_training.preplexity(valid_data=validation_dev, train_data=train_dev, lamda=1))

    optimal_lamda, min_prep = lidetone_model_training.find_optimal_lamda(valid_data=validation_dev,
                                                                         train_data=train_dev)
    output_list.append(optimal_lamda)
    output_list.append(min_prep)


def held_out(unigram_model, events, output_list):
    train_data, test_data = held_out_training.split_held_outtrain_validation(events=events)

    train_data_len = len(train_data)
    test_data_len = len(test_data)
    output_list.append(train_data_len)
    output_list.append(test_data_len)


def run(arguments):
    output_list = []
    unigram_model = initator(arguments, output_list)
    if unigram_model is None:
        print("Input us incorrect!")
        return None
    events = preprocessing(unigram_model, output_list)
    lidetone(unigram_model, events, output_list)
    held_out(unigram_model, events, output_list)

    print(output_list)
    # TODO imeplemnt function that iterates output and writes to file in requested format OutputX: Y


if __name__ == "__main__":
    run(sys.argv[1:])
