from code.consts import *
import code.file_utils as file_utils


def is_input_valid(arguemnts):
    if len(arguemnts) != 4:
        return False
    develop_file = arguemnts[0]
    test_file = arguemnts[1]
    output_file = arguemnts[3]
    if file_utils.is_file_exists(DIRECTORY_PATH, develop_file) and file_utils.is_file_exists(DIRECTORY_PATH,
                                                                                             test_file) and not file_utils.is_file_exists(
        DIRECTORY_PATH, output_file):
        return True
    return False
