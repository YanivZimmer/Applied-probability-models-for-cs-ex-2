import file_utils as file_utils
import os.path


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
        if file_utils.is_file_exists(self.directory_path, self.develop_file) and file_utils.is_file_exists(
                self.directory_path, self.test_file) and not file_utils.is_file_exists(self.directory_path,
                                                                                       self.output_file):
            return True
        return False

    def develop_path(self):
        return os.path.join(self.directory_path, self.develop_file)

    def test_path(self):
        return os.path.join(self.directory_path, self.test_file)
    
    def output_path(self):
        return os.path.join(self.directory_path, self.output_file)
