'''
Created on Sep 23, 2015

@author: Philip Schulz
@modifications: Jakub Dotlacil, April 26, 2018
'''

import sys
import argparse
from datetime import datetime
# TODO: replace <package> by the name of the package that you store these files in
from naive_bayes_solution import NaiveBayes
from os import listdir, remove, system
from os.path import isfile, join

# TODO: Please the command that you use to call Python in the terminal here
# (most likely, the command is python or python3)
my_python = ""

def train_model(corpus_dir, classifier):
    '''Train a classifier on a training corpus where labels are provided.

    :param corpus_dir: The path to the training folder
    :param classifier: The classifier to be trained
    '''
    print('Starting training at {}'.format(datetime.now()))

    for directory in listdir(corpus_dir):
        print("Training on label {}".format(directory))
        directory_path = join(corpus_dir, directory)
        for text_file in listdir(directory_path):
            file_path = join(directory_path, text_file)
            classifier.update_label_count(directory)
            try:
                with open(file_path) as data_file:
                    classifier.train(data_file, directory)
            except IOError as e:
                print(e)
                print("It seems that the text_file {} is damaged.".format(text_file))
                sys.exit(0)

    print("Starting to smooth and normalise at {}".format(datetime.now()))
    classifier.smooth_feature_counts()
    classifier.log_normalise_label_probs()
    classifier.log_normalise_feature_probs()

    print("Finished training at {}".format(datetime.now()))

def make_predictions(predictions_file, test_dir, classifier):
    '''Make predictions on data with missing label

    :param predictions_file: The file to which the ouput predictions should be written
    :param test_dir: The path to the directory containing the test items
    :param classifier: A trained classifier
    '''
    print("Start making predictions at {}".format(datetime.now()))
    
    if isfile(predictions_file):
        remove(predictions_file)
    
    for test_file in listdir(test_dir):
        try:
            with open(join(test_dir, test_file)) as test, open(predictions_file, "a") as out:
                prediction = classifier.predict(test)
                out.write(test_file + "\t{}\n".format(prediction))
        except IOError as e:
            print(e)
            print("Something went wrong while reading test file {}".format(test_file))
            sys.exit(0)
            
    print("Finished making predictions at {}".format(datetime.now()))

def main():
    '''Standard method in Python that does not need a docstring. Don't worry about it for now, we will get to know it
    more deeply in week 6.
    '''
    
    commandline_parser = argparse.ArgumentParser("Add description.")
    
    commandline_parser.add_argument("--training-corpus-directory", required = True, help="Specifies the path to the directory where the corpus is stored.")
    commandline_parser.add_argument("--test-set-directory", required = True, help="Specifies the path to the directory where the test files are stored.")
    commandline_parser.add_argument("--keys", help="Specifies the path to the file holding the true labels for the test-set.")

    args = vars(commandline_parser.parse_args())
    corpus_dir = args["training_corpus_directory"]
    test_dir = args["test_set_directory"]
    keys_file = args["keys"]

    nb_classifier = NaiveBayes()

    if corpus_dir:
        train_model(corpus_dir, nb_classifier)

    if test_dir:
        make_predictions("predictions.txt", test_dir, nb_classifier)

    if keys_file:
        system("{} accuracy_checker.py {} predictions.txt".format(my_python, keys_file))

if __name__ == '__main__':
    main()
    
