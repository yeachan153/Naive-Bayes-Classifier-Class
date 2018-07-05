from collections import Counter
import math
import copy
import operator


class NaiveBayes(object):
    def __init__(self):
        # Per class (e.g. alt.atheism), how many text files are from the class?
        self.label_counts = Counter()

        # The key is the class, the value is a counter of different words.
        self.feature_counts = dict()

        # The following dictionary will be used to collect
        # prior probabilities of the classes.
        self.label_probs = dict()

        # The following dictionary will be used to collect word
        # probabilities given a class.
        self.feature_probs = dict()

        # A set that contains all words encountered during training.
        self.vocabulary = set()

    def train(self, data, label):
        '''
        Train the classifier by counting features in the data set.

        :param data: A stream of string data from which to extract features
        :param label: The label of the data
        '''
        for line in data:
            self.add_feature_counts(line.split(), label)

    def add_feature_counts(self, features, label):
        '''
        Count the features in a feature list.

        :param features: a list of words.
        :param label: the class of the data file from which the features were extracted.
        '''

        # This method updates feature_counts by features given the class. It
        # should also update vocabulary with features.
        if label not in self.feature_counts:
            self.feature_counts[label] = Counter(features)
        elif label in self.feature_counts:
            self.feature_counts[label].update(features)

        # Vocabulary Update
        self.vocabulary.update(features)

    def smooth_feature_counts(self, smoothing=1):
        '''Smooth the collected feature counts

        :param smoothing: The smoothing constant
        '''
        for each_class in self.feature_counts:
            for each_word in self.vocabulary:
                self.feature_counts[each_class].update([each_word] * smoothing)

    def update_label_count(self, label):
        '''
        Increase the count for the supplied class by 1.

        :param label: The class whose count is to be increased.
        '''
        self.label_counts.update([label])

    def log_normalise_label_probs(self):
        '''
        Take label counts in label_counts (how many files are inside class A),
        normalize them to probabilities, transform them to logprobs and update label_probs
        with the logprobs.
        '''
        total_counts = sum(self.label_counts.values())
        copy_label_counts = self.label_counts.copy()
        for key in copy_label_counts:
            copy_label_counts[key] = math.log(copy_label_counts[key] / total_counts)
        self.label_probs.update(copy_label_counts)

    def log_normalise_feature_probs(self):
        '''
        Take feature counts in feature_counts and for each label, normalize
        them to probabilities and turn them into logprobs. update
        feature_probs with the created logprobs.
        '''
        total_count_list = [sum(self.feature_counts[each_key].values()) for each_key in self.feature_counts]
        # Deep copy necessary to copy the counter structure inside dictionary
        feature_copy = copy.deepcopy(self.feature_counts)
        for idx, each_key in enumerate(feature_copy):
            for each_value in feature_copy[each_key]:
                feature_copy[each_key][each_value] = math.log(
                    feature_copy[each_key][each_value] / total_count_list[idx])
        self.feature_probs.update(feature_copy)

    def predict(self, data):
        '''
        Predict the most probable label according to the model on a stream of data.

        :param data: A stream of string data from which to extract features
        :return: the most probable label for the data (type string)
        '''
        data = data.read()
        data = data.split()
        cleaned_data = [each_word for each_word in data if each_word in self.vocabulary]

        labels = [each for each in self.label_probs]

        list1 = [self.feature_probs[each_key][each_word] for each_key in self.feature_probs for each_word in
                 cleaned_data]
        chunks = [list1[x:x + len(cleaned_data)] for x in range(0, len(list1), len(cleaned_data))]

        probability_dict = {}
        for idx, each_key in enumerate(self.feature_probs):
            for each_word in cleaned_data:
                for each_label in labels:
                    if each_key not in probability_dict:
                        probability_dict[each_key] = sum(chunks[idx]) + self.label_probs[each_label]

        return max(probability_dict.items(), key=operator.itemgetter(1))[0]