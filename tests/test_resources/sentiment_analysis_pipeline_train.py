import numpy
from timeit import default_timer as timer
import willump.evaluation.willump_executor
import csv
import sklearn.linear_model
import pickle

g_vocab_dict = {}


def willump_frequency_count(input_words, vocab_dict):
    output = numpy.zeros((1, len(vocab_dict)), dtype=numpy.int64)
    for word in input_words:
        if word in vocab_dict:
            index = vocab_dict[word]
            output[0, index] += 1
    return output


@willump.evaluation.willump_executor.willump_execute
def process_string(input_string, time_of_day):
    output_strings = input_string.split()
    for i in range(len(output_strings)):
        output_strings[i] = output_strings[i].lower()
        output_strings[i] = output_strings[i].replace(".", "")
        output_strings[i] = output_strings[i].replace(",", "")
        output_strings[i] = output_strings[i].replace("#", "")
        # output_strings[i] = output_strings[i].replace("@", "")
        # output_strings[i] = output_strings[i].replace("\'", "")
        # output_strings[i] = output_strings[i].replace("!", "")
        # output_strings[i] = output_strings[i].replace("\"", "")
        # output_strings[i] = output_strings[i].replace("$", "")
        # output_strings[i] = output_strings[i].replace("%", "")
        # output_strings[i] = output_strings[i].replace("&", "")
        # output_strings[i] = output_strings[i].replace("/", "")
        # output_strings[i] = output_strings[i].replace("`", "")
        # output_strings[i] = output_strings[i].replace("(", "")
        # output_strings[i] = output_strings[i].replace(")", "")
        # output_strings[i] = output_strings[i].replace("*", "")
        # output_strings[i] = output_strings[i].replace("+", "")
        # output_strings[i] = output_strings[i].replace("-", "")
        # output_strings[i] = output_strings[i].replace("/", "")
        # output_strings[i] = output_strings[i].replace(":", "")
        # output_strings[i] = output_strings[i].replace(";", "")
        # output_strings[i] = output_strings[i].replace("<", "")
        # output_strings[i] = output_strings[i].replace("=", "")
        # output_strings[i] = output_strings[i].replace(">", "")
        # output_strings[i] = output_strings[i].replace("?", "")
        # output_strings[i] = output_strings[i].replace("[", "")
        # output_strings[i] = output_strings[i].replace("\\", "")
        # output_strings[i] = output_strings[i].replace("]", "")
        # output_strings[i] = output_strings[i].replace("^", "")
        # output_strings[i] = output_strings[i].replace("_", "")
        # output_strings[i] = output_strings[i].replace("{", "")
        # output_strings[i] = output_strings[i].replace("|", "")
        # output_strings[i] = output_strings[i].replace("}", "")
        # output_strings[i] = output_strings[i].replace("~", "")
    output_vec = willump_frequency_count(output_strings, g_vocab_dict)
    output_vec = numpy.append(output_vec, time_of_day)
    return output_vec


def main():
    global g_vocab_dict
    with open("tests/test_resources/top_1k_english_words.txt", "r") as vocab_file:
        for i, line in enumerate(vocab_file.read().splitlines()):
            g_vocab_dict[line] = i
    tweet_labels = []
    processed_inputs = []
    process_string("a b c", 5)
    process_string("a b c", 5)
    process_string("a b c", 5)
    dataset_nlines = 0
    with open("tests/test_resources/twitter.200000.processed.noemoticon.csv", "r", encoding="latin-1") as csvfile:
        csvreader = csv.reader(csvfile)
        start = timer()
        for row in csvreader:
            processed_input = process_string(row[5], int(row[2][11:13]))
            processed_inputs.append(processed_input)
            tweet_labels.append(int(row[0]))
            dataset_nlines += 1
        end = timer()
    print(processed_inputs[0])
    print("Total processing time: {0}".format(end - start))
    print("Processing latency: {0}".format((end - start) / dataset_nlines))
    model = sklearn.linear_model.LogisticRegression()
    model = model.fit(processed_inputs, tweet_labels)
    pickle.dump(model, open("tests/test_resources/sa_model.pk", 'wb'))


if __name__ == '__main__':
    main()
