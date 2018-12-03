import random
import numpy
from timeit import default_timer as timer
import willump.evaluation.willump_executor

g_vocab_dict = {}


def willump_frequency_count(input_words, vocab_dict):
    output = numpy.zeros(len(vocab_dict), dtype=numpy.int64)
    for word in input_words:
        if word in vocab_dict:
            index = vocab_dict[word]
            output[index] += 1
    return output


@willump.evaluation.willump_executor.willump_execute
def split_string(input_string):
    output_strings = input_string.split()
    for i in range(len(output_strings)):
        output_strings[i] = output_strings[i].lower()
        output_strings[i] = output_strings[i].replace(".", "")
        output_strings[i] = output_strings[i].replace(",", "")
        # output_strings[i] = output_strings[i].replace("\'", "")
        # output_strings[i] = output_strings[i].replace("!", "")
        # output_strings[i] = output_strings[i].replace("\"", "")
        # output_strings[i] = output_strings[i].replace("#", "")
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
        # output_strings[i] = output_strings[i].replace("@", "")
        # output_strings[i] = output_strings[i].replace("[", "")
        # output_strings[i] = output_strings[i].replace("\\", "")
        # output_strings[i] = output_strings[i].replace("]", "")
        # output_strings[i] = output_strings[i].replace("^", "")
        # output_strings[i] = output_strings[i].replace("_", "")
        # output_strings[i] = output_strings[i].replace("{", "")
        # output_strings[i] = output_strings[i].replace("|", "")
        # output_strings[i] = output_strings[i].replace("}", "")
        # output_strings[i] = output_strings[i].replace("~", "")
    output_strings = willump_frequency_count(output_strings, g_vocab_dict)
    return output_strings


def main():
    global g_vocab_dict
    with open("tests/test_resources/top_1k_english_words.txt") as vocab_file:
        for i, line in enumerate(vocab_file.read().splitlines()):
            g_vocab_dict[line] = i
    split_string("a b c")
    split_string("a b c")
    split_string("a b c")
    with open("tests/test_resources/hamlet.txt") as hamlet_file:
        big_string = hamlet_file.read()
    start = timer()
    freq_vec = split_string(big_string)
    end = timer()
    print(freq_vec)
    print("Splitting time: {0}".format(end - start))


if __name__ == '__main__':
    random.seed(0)
    main()
