import random
from timeit import default_timer as timer
import willump.evaluation.willump_executor


@willump.evaluation.willump_executor.willump_execute
def split_string(input_string):
    output_strings = input_string.split()
    for i in range(len(output_strings)):
        output_strings[i] = output_strings[i].lower()
    return output_strings


def main():
    split_string("a b c")
    split_string("a b c")
    split_string("a b c")
    with open("tests/test_resources/hamlet.txt") as hamlet_file:
        big_string = hamlet_file.read()
    start = timer()
    word_list = split_string(big_string)
    end = timer()
    print("Splitting time: {0}".format(end - start))


if __name__ == '__main__':
    random.seed(0)
    main()
