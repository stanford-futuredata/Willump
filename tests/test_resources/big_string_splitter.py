import string
import random
from timeit import default_timer as timer
import willump.evaluation.willump_executor


@willump.evaluation.willump_executor.willump_execute
def split_string(input_string):
    output_strings = input_string.split()
    # for i in range(len(output_strings)):
    #     output_strings[i] = output_strings[i].lower()
    return output_strings


def main():
    split_string("a b c")
    split_string("a b c")
    split_string("a b c")
    big_string = ""

    for _ in range(1000000):
        if random.randint(0, 3) == 0:
            big_string += " "
        else:
            big_string += random.choice(string.ascii_letters)
    start = timer()
    word_list = split_string(big_string)
    end = timer()
    print("Splitting time: {0}".format(end - start))


if __name__ == '__main__':
    random.seed(0)
    main()
