from collections import Counter


def print_counter(counter: Counter[str]):
    for key in counter.keys():
        print(key, counter[key])
    

def print_counters(counter_1: Counter[str], counter_2: Counter[str]):
    keys = set(counter_1.keys())
    keys2 = set(counter_2.keys())
    keys = keys.union(keys2)
    for key in keys:
        a = 0
        b = 0
        if key in counter_1:
            a = counter_1[key]
        if key in counter_2:
            b = counter_2[key]
        print(key, a, b)


def compare_counters(counter_1: Counter[str], counter_2: Counter[str]):
    keys = set(counter_1.keys())
    keys2 = set(counter_2.keys())
    keys = keys.union(keys2)
    for key in keys:
        a = 0
        b = 0
        if key in counter_1:
            a = counter_1[key]
        if key in counter_2:
            b = counter_2[key]
        if a == b:
            print(key, a, b)
        else:
            assert(False)
    print(counter_1 == counter_2)