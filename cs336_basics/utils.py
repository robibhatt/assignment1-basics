from collections import Counter
from collections.abc import Iterable
import regex as re
import random
import string


def random_string(length: int)->str:
    alphabet = string.ascii_letters + string.digits
    return ''.join(random.choice(alphabet) for _ in range(length))


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


def create_special_token_string(special_tokens:Iterable[str],
                                include_specials:bool)->str:
    special_token_strings = sorted([re.escape(spec) for spec in special_tokens], 
                                       key=len, 
                                       reverse=True)
    
    if include_specials:
        return '(' + "|".join(special_token_strings) + ')'
    else:
        return "|".join(special_token_strings)
