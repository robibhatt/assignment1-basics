from cs336_basics.counter_heap import get_heap_format, get_count_and_pair, create_max_heap
from cs336_basics.utils import random_string
import random


def test_heap_format():
    strings = [random_string(20) for i in range(3)]
    strings.append('aaaa')
    strings.append('aaa')
    byte_strings = [s.encode('utf-8') for s in strings]
    counts = [random.randint(1, 100) for i in range(4)]

    for count_0 in counts:
        for count_1 in counts:
            for byte_a in byte_strings:
                for byte_b in byte_strings:
                    for byte_c in byte_strings:
                        for byte_d in byte_strings:
                            first_guy = (count_0, (byte_a, byte_b))
                            second_guy = (count_1, (byte_c, byte_d))
                            fh = get_heap_format(*first_guy)
                            sh = get_heap_format(*second_guy)
                            if first_guy == second_guy:
                                assert(fh == sh)
                            elif first_guy < second_guy:
                                assert(fh > sh)
                            else:
                                assert(fh < sh)
                            first = get_count_and_pair(fh)
                            second = get_count_and_pair(sh)
                            assert(first == first_guy)
                            assert(second == second_guy)