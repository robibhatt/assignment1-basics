from collections import Counter
import heapq


def get_heap_format(
    pair_count: int,
    pair: tuple[bytes, bytes]    
)-> tuple[int, tuple[bytes, bytes]]:
    """
    A useful conversion so that get_heap_format(a) < get_heap_format(b) iff b > a
    """
    bytes_list0 = list(pair[0])
    new_list0 = [mb for b in bytes_list0[:-1] for mb in (255-b, 0)]
    new_list0.append(255-bytes_list0[-1])
    new_list0.append(1)
    bytes_list1 = list(pair[1])
    new_list1 = [mb for b in bytes_list1[:-1] for mb in (255-b, 0)]
    new_list1.append(255-bytes_list1[-1])
    new_list1.append(1)
    return(-pair_count, (bytes(new_list0), bytes(new_list1)))


def get_count_and_pair(heap_format: tuple[int, tuple[bytes, bytes]]) -> tuple[int, tuple[bytes, bytes]]:
    """
    inverts the previous operation
    """
    neg_count, pair = heap_format
    first_bytes, second_bytes = pair
    return (-neg_count, (bytes([255-b for b in list(first_bytes)[::2]]), bytes([255-b for b in list(second_bytes)[::2]])))


def create_max_heap(pair_counter: Counter[tuple[bytes, bytes]])-> list[tuple[int, tuple[bytes, bytes]]]:
    """
    makes a max heap out of the pair_counter
    """
    max_heap = [get_heap_format(pair_count = pair_counter[pair], pair=pair) for pair in pair_counter]
    heapq.heapify(max_heap)
    return max_heap


class CounterHeap:

    def __init__(self, pair_counter: Counter[tuple[bytes, bytes]]):
        self.pair_counter = pair_counter
        max_heap = [get_heap_format(pair_count = pair_counter[pair], pair=pair) for pair in pair_counter]
        heapq.heapify(max_heap)
        self.max_heap = max_heap

    def get_merge(self)-> tuple[bytes, bytes]:
        (count, pair) = get_count_and_pair(self.max_heap[0])
        while self.pair_counter[pair] != count:
            heapq.heappop(self.max_heap)
            if len(self.max_heap)>0:
                (count, pair) = get_count_and_pair(self.max_heap[0])
            else:
                return None
        if count == 0:
            return None
        return pair
    
    def modify_counts(self, new_counts: Counter[tuple[bytes, bytes]]):
        for pair in new_counts:
            new_count = new_counts[pair]
            if self.pair_counter[pair] != new_count:
                self.pair_counter[pair] = new_count
                heap_format = get_heap_format(pair_count=new_count, pair=pair)
                heapq.heappush(self.max_heap, heap_format)

    def update_counts(self, counts_delta: Counter[tuple[bytes, bytes]]):
        for pair in counts_delta:
            delta = counts_delta[pair]
            if delta != 0:
                new_count = self.pair_counter[pair] + delta
                self.pair_counter[pair] = new_count
                heap_format = get_heap_format(pair_count=new_count, pair=pair)
                heapq.heappush(self.max_heap, heap_format)