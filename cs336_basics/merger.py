from collections import Counter
from cs336_basics.utils import print_counter
from cs336_basics.counter_heap import CounterHeap


class BytesNode:

    def __init__(self, bytes_value: bytes, rep_factor=0):
        self.bytes_value = bytes_value
        self.prev = None
        self.nxt = None
        self.rep_factor = rep_factor


def get_nodes(
    tokenized_pretokens:dict[str, list[bytes]],
    pretokenized_counts:Counter[str]=None
) -> dict[str, BytesNode]:
    """
    For each pretoken, converts a list of bytes into a double linked list of nodes.
    The pretokenized_counts are used to allow each node to know its own weight. 
    """
    
    if pretokenized_counts is None:
        pretokenized_counts = Counter()
        for pretoken in tokenized_pretokens:
            pretokenized_counts[pretoken] = None
    
    pretoken_nodes = {}
    for pretoken in tokenized_pretokens:
        prev_node = BytesNode(bytes_value=None, rep_factor=pretokenized_counts[pretoken])
        pretoken_nodes[pretoken] = prev_node
        for b in tokenized_pretokens[pretoken]:
            node = BytesNode(bytes_value=b, rep_factor=pretokenized_counts[pretoken])
            prev_node.nxt = node
            node.prev = prev_node
            prev_node = node

    return pretoken_nodes


def get_bytes_nodes(pretoken_nodes:dict[str, BytesNode]) -> dict[bytes, set[BytesNode]]:
    bytes_nodes = {}
    for pretoken in pretoken_nodes:
        node = pretoken_nodes[pretoken]
        while node.nxt is not None:
            node = node.nxt
            b = node.bytes_value
            if b not in bytes_nodes:
                bytes_nodes[b] = set([])
            bytes_nodes[b].add(node)
    return bytes_nodes


def from_nodes(
    pretoken_nodes:dict[str, BytesNode]
)-> dict[str, list[bytes]]:
    
    tokenized_pretokens = {}
    for pretoken in pretoken_nodes:
        tokenized_pretokens[pretoken] = []
        node = pretoken_nodes[pretoken]
        while node is not None:
            if node.bytes_value is not None:
                tokenized_pretokens[pretoken].append(node.bytes_value)
            node = node.nxt

    return tokenized_pretokens


def initialize_pair_counter(pretoken_nodes:dict[str, BytesNode])->Counter[tuple[bytes, bytes]]:
    pair_counter = Counter()
    for pretoken in pretoken_nodes:
        current_bytes = None
        node = pretoken_nodes[pretoken]
        node = node.nxt
        if node is not None:
            while node.nxt is not None:
                next_node = node.nxt
                pair_counter[(node.bytes_value, next_node.bytes_value)] += node.rep_factor
                node = next_node
    return pair_counter


def smart_step(
    bytes_nodes:dict[bytes, set[BytesNode]],
    current_tokens:list[bytes],
    best_pair_heap:CounterHeap,
    merges:list[tuple[bytes, bytes]]
) -> None:
    
    best_pair = best_pair_heap.get_merge()
    if best_pair is None:
        return None

    (first_bytes, second_bytes) = best_pair
    new_token_bytes = first_bytes + second_bytes
    current_tokens.append(new_token_bytes)

    merges.append((first_bytes, second_bytes))

    del_set = set([])

    if first_bytes != second_bytes:
        # modify all of the linked lists
        for first_bytes_node in bytes_nodes[first_bytes]:
            succ = first_bytes_node.nxt
            if succ is not None and succ.bytes_value == second_bytes:
                new_node = BytesNode(bytes_value=new_token_bytes, rep_factor=first_bytes_node.rep_factor)
                if new_token_bytes not in bytes_nodes:
                    bytes_nodes[new_token_bytes] = set([])
                bytes_nodes[new_token_bytes].add(new_node)
                assert(new_node is not None)
                assert(new_node.rep_factor is not None)
                new_node.nxt = succ.nxt
                if succ.nxt is not None:
                    succ.nxt.prev = new_node
                new_node.prev = first_bytes_node.prev
                assert new_node.prev is not None
                new_node.prev.nxt = new_node

                del_set.add(first_bytes_node)
                del_set.add(succ)

        for node in del_set:
            bytes_nodes[node.bytes_value].remove(node)

    else:
        new_first_bytes_set = set([])
        while len(bytes_nodes[first_bytes]) > 0:
            node = next(iter(bytes_nodes[first_bytes]))
            assert(node is not None)
            while node.prev.bytes_value == node.bytes_value:
                node = node.prev
            assert(node is not None)
            while (node is not None) and (node.nxt is not None) and (node.bytes_value == first_bytes) and (node.nxt.bytes_value == first_bytes):

                # create the new node
                new_node = BytesNode(bytes_value=new_token_bytes, rep_factor=node.nxt.rep_factor)
                if new_token_bytes not in bytes_nodes:
                    bytes_nodes[new_token_bytes] = set([])
                bytes_nodes[new_token_bytes].add(new_node)
                assert(new_node is not None)

                # add its prev link
                new_node.prev = node.prev
                node.prev.nxt = new_node

                # add its nxt link
                new_node.nxt = node.nxt.nxt
                if node.nxt.nxt is not None:
                    node.nxt.nxt.prev = new_node

                # remove the node and its successor from our set of nodes to check
                bytes_nodes[first_bytes].remove(node)
                bytes_nodes[first_bytes].remove(node.nxt)
                del_set.add(node)
                del_set.add(node.nxt)

                # on to the next node worth considering
                node = new_node.nxt
            if node is not None and node.bytes_value == first_bytes:
                bytes_nodes[first_bytes].remove(node)
                new_first_bytes_set.add(node)
                assert(node is not None)
        bytes_nodes[first_bytes] = new_first_bytes_set

    # delete the removed nodes
    del_set.clear()
    del del_set

    # time to handle the pair counter
    touched_nodes = set([])
    pair_counter_delta = Counter()

    for bytes_value in [first_bytes, second_bytes, new_token_bytes]:
        for node in bytes_nodes[bytes_value]:
            prev_node = node.prev
            nxt_node = node.nxt
            if prev_node.bytes_value is not None:
                if prev_node not in touched_nodes:
                    pair_counter_delta[(prev_node.bytes_value, node.bytes_value)] += node.rep_factor
                    touched_nodes.add(prev_node)
            if nxt_node is not None:
                if node not in touched_nodes:
                    pair_counter_delta[(node.bytes_value,nxt_node.bytes_value)] += node.rep_factor
                    touched_nodes.add(node)

    for bytes_value in [first_bytes, second_bytes, new_token_bytes]:
        for current_token in current_tokens:
            pair_counter_delta[(bytes_value, current_token)] += 0
            pair_counter_delta[(current_token, bytes_value)] += 0

    best_pair_heap.modify_counts(new_counts=pair_counter_delta)


def get_tokens_and_merges(
    pretokenized_counts: Counter[str],
    number_of_merges: int
)->tuple[list[bytes], list[tuple[bytes, bytes]]]:

    # setting up the tokens that we will return
    current_tokens = [bytes([b]) for b in range(256)]

    # taking each word and representing as a list of bytes. these will change over time
    tokenized_pretokens = {}
    for pretoken in pretokenized_counts:
        tokenized_pretokens[pretoken] = [bytes([b]) for b in pretoken.encode('utf-8')]

    # create nodes that correspond to the smart tokenized pretokens
    pretoken_nodes = get_nodes(tokenized_pretokens=tokenized_pretokens, 
                               pretokenized_counts=pretokenized_counts)
    
    # creates a dictionary that tells which nodes correspond to each bytes_value
    bytes_nodes = get_bytes_nodes(pretoken_nodes=pretoken_nodes)

    # intiailizing a count over pairs of bytes frequencies
    pair_counter = initialize_pair_counter(pretoken_nodes=pretoken_nodes)

    # turning the pair_counter into a heap
    best_pair_heap = CounterHeap(pair_counter)

    # setting up merges
    merges = []

    for i in range(number_of_merges):
        smart_step(
            bytes_nodes=bytes_nodes,
            current_tokens=current_tokens,
            best_pair_heap=best_pair_heap,
            merges=merges
        )

    return (current_tokens, merges)
