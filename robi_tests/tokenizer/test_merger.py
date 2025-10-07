from collections import Counter
from cs336_basics.utils import random_string
import random
from cs336_basics.tokenizer.counter_heap import CounterHeap
from cs336_basics.tokenizer.merger import BpeMerger, BytesNode


def initialize_pair_counter(pretoken_nodes:dict[str, BytesNode])->Counter[tuple[bytes, bytes]]:
    pair_counter = Counter()
    for pretoken in pretoken_nodes:
        node = pretoken_nodes[pretoken]
        node = node.nxt
        if node is not None:
            while node.nxt is not None:
                next_node = node.nxt
                pair_counter[(node.bytes_value, next_node.bytes_value)] += node.rep_factor
                node = next_node
    return pair_counter


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


def test_nodes():
    pretokenized_counts = Counter()
    pretokenized_counts['eat'] = 6
    pretokenized_counts['stale'] = 3
    pretokenized_counts['fleat'] = 7
    pretokenized_counts['farts'] = 7
    pretokenized_counts['attack'] = 7
    pretokenized_counts['aaaaaaaaa'] = 700

    tokenized_pretokens = {}
    current_tokens = [bytes([b]) for b in range(256)]
    for pretoken in pretokenized_counts:
        tokenized_pretokens[pretoken] = [bytes([b]) for b in pretoken.encode('utf-8')]
                                         
    pretoken_nodes = get_nodes(tokenized_pretokens)
    new_tokenized_pretokens = from_nodes(pretoken_nodes)
    assert(tokenized_pretokens == new_tokenized_pretokens)


def brute_force_step(
    pretokenized_counts:Counter[str], 
    tokenized_pretokens:dict[str, list[bytes]],
    current_tokens:list[bytes],
    verbose=False
) -> None:
    
    pair_counter = Counter()
    for pretoken in pretokenized_counts:
        current_bytes = None
        for next_bytes in tokenized_pretokens[pretoken]:
            if current_bytes is not None:
                pair_counter[(current_bytes, next_bytes)] += pretokenized_counts[pretoken]
            current_bytes = next_bytes

    
    best_pair = None
    for pair in pair_counter:
        if verbose:
            if best_pair is None:
                print(None, [b for b in pair[0]], [b for b in pair[1]])
                print(0, pair_counter[pair])
            else:
                print([b for b in best_pair[0]], [b for b in best_pair[1]], [b for b in pair[0]], [b for b in pair[1]])
                print(pair_counter[best_pair], pair_counter[pair])
        if best_pair is None:
            best_pair = pair 
        elif pair_counter[pair] > pair_counter[best_pair]:
            best_pair = pair
        elif pair_counter[pair] == pair_counter[best_pair] and pair > best_pair:
            best_pair = pair
        else:
            pass

    (first_bytes, second_bytes) = best_pair
    new_token_bytes = first_bytes + second_bytes
    current_tokens.append(new_token_bytes)

    for pretoken in tokenized_pretokens:
        new_list = []
        prev_bytes = None
        for b in tokenized_pretokens[pretoken]:
            if prev_bytes is None:
                prev_bytes = b
            else:
                if (prev_bytes + b) == new_token_bytes:
                    new_list.append(new_token_bytes)
                    prev_bytes = None
                else:
                    new_list.append(prev_bytes)
                    prev_bytes = b
        if prev_bytes is not None:
            new_list.append(prev_bytes)
        tokenized_pretokens[pretoken] = new_list


def test_brute_force_step():
    pretokenized_counts = Counter()
    pretokenized_counts['aaaaaaaaa'] = 700

    tokenized_pretokens = {}
    current_tokens = [bytes([b]) for b in range(256)]
    for pretoken in pretokenized_counts:
        tokenized_pretokens[pretoken] = [bytes([b]) for b in pretoken.encode('utf-8')]
    brute_force_step(
        pretokenized_counts,
        tokenized_pretokens,
        current_tokens
    )
    brute_force_step(
        pretokenized_counts,
        tokenized_pretokens,
        current_tokens,
    )
    brute_force_step(
        pretokenized_counts,
        tokenized_pretokens,
        current_tokens,
    )
    assert([[b for b in bytes] for bytes in current_tokens[256:]] == [[97, 97], [97, 97, 97, 97], [97, 97, 97, 97, 97, 97, 97, 97]])


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

def test_smart_step():

    # just a bunch of words withcounts
    pretokenized_counts = Counter()
    for i in range(10):
        word = random_string(12)
        pretokenized_counts[word] = random.randint(1, 10)

    # repeating the variable for the smart version
    smart_pretokenized_counts = pretokenized_counts

    # taking each word and representing as a list of bytes. these will change over time
    tokenized_pretokens = {}
    current_tokens = [bytes([b]) for b in range(256)]
    for pretoken in pretokenized_counts:
        tokenized_pretokens[pretoken] = [bytes([b]) for b in pretoken.encode('utf-8')]



    # doing the same thing a second time for the smart version
    smart_tokenized_pretokens = {}
    smart_current_tokens = [bytes([b]) for b in range(256)]
    for pretoken in smart_pretokenized_counts:
        smart_tokenized_pretokens[pretoken] = [bytes([b]) for b in pretoken.encode('utf-8')]

    # create nodes that correspond to the smart tokenized pretokens
    pretoken_nodes = get_nodes(tokenized_pretokens=smart_tokenized_pretokens, 
                               pretokenized_counts=pretokenized_counts)
    
    # creates a dictionary that tells which nodes correspond to each bytes_value
    bytes_nodes = get_bytes_nodes(pretoken_nodes=pretoken_nodes)

    # intiailizing a count over pairs of bytes frequencies
    pair_counter = initialize_pair_counter(pretoken_nodes=pretoken_nodes)

    # turning the pair_counter into a heap
    best_pair_heap = CounterHeap(pair_counter)

    # setting up merges, which we will return later
    merges = []

    for i in range(50):
        brute_force_step(
            pretokenized_counts,
            tokenized_pretokens,
            current_tokens
        )

        smart_step(
            merges=merges,
            bytes_nodes=bytes_nodes,
            current_tokens=smart_current_tokens,
            best_pair_heap=best_pair_heap
        )

        smart_tokenized_pretokens = from_nodes(pretoken_nodes)

        assert(pretokenized_counts == smart_pretokenized_counts)
        assert(tokenized_pretokens == smart_tokenized_pretokens)
        assert(current_tokens == smart_current_tokens)

    for pretoken in pretoken_nodes:
        node = pretoken_nodes[pretoken]
        while node.nxt is not None:
            node = node.nxt
            assert(node.rep_factor == pretokenized_counts[pretoken])


def get_tokens_and_merges_old(
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



def test_merger_class_merges():

    # setting up brute force variables

    # just a bunch of words withcounts
    pretokenized_counts = Counter()
    for i in range(10):
        word = random_string(12)
        pretokenized_counts[word] = random.randint(1, 10)

    # taking each word and representing as a list of bytes. these will change over time
    tokenized_pretokens = {}
    current_tokens = [bytes([b]) for b in range(256)]
    for pretoken in pretokenized_counts:
        tokenized_pretokens[pretoken] = [bytes([b]) for b in pretoken.encode('utf-8')]

    # setting up the smart step variables

    # prevent variable polution and make our own thing here
    smart_pretokenized_counts = Counter()
    for word in pretokenized_counts:
        smart_pretokenized_counts[word] = pretokenized_counts[word]

    # creating a merger_class object which we will use
    merger_pretokenized_counts = Counter()
    for word in pretokenized_counts:
        merger_pretokenized_counts[word] = pretokenized_counts[word]
    
    bpe_merger = BpeMerger(pretoken_counts=merger_pretokenized_counts,
                           encode_only=False)

    # creating the initial byte representations for words
    smart_tokenized_pretokens = {}
    smart_current_tokens = [bytes([b]) for b in range(256)]
    for pretoken in smart_pretokenized_counts:
        smart_tokenized_pretokens[pretoken] = [bytes([b]) for b in pretoken.encode('utf-8')]

    # create nodes that correspond to the smart tokenized pretokens
    pretoken_nodes = get_nodes(tokenized_pretokens=smart_tokenized_pretokens, 
                               pretokenized_counts=smart_pretokenized_counts)
    
    # creates a dictionary that tells which nodes correspond to each bytes_value
    bytes_nodes = get_bytes_nodes(pretoken_nodes=pretoken_nodes)

    # intiailizing a count over pairs of bytes frequencies
    pair_counter = initialize_pair_counter(pretoken_nodes=pretoken_nodes)

    # turning the pair_counter into a heap
    best_pair_heap = CounterHeap(pair_counter)

    # setting up merges, which we will return later
    merges = []

    for i in range(50):
        brute_force_step(
            pretokenized_counts,
            tokenized_pretokens,
            current_tokens
        )

        smart_step(
            merges=merges,
            bytes_nodes=bytes_nodes,
            current_tokens=smart_current_tokens,
            best_pair_heap=best_pair_heap
        )

        # have the merger make the merge
        bpe_merge = bpe_merger.get_best_merge()
        bpe_merger.merge(merge_pair = bpe_merge)
        bpe_merger_tokenized_pretokens = {}
        for pretoken in smart_tokenized_pretokens:
            bpe_merger_tokenized_pretokens[pretoken] = bpe_merger.tokenize(pretoken=pretoken)

        # get the representation of each pretoken as a list of bytes
        smart_tokenized_pretokens = from_nodes(pretoken_nodes)

        # check the merger
        assert(bpe_merger_tokenized_pretokens == smart_tokenized_pretokens)
        assert(bpe_merger.get_vocab_list() == smart_current_tokens)
        assert(bpe_merger.get_merge_history() == merges)

        assert(pretokenized_counts == smart_pretokenized_counts)
        assert(tokenized_pretokens == smart_tokenized_pretokens)
        assert(current_tokens == smart_current_tokens)

    for pretoken in pretoken_nodes:
        node = pretoken_nodes[pretoken]
        while node.nxt is not None:
            node = node.nxt
            assert(node.rep_factor == pretokenized_counts[pretoken])