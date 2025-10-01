from cs336_basics.merger import smart_step, from_nodes, get_nodes, get_bytes_nodes, initialize_pair_counter
from collections import Counter
from cs336_basics.utils import random_string
from cs336_basics.counter_heap import CounterHeap
import random


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

  