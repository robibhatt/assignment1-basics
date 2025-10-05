from cs336_basics.merger import BpeMerger
from cs336_basics.quick_merger import QuickMerger
from collections import Counter
from cs336_basics.utils import random_string
import random


def test_merger_class_merges():

    # setting up brute force variables

    # just a bunch of words withcounts
    quick_pretokenized_counts = Counter()
    for i in range(10):
        word = random_string(12)
        quick_pretokenized_counts[word] = random.randint(1, 10)

    # create the quick merger
    quick_merger = QuickMerger(pretoken_counts=quick_pretokenized_counts)

    # remake the same pretokenized counts
    merger_pretokenized_counts = Counter()
    for word in quick_pretokenized_counts:
        merger_pretokenized_counts[word] = quick_pretokenized_counts[word]

    # make the 'slow' merger
    bpe_merger = BpeMerger(pretoken_counts=merger_pretokenized_counts,
                           encode_only=False)

 
    for i in range(50):

        # have the slow merger make the merge
        bpe_merge = bpe_merger.get_best_merge()
        bpe_merger.merge(merge_pair = bpe_merge)
        bpe_merger_tokenized_pretokens = {}
        for pretoken in merger_pretokenized_counts:
            bpe_merger_tokenized_pretokens[pretoken] = bpe_merger.tokenize(pretoken=pretoken)

        # have the fast merger make the merge
        quick_merge = quick_merger.get_best_merge()
        quick_merger.merge(merge_pair=quick_merge)
        quick_merger_tokenized_pretokens = {}
        for pretoken in quick_pretokenized_counts:
            quick_merger_tokenized_pretokens[pretoken] = quick_merger.tokenize(pretoken=pretoken)

        # check the merger
        assert(quick_merger_tokenized_pretokens == bpe_merger_tokenized_pretokens)
        assert(bpe_merger.get_vocab_list() == quick_merger.get_vocab_list())
        assert(bpe_merger.get_merge_history() == quick_merger.get_merge_history())