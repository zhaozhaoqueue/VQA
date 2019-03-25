from data_provider import VQADataProvider
import re


def make_vocab(instance_dic, vocab_size=-1):
    """
    Returns a dictionary that maps words to indices.
    The output dictionary is trimmed and reindexed by <vocab_size>
    Use <vocab_size> = -1 to make question vocabulary (no trimming)
    """
    vocab = {'': 0}  # answer vocabulary | OOV is indexed as 0
    n_vocab = {'': 1000000}  # the frequency of each word in adict | OOV count is initialized with 0
    vid = 1  # the size of the vocabulary dictionary (numerate from 1)

    word_pool = []
    for instance_id in instance_dic.keys():
        instance = instance_dic[instance_id]
        for key in instance.keys():
            target_key = re.match(r'(.*)str\Z', key)   # the target key string should end with 'str'
            #  the target key is one of ('qstr', 'astr', 'qastr') exported from <load_vqa_csv>
            if target_key is not None:
                target_key = target_key.group()
                word_pool += VQADataProvider.seq_to_list(instance[target_key])

    for word in word_pool:
        # create dict
        if vocab.has_key(word):
            n_vocab[word] += 1
        else:
            n_vocab[word] = 1
            vocab[word] = vid
            vid += 1

    # debug
    nlist = []  # word frequency list in ascending order
    for k, v in sorted(n_vocab.items(), key=lambda x: x[1]):  # sort by the 2nd attribute (frequency)
        nlist.append((k, v))

    # Fill up the vocabulary dictionary with <vocab_size> most frequent words, delete the remaining

    n_del_word = 0  # total count of loss after deletion
    n_valid_word = 0  # total count of coverage after deletion
    vocab_nid = {}  # Trimmed and re-indexed answer vocabulary dictionary

    if vocab_size == -1:
        vocab_size = len(nlist)

    for i, w in enumerate(nlist[:-vocab_size]):  # Trim
        del vocab[w[0]]
        n_del_word += w[1]
    print('%s words beyond vocab_size are abandoned' % n_del_word)

    for i, w in enumerate(nlist[-vocab_size:]):  # Re-index
        n_valid_word += w[1]
        vocab_nid[w[0]] = i
    print('%s words are made into vocabulary' % n_valid_word)
    return vocab_nid


def make_vocab_files(opt):
    """
    Produce the question and answer vocabulary files.
    """
    print ('Making question vocab from ' + opt.QUESTION_VOCAB_SPACE + ' split ......')  # opt.QUESTION_VOCAB_SPACE = 'train' (default)
    qdic, _, _ = VQADataProvider.load_data(opt.QUESTION_VOCAB_SPACE)
    question_vocab_tmp = make_vocab(qdic)
    print ('Making answer vocab from ' + opt.ANSWER_VOCAB_SPACE + ' split .......')  # opt.ANSWER_VOCAB_SPACE = 'train' (default)
    _, adic, _ = VQADataProvider.load_data(opt.ANSWER_VOCAB_SPACE)
    answer_vocab_tmp = make_vocab(adic, opt.ANSWER_VOCAB_SIZE)  # opt.NUM_OUTPUT_UNITS = 3000 (default)
    return question_vocab_tmp, answer_vocab_tmp

