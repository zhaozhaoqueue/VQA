# -*- coding: utf-8 -*-
import numpy as np
import re, csv, random, json
import config
import torch.utils.data as data
import spacy

QID_KEY_SEPARATOR = '/'
ZERO_PAD = '_PAD'  # ???
MED_EMBEDDING_SIZE = 200

MED_folder = '/home/tryn/MedVQA/data/MED Word Embedding'
TXT = '/MED_Word2Vec.txt'


class VQADataProvider:

    def __init__(self, opt, folder='./result', batchsize=64, max_length=15, mode='train'):
        self.opt = opt  # output　related argument
        self.batchsize = batchsize
        self.d_vocabulary = None
        self.batch_index = None
        self.batch_len = None
        self.rev_adict = None
        self.max_length = max_length
        self.mode = mode
        self.qdic, self.adic = VQADataProvider.load_data(mode)

        with open('./%s/vdict.json' % folder, 'r') as f:
            self.vdict = json.load(f)  # questions vocabulary dictionary {(<'word'>, <index>}
        with open('./%s/adict.json' % folder, 'r') as f:
            self.adict = json.load(f)  # answers vocabulary dictionary {(<'word'>, <index>}

        self.n_ans_vocabulary = len(self.adict)
        self.nlp = spacy.load('en_vectors_web_lg')
        self.MED = VQADataProvider.loadMED_Embedding('%s/%s' % (MED_folder, TXT))
        # MED is a pre-trained medical word embedding. Note the values are np arrays not lists for quick loading
        
    @staticmethod        
    def loadMED_Embedding(path='%s/%s' % (MED_folder, TXT)):
        print("Loading MED Embedding")
        f = open(path, 'r')
        MED = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            MED[word] = embedding
        print("Done." + str(len(MED)) + " words loaded!")
        return MED

    @staticmethod
    def load_vqa_csv(data_split):
        """
        Parses the question and answer csv files for the given data split.
        :param data_split= 'train'/ 'val'/ 'test'
        :return question dictionary and answer dictionary
        """
        qdic, adic, qa_pair= {}, {}, {}
        # assert data_split in config.DATA_PATHS.keys(), 'unknown data split'
        with open(config.DATA_PATHS[data_split]['QA_file']) as csvfile:
            QA = csv.reader(csvfile, delimiter='\t', quotechar='\n')
            for rows in QA:
                if len(rows) != 4:
                    # Ideally, rows = [<qa_id>, <image_id>, <question>, <answer>]
                    print('Row ' + str(rows[0]) + ' is not in QA form:' + str(rows))
                else:
                    qdic[data_split + QID_KEY_SEPARATOR + str(rows[0])] = \
                        {'qstr': rows[2], 'iid': rows[1]}
                    if 'test' not in data_split:
                        adic[data_split + QID_KEY_SEPARATOR + str(rows[0])] = \
                            {'astr': rows[3], 'iid': rows[1]}
                        qa_pair[data_split + QID_KEY_SEPARATOR + str(rows[0])] = \
                            {'qastr': rows[2] + ' ' +rows[3], 'iid': rows[1]}
        print('parsed ' + str(len(qdic)) + ' Q&A instances for ' + str(data_split))
        return qdic, adic, qa_pair

    @staticmethod
    def load_data(data_split_str):
        """
        :param data_split_str: 'train','val', 'test', 'train+val'
        :return: combined question dictionary and answer dictionary
        """
        all_qdic, all_adic, all_qa_pair = {}, {}, {}
        for data_split in data_split_str.split('+'):
            assert data_split in config.DATA_PATHS.keys(), 'unknown data split'
            qdic, adic, qa_pair= VQADataProvider.load_vqa_csv(data_split)
            all_qdic.update(qdic)
            all_adic.update(adic)
            all_qa_pair.update(qa_pair)
        return all_qdic, all_adic, all_qa_pair

    def getQuesIds(self):
        return self.qdic.keys()

    @staticmethod
    def getStrippedQuesId(qid):
        return qid.split(QID_KEY_SEPARATOR)[1]

    def getImgId(self, qid):
        return self.qdic[qid]['iid']

    def getQuesStr(self, qid):
        return self.qdic[qid]['qstr']

    def getAnsObj(self, qid):
        if self.mode == 'test':
            return -1
        return self.adic[qid]

    @staticmethod
    def seq_to_list(s):
        t_str = s.lower()
        for i in [r'\?', r'\!', r'\'', r'\"', r'\$', r'\:', r'\@', r'\(', r'\)', r'\,', r'\.', r'\;', r'\/']:
            t_str = re.sub(i, ' ', t_str)
        for i in [r'\-']:
            t_str = re.sub(i, '', t_str)
        q_list = re.sub(r'\?', '', t_str.lower()).split(' ')
        q_list = list(filter(lambda x: len(x) > 0, q_list))
        return q_list

    def extract_answer(self, answer_obj):
        """ Return the most popular answer in string."""
        if self.mode == 'test':
            return -1
        answer_list = [answer_obj[i]['answer'] for i in range(10)]
        dic = {}
        for ans in answer_list:
            if dic.has_key(ans):
                dic[ans] += 1
            else:
                dic[ans] = 1
        max_key = max((v, k) for (k, v) in dic.items())[1]
        return max_key

    def extract_answer_prob(self, answer_obj):
        """ Return the most popular answer in string.
            Answers are filtered so that they are all from adict.
        """
        if self.mode == 'test':
            return -1

        answer_list = [ans['answer'] for ans in answer_obj]
        prob_answer_list = []
        for ans in answer_list:
            if self.adict.has_key(ans):
                prob_answer_list.append(ans)

        if len(prob_answer_list) == 0:
            if self.mode == 'val' or self.mode == 'test':
                return 'hoge'  # what does it mean?
            else:
                raise Exception("This should not happen.")
        else:
            return random.choice(prob_answer_list)

    def extract_answer_list(self, answer_obj):
        answer_list = [ans['answer'] for ans in answer_obj]
        prob_answer_vec = np.zeros(self.opt.NUM_OUTPUT_UNITS)
        for ans in answer_list:
            if self.adict.has_key(ans):
                index = self.adict[ans]
                prob_answer_vec[index] += 1
        return prob_answer_vec / np.sum(prob_answer_vec)

    def qlist_to_vec(self, max_length, q_list):
        """
        Converts a list of words into a format suitable for the embedding layer.
        :param
        max_length -- the maximum length of a question sequence
        q_list -- a list of words which are the tokens in the question

        :returns
        qvec -- A max_length length vector containing one-hot indices for each word
        cvec -- A max_length length sequence continuation indicator vector: padded --> 0; non-padded --> 1
        """
        qvec = np.zeros(max_length)
        cvec = np.zeros(max_length)
        MED_matrix = np.zeros((max_length, MED_EMBEDDING_SIZE))
        """  pad on the left   """
        # for i in xrange(max_length):
        #     if i < max_length - len(q_list):
        #         cvec[i] = 0
        #     else:
        #         w = q_list[i-(max_length-len(q_list))]
        #         # is the word in the vocabulary?
        #         if self.vdict.has_key(w) is False:
        #             w = ''
        #         qvec[i] = self.vdict[w]
        #         cvec[i] = 0 if i == max_length - len(q_list) else 1
        """  pad on the right   """
        for i in range(max_length):  # use range() in python 3
            if i >= len(q_list):
                pass
            else:
                w = q_list[i]
                if w not in self.MED:
                    self.MED[w] = self.nlp(u'%s' % w).vector  # use embeddings from spacy to handle out-of-vocabulary
                MED_matrix[i] = self.MED[w]
                if self.vdict.has_key(w) is False:
                    w = ''  # Not update vocabulary in here
                qvec[i] = self.vdict[w]  # words out of question vocabulary will be indexed with 0
                cvec[i] = 1
        return qvec, cvec, MED_matrix

    def answer_to_vec(self, ans_str):
        """ Return answer id if the answer is included in vocabulary otherwise 0 """
        if self.mode == 'test':
            return -1

        if self.adict.has_key(ans_str):
            ans = self.adict[ans_str]
        else:
            ans = self.adict['']
        return ans

    def vec_to_answer(self, ans_symbol):
        """ Return answer id if the answer is included in vocabulary otherwise 0 """
        if self.rev_adict is None:
            rev_adict = {}
            for k, v in self.adict.items():  # flip key-value pairs
                rev_adict[v] = k
            self.rev_adict = rev_adict

        return self.rev_adict[ans_symbol]

    def create_batch(self, qid_list):

        qvec = (np.zeros(self.batchsize * self.max_length)).reshape(self.batchsize, self.max_length)
        cvec = (np.zeros(self.batchsize * self.max_length)).reshape(self.batchsize, self.max_length)
        ivec = np.zeros((self.batchsize, 2048, self.opt.IMG_FEAT_SIZE))
        if self.mode == 'val' or self.mode == 'test':
            avec = np.zeros(self.batchsize)  # ??? it should be self.opt.NUM_OUTPUT_UNITS ???
        else:
            avec = np.zeros((self.batchsize, self.opt.NUM_OUTPUT_UNITS))
        MED_matrix = np.zeros((self.batchsize, self.max_length, MED_EMBEDDING_SIZE))

        for i, qid in enumerate(qid_list):

            # load raw question information
            q_str = self.getQuesStr(qid)
            q_ans = self.getAnsObj(qid)
            q_iid = self.getImgId(qid)

            # convert question to vec
            q_list = V.DataProvider.seq_to_list(q_str)
            t_qvec, t_cvec, t_MED_matrix = self.qlist_to_vec(self.max_length, q_list)

            # we don't have information from Genome, so the following "if" will fail
            try:
                qid_split = qid.split(QID_KEY_SEPARATOR)
                data_split = qid_split[0]
                if data_split == 'genome':
                    t_ivec = np.load(config.DATA_PATHS['genome']['features_prefix'] + str(q_iid) + '.jpg.npz')['x']  # the image data are in npz (numpy) form.
                else:
                    t_ivec = \
                        np.load(config.DATA_PATHS[data_split]['features_prefix'] + str(q_iid).zfill(12) + '.jpg.npz')[
                            'x']

                # reshape t_ivec to D x FEAT_SIZE
                if len(t_ivec.shape) > 2:
                    t_ivec = t_ivec.reshape((2048, -1))  # restructure matrix into 2-D with fixed 2048 rows and enough to hold all elements
                t_ivec = (t_ivec / np.sqrt((t_ivec ** 2).sum()))
            except:
                t_ivec = 0.
                print('data not found for qid : ', q_iid, self.mode)

            # convert answer to vec
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                q_ans_str = self.extract_answer(q_ans)
                t_avec = self.answer_to_vec(q_ans_str)
            else:
                t_avec = self.extract_answer_list(q_ans)

            qvec[i, ...] = t_qvec
            cvec[i, ...] = t_cvec
            ivec[i, :, 0:t_ivec.shape[1]] = t_ivec
            avec[i, ...] = t_avec
            MED_matrix[i, ...] = t_MED_matrix

        return qvec, cvec, ivec, avec, MED_matrix

    def get_batch_vec(self):
        if self.batch_len is None:
            self.n_skipped = 0
            qid_list = self.getQuesIds()
            random.shuffle(qid_list)
            self.qid_list = qid_list
            self.batch_len = len(qid_list)
            self.batch_index = 0
            self.epoch_counter = 0

        def has_at_least_one_valid_answer(t_qid):
            answer_obj = self.getAnsObj(t_qid)
            answer_list = [ans['answer'] for ans in answer_obj]
            for ans in answer_list:
                if self.adict.has_key(ans):
                    return True

        counter = 0
        t_qid_list = []
        t_iid_list = []
        while counter < self.batchsize:
            t_qid = self.qid_list[self.batch_index]
            t_iid = self.getImgId(t_qid)
            if self.mode == 'val' or self.mode == 'test':
                t_qid_list.append(t_qid)
                t_iid_list.append(t_iid)
                counter += 1
            elif has_at_least_one_valid_answer(t_qid):
                t_qid_list.append(t_qid)
                t_iid_list.append(t_iid)
                counter += 1
            else:
                self.n_skipped += 1

            if self.batch_index < self.batch_len - 1:
                self.batch_index += 1
            else:
                self.epoch_counter += 1
                qid_list = self.getQuesIds()
                random.shuffle(qid_list)
                self.qid_list = qid_list
                self.batch_index = 0
                print("%d questions were skipped in a single epoch" % self.n_skipped)
                self.n_skipped = 0

        t_batch = self.create_batch(t_qid_list)
        return t_batch + (t_qid_list, t_iid_list, self.epoch_counter)


class VQADataset(data.Dataset):

    def __init__(self, mode, batchsize, folder, opt):
        self.batchsize = batchsize
        self.mode = mode
        self.folder = folder
        if self.mode == 'val' or self.mode == 'test':
            pass
        else:
            self.dp = VQADataProvider(opt, batchsize=self.batchsize, mode=self.mode, folder=self.folder)

    def __getitem__(self, idx):
        if self.mode == 'val' or self.mode == 'test':
            pass
        else:
            word, cont, feature, answer, MED_matrix, _, _, epoch = self.dp.get_batch_vec()
        word_length = np.sum(cont, axis=1)
        return word, word_length, feature, answer, MED_matrix, epoch

    def __len__(self):
        return 150000
