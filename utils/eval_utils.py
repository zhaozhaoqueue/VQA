import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import os
import sys
import json
import re
import shutil
from PIL import Image
from PIL import ImageFont, ImageDraw
import torch
import torch.nn as nn
from torch.autograd import Variable
from data_provider import VQADataProvider
import config
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
from scipy import spatial
from numpy.random import choice
from utils import evaluator_unit

plt.switch_backend('agg')


# def visualize_failures(stat_list, mode):

#     def save_qtype(qtype_list, save_filename, mode):

#         if mode == 'val':
#             savepath = os.path.join('./eval', save_filename)
#             # TODO
#             img_pre = '/home/dhpseth/vqa/02_tools/VQA/Images/val2014'

#         elif mode == 'test':
#             savepath = os.path.join('./test', save_filename)
#             # TODO
#             img_pre = '/home/dhpseth/vqa/02_tools/VQA/Images/test2015'
#         else:
#             raise Exception('Unsupported mode')
#         if os.path.exists(savepath):
#             shutil.rmtree(savepath)  # delete the entire dir tree !!
#         if not os.path.exists(savepath):
#             os.makedirs(savepath)

#         for qt in qtype_list:
#             count = 0
#             for t_question in stat_list:
#                 #print count, t_question
#                 if count < 40/len(qtype_list):
#                     t_question_list = t_question['q_list']
#                     saveflag = False
#                     #print 'debug****************************'
#                     #print qt
#                     #print t_question_list
#                     #print t_question_list[0] == qt[0]
#                     #print t_question_list[1] == qt[1]
#                     if t_question_list[0] == qt[0] and t_question_list[1] == qt[1]:
#                         saveflag = True
#                     else:
#                         saveflag = False
                               
#                     if saveflag == True:
#                         t_iid = t_question['iid']
#                         if mode == 'val':
#                             t_img = Image.open(os.path.join(img_pre, 'COCO_val2014_' + str(t_iid).zfill(12) + '.jpg'))
#                         elif mode == 'test-dev' or 'test':
#                             t_img = Image.open(os.path.join(img_pre, 'COCO_test2015_' + str(t_iid).zfill(12) + '.jpg'))

#                         # for caption
#                         #print t_iid
#                         #annIds = caps.getAnnIds(t_iid)
#                         #anns = caps.loadAnns(annIds)
#                         #cap_list = [ann['caption'] for ann in anns]
#                         ans_list = t_question['ans_list']
#                         draw = ImageDraw.Draw(t_img)
#                         for i in range(len(ans_list)):
#                             try:
#                                 draw.text((10,10*i), str(ans_list[i]))
#                             except:
#                                 pass

#                         ans = t_question['answer']
#                         pred = t_question['pred']
#                         if ans == -1:
#                             pre = ''
#                         elif ans == pred:
#                             pre = 'correct  '
#                         else:
#                             pre = 'failure  '
#                         #print ' aaa ', ans, pred
#                         ans = re.sub( '/', ' ', str(ans))
#                         pred = re.sub( '/', ' ', str(pred))
#                         img_title = pre + str(' '.join(t_question_list)) + '.  a_' + \
#                             str(ans) + ' p_' + str(pred) + '.png'
#                         count += 1
#                         print(os.path.join(savepath,img_title))
#                         t_img.save(os.path.join(savepath,img_title))

#     print 'saving whatis'
#     qt_color_list = [['what','color']]
#     save_qtype(qt_color_list, 'colors', mode)

#     print 'saving whatis'
#     qt_whatis_list = [['what','is'],['what','kind'],['what','are']]
#     save_qtype(qt_whatis_list, 'whatis', mode)

#     print 'saving is'
#     qt_is_list = [['is','the'], ['is','this'],['is','there']]
#     save_qtype(qt_is_list, 'is', mode)

#     print 'saving how many'
#     qt_howmany_list =[['how','many']]
#     save_qtype(qt_howmany_list, 'howmany', mode)

# def exec_validation(model, opt, mode, folder, it, visualize=False):
#     model.eval()
#     criterion = nn.NLLLoss()
#     dp = VQADataProvider(opt, batchsize=opt.VAL_BATCH_SIZE, mode=mode, folder=folder)
#     epoch = 0
#     pred_list = []
#     testloss_list = []
#     stat_list = []
#     total_questions = len(dp.getQuesIds())

#     print ('Validating...')
#     while epoch == 0:
#         t_word, word_length, t_img_feature, t_answer, t_glove_matrix, t_qid_list, t_iid_list, epoch = dp.get_batch_vec() 
#         word_length = np.sum(word_length,axis=1)
#         data = Variable(torch.from_numpy(t_word)).cuda().long()
#         word_length = torch.from_numpy(word_length).cuda()
#         img_feature = Variable(torch.from_numpy(t_img_feature)).cuda().float()
#         label = Variable(torch.from_numpy(t_answer)).cuda()
#         glove = Variable(torch.from_numpy(t_glove_matrix)).cuda().float()
#         pred = model(data, word_length, img_feature, glove, mode)
#         pred = (pred.data).cpu().numpy()
#         if mode == 'test-dev' or 'test':
#             pass
#         else:
#             loss = criterion(pred, label.long())
#             loss = (loss.data).cpu().numpy()
#             testloss_list.append(loss)
#         t_pred_list = np.argmax(pred, axis=1)
#         t_pred_str = [dp.vec_to_answer(pred_symbol) for pred_symbol in t_pred_list]
        
#         for qid, iid, ans, pred in zip(t_qid_list, t_iid_list, t_answer.tolist(), t_pred_str):
#             pred_list.append((pred,int(dp.getStrippedQuesId(qid))))
#             if visualize:
#                 q_list = dp.seq_to_list(dp.getQuesStr(qid))
#                 if mode == 'test-dev' or 'test':
#                     ans_str = ''
#                     ans_list = ['']*10
#                 else:
#                     ans_str = dp.vec_to_answer(ans)
#                     ans_list = [ dp.getAnsObj(qid)[i]['answer'] for i in range(10)]
#                 stat_list.append({\
#                                     'qid'   : qid,
#                                     'q_list' : q_list,
#                                     'iid'   : iid,
#                                     'answer': ans_str,
#                                     'ans_list': ans_list,
#                                     'pred'  : pred })
#         percent = 100 * float(len(pred_list)) / total_questions
#         sys.stdout.write('\r' + ('%.2f' % percent) + '%')
#         sys.stdout.flush()

#     print ('Deduping arr of len', len(pred_list))
#     deduped = []
#     seen = set()
#     for ans, qid in pred_list:
#         if qid not in seen:
#             seen.add(qid)
#             deduped.append((ans, qid))
#     print ('New len', len(deduped))
#     final_list=[]
#     for ans,qid in deduped:
#         final_list.append({u'answer': ans, u'question_id': qid})

#     if mode == 'val':
#         mean_testloss = np.array(testloss_list).mean()
#         valFile = './%s/val2015_resfile'%folder
#         with open(valFile, 'w') as f:
#             json.dump(final_list, f)
#         if visualize:
#             visualize_failures(stat_list,mode)
#         annFile = config.DATA_PATHS['val']['ans_file']
#         quesFile = config.DATA_PATHS['val']['ques_file']
#         vqa = VQA(annFile, quesFile)
#         vqaRes = vqa.loadRes(valFile, quesFile)
#         vqaEval = VQAEval(vqa, vqaRes, n=2)
#         vqaEval.evaluate()
#         acc_overall = vqaEval.accuracy['overall']
#         acc_perQuestionType = vqaEval.accuracy['perQuestionType']
#         acc_perAnswerType = vqaEval.accuracy['perAnswerType']
#         return mean_testloss, acc_overall, acc_perQuestionType, acc_perAnswerType
#     elif mode == 'test-dev':
#         filename = './%s/vqa_OpenEnded_mscoco_test-dev2015_%s-'%(folder,folder)+str(it).zfill(8)+'_results'
#         with open(filename+'.json', 'w') as f:
#             json.dump(final_list, f)
#         if visualize:
#             visualize_failures(stat_list,mode)
#     elif mode == 'test':
#         filename = './%s/vqa_OpenEnded_mscoco_test2015_%s-'%(folder,folder)+str(it).zfill(8)+'_results'
#         with open(filename+'.json', 'w') as f:
#             json.dump(final_list, f)
#         if visualize:
#             visualize_failures(stat_list,mode)
# def drawgraph(results, folder,k,d,prefix='std',save_question_type_graphs=False):
#     # 0:it
#     # 1:trainloss
#     # 2:testloss
#     # 3:oa_acc
#     # 4:qt_acc
#     # 5:at_acc

#     # training curve
#     it = np.array([l[0] for l in results])
#     loss = np.array([l[1] for l in results])
#     valloss = np.array([l[2] for l in results])
#     valacc = np.array([l[3] for l in results])

#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     ax2 = ax1.twinx()

#     ax1.plot(it,loss, color='blue', label='train loss')
#     ax1.plot(it,valloss, '--', color='blue', label='test loss')
#     ax2.plot(it,valacc, color='red', label='acc on val')
#     plt.legend(loc='lower left')

#     ax1.set_xlabel('Iterations')
#     ax1.set_ylabel('Loss Value')
#     ax2.set_ylabel('Accuracy on Val [%]')

#     plt.savefig('./%s/result_it_%d_acc_%2.2f_k_%d_d_%d_%s.png'%(folder,it[-1],valacc[-1],k,d,prefix))
#     plt.clf()
#     plt.close("all")

#     # question type
#     it = np.array([l[0] for l in results])
#     oa_acc = np.array([l[3] for l in results])
#     qt_dic_list = [l[4] for l in results]

#     def draw_qt_acc(target_key_list, figname):
#         fig = plt.figure()
#         for k in target_key_list:
#             print k,type(k)
#             t_val = np.array([qt_dic[k] for qt_dic in qt_dic_list])
#             plt.plot(it, t_val, label=str(k))
#         plt.legend(fontsize='small')
#         plt.ylim(0,100.)
#         #plt.legend(prop={'size':6})

#         plt.xlabel('Iterations')
#         plt.ylabel('Accuracy on Val [%]')

#         plt.savefig(figname,dpi=200)
#         plt.clf()
#         plt.close("all")

#     if save_question_type_graphs:
#         s_keys = sorted(qt_dic_list[0].keys())
#         draw_qt_acc(s_keys[ 0:13]+[s_keys[31],],  './ind_qt_are.png')
#         draw_qt_acc(s_keys[13:17]+s_keys[49:], './ind_qt_how_where_who_why.png')
#         draw_qt_acc(s_keys[17:31]+[s_keys[32],],  './ind_qt_is.png')
#         draw_qt_acc(s_keys[33:49],             './ind_qt_what.png')
#         draw_qt_acc(['what color is the','what color are the','what color is',\
#             'what color','what is the color of the'],'./qt_color.png')
#         draw_qt_acc(['how many','how','how many people are',\
#             'how many people are in'],'./qt_number.png')
#         draw_qt_acc(['who is','why','why is the','where is the','where are the',\
#             'which'],'./qt_who_why_where_which.png')
#         draw_qt_acc(['what is the man','is the man','are they','is he',\
#             'is the woman','is this person','what is the woman','is the person',\
#             'what is the person'],'./qt_human.png')


def eval_with_validation(model, EMB, opt, answer_vocab):
    # def q_to_vec(q_list, MED):
    #     max_length = opt.MAX_WORDS_IN_QUESTION
    #     MED_matrix = np.zeros((max_length, 200))
    #     for i in range(max_length):  # if the words are
    #         if i >= len(q_list):
    #             # pass
    #             break
    #         else:
    #             w = q_list[i]
    #             if w not in MED.wv.vocab:
    #                 MED_matrix[i] = np.zeros(self.MED_EMBEDDING_SIZE)
    #             else:
    #                 MED_matrix[i] = self.MED[w]
    #     return MED_matrix
    model.eval()
    loss_func = nn.NLLLoss()
    # simple validation metrics computation, load all validation data one time
    # later change the data provider to load validation data with batch
    # MED = VQADataProvider.loadMED_Embedding(opt.EMBEDDING_PATH)
    # _, _, qai_pair_dict = VQADataProvider.load_vqa_csv("valid")
    # q_MED_matrix = np.zeros((len(qai_pair_dict), opt.MAX_WORDS_IN_QUESTION, 200))
    # img_matrix = np.zeros((len(qai_pair_dict), 3, 224, 224))

    # pred_dict, gt_dict = {}, {}
    # gt, pred_list = [], []
    # for i, instance_id in enumerate(qai_pair_dict.keys()):
    #     q_str = qai_pair_dict[instance_id]["qstr"]
    #     a_str = qai_pair_dict[instance_id]["astr"]
    #     img_id = qai_pair_dict[instance_id]["iid"]
    #     gt.append(a_str)
    #     q_list = VQADataProvider.seq_to_list(q_str)
    #     q_MED_matrix[i, ...] = q_to_vec(q_list)
    #     img = Image.open(os.path.join("/home/lshi/vqa/valid/VQAMed2018Valid-images", "%s.jpg"%img_id)).convert('RGB')
    #     img_matrix[i, ...] = config.transform(img)

    # use GPU
    # q_MED_matrix = Variable(q_MED_matrix).cuda().float()
    # img_matrix = Variable(img_matrix).cuda().float()

    # q_a_i_df = pd.read_csv("/home/lshi/vqa/valid/VQAMed2018valid-QA.csv", delimiter="\t", header=None, names=["index", "img_id", "question", "answer"])[["img_id", "question", "answer"]]
    q_a_i_df = pd.read_csv("/home/lshi/vqa/valid/VQAMed2018valid-QA.csv", delimiter="\t", header=None, names=["q_id", "img_id", "question", "answer"])
    qa_num = q_a_i_df.shape[0]
    
    qs_MED_matrix = np.zeros((qa_num, opt.MAX_WORDS_IN_QUESTION, 200))
    imgs_matrix = np.zeros((qa_num, 3, 224, 224))

    gt, pred_list = [], []
    for i in range(qa_num):
        q_list = VQADataProvider.seq_to_list(q_a_i_df["question"][i])
        q_matrix = VQADataProvider.qlist_to_matrix(q_list, opt.MAX_WORDS_IN_QUESTION, EMB)
        qs_MED_matrix[i] = q_matrix

        image_path = "/home/lshi/vqa/valid/VQAMed2018valid-images/%s.jpg"%q_a_i_df["img_id"][i]
        image = Image.open(image_path).convert('RGB')
        img_matrix = config.transform(image)
        imgs_matrix[i] = img_matrix
        gt.append(q_a_i_df["answer"][i])

    # ans_voc = {}
    # for k, v in answer_vocab.items():
    #     ans_voc[v] = k
    
    # for sampling answer
    ans_voc_sampling = []
    for k, _ in answer_vocab.items():
        ans_voc_sampling.append(k)

    # use CPU
    qs_MED_matrix = Variable(torch.from_numpy(qs_MED_matrix)).float()
    imgs_matrix = Variable(torch.from_numpy(imgs_matrix)).float()
    pred = model(ivec=imgs_matrix, q_MED_Matrix=qs_MED_matrix, mode="val")
    pred = pred.data.numpy()

    # save the pred to check
    np.savetxt("/home/lshi/vqa/pred_log_softmax.csv", pred, delimiter=",")
    
    # pred_ind = pred.argsort(axis=1)[:, -opt.MAX_WORDS_IN_ANSWER: ]
    # # with open("/home/lshi/vqa/vocab/question.json", "r") as f:
    # #     q_vocab = json.load(f)
    # for i in range(pred_ind.shape[0]):
    #     words = [ans_voc[x] for x in pred_ind[i, :]]
    #     pred_list.append(" ".join(words))

    # use sampling method to product the predicted answer
    pred_positive = np.exp(pred)

    for i in range(pred.shape[0]):
        a_idx = choice(opt.NUM_OUTPUT_UNITS, size=opt.MAX_WORDS_IN_ANSWER, replace=False, p=pred_positive[i, :])
        words = [ans_voc_sampling[x] for x in a_idx]
        if("<END>" in words):
            words = words[:words.index("<END>")]
        if("<START>" in words):
            words.remove("<START>")
        if("<UNKNOWN>" in words):
            words.remove("<UNKNOWN>")
        if("<break>" in words):
            words.remove("<break>")
        pred_list.append(" ".join(words))

    # save the prediction to a csv file
    pred_df = q_a_i_df[["q_id", "img_id"]]
    pred_df["pred"] = pred_list
    pred_df.to_csv("/home/lshi/vqa/valid/prediction.csv", sep="\t", header=None, index=None)

    # compute two metrics
    gt_file_path = "/home/lshi/vqa/valid/VQAMed2018valid-QA.csv"
    submission_file_path = "/home/lshi/vqa/valid/prediction.csv"
    _client_payload = {}
    _client_payload["submission_file_path"] = submission_file_path
    evaluator = evaluator_unit.VqaMedEvaluator(gt_file_path)
    result = evaluator._evaluate(_client_payload)
    return result


    # wbss = compute_wbss(gt, pred_list)
    # bleu = compute_bleu(gt, pred_list)
    # return bleu #, wbss


# use the provided eval unit to compute bleu and wbss

# def compute_wbss(gt, pred_list):
#     def gen_input(str_list):
#     pass

# def compute_bleu(gt, pred_list):
#     nltk.download('punkt')
#     nltk.download('stopwords')

#     # English Stopwords
#     stops = set(stopwords.words("english"))

#     # Stemming
#     stemmer = SnowballStemmer("english")

#     # Remove punctuation from string
#     translator = str.maketrans('', '', string.punctuation)

#     max_score = len(gt)
#     current_score = 0

#     for i in range(len(gt)):
#         # lower case
#         pred_caption = pred_list[i].lower()
#         gt_caption = gt[i].lower()
#         # remove punctutation
#         pred_words = nltk.tokenize.word_tokenize(pred_caption.translate(translator))
#         gt_words = nltk.tokenize.word_tokenize(gt_caption.translate(translator))
#         # remove stopwords
#         pred_words = [word for word in pred_words if word.lower() not in stops]
#         gt_words = [word for word in gt_words if word.lower() not in stops]
#         # apply stemming
#         pred_words = [stemmer.stem(word) for word in pred_words]
#         gt_words = [stemmer.stem(word) for word in gt_words]

#         try:
#             # If both the GT and candidate are empty, assign a score of 1 for this caption
#             if len(gt_words) == 0 and len(pred_words) == 0:
#                 bleu_score = 1
#             # Calculate the BLEU score
#             else:
#                 bleu_score = nltk.translate.bleu_score.sentence_bleu([gt_words], pred_words, smoothing_function=SmoothingFunction().method0)
#         # Handle problematic cases where BLEU score calculation is impossible
#         except ZeroDivisionError:
#             pass

#         current_score += bleu_score
#     return current_score / max_score



