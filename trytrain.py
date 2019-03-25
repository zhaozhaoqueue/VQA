import numpy as np
import datetime
import sys
sys.path.append("/home/lshi/vqa")
sys.path.append("/home/lshi/vqa/utils")
sys.path.append("/home/lshi/vqa/models")

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision

from models.mfh_coatt_Med import mfh_coatt_Med
import config
from utils.data_provider import VQADataset
# from utils.eval_utils import exec_validation
from utils.eval_utils import eval_with_validation

# set folder
folder = "vqa/train"

# read hyper parameters from configuration
opt = config.parse_opt()

def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def convert_answer_vec(row):
	# input is a array (vector)
	fre_dic = {}
	for d in np.nditer(row):
		if d not in fre_dic:
			fre_dic[d] = 1
		else:
			fre_dic[d] += 1
	ans_vec = np.zeros(opt.NUM_OUTPUT_UNITS)
	for k, v in fre_dic.items():
		ans_vec[k] = v
	return ans_vec/np.sum(ans_vec)


# provide data
train_data 	= VQADataset(mode="train", opt=opt)
train_loader = data.DataLoader(dataset=train_data, shuffle=False, batch_size=opt.BATCH_SIZE, num_workers=1)

# construct model
model = mfh_coatt_Med(opt)

# use GPU
# model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=opt.INIT_LEARNING_RATE)
loss_func = nn.KLDivLoss()
train_loss = np.zeros((opt.EPOCH, opt.MAX_ITERATIONS + 1))
results = []
for epoch in range(opt.EPOCH):
	# for step, (qvec, q_length, q_MED_matrix, avec, a_length, a_MED_matrix, ivec, e) in enumerate(train_loader):
	for step, (q_EMB_matrix, img_matrix, a_vec) in enumerate(train_loader):
		model.train()
		print("\n%d step detail" % step)
		'''
		data prepare
		'''
		# qvec = np.squeeze(qvec, axis=0)
		# q_length = np.squeeze(q_length, axis=0)
		# a_MED_matrix = np.squeeze(a_MED_matrix, axis=0)
		# print("qvec shape: ", qvec.shape)
		# print("q_length shape: ", q_length.shape)
		# print("q_MED_matrix shape: ", q_MED_matrix.shape)
		# print("avec shape: ", avec.shape)
		# avec = np.squeeze(avec, axis=0)
		# if a_max_length is 15, avec is batch_size x 15 matrix
		# for one vector (row) in avec, each number is the index in the answer vocabulary
		# so we need to convert it to a 3000-dimension vector where 3000 is the output of model, as well as the number of words in answer vocabulary
		# the number in the new vector is the frequency rate of a word in that question. The index of the new vector is the index of the word in the answer vocabulary
		# answer = np.apply_along_axis(convert_answer_vec, 1, avec)
		# print("answer shape: ", answer.shape)

		# a_MED_matrix = np.squeeze(a_MED_matrix, axis=0)
		# ivec = np.squeeze(ivec, axis=0)
		# print("a_length shape: ", a_length.shape)
		# print("a_MED_matrix shape: ", a_MED_matrix.shape)
		# print("ivec shape: ", ivec.shape)
		# e = e.numpy()
		# print("if step is equal to e: ", step, e)
		
		# use GPU
		# qvec = Variable(qvec).cuda().long()
		# ivec = Variable(ivec).cuda().float()
		# q_MED_matrix = Variable(q_MED_matrix).cuda().float()
		# answer = Variable(answer).cuda().float()

		# use CPU
		# qvec = Variable(qvec).long()
		# ivec = Variable(ivec).float()
		# q_MED_matrix = Variable(q_MED_matrix).float()
		# answer = Variable(answer).float()

		# print("question matrix shape with batch size: ", q_EMB_matrix.shape)
		# print("image matrix shape with batch size: ", img_matrix.shape)
		# print("answer vector shape with batch size: ", a_vec.shape)

		q_EMB_matrix = Variable(q_EMB_matrix).float()
		img_matrix = Variable(img_matrix).float()
		a_vec = Variable(a_vec).float()

		optimizer.zero_grad()
		pred = model(img_matrix, q_EMB_matrix, mode="train")
		loss = loss_func(pred, a_vec)
		loss.backward()
		optimizer.step()
		train_loss[epoch, step] = loss.data.numpy()

		# adjust lr
		if(step%opt.DECAY_STEPS == 0 and step != 0):
			adjust_learning_rate(optimizer, opt.DECAY_RATE)
		if(step%10==0 and step!=0):
			print("%d step, loss is "%step)
			print(loss.data.numpy(), "\n")

		if(step%opt.PRINT_INTERVAL == 0 and step != 0):
			now = str(datetime.datetime.now())
			c_mean_loss = train_loss[epoch, step-opt.PRINT_INTERVAL:step].mean()/opt.BATCH_SIZE
			print('{}\tTrain Epoch {}\t: {}\tIter: {}\tLoss: {:.4f}'.format(now, epoch, step, c_mean_loss))

		# # validate accuracy
		# # this step takes too many memory
		# if(step%opt.VAL_INTERVAL == 0 and step != 0):
		# 	bleu = eval_with_validation(model, opt)
		# 	results.append((step, bleu))
		# 	print("%d step bleu score on validation dataset: %d" %(step, bleu))

		# if(step%opt.VAL_INTERVAL == 0 and step != 0):
		# 	test_loss, acc_overall, acc_per_ques, acc_per_ans = exec_validation(model, opt, mode='val', folder=folder, it=step)
		# 	print ('Test loss:', test_loss)
		#           print ('Accuracy:', acc_overall)
		#           print ('Test per ans', acc_per_ans)
		#           results.append([step, c_mean_loss, test_loss, acc_overall, acc_per_ques, acc_per_ans])
		#           best_result_idx = np.array([x[3] for x in results]).argmax()
		#           print ('Best accuracy of', results[best_result_idx][3], 'was at iteration', results[best_result_idx][0])
	result = eval_with_validation(model, train_data.EMB, opt, train_data.answer_vocab)
	results.append((epoch, result))
	print("%d epoch score on validation dataset: %d" %(epoch, result))

print("Training finished")
print("bleu and wbss score for each epoch: ", results)
