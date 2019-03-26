import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
# import sys
# sys.path.append("..")

'''
Pre-trained Image NN
'''
model_conv = torchvision.models.resnet152(pretrained=True)

for param in model_conv.parameters():
    param.requires_grad = False             # just use the pre-trained network to extract features from images
# Parameters of newly constructed modules have requires_grad=True by default


class mfh_coatt_Med(nn.Module):
    def __init__(self, opt):
        super(mfh_coatt_Med, self).__init__()
        self.opt = opt
        self.batch_size = self.opt.BATCH_SIZE
        self.JOINT_EMB_SIZE = opt.MFB_FACTOR_NUM * opt.MFB_OUT_DIM  # mfh: a layer of mfb's.
        # self.Embedding = nn.Embedding(opt.quest_vob_size, 200)  # 200 is the dim of MED embedding

        self.LSTM = nn.LSTM(input_size=200, hidden_size=opt.LSTM_UNIT_NUM, num_layers=1, batch_first=False)  # 200 is the embedding dim

        self.Softmax = nn.Softmax(dim=1)

        self.Linear1_q_proj = nn.Linear(opt.LSTM_UNIT_NUM*opt.NUM_QUESTION_GLIMPSE, self.JOINT_EMB_SIZE)
        self.Linear2_q_proj = nn.Linear(opt.LSTM_UNIT_NUM*opt.NUM_QUESTION_GLIMPSE, self.JOINT_EMB_SIZE)
        self.Linear3_q_proj = nn.Linear(opt.LSTM_UNIT_NUM*opt.NUM_QUESTION_GLIMPSE, self.JOINT_EMB_SIZE)

        self.img_feature_extr = nn.Sequential(*list(model_conv.children())[:-2])  # b_size x IMAGE_CHANNEL x IMAGE_WIDTH x IMAGE_WIDTH, IMAGE_CHANNEL=2048

        self.Conv1_i_proj = nn.Conv2d(opt.IMAGE_CHANNEL, self.JOINT_EMB_SIZE, 1)
        self.Linear2_i_proj = nn.Linear(opt.IMAGE_CHANNEL*opt.NUM_IMG_GLIMPSE, self.JOINT_EMB_SIZE)
        self.Linear3_i_proj = nn.Linear(opt.IMAGE_CHANNEL*opt.NUM_IMG_GLIMPSE, self.JOINT_EMB_SIZE)

        self.Dropout_L = nn.Dropout(p=opt.LSTM_DROPOUT_RATIO)
        self.Dropout_M = nn.Dropout(p=opt.MFB_DROPOUT_RATIO)

        self.Conv1_Qatt = nn.Conv2d(opt.LSTM_UNIT_NUM, 512, 1)  # (in_channels, out_channels, kernel_size)

        self.Conv2_Qatt = nn.Conv2d(512, opt.NUM_QUESTION_GLIMPSE, 1)

        self.Conv1_Iatt = nn.Conv2d(opt.MFB_OUT_DIM, 512, 1)

        self.Conv2_Iatt = nn.Conv2d(512, opt.NUM_IMG_GLIMPSE, 1)

        self.Linear_predict = nn.Linear(opt.MFB_OUT_DIM*2, opt.NUM_OUTPUT_UNITS)

    # since qvec is not used, remove this argument
    def forward(self, ivec, q_MED_Matrix, mode):
        # if mode == 'val' or mode == 'test':
        #     # self.batch_size = self.opt.VAL_BATCH_SIZE
        #     # load all validation data one time
        #     # later change it to batch type
        #     self.batch_size = q_MED_Matrix.size[0]
        # else:  # model == 'train'
        #     self.batch_size = self.opt.BATCH_SIZE
        self.batch_size = q_MED_Matrix.size()[0]

        q_MED_Matrix = q_MED_Matrix.permute(1, 0, 2)                # type float, q_max_len x b_size x emb_size
        lstm1, _ = self.LSTM(q_MED_Matrix)                     # q_max_len x b_size x hidden_size
        lstm1_droped = self.Dropout_L(lstm1)                    # q_max_len x b_size x hidden_size
        lstm1_resh = lstm1_droped.permute(1, 2, 0)                     # b_size x hidden_size x q_max_len
        lstm1_resh2 = torch.unsqueeze(lstm1_resh, 3)              # b_size x hidden_size x q_max_len x 1

        '''
        Question Attention
        '''        
        qatt_conv1 = self.Conv1_Qatt(lstm1_resh2)                   # b_size x 512 x q_max_len x 1           ; 512 is the output dim of Conv1_Qatt layer
        qatt_relu = F.relu(qatt_conv1)
        qatt_conv2 = self.Conv2_Qatt(qatt_relu)                     # b_size x opt.NUM_QUESTION_GLIMPSE x q_max_len x 1;
        qatt_conv2 = qatt_conv2.view(self.batch_size*self.opt.NUM_QUESTION_GLIMPSE, -1)  # reshape
        # qatt_conv2 = qatt_conv2.view(-1, 200*1)  # reshape          # b_size*opt.NUM_QUESTION_GLIMPSE x 200
        qatt_softmax = self.Softmax(qatt_conv2)
        qatt_softmax = qatt_softmax.view(self.batch_size, self.opt.NUM_QUESTION_GLIMPSE, -1, 1)  # reshape
        qatt_feature_list = []
        for i in range(self.opt.NUM_QUESTION_GLIMPSE):
            t_qatt_mask = qatt_softmax.narrow(1, i, 1)              # b_size x 1 x q_max_len x 1            ; narrow(dimension, start, length)
            t_qatt_mask = t_qatt_mask * lstm1_resh2                 # b_size x hidden_size x q_max_len x 1
            t_qatt_mask = torch.sum(t_qatt_mask, 2, keepdim=True)   # b_size x hidden_size x 1 x 1
            qatt_feature_list.append(t_qatt_mask)
        qatt_feature_concat = torch.cat(qatt_feature_list, 1)       # b_size x (hidden_size * NUM_QUESTION_GLIMPSE) x 1 x 1

        '''
        Extract Image Features with pre-trained NN
        '''
        img_feature = self.img_feature_extr(ivec)                        # b_size x IMAGE_CHANNEL x IMAGE_WIDTH x IMAGE_WIDTH
        # print(img_feature.size())
        # print(model_conv)

        '''
        Image Attention with MFB
        '''
        q_feat_resh = torch.squeeze(qatt_feature_concat)                                # b_size x (hidden_size * NUM_QUESTION_GLIMPSE)
        iatt_q_proj = self.Linear1_q_proj(q_feat_resh)                                  # b_size x JOINT_EMB_SIZE
        iatt_q_resh = iatt_q_proj.view(self.batch_size, self.JOINT_EMB_SIZE, 1, 1)      # b_size x JOINT_EMB_SIZE x 1 x 1

        i_feat_resh = img_feature.view(self.batch_size, self.opt.IMAGE_CHANNEL, self.opt.IMG_FEAT_SIZE, 1)  # b_size x IMAGE_CHANNEL x IMG_FEAT_SIZE x 1
        iatt_i_conv = self.Conv1_i_proj(i_feat_resh)                                     # b_size x JOINT_EMB_SIZE x IMG_FEAT_SIZE x 1

        iatt_iq_eltwise = iatt_q_resh * iatt_i_conv                                     # b_size x JOINT_EMB_SIZE x IMG_FEAT_SIZE x 1
        iatt_iq_droped = self.Dropout_M(iatt_iq_eltwise)                                # b_size x JOINT_EMB_SIZE x IMG_FEAT_SIZE x 1
        iatt_iq_permute1 = iatt_iq_droped.permute(0, 2, 1, 3).contiguous()                 # b_size x IMG_FEAT_SIZE x JOINT_EMB_SIZE x 1
        iatt_iq_resh = iatt_iq_permute1.view(self.batch_size, self.opt.IMG_FEAT_SIZE, self.opt.MFB_OUT_DIM, self.opt.MFB_FACTOR_NUM)
        iatt_iq_sumpool = torch.sum(iatt_iq_resh, 3, keepdim=True)                      # b_size x IMG_FEAT_SIZE x MFB_OUT_DIM x 1
        iatt_iq_permute2 = iatt_iq_sumpool.permute(0, 2, 1, 3)                             # b_size x MFB_OUT_DIM x IMG_FEAT_SIZE x 1
        iatt_iq_sqrt = torch.sqrt(F.relu(iatt_iq_permute2)) - torch.sqrt(F.relu(-iatt_iq_permute2))
        iatt_iq_sqrt = iatt_iq_sqrt.view(self.batch_size, -1)                           # b_size x (MFB_OUT_DIM x IMG_FEAT_SIZE)
        iatt_iq_l2 = F.normalize(iatt_iq_sqrt)
        iatt_iq_l2 = iatt_iq_l2.view(self.batch_size, self.opt.MFB_OUT_DIM, self.opt.IMG_FEAT_SIZE, 1)  # b_size x MFB_OUT_DIM x IMG_FEAT_SIZE x 1

        # 2 conv layers 1000 -> 512 -> 2
        iatt_conv1 = self.Conv1_Iatt(iatt_iq_l2)                    # b_size x 512 x IMG_FEAT_SIZE x 1
        iatt_relu = F.relu(iatt_conv1)
        iatt_conv2 = self.Conv2_Iatt(iatt_relu)                     # b_size x 2 x IMG_FEAT_SIZE x 1
        iatt_conv2 = iatt_conv2.view(self.batch_size*self.opt.NUM_IMG_GLIMPSE, -1)
        iatt_softmax = self.Softmax(iatt_conv2)
        iatt_softmax = iatt_softmax.view(self.batch_size, self.opt.NUM_IMG_GLIMPSE, -1, 1)
        iatt_feature_list = []
        for i in range(self.opt.NUM_IMG_GLIMPSE):
            t_iatt_mask = iatt_softmax.narrow(1, i, 1)              # b_size x 1 x IMG_FEAT_SIZE x 1
            t_iatt_mask = t_iatt_mask * i_feat_resh                 # b_size x IMAGE_CHANNEL x IMG_FEAT_SIZE x 1
            t_iatt_mask = torch.sum(t_iatt_mask, 2, keepdim=True)   # b_size x IMAGE_CHANNEL x 1 x 1
            iatt_feature_list.append(t_iatt_mask)
        iatt_feature_concat = torch.cat(iatt_feature_list, 1)       # b_size x (IMAGE_CHANNEL*2) x 1 x 1
        iatt_feature_concat = torch.squeeze(iatt_feature_concat)    # b_size x (IMAGE_CHANNEL*2)
        '''
        Fine-grained Image-Question MFH fusion
        '''

        mfb_q_o2_proj = self.Linear2_q_proj(q_feat_resh)               # b_size x 5000
        mfb_i_o2_proj = self.Linear2_i_proj(iatt_feature_concat)        # b_size x 5000
        mfb_iq_o2_eltwise = torch.mul(mfb_q_o2_proj, mfb_i_o2_proj)          # b_size x 5000
        mfb_iq_o2_drop = self.Dropout_M(mfb_iq_o2_eltwise)
        mfb_iq_o2_resh = mfb_iq_o2_drop.view(self.batch_size, 1, self.opt.MFB_OUT_DIM, self.opt.MFB_FACTOR_NUM)   # N x 1 x MFB_OUT_DIM x MFB_FACTOR_NUM
        mfb_iq_o2_sumpool = torch.sum(mfb_iq_o2_resh, 3, keepdim=True)    # b_size x 1 x MFB_OUT_DIM x 1
        mfb_o2_out = torch.squeeze(mfb_iq_o2_sumpool)                     # b_size x MFB_OUT_DIM
        mfb_o2_sign_sqrt = torch.sqrt(F.relu(mfb_o2_out)) - torch.sqrt(F.relu(-mfb_o2_out))
        mfb_o2_l2 = F.normalize(mfb_o2_sign_sqrt)

        mfb_q_o3_proj = self.Linear3_q_proj(q_feat_resh)               # b_size x 5000
        mfb_i_o3_proj = self.Linear3_i_proj(iatt_feature_concat)        # b_size x 5000
        mfb_iq_o3_eltwise = torch.mul(mfb_q_o3_proj, mfb_i_o3_proj)          # b_size x 5000
        mfb_iq_o3_eltwise = torch.mul(mfb_iq_o3_eltwise, mfb_iq_o2_drop)
        mfb_iq_o3_drop = self.Dropout_M(mfb_iq_o3_eltwise)
        mfb_iq_o3_resh = mfb_iq_o3_drop.view(self.batch_size, 1, self.opt.MFB_OUT_DIM, self.opt.MFB_FACTOR_NUM)   # b_size x 1 x MFB_OUT_DIM x MFB_FACTOR_NUM
        mfb_iq_o3_sumpool = torch.sum(mfb_iq_o3_resh, 3, keepdim=True)    # b_size x 1 x MFB_OUT_DIM x 1
        mfb_o3_out = torch.squeeze(mfb_iq_o3_sumpool)                     # b_size x MFB_OUT_DIM
        mfb_o3_sign_sqrt = torch.sqrt(F.relu(mfb_o3_out)) - torch.sqrt(F.relu(-mfb_o3_out))
        mfb_o3_l2 = F.normalize(mfb_o3_sign_sqrt)

        mfb_o23_l2 = torch.cat((mfb_o2_l2, mfb_o3_l2), 1)               # b_size x (MFB_OUT_DIM * MFH_ORDER)
        prediction = self.Linear_predict(mfb_o23_l2)
        # add a sigmoid function
        prediction = F.sigmoid(prediction)
        prediction = F.log_softmax(prediction, dim=1)
        return prediction
