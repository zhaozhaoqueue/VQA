import argparse
from PIL import Image
import numpy as np

DATA_PATHS = "..."

# input: a Image object
# return: numpy array 3d
def transform(img):
    size = max(img.size)
    new_img = Image.new("RGB", (size, size))
    new_img.paste(img, (int((size - img.size[0])/2), int((size - img.size[1])/2)))
    new_img = new_img.resize((224, 224))
    return np.transpose(np.array(new_img)/255.0, [2, 0, 1])
	# pass

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--TRAIN_GPU_ID', type=int, default=0)
    parser.add_argument('--TEST_GPU_ID', type=int, default=3)
    parser.add_argument('--SEED', type=int, default=-1)
    parser.add_argument('--BATCH_SIZE', type=int, default=32) #
    parser.add_argument('--VAL_BATCH_SIZE', type=int, default=32) #
    parser.add_argument('--NUM_OUTPUT_UNITS', type=int, default=3000) # this should be equal to the size of vocabulary size
    parser.add_argument('--MAX_WORDS_IN_QUESTION', type=int, default=15) #
    parser.add_argument('--MAX_WORDS_IN_ANSWER', type=int, default=8) # self defined
    parser.add_argument('--MAX_ITERATIONS', type=int, default=100000) #
    parser.add_argument('--PRINT_INTERVAL', type=int, default=10) #
    parser.add_argument('--TESTDEV_INTERVAL', type=int, default=100000)
    parser.add_argument('--CHECKPOINT_INTERVAL', type=int, default=5000)
    parser.add_argument('--RESUME', type=bool, default=False)
    # parser.add_argument('--RESUME_PATH', type=str, default='./data/mfh_coatt_glove_iter_30000.pth')
    parser.add_argument('--VAL_INTERVAL', type=int, default=5000) #
    parser.add_argument('--IMAGE_CHANNEL', type=int, default=2048)
    parser.add_argument('--INIT_LEARNING_RATE', type=float, default=0.0007) #
    parser.add_argument('--DECAY_STEPS', type=int, default=40000) #
    parser.add_argument('--DECAY_RATE', type=float, default=0.5) #
    parser.add_argument('--MFB_FACTOR_NUM', type=int, default=5) #
    parser.add_argument('--MFB_OUT_DIM', type=int, default=1000) #
    parser.add_argument('--LSTM_UNIT_NUM', type=int, default=1024)
    parser.add_argument('--LSTM_DROPOUT_RATIO', type=float, default=0.3)
    parser.add_argument('--MFB_DROPOUT_RATIO', type=float, default=0.1)
    parser.add_argument('--TRAIN_DATA_SPLITS', type=str, default='train')
    parser.add_argument('--QUESTION_VOCAB_SPACE', type=str, default='train')
    parser.add_argument('--ANSWER_VOCAB_SPACE', type=str, default='train')

    parser.add_argument('--NUM_IMG_GLIMPSE', type=int, default=2)
    parser.add_argument('--NUM_QUESTION_GLIMPSE', type=int, default=2)
    parser.add_argument('--IMG_FEAT_SIZE', type=int, default=49)

    parser.add_argument('--EPOCH', type=int, default=3) # self defined
    parser.add_argument('--EMBEDDING_PATH', type=str, default="/home/lshi/vqa/embedding/BioWordVec_PubMed_MIMICIII_d200.vec.bin")

    args = parser.parse_args()
    return args