import os
# import skimage.io as io
from PIL import Image
import csv
import numpy as np
from gensim.models import KeyedVectors # load word2vev
import json
from data_provider import VQADataProvider

root_path = os.path.join(os.getenv("HOME"), "vqa")

# folder = "../../data/train/VQAMed2018Train-images"
# img_list = os.listdir(folder)

# img_size_set = set()

# for t in img_list:
#     img = Image.open(os.path.join(folder, t))
#     img_size_set.add(img.size)

# print("number of unique image sizes in training: ", len(img_size_set))
# print("all image sizes for training: ")
# print(img_size_set)
# print()

# folder = "../../data/valid/VQAMed2018Valid-images"
# img_list = os.listdir(folder)
# for t in img_list:
# 	img = Image.open(os.path.join(folder, t))
# 	img_size_set.add(img.size)
# print("number of image sizes including valid: ", len(img_size_set))

# folder = "../../data/test/VQAMed2018Test-images"
# img_list = os.listdir(folder)
# for t in img_list:
# 	img = Image.open(os.path.join(folder, t))
# 	img_size_set.add(img.size)
# print("number of image sizes including valid and test: ", len(img_size_set))
# print()
# print("all sizes")
# print(img_size_set)
def image_size_list(folder):
	folder_name = os.path.join(root_path, folder, "VQAMed2018%s-images" % folder)
	img_size_set = set()
	img_list = os.listdir(folder_name)

	for t in img_list:
		img = Image.open(os.path.join(folder_name, t))
		img_size_set.add(img.size)
	return img_size_set

def img_ratio_check(folder, ratio):
	# ratio should be a integer
	folder_name = os.path.join(root_path, folder, "VQAMed2018%s-images" % folder)
	img_list = os.listdir(folder_name)
	out_ratio_ls = []
	for t in img_list:
		img = Image.open(os.path.join(folder_name, t))
		r = img.size[0]/img.size[1]
		if(r>ratio or r<(1/ratio)):
			out_ratio_ls.append((t, r))
	print("There are %d images are over the ratio in the %s images folder" % (len(out_ratio_ls), folder))
	return out_ratio_ls



def image_name_check(folder):
	img_folder = os.path.join(root_path, folder, "VQAMed2018%s-images" % folder)
	# img_file = os.path.join("..", folder, "VQAMed2018%s-images-List.txt" % folder)
	qa_file =os.path.join(root_path, folder, "VQAMed2018%s-QA.csv" % folder)

	img_list = os.listdir(img_folder)
	img_list = set(img_list)
	
	# img_name_list = []
	# with open(img_file, "r") as f:
	# 	for line in f:
	# 		img_name_list.append("%s.jpg" % line.strip())
	# img_name_list = set(img_name_list)

	qa_img_list = []
	with open(qa_file, "r") as f:
		QA = csv.reader(f, delimiter='\t', quotechar='\n')
		for row in QA:
			qa_img_list.append("%s.jpg" % row[1].strip())
	qa_img_list = set(qa_img_list)

	# if(len(qa_img_list) != len(img_name_list)):
	# print("difference between qa and name list:")
	# print(img_name_list - qa_img_list)
	# print()
	# if(len(img_list) != len(qa_img_list)):
	print("difference between qa and images: ")
	print(img_list - qa_img_list)

def img_size_pixel_check(file):
	img = Image.open(file)
	img_size = img.size
	img_pixel = np.array(img).shape
	print(file)
	print("image size (width, weight): ", img_size)
	print("image pixel (3d): ", img_pixel)


def check_embedding(embedding_file, vocab_file, qa_file):
	EMBEDDING = KeyedVectors.load_word2vec_format(os.path.join(root_path, "embedding", embedding_file), binary=True).wv.vocab
	
	voc_counter = 0
	with open(os.path.join(root_path, "vocab", vocab_file), "r") as f:
		vocab = json.load(f)
	for w in vocab:
		if(w in EMBEDDING):
			voc_counter += 1

	tot_counter = 0
	word_fre = {}
	with open(os.path.join(root_path, "train", qa_file), "r") as f:
		QA = csv.reader(f, delimiter='\t', quotechar='\n')
		for row in QA:
			ws = VQADataProvider.seq_to_list(row[2])
			for w in ws:
				if(w not in word_fre):
					word_fre[w] = 1
				else:
					word_fre[w] += 1
	for w in word_fre:
		if(w in EMBEDDING):
			tot_counter += word_fre[w]

	return voc_counter/len(vocab), tot_counter/sum(word_fre.values())


if __name__ == "__main__":
	# image_name_check("train")
	# image_name_check("valid")
	# img_size_pixel_check("../train/VQAMed2018Train-images/1750-1172-5-4-1.jpg")
	# question_emb_proportion, total_emb_proportion = check_embedding("BioWordVec_PubMed_MIMICIII_d200.vec.bin", "question.json", "VQAMed2018Train-QA.csv")
	# answer_emb_proportion = check_embedding("BioWordVec_PubMed_MIMICIII_d200.vec.bin", "answer.json")
	# print("vocab embedding: ", question_emb_proportion)
	# print("total embedding: ", total_emb_proportion)
	# print("answer: ", answer_emb_proportion)
	o_r_ls_train = img_ratio_check("train", 3)
	print("\ntraining over ratio images:")
	print(o_r_ls_train)
	o_r_ls_valid = img_ratio_check("valid", 3)
	print("\nvalidation over ratio images:")
	print(o_r_ls_valid)
	o_r_ls_test = img_ratio_check("test", 3)
	print("\ntest over ratio images:")
	print(o_r_ls_test)