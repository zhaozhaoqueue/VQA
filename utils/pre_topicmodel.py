import os
import json
import csv
import data_provider

q_a_i_path = "/home/leishi/vqa/train/VQAMed2018train-QA.csv"
q_voc_path = "/home/leishi/vqa/vocab/question.json"
wordlist_path = "/home/leishi/vqa/TopicModel/word.txt"
doc_path = "/home/leishi/vqa/TopicModel/sentences.txt"

def gen_txt():
	with open(q_voc_path, "r") as f:
		q_dic = json.load(f)

	word_list = []

	exc_list = ["<break>", "<END>", "<START>", "<UNKNOWN>"]

	for k, _ in q_dic.items():
		# exclude <break>, <END>, <START>, <UNKNOWN>
		if(k not in exc_list):
			word_list.append(k)


	sent_list = []
	with open(q_a_i_path, "r") as csvfile:
		QA = csv.reader(csvfile, delimiter="\t", quotechar='\n')
		for row in QA:
			sent_list.append(data_provider.VQADataProvider.seq_to_list(row[2]))


	sent_idx_list = []
	for sent in sent_list:
		sent_idx_list.append([word_list.index(x) for x in sent if x not in exc_list])

	with open(wordlist_path, "w") as f:
		for item in word_list:
			f.write("%s\n"%item)

	with open(doc_path, "w") as f:
		for sent in sent_idx_list:
			f.write(" ".join([str(i) for i in sent]) + "\n")

if __name__ == "__main__":
	gen_txt()