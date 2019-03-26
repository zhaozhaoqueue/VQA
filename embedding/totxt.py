from gensim.models.keyedvectors import KeyedVectors

def bin_to_txt():
	MED = KeyedVectors.load_word2vec_format("BioWordVec_PubMed_MIMICIII_d200.vec.bin", binary=True)
	MED.save_word2vec_format("MED_Word2Vec.txt", binary=False)

if __name__ == "__main__":
	bin_to_txt()
