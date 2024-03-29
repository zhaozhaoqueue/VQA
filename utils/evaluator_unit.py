import codecs
import csv
import string
import nltk
import warnings
from nltk.translate.bleu_score import SmoothingFunction
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
from scipy import spatial

"""
Evaluator class
Evaluates one single runfile
_evaluate method is called by the CrowdAI framework and returns an object holding up to 2 different scores
"""


class VqaMedEvaluator:
    # Used for Bleu in NLTK
    remove_stopwords = True
    stemming = True
    case_sensitive = False

    """
    Constructor
    Parameter 'answer_file_path': Path of file containing ground truth
    """

    def __init__(self, answer_file_path, debug_mode=False):
        # Ground truth file
        self.answer_file_path = answer_file_path
        # Ground truth data
        self.gt = self.load_gt()
        # Used for WUPS
        self.word_pair_dict = {}
        # ...

    """
    This is the only method that will be called by the framework
    Parameter 'submission_file_path': Path of the submitted runfile
    returns a _result_object that can contain up to 2 different scores
    """

    def _evaluate(self, client_payload, context={}):
        submission_file_path = client_payload['submission_file_path']
        # Load predictions
        predictions = self.load_predictions(submission_file_path)
        # Compute first score
        wbss = self.compute_wbss(predictions)
        # Compute second score
        bleu = self.compute_bleu(predictions)

        # Create object that is returned to the CrowdAI framework
        # _result_object = {
        #  "wbss": wbss,
        #  "bleu" : bleu
        # }

        _result_object = {
            "score": wbss,
            "score_secondary": bleu
        }

        return _result_object

    """
    Load and return ground truth data
    """

    def load_gt(self):
        # return gt
        results = []
        for line in codecs.open(self.answer_file_path, 'r', 'utf-8'):
            QID = line.split('\t')[0]
            ImageID = line.split('\t')[1]
            ans = line.split('\t')[2].strip()
            results.append((QID, ImageID, ans))
        return results

    """
    Loads and returns a predictions object (dictionary) that contains the submitted data that will be used in the _evaluate method
    Parameter 'submission_file_path': Path of the submitted runfile
    Validation of the runfile format will also be handled here
    THE VALIDATION PART CAN BE IMPLEMENTED BY IVAN IF YOU WISH (ivan.eggel@hevs.ch)
    """

    def load_predictions(self, submission_file_path):

        qa_ids_testset = [tup[0] for tup in self.gt]
        image_ids_testset = [tup[1] for tup in self.gt]
        predictions = []
        occured_qaid_imageid_pairs = []

        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            lineCnt = 0
            occured_images = []
            for row in reader:
                lineCnt += 1
                # Not 2 nor 3 tab separated tokens on line => Error
                if len(row) != 3 and len(row) != 2:
                    raise Exception(
                        "Wrong format: Each line must consist of an QA-ID followed by a tab, an Image ID, a tab and an answer ({}), where the answer can be empty {}"
                        .format("<QA-ID><TAB><Image-ID><TAB><Answer>", self.line_nbr_string(lineCnt)))

                qa_id = row[0]
                image_id = row[1]

                # QA-ID - Image-ID pair does not match with test set => Error
                try:
                    i = qa_ids_testset.index(qa_id)
                    expected_image_id = image_ids_testset[i]
                    if image_id != expected_image_id:
                        raise Exception
                except:
                    raise Exception(
                        "QA-ID '{}' with Image-ID '{}' does not represent a valid QA-ID - IMAGE ID pair in the testset {}"
                        .format(qa_id, image_id, self.line_nbr_string(lineCnt)))

                # QA-ID - Image-ID already appeared => Error
                if (qa_id, image_id) in occured_qaid_imageid_pairs:
                    raise Exception(
                        "The QA-ID '{}' with Image-ID '{}' pair appeared more than once in the submission file {}"
                        .format(qa_id, image_id, self.line_nbr_string(lineCnt)))

                answer = row[2] if (len(row) == 3) else ""

                predictions.append((qa_id, image_id, answer))
                occured_qaid_imageid_pairs.append((qa_id, image_id))

            # Not all QA-ID Image-ID pairs included => Error
            if len(predictions) != len(self.gt):
                raise Exception(
                    "Number of QA-ID - Image-ID pairs in submission file does not correspond with number of QA-ID - Image-ID pairs in testset")

        return predictions

    """
    Compute and return the primary score
    Parameter 'predictions' : predictions object generated by the load_predictions method
    NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
    We assume that the predictions in the parameter are valid
    Valiation should be handled in the load_predictions method
    """

    def compute_wbss(self, predictions):
        # nltk.download('wordnet')
        count = 0
        totalscore_wbss = 0.0
        for tuple1, tuple2 in zip(self.gt, predictions):
            QID1 = tuple1[0]
            QID2 = tuple2[0]
            imageID1 = tuple1[1]
            imageID2 = tuple2[1]
            ans1 = tuple1[2]
            ans2 = tuple2[2]
            assert (QID1 == QID2)
            assert (imageID1 == imageID2)

            count += 1
            QID = QID1

            if ans1 == ans2:
                score_wbss = 1.0
            elif ans2.strip() == "":  # Added by Ivan (Handle case of empty answer)
                score_wbss = 0
            else:
                score_wbss = self.calculateWBSS(ans1, ans2)

            totalscore_wbss += score_wbss

        return totalscore_wbss / float(count)

    def calculateWBSS(self, S1, S2):
        if S1 is None or S2 is None:
            return 0.0
        dictionary = self.constructDict(S1.split(), S2.split())
        vector1 = self.getVector_wordnet(S1, dictionary)
        vector2 = self.getVector_wordnet(S2, dictionary)
        cos_similarity = self.calculateCosineSimilarity(vector1, vector2)
        return cos_similarity

    def getVector_wordnet(self, S, dictionary):
        vector = [0.0] * len(dictionary)
        for index, word in enumerate(dictionary):
            # update score for vector[index]
            for wordinS in S.split():
                if wordinS == word:
                    score = 1.0
                else:
                    score = self.wups_score(word, wordinS)
                if score > vector[index]:
                    vector[index] = score
        return vector

    def constructDict(self, list1, list2):
        return list(set(list1 + list2))

    def wups_score(self, word1, word2):
        score = 0.0
        score = self.wup_measure(word1, word2)
        return score

    def wup_measure(self, a, b, similarity_threshold=0.925, debug=False):
        """
        Returns Wu-Palmer similarity score.
        More specifically, it computes:
            max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
            where interp is a 'interpretation field'
        """
        if debug: print('Original', a, b)
        # if word_pair_dict.has_key(a+','+b):
        if a + ',' + b in self.word_pair_dict.keys():
            return self.word_pair_dict[a + ',' + b]

        def get_semantic_field(a):
            return wn.synsets(a, pos=wn.NOUN)

        if a == b: return 1.0

        interp_a = get_semantic_field(a)
        interp_b = get_semantic_field(b)
        if debug: print(interp_a)

        if interp_a == [] or interp_b == []:
            return 0.0

        if debug: print('Stem', a, b)
        global_max = 0.0
        for x in interp_a:
            for y in interp_b:
                local_score = x.wup_similarity(y)
                if debug: print('Local', local_score)
                if local_score > global_max:
                    global_max = local_score
        if debug: print('Global', global_max)

        # we need to use the semantic fields and therefore we downweight
        # unless the score is high which indicates both are synonyms
        if global_max < similarity_threshold:
            interp_weight = 0.1
        else:
            interp_weight = 1.0

        final_score = global_max * interp_weight
        self.word_pair_dict[a + ',' + b] = final_score
        return final_score

    def calculateCosineSimilarity(self, vector1, vector2):
        return 1 - spatial.distance.cosine(vector1, vector2)

    """
    Compute and return the secondary score
    Parameter 'predictions' : predictions object generated by the load_predictions method
    NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
    We assume that the predictions in the parameter are valid
    Valiation should be handled in the load_predictions method
    """

    def compute_bleu(self, predictions):

        # Hide warnings
        warnings.filterwarnings('ignore')

        # NLTK
        # Download Punkt tokenizer (for word_tokenize method)
        # Download stopwords (for stopword removal)
        # nltk.download('punkt')
        # nltk.download('stopwords')

        # English Stopwords
        stops = set(stopwords.words("english"))

        # Stemming
        stemmer = SnowballStemmer("english")

        # Remove punctuation from string
        translator = str.maketrans('', '', string.punctuation)

        candidate_pairs = self.readresult(predictions)

        gt_pairs = self.readresult(self.gt)

        # Define max score and current score
        max_score = len(gt_pairs)
        current_score = 0

        i = 0
        for image_key in candidate_pairs:

            # Get candidate and GT caption
            candidate_caption = candidate_pairs[image_key]
            gt_caption = gt_pairs[image_key]

            # Optional - Go to lowercase
            if not VqaMedEvaluator.case_sensitive:
                candidate_caption = candidate_caption.lower()
                gt_caption = gt_caption.lower()

            # Split caption into individual words (remove punctuation)
            candidate_words = nltk.tokenize.word_tokenize(candidate_caption.translate(translator))
            gt_words = nltk.tokenize.word_tokenize(gt_caption.translate(translator))

            # Optional - Remove stopwords
            if VqaMedEvaluator.remove_stopwords:
                candidate_words = [word for word in candidate_words if word.lower() not in stops]
                gt_words = [word for word in gt_words if word.lower() not in stops]

            # Optional - Apply stemming
            if VqaMedEvaluator.stemming:
                candidate_words = [stemmer.stem(word) for word in candidate_words]
                gt_words = [stemmer.stem(word) for word in gt_words]

            # Calculate BLEU score for the current caption
            try:
                # If both the GT and candidate are empty, assign a score of 1 for this caption
                if len(gt_words) == 0 and len(candidate_words) == 0:
                    bleu_score = 1
                # Calculate the BLEU score
                else:
                    bleu_score = nltk.translate.bleu_score.sentence_bleu([gt_words], candidate_words,
                                                                         smoothing_function=SmoothingFunction().method0)
            # Handle problematic cases where BLEU score calculation is impossible
            except ZeroDivisionError:
                pass
                # raise Exception('Problem with {} {}', gt_words, candidate_words)

            # Increase calculated score
            current_score += bleu_score

        return current_score / max_score

    def readresult(self, tuples):
        pairs = {}
        for row in tuples:
            pairs[row[0]] = row[2]
        return pairs

    def line_nbr_string(self, line_nbr):
        return "(Line nbr {})".format(line_nbr)


"""
Test evaluation a runfile
provide path to ground truth file in constructor
call _evaluate method with path of submitted file as argument
"""
if __name__ == "__main__":
    # Ground truth file
    gt_file_path = "/home/tryn/Desktop/result.csv"
    # Submission file
    submission_file_path = '/home/tryn/Desktop/result.csv'
    # submission_file_path = "runs/01_ok_run.csv"
    # submission_file_path = "runs/02_not_2_or_3_tokens.csv"
    # submission_file_path = "runs/03_wrong_qaid_imageid_pair.csv"
    # submission_file_path = "runs/04_qaid_imageid_pair_more_than_once.csv"
    # submission_file_path = "runs/05_not_all_qaid_imageid_pairs.csv"

    _client_payload = {}
    _client_payload["submission_file_path"] = submission_file_path

    # Create instance of Evaluator
    evaluator = VqaMedEvaluator(gt_file_path)
    # Call _evaluate method
    result = evaluator._evaluate(_client_payload)
    print(result)
