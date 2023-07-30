# A Simple But Tough-To-Beat Baseline For Sentence Embeddings" by Arora et al. (2017)
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
import numpy as np

# implimentation of Tough-to-beat baseline for sentence embeddings


def smooth_inverse_frequency(sentences, embedding_model, a=0.001):
    word_counts = {}
    # Count the number of times each word appears in the corpus and store in total_count
    for index, word in enumerate(embedding_model.wv.index_to_key):
        word_counts[word] = embedding_model.wv.get_vecattr(word, "count")
    total_count = sum(word_counts.values())
    sentence_vectors = []
    for sentence in sentences:
        sentence_vector = np.zeros(embedding_model.vector_size)
        words_in_sentence = len(sentence)

        for word in sentence:
            # check if word is in the vocabulary of the embedding model
            if word in embedding_model.wv.index_to_key:
                weight = a / (a + word_counts[word]/total_count)
                sentence_vector += weight * embedding_model.wv[word]
        sentence_vectors.append(sentence_vector / words_in_sentence)

    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(np.array(sentence_vectors))
    u = svd.components_

    sentence_vectors = np.array(sentence_vectors) - \
        np.array(sentence_vectors).dot(u.T) * u

    return sentence_vectors


# Assuming sentences is a list of tokenized sentences and model is a pretrained Word2Vec model
# get full path to where we are running this program
# print(os.path.dirname(os.path.abspath(__file__)))
embedding_model = Word2Vec.load("W2Vec_model")
sentences = [["garage", "room", "ready", "first", "floor"],
             ["kitchen", "is", "another", "room"]]

sentence_vectors = smooth_inverse_frequency(
    sentences, embedding_model, a=0.001)
print(sentence_vectors)
