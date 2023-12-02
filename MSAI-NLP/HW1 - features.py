from collections import OrderedDict, Counter
from pandas.io.formats.format import TextAdjustment
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np
from sklearn.preprocessing import normalize, MinMaxScaler


class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow
        inner_bow = {}

        for string in X:
            for word in set(' '.join(X).split()):                
                inner_bow[word] = string.count(word)
        ss = list(inner_bow.items())
        ss.sort(key = lambda x: x[1], reverse = True)

        self.bow = [i for i, _ in ss][:self.k]         

        # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """
        txt = text.split()
        vect = [0 for i in range(len(self.bow))]        
        for i in range(len(self.bow)):
            word = self.bow[i]
            vect[i] = txt.count(word)            

        result = vect        
        return np.array(result, "float32")        

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow


class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize        
        self.idf = OrderedDict()

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
               
        count_dict = {}

        num_of_docs = len(X)
  
        for string in X:
            for word in set(string.split()):
                if word in count_dict.keys():
                    count_dict[word] += 1
                else:
                    count_dict[word] = 1
        count_dict = dict(sorted(count_dict.items(), key=lambda item: -item[1]))
        count_dict = {word: np.log(num_of_docs / (count + 1)) for word, count in count_dict.items()}
        self.idf = OrderedDict(list(count_dict.items())[:self.k])        
        
        return self

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """
        words = Counter([word for word in text.split()])
        vector = []
        for word in self.idf.keys():
            if word in words.keys():
                vector.append(words[word] * self.idf[word])
            else:
                vector.append(0)
        if normalize:
            result = list(normalize(np.array(vector).reshape(1, -1))[0,:])        
        return np.array(vector, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])