from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from IPython.display import clear_output
import pandas as pd
import numpy as np
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags, STOPWORDS
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from lbl2vec import Lbl2Vec
from lbl2vec import Lbl2TransformerVec
from sentence_transformers import SentenceTransformer
from transformers import AutoModel


class Textclassifier:
    """
    This class purpose is to classify text data with unsupervised lbl2vec method

    Attributes:
        data:
        classes:
        keywords_dict:
        _model:

    Methods:
        find_keywords: find keywords with applying simple clustering methods
        _keyword_selection_menu: ask user to link keywords with classes
        _tokenize:
        _generate_class_names:
    """
    def __init__(self, data: list, classes: list, keywords_dict: dict = None):
        """
        Textclassifier init method.

        :param data: list with texts to classify.
        :param classes: list with class names present in data.
        :param keywords_dict: dict with keywords for each class.
        """
        self.data = data
        self.classes = classes
        self.keywords_dict = keywords_dict

        self._model = None

    @staticmethod
    def _remove_stop_words(text: str, stop_words: list = STOPWORDS, min_len: int = 3, max_len: int = 15) -> str:
        """
        Clean text from stop words.

        :param text: text that is going to be pre-processed.
        :param stop_words: list with stop words.
        :param min_len: minimal length of words
        :param max_len: maximal length of words
        :return: string after removing stop words.
        """
        text_words = text.split()
        new_text = [word for word in text_words if word not in stop_words and max_len >= len(word) >= min_len]
        return " ".join(new_text)

    def find_keywords(self, model: str = "Kmeans", k_clusters: int = 30,
                      n_keywords: int = 10, random_seed: int = 123) -> None:
        """
        Propose keyword groups for current data. Cluster text data and find
        typical words for each cluster. Keyword sets found by this function will
        be used as definitions of classes in lbl2vec model

        :param model: Model used for text clustering, currently only support of "Kmeans".
        :param k_clusters: number of clusters
        :param n_keywords: number of keywords retrieved from cluster
        :param random_seed: randomization seed number
        :return: None
        """
        # remove stop words given by gensim library
        processed_text = []
        for text in self.data:
            processed_text.append(Textclassifier._remove_stop_words(text))

        # vectorize text list
        vectorizer = TfidfVectorizer(stop_words='english')
        data_vectorized = vectorizer.fit_transform(processed_text)

        # TODO: add kwargs for additional parameters of KMeans
        # fit model
        model = KMeans(n_clusters=k_clusters, init='k-means++', max_iter=300,
                       n_init=5, random_state=random_seed)
        model.fit(data_vectorized)

        # find closes words to centroid and use it as keywords for our classes
        terms = vectorizer.get_feature_names_out()
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]

        header = []
        for k in range(n_keywords):
            header.append(f'keyword_{k+1}')

        keywords = []
        for i in range(k_clusters):
            class_keywords = []
            for ind in order_centroids[i, :n_keywords]:
                class_keywords.append(terms[ind])
            keywords.append(class_keywords.copy())

        # ask user to link keywords with classes, if not suitable for any class
        # drop keywords otherwise keywords must be assigned to one of the classes
        self.keywords_dict = Textclassifier._keyword_selection_menu(keywords,
                                                                    self.classes,
                                                                    self.keywords_dict)

    @staticmethod
    def _keyword_selection_menu(keyword_list: list, classes: list,
                                classes_dict: dict = None) -> dict:
        """
        Ask user to link keywords with predefined classes.

        :param keyword_list: list with keywords
        :param classes: list with classes
        :param classes_dict: dict with already linked keywords
        :return: dict with keywords linked to each class
        """
        if classes_dict:
            collected_dict = classes_dict
            not_in_dict = [x for x in classes if x not in classes_dict.keys()]
            for clss in not_in_dict:
                collected_dict[clss] = []
        else:
            collected_dict = {}
            for clss in classes:
                collected_dict[clss] = []

        for keywords in keyword_list:
            # get user decision about keywords
            while True:
                print(keywords)
                print("Please choose one of these options:")
                print("0.) Drop keywords")
                for index, clss in enumerate(classes):
                    print(f'{index + 1}.) {clss}')
                try:
                    selection = int(input("> "))
                except ValueError:
                    clear_output(wait=True)
                    print("Please type one of the valid options AS a NUMBER. Let's try it again...")
                    continue
                if 0 <= selection <= len(classes):
                    clear_output(wait=True)
                    break
                else:
                    clear_output(wait=True)
                    print("Please type one of the valid options. Let's try it again...")
            if selection != 0:
                collected_dict[classes[selection - 1]].append(keywords)

        return collected_dict

    @staticmethod
    def _tokenize(doc):
        return simple_preprocess(strip_tags(doc), deacc=True, min_len=2, max_len=15)

    @staticmethod
    def _generate_class_names(number_classes: int, start_numbering: int = 0, confirmed_classes: bool = False) -> list:
        """
        Generate automatic labels for classes

        :param number_classes: number of classes that we have keywords for
        :param start_numbering: number where function will start class numbering
        :param confirmed_classes: True if we want to apply naming for already selected classes
        :return: list with class names
        """
        if confirmed_classes:
            name_prefix = "confirmed_class_"
        else:
            name_prefix = "class_"

        class_names = []
        for i in range(start_numbering, number_classes):
            class_names.append(name_prefix + str(i))

        return class_names

    def train_lbl2vec(self, keywords_list: list, class_names: list, pretrained: bool = False, epochs: int = 10) -> None:
        """
        Train lbl2vec model.

        :param keywords_list: list with keyword lists
        :param class_names: list with class names
        :param pretrained: True if we want to pretrain Doc2Vec model
        :param epochs: number of epochs for model training
        :return: None
        """
        data_train = []
        for index, text in enumerate(self.data):
            data_train.append(TaggedDocument(Textclassifier._tokenize(text), [str(index)]))

        if pretrained:
            doc2vec_model = Doc2Vec(documents=data_train, dbow_words=1, dm=0,
                                    epochs=epochs)
            # init lbl2vec model

            lbl2vec_model = Lbl2Vec(keywords_list=keywords_list, doc2vec_model=doc2vec_model,
                                    label_names=class_names, epochs=epoch, similarity_threshold=0.30, min_num_docs=100)
        else:
            # init lbl2vec model
            lbl2vec_model = Lbl2Vec(keywords_list=keywords_list, tagged_documents=data_train,
                                    label_names=class_names, epochs=epochs, similarity_threshold=0.30, min_num_docs=100)
        # train model
        lbl2vec_model.fit()
        self._model = lbl2vec_model

    def train_lbl2transformervec(self, keywords_list: list, class_names: list, transformer_model: str = None) -> None:
        """
        Train Lbl2TransformerVec model.

        :param keywords_list: list with keyword lists
        :param class_names: list with class names
        :param transformer_model: type of transformer-based language model used
            during training
        :return: None
        """

        if transformer_model == "SBERT":
            # select sentence-tranformers model
            transformer_model = SentenceTransformer("all-mpnet-base-v2")

            # init model
            lbl2transformervec_model = Lbl2TransformerVec(transformer_model=transformer_model,
                                                          keywords_list=keywords_list, documents=self.data,
                                                          label_names=class_names,
                                                          similarity_threshold=0.30, min_num_docs=100)

            # train model
            lbl2transformervec_model.fit()
            self._model = lbl2transformervec_model

        elif transformer_model == "SimCSE":
            # select SimCSE model
            transformer_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-base")

            # init model
            lbl2transformervec_model = Lbl2TransformerVec(transformer_model=transformer_model,
                                                          keywords_list=keywords_list, documents=self.data,
                                                          label_names=class_names,
                                                          similarity_threshold=0.30, min_num_docs=100)

            # train model
            lbl2transformervec_model.fit()
            self._model = lbl2transformervec_model

        else:
            # init model using the default transformer-embedding model ("sentence-transformers/all-MiniLM-L6-v2")
            lbl2transformervec_model = Lbl2TransformerVec(keywords_list=keywords_list, documents=self.data,
                                                          label_names=class_names,
                                                          similarity_threshold=0.30, min_num_docs=100)

            # train model
            lbl2transformervec_model.fit()
            self._model = lbl2transformervec_model

    def train_model(self, pretrained: bool = False, epochs: int = 10, transformer: bool = False,
                    transformer_model: str = None) -> None:
        """
        Perform classification with lbl2vec model using selected keywords.

        :param pretrained: True if it is desired to use pretrained model
        :param epochs: number of epochs to train model
        :param transformer: True if it is desired to use Lbl2TransformerVec
        :param transformer_model: if None transformer model will be trained from
            scratch, other options "SBERT" and "SimCSE"
        :return: None
        """
        # collapse dict into list
        keywords_list = []
        for clss in self.keywords_dict:
            keywords_list += self.keywords_dict[clss]

        # for purpose of model fitting create generic class labels (each keyword
        # list represents class for lbl2vec
        class_names = Textclassifier._generate_class_names(len(keywords_list))

        if transformer:
            self.train_lbl2transformervec(keywords_list, class_names, transformer_model)
        else:
            self.train_lbl2vec(keywords_list, class_names, pretrained, epochs)

    @staticmethod
    def _case_when(df: pd.DataFrame, col: str, conditions: list, values: list) -> np.ndarray:
        """
        Apply case when conditions to column.

        :param df: dataframe
        :param col: column name to apply case when conditions
        :param conditions: list with conditions
        :param values: list with result values
        :return: numpy ndrray with result of case when
        """
        current_condition = conditions[0]
        del conditions[0]
        current_value = values[0]
        del values[0]

        if conditions and values:
            return np.where(df[col] == current_condition,
                            current_value,
                            Textclassifier._case_when(df, col, conditions, values))
        else:
            return np.where(df[col] == current_condition,
                            current_value,
                            -1)

    def _link_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Join class names to data.

        :param df_classified: dataframe with calculated similarities for all keyword sets
        :return: dataframe with added classes
        """
        df_classified = df.copy()
        conditions = []
        values = []
        no_class = 0

        for index, clss in enumerate(self.keywords_dict):
            keywords = self.keywords_dict[clss]
            conditions.append(["class_" + str(x) for x in list(range(no_class, no_class + len(keywords)))])
            no_class += len(keywords)
            values.append([index + 1 for x in range(len(keywords))])

        conditions_adj = []
        values_adj = []
        for cond in conditions:
            conditions_adj += cond
        for val in values:
            values_adj += val

        df_classified["predicted_class"] = Textclassifier._case_when(df_classified,
                                                                     "most_similar_label",
                                                                     conditions_adj,
                                                                     values_adj)
        return df_classified

    def classify_data(self, data: list = None, transformer: bool = False) -> pd.DataFrame:
        """
        Classify text

        :param data: list with data, if not specified classification will be done
            on training data (self.data)
        :param transformer: True if model is transformer
        :return: DataFrame with prediction and calculated similarities
        """
        # collapse dict into list
        keywords_list = []
        for clss in self.keywords_dict:
            keywords_list += self.keywords_dict[clss]
        class_names = Textclassifier._generate_class_names(len(keywords_list))

        if not data:
            # in case we just want to classify data used for model training
            df_similarities = self._model.predict_model_docs()
        else:
            # prediction of new documents
            if transformer:
                df_similarities = self._model.predict_new_docs(documents=data)
            else:
                data_tokenized = []
                for index, text in enumerate(data):
                    data_tokenized.append(TaggedDocument(Textclassifier._tokenize(text), [str(index)]))

                df_similarities = self._model.predict_new_docs(tagged_docs=data_tokenized)

        # add original text into dataframe
        df_similarities['data'] = data
        ordered_cols = ['doc_key', 'data', 'most_similar_label', 'highest_similarity_score'] + class_names

        # add predicted_class column to dataframe
        return self._link_classes(df_similarities[ordered_cols])


if __name__ == "__main__":
    df_yahoo = pd.read_csv("data/test.csv", names=["class", "title", "question", "answer"])
    # exclude empty answers and one-word answers
    df_yahoo = df_yahoo[df_yahoo.answer.str.len() > 0]
    word_counts = df_yahoo["answer"].str.split().str.len()
    df_yahoo = df_yahoo[word_counts > 1]
    
    class_names = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference",
                   "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music",
                   "Family & Relationships", "Politics & Government"]
    
    # initialize classifier
    classifier = Textclassifier(data=df_yahoo["answer"].to_list(), classes=class_names)
    # identify keywords and link it with classes
    classifier.find_keywords(model="Kmeans", k_clusters=100, n_keywords=10, random_seed=123456)
    # classify whole dataset
    classifier.train_lbl2vec(epochs=5)
    df_classified = classifier.classify_data()
