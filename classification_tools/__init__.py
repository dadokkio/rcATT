##########################################################
#            BASIC CLASSIFICATION FUNCTIONS              #
##########################################################
# rcATT is a tool to prediction tactics and techniques
# from the ATT&CK framework, using multilabel text
# classification and post processing.
# Version:    1.00
# Author:     Valentine Legoy
# Date:       2019_10_22
# Important global constants and functions for
# classifications: training and prediction.
import ast
import joblib
import configparser
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectPercentile

from nltk.corpus import stopwords

import classification_tools.preprocessing as prp
import classification_tools.postprocessing as pop

##########################################################
#       LABELS AND DATAFRAME LISTS AND RELATIONSHIP      #
##########################################################

config = configparser.ConfigParser()
config.read("classification_tools/rcatt.ini")

TEXT_FEATURES = config["VARIABLES"]["TEXT_FEATURES"].split(",")
CODE_TACTICS = config["VARIABLES"]["CODE_TACTICS"].split(",")
NAME_TACTICS = config["VARIABLES"]["NAME_TACTICS"].split(",")
CODE_TECHNIQUES = config["VARIABLES"]["CODE_TECHNIQUES"].split(",")
NAME_TECHNIQUES = config["VARIABLES"]["NAME_TECHNIQUES"].split(",")
STIX_IDENTIFIERS = config["VARIABLES"]["STIX_IDENTIFIERS"].split(",")
TACTICS_TECHNIQUES_RELATIONSHIP_DF = ast.literal_eval(
    config["VARIABLES"]["RELATIONSHIP"]
)
ALL_TTPS = config["VARIABLES"]["ALL_TTPS"].split(",")

TRAINING_FILE = config["PATH"]["TRAINING_FILE"]
ADDED_FILE = config["PATH"]["ADDED_FILE"]


##########################################################
#             RETRAIN AND PREDICT FUNCTIONS              #
##########################################################


def train(cmd):
    """
    Train again rcATT with a new dataset
    """

    # stopwords with additional words found during the development
    stop_words = stopwords.words("english")
    new_stop_words = [
        "'ll",
        "'re",
        "'ve",
        "ha",
        "wa",
        "'d",
        "'s",
        "abov",
        "ani",
        "becaus",
        "befor",
        "could",
        "doe",
        "dure",
        "might",
        "must",
        "n't",
        "need",
        "onc",
        "onli",
        "ourselv",
        "sha",
        "themselv",
        "veri",
        "whi",
        "wo",
        "would",
        "yourselv",
    ]
    stop_words.extend(new_stop_words)

    # load all possible data
    train_data_df = pd.read_csv(TRAINING_FILE, encoding="ISO-8859-1")
    train_data_added = pd.read_csv(ADDED_FILE, encoding="ISO-8859-1")
    train_data_df.append(train_data_added, ignore_index=True)

    train_data_df = prp.processing(train_data_df)

    print(train_data_df.head())
    print(train_data_df.columns, train_data_df.index)

    reports = train_data_df[TEXT_FEATURES]
    tactics = train_data_df[CODE_TACTICS]
    techniques = train_data_df[CODE_TECHNIQUES]

    if cmd:
        pop.print_progress_bar(0)

    # Define a pipeline combining a text feature extractor with multi label classifier for tactics prediction
    pipeline_tactics = Pipeline(
        [
            ("columnselector", prp.TextSelector(key="processed")),
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=prp.LemmaTokenizer(), stop_words=stop_words, max_df=0.90
                ),
            ),
            ("selection", SelectPercentile(chi2, percentile=50)),
            (
                "classifier",
                OneVsRestClassifier(
                    LinearSVC(
                        penalty="l2",
                        loss="squared_hinge",
                        dual=True,
                        class_weight="balanced",
                    ),
                    n_jobs=1,
                ),
            ),
        ]
    )

    # train the model for tactics
    pipeline_tactics.fit(reports, tactics)

    if cmd:
        pop.print_progress_bar(2)

    # Define a pipeline combining a text feature extractor with multi label classifier for techniques prediction
    pipeline_techniques = Pipeline(
        [
            ("columnselector", prp.TextSelector(key="processed")),
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=prp.StemTokenizer(),
                    stop_words=stop_words,
                    min_df=2,
                    max_df=0.99,
                ),
            ),
            ("selection", SelectPercentile(chi2, percentile=50)),
            (
                "classifier",
                OneVsRestClassifier(
                    LinearSVC(
                        penalty="l2",
                        loss="squared_hinge",
                        dual=False,
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                    n_jobs=1,
                ),
            ),
        ]
    )

    # train the model for techniques
    pipeline_techniques.fit(reports, techniques)

    if cmd:
        pop.print_progress_bar(4)

    pop.find_best_post_processing(cmd)

    # Save model
    joblib.dump(pipeline_tactics, "classification_tools/data/pipeline_tactics.joblib")
    joblib.dump(
        pipeline_techniques, "classification_tools/data/pipeline_techniques.joblib"
    )


def predict(report_to_predict, post_processing_parameters):
    """
    Predict tactics and techniques from a report in a txt file.
    """

    # loading the models
    pipeline_tactics = joblib.load("classification_tools/data/pipeline_tactics.joblib")
    pipeline_techniques = joblib.load(
        "classification_tools/data/pipeline_techniques.joblib"
    )

    report = prp.processing(pd.DataFrame([report_to_predict], columns=["Text"]))[
        TEXT_FEATURES
    ]

    # predictions
    predprob_tactics = pipeline_tactics.decision_function(report)
    pred_tactics = pipeline_tactics.predict(report)

    predprob_techniques = pipeline_techniques.decision_function(report)
    pred_techniques = pipeline_techniques.predict(report)

    if post_processing_parameters[0] == "HN":
        # hanging node thresholds retrieval and hanging node performed on predictions if in parameters
        pred_techniques = pop.hanging_node(
            pred_tactics,
            predprob_tactics,
            pred_techniques,
            predprob_techniques,
            post_processing_parameters[1][0],
            post_processing_parameters[1][1],
        )
    elif post_processing_parameters[0] == "CP":
        # confidence propagation performed on prediction if in parameters
        pred_techniques, predprob_techniques = pop.confidence_propagation(
            predprob_tactics, pred_techniques, predprob_techniques
        )

    return pred_tactics, predprob_tactics, pred_techniques, predprob_techniques
