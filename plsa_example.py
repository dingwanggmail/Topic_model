from plsa import Corpus, Pipeline, Visualize
from plsa.pipeline import DEFAULT_PIPELINE
from plsa.algorithms import PLSA

csv_file = '/disk2/wangding/plsa_data/Full-Economic-News-DFE-839861.csv'
pipeline = Pipeline(*DEFAULT_PIPELINE)

corpus = Corpus.from_csv(csv_file, pipeline)

n_topics = 5

plsa = PLSA(corpus, n_topics, True)

result = plsa.fit()

result = plsa.best_of(5)

visualize = Visualize(result)