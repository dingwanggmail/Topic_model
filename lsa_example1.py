__author__ = 'dyami'

import pandas as pd
import numpy as np

corpus = {'c1': 'Human machine interface for ABC computer applications',
          'c2': 'A survey of user opinion of computer system response time',
            'c3': 'The EPS user interface management system',
            'c4': 'System and human system engineering testing of EPS',
            'c5': 'Relation of user perceived response time to error measurement',
            'm1': 'The generation of random, binary, ordered trees',
            'm2': 'The intersection graph of paths in trees',
            'm3': 'Graph minors IV: Widths of trees and well-quasi-ordering',
            'm4': 'Graph minors: A survey'}

bow_dict = {}

def generate_counts(s):
    mydict = {}
    for j in s.split(' '):
        mys = j.strip(' ').strip(':').lower()
        if mys in mydict:
            mydict[mys] += 1
        else:
            mydict[mys] = 1
    return mydict

for k,v in corpus.items():
    bow_dict[k] = generate_counts(v)

bow_df = pd.DataFrame(bow_dict).fillna(0)

stop_words = {'and', 'a', 'the', 'in', 'of', 'on'}
bow_list = [w for w in bow_df.index if w not in stop_words and bow_df.ix[w].sum() > 1]
bow_list = ['human', 'interface', 'computer', 'user', 'system', 'response', 'time', 'eps', 'survey', 'trees', 'graph', 'minors']

td_matrix_df = bow_df.ix[bow_list]

U,Sigma,V_T = np.linalg.svd(td_matrix_df)

pd.options.display.float_format = '{:.2f}'.format
pd.options.display.width = 180
U_df = pd.DataFrame(U, index=bow_list, columns=['concept'+str(i) for i in xrange(U.shape[1])])
V_T_df = pd.DataFrame(V_T, columns = td_matrix_df.columns, index=['topic'+str(i) for i in xrange(V.shape[1])])
Sigma_matrix = np.zeros([U.shape[1], V_T.shape[0]])
Sigma_matrix[0:len(Sigma), 0:len(Sigma)] = np.diag(Sigma)
Sigma_df = pd.DataFrame(Sigma_matrix)

#test SVD
k=10; pd.DataFrame(U[:, :k].dot(Sigma_matrix[:k,:]).dot(V_T))
k=9; pd.DataFrame(np.linalg.inv(Sigma_matrix[:k,:k]).dot(U[:, :k].T.dot(td_matrix_df.values)))


#XX.T
pd.DataFrame(td_matrix_df.values.dot(td_matrix_df.values.transpose()))
#Sigma'
pd.DataFrame(Sigma_matrix.dot(Sigma_matrix.T))

#new doc
doc_cm = 'The difference between how a computer thinks about trees and how human think about trees'
doc_cm_dict = {}
for i in doc_cm.lower().split():
    if i in bow_list:
        if i in doc_cm_dict:
            doc_cm_dict[i] += 1
        else:
            doc_cm_dict[i] = 1

doc_cm_df = pd.DataFrame(doc_cm_dict, index=['cm']).T.ix[bow_list].fillna(0)#term coordinates
doc_cm_concept_df = U_df.T.dot(doc_cm_df)#concepts coordinates

k=9; pd.DataFrame(np.linalg.inv(Sigma_matrix[:k,:k]).dot(U[:, :k].T.dot(doc_cm_df.values)))

k=9; pd.DataFrame(U[:, :k].dot(Sigma_matrix[:k,:]).dot(V_T))

k = td_matrix_df.shape[1]#k = m

doc_cm_concept_df_k = doc_cm_concept_df[:k]

Sigma_inv = (Sigma**-1)[:k]
Sigma_inv_matrix = np.diag(Sigma_inv)
Sigma_inv_df = pd.DataFrame(Sigma_inv_matrix)


