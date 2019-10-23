from __future__ import print_function
import sklearn
# Import all of the scikit learn stuff
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
import pandas as pd
import numpy as np

# function to get unique values
def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
            # print list
    return unique_list

def text_cleaner(text):
    for items in punctuation:
        text = text.replace(items, " ")
    text = text.replace("\n"," ")
    return text


def create_similarity_matrix(textData):
    vectorizer = CountVectorizer(min_df=1, stop_words='english')
    dtm = vectorizer.fit_transform(textData)

    # Fit LSA. Use algorithm = “randomized” for large datasets
    if dtm.shape[1] <= 20:
        ncomp = dtm.shape[1]-1
    else:
        ncomp = 20
    lsa = TruncatedSVD(ncomp)
    dtm_lsa = lsa.fit_transform(dtm)
    dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)

    # Compute document similarity using LSA components
    similarity = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T)
    simMatrix = pd.DataFrame(similarity)
    # simMatrix.to_csv(filename, index=True, header=False, mode='a')
    return simMatrix

punctuation = ("*",".","!","?",",",":",";","'",'"',"-","--","@","#",'$',"%","^","&","+","=","_",")","(","/","\\","-rrb-","-lrb-")


# setting up variables for loop
filename = 'Cycle2_Sim.csv'
analysts = ['B', 'C', 'E', 'F']
# analysts = ['A', 'C', 'D', 'E', 'F']

hypotheses_df = pd.read_csv('C:/Users/ashley/desktop/Cycle2_Hypotheses.csv')
problems = unique(list(hypotheses_df['Problem']))

for analyst in analysts:
    single_analyst_df = hypotheses_df[hypotheses_df['Analyst'] == analyst]
    print(analyst)
    for problem in problems:
        single_problem_df = single_analyst_df[single_analyst_df['Problem'] == problem]
        single_problem_hyp = list(single_problem_df['Content'])

        cleaned_hyp = []
        for hyp in single_problem_hyp:
            cleaned_hyp.append(text_cleaner(hyp))
        print(problem)
        sim_matrix = create_similarity_matrix(cleaned_hyp)
        lower_sim_matrix = np.tril(sim_matrix)

        sim_list = []
        for i in list(range(0, len(lower_sim_matrix))):
            for j in list(range(0, len(lower_sim_matrix))):
                if i != j and lower_sim_matrix[i][j] != 0:
                    sim_list.append(lower_sim_matrix[i][j])
        problem_out = [analyst, problem, np.mean(sim_list), len(lower_sim_matrix)]
        output_df = pd.DataFrame([problem_out])
        output_df.to_csv(filename, index=False, header=False, mode='a')
