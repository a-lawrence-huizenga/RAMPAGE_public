import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from functools import reduce
import operator
from collections import OrderedDict
import psycopg2
import matplotlib.pyplot as plt
from sklearn import manifold


# This code creates multidimensional scaling plots for different probes with word activations
# Outputs plots as pngs and outputs words with plot positions as csv
# This code requires the word sql table and context sql table to have been created from the other code files

# User defined functions

# This function finds the vectors in the sql table for the probe words
def get_probe_vectors(probe, word_database):  # probe can be a single word or list of words, database is sql table
    cur = db_connection.cursor()
    if isinstance(probe, list):
        probe = tuple(probe)  # changing probe list to tuple to be added to search sql table with word vectors
        # querying word sql table to find all vectors for the words in the probe
        cur.execute(f"SELECT vector FROM {word_database} WHERE word IN {probe};")
    if isinstance(probe, tuple):
        # querying word sql table to find all vectors for the words in the probe
        cur.execute(f"SELECT vector FROM {word_database} WHERE word IN {probe};")
    else:
        # querying word sql table to find vector for the word in the probe
        cur.execute(f"SELECT vector FROM {word_database} WHERE word = '{probe}';")
    probe_vectors = cur.fetchall()  # getting all word vectors from query
    cur.close()
    probe_vectors = np.array(probe_vectors)  # putting word vectors for the probe into an array
    return probe_vectors


# This function finds the number of contexts/documents in the context sql database
def get_num_contexts(context_database):  # context_database is a sql table
    cur = db_connection.cursor()
    cur.execute(f"SELECT count(*) FROM {context_database}")  # finds number of rows in database
    numContexts = int(cur.fetchall()[0][0])  # outputs number of rows in database
    cur.close()
    return numContexts


# This function calculates the activation of each context/document to the probe which can be a single word or multiple
def get_context_activation(probe_vectors, context_database):
    # probe_vectors are the vectors for the words in a single probe that can have multiple items
    # context_database is the sql table with context vectors
    numContexts = get_num_contexts(context_database)  # calculates number of contexts
    contextActAll = []
    for j in list(range(0, len(probe_vectors))):  # going through each probe vector
        contextAct1 = []
        word = probe_vectors[j][0]  # taking the word vector for the probe
        for i in list(range(1, numContexts + 1)):  # going through each context/document
            cur = db_connection.cursor()
            # querying the context database for a single context vector
            cur.execute(f"SELECT vector FROM {context_database} WHERE id = {i};")
            one_context = cur.fetchone()  # outputs a single context vector
            cur.close()
            one_context = one_context[0]  # takes the context vector
            dot = np.dot(word, one_context)  # taking the dot product between the word vector and context vector
            norm1 = np.linalg.norm(word)  # norming the word vector
            norm2 = np.linalg.norm(one_context)  # norming the context vector
            cos = dot / (norm1 * norm2)  # dividing dot product by norms multiplied together to get cosine similarity
            act = cos ** 3  # cubing cosine similarity
            contextAct1.append(act)  # appending the activation of context to the list of context activations for probe
        contextActAll.append(contextAct1)  # appending act of context to probe to all context activations for all probes
    return contextActAll


# This function returns the echo content for each probe
def get_echo_content(contextActAll, context_database):
    # contextActAll is context activation of each context to each item in a single probe
    numContexts = get_num_contexts(context_database)  # calculates number of contexts
    transposeContextActAll = np.transpose(contextActAll)  # transposes acts so grouped by context and not probe item
    probeTotalAct = np.prod(transposeContextActAll, axis=1)  # multiplies all acts to each probe item for each context
    WeightedTrace = []
    for i in list(range(0, numContexts)):  # going through each context/document
        cur = db_connection.cursor()  # connecting to sql database
        # querying sql database for each context
        cur.execute(f"SELECT vector FROM {context_database} WHERE id = {i+1};")
        one_context = cur.fetchone()  # outputs a single context vector
        cur.close()
        one_context = np.array(one_context[0])  # takes the context vector as array
        WeightedTrace1 = probeTotalAct[i] * one_context  # multiplies context vector by total context act for probe
        WeightedTrace.append(WeightedTrace1)  # adds weighted trace to list with all weighted traces for each context
    echoContent = np.sum(WeightedTrace, axis=0)  # sums weighted traces to get echo content for probe
    return echoContent


# Function to calculate the similarity of two echo contents
def get_echoContent_sim(echoContent1, echoContent2):  # reads in two echo contents
    dot = np.dot(echoContent1, echoContent2)  # taking the dot product between the two echo contents
    norm1 = np.linalg.norm(echoContent1)  # norming the first echo content
    norm2 = np.linalg.norm(echoContent2)  # norming the second echo content
    cos = dot / (norm1 * norm2)  # calculating cosine similarity
    echoSim = cos ** 3  # cubing cosine similarity to get echo similarity
    return echoSim


# Function to calculate activation of the echocontent of a probe to the word vectors to find word activations
def get_word_activation(echocontent, word_database, wordIDs):
    # reads in echo content calculated for a probe and the word database
    wordActAll = []
    for ID in wordIDs:  # going through each word
        cur = db_connection.cursor()
        # querying the word database for a single word vector
        cur.execute(f"SELECT vector FROM {word_database} WHERE id = {ID};")
        one_word = cur.fetchone()  # outputs a single word vector
        print(ID)
        print(one_word)
        cur.close()
        one_word = one_word[0]  # takes the word vector as array
        dot = np.dot(echocontent, one_word)  # taking the dot product between the echo content and word vector
        norm1 = np.linalg.norm(echocontent)  # norming the echo content
        norm2 = np.linalg.norm(one_word)  # norming the word vector
        cos = dot / (norm1 * norm2)  # dividing dot product by norms multiplied together to get cosine similarity
        act = cos ** 3  # cubing cosine similarity
        cur = db_connection.cursor()
        # querying word database for corresponding word
        cur.execute(f"SELECT word FROM {word_database} WHERE id = {ID};")
        word = cur.fetchone()  # outputs a single word
        cur.close()
        wordActAll.append([word[0], act])  # appends word activation and word to list of all words activations
    return wordActAll


#  Function to get activations of each word to the echocontent of the probe
def get_allword_activations(probe, context_database, word_database, wordIDs):
    # reads in a single probe which can contain multiple items and the name of the context database and word database
    probeVec = get_probe_vectors(probe, word_database)  # gets the vectors for words in the probe
    contextActAll = get_context_activation(probeVec, context_database)  # Calculate act of each context to each probe
    echoContent = get_echo_content(contextActAll, context_database)  # Determine echo content
    wordActivations = get_word_activation(echoContent, word_database, wordIDs)  # gets activation of each word to echo content
    wordActivations.sort(key=lambda x: x[1], reverse=True)  # sorts word activations by most activated
    return wordActivations


####################################################################################################################
# Connecting to sql database
db_connection = psycopg2.connect(host="localhost", database="rampage", user="postgres", password="want2Thru-hike")

# Reads in any number of sets of probes and writes a file with the activations of each word to the probe
probes = [['LEADER', 'EPU_ECONOMY'], ['EPU_POLICY', 'EPU_ECONOMY'], ['LEADER', 'EPU_POLICY', 'EPU_ECONOMY']]
cur = db_connection.cursor()
cur.execute(f"SELECT id FROM words_vectors_Sample_GKG_Export_tags;")
wordIDsMessy = cur.fetchall()
cur.close()
wordIDs = list(map(lambda x: x[0], wordIDsMessy))

# get all activations of each probe to each word
all_activations = []
for probe in probes:  # going through one probe at a time
    # calculating the activation of each word to the probe
    all_activations1 = [probe, get_allword_activations(probe, 'context_vectors_Sample_GKG_Export_tags',
                                                       'words_vectors_Sample_GKG_Export_tags', wordIDs)]
    all_activations.append(all_activations1)  # appending activations of each probe for each word together
cur.close()

# Get best n activated words to each probe
n = 10  # setting number of words to take for plots and tables
justWords = []
for j in list(range(0, len(all_activations))):  # going through each probes
    for i in list(range(0, n)):  # going through the first n activations for each probe
        # taking the word value and appending it to list of most activated words
        justWords.append(all_activations[j][1][i][0])
activatedWords = list(set(justWords))  # removing duplicated words from list of most activated words

# Calculate echo content for most activated words
# gets vectors for activated words
actWordVectors = get_probe_vectors(activatedWords, 'words_vectors_Sample_GKG_Export_tags')
actEchoContent = []
for words in actWordVectors:  # going through each word in activated words
    # Calculate activation of each context to each word
    contextActAll = get_context_activation([words], 'context_vectors_Sample_GKG_Export_tags')
    echoContent = get_echo_content(contextActAll, 'context_vectors_Sample_GKG_Export_tags')  # Determine echo content
    actEchoContent.append(echoContent)  # appending echo content for each word to list for all words

# Generate position of words in scatter based on MDS of echo sim matrix
# Calculating similarity of echo contents of all activated words with each other to make similarity matrix
simMatrix = []
for item in actEchoContent:  # going through echo content of each activated word
    simMatrix1 = []
    for word in actEchoContent:  # going through echo content of each activated word
        echoSim = get_echoContent_sim(word, item)  # calculating echo similarity of activated words
        simMatrix1.append(1-echoSim)  # appending 1 minus echo sim to list of echo sim for single word
    simMatrix.append(simMatrix1)  # appending echo sim of each word to each other word
# Calculating mds
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(simMatrix).embedding_  # outputs positions in mds

# Activations for each probe to most activated words
activationsAllProbes = []
for probe in all_activations:  # goes through activations for each probe to each word
    activationsforWordsinList = []
    for words in activatedWords:  # goes through each word in the set of activated words
        for items in probe[1]:  # goes through each activations of each word to the probe
            if items[0] == words:  # appends the activation of that word to the probe if in the list of activated words
                activationsforWordsinList.append(items[1])
    #  norming activation of words to each probe so that they are in the range 0 to 1
    normed_activationsforWords = []
    for item in activationsforWordsinList:  # going through activations to probe of each activated word
        # norming the activations
        normed_activations = (item - min(activationsforWordsinList))/(max(activationsforWordsinList)
                                                                      - min(activationsforWordsinList))
        normed_activationsforWords.append(normed_activations)  # adding normed activations to list of all normed acts
    activationsAllProbes.append(normed_activationsforWords)  # adding normed acts to  probe to list for all probes

#Write positions and activations to csv
for i in list(range(0, len(activatedWords))):  # going through each word in the list of activated words
    word_pos = np.append(activatedWords[i], pos[i])  # appending the position of word to the word
    probeActs = []
    for probe in activationsAllProbes:  # going through activations of each probes activations
        act = probe[i]  # getting the activation
        probeActs.append(act)  # adding activation to list of activations
    word_info = np.append(word_pos, probeActs)  # adding activation to the word and position
    # Adding columns names to dataframe... these don't currently work in the csv
    columNames = ['Word', 'X', 'Y']
    for item in probes:
        columNames.append(f'{item}')
    # creating a dataframe with word, position, and activation to each probe
    df_pos = pd.DataFrame([word_info], columns=columNames)
    # exporting the data frame to a csv file
    df_pos.to_csv('position_most_activated_Sample_GKG_Export_tags.csv', index=False, header=False, mode='a')  # check name

# Making plots based on MDS positions and activations for each probe set
for i in list(range(0, len(probes))):  # going through each probe
    filename = 'MDS_Plot_Sample_GKG_Export_tags'  # change file name for data set
    for item in probes[i]:  # adds probe words to file name
        filename = str(filename) + '_' + str(item)
    filename = filename + '.png'
    fig = plt.figure()  # creates figure for plot
    # creates  mds plot, c is color, s is size, alpha is opacity, cmap is color scale to use
    plt.scatter(pos[:, 0], pos[:, 1], c=activationsAllProbes[i], s=100, lw=0, alpha=.75, cmap='seismic')
    plt.colorbar()
    #plt.show()
    fig.savefig(filename)  # saves plot to file
