import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from functools import reduce
import operator
from collections import OrderedDict
import psycopg2


# This code calculates the similarity between the echo content from two different probes
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
    probe_vectors = cur.fetchall() # getting all word vectors from query
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
        word = probe_vectors[j][0]  # taking the word for the probe
        for i in list(range(1, numContexts + 1)):  # going through each context/document
            cur = db_connection.cursor()
            # querying the context database for a single context vector
            cur.execute(f"SELECT vector FROM {context_database} WHERE id = {i};")
            one_context = cur.fetchone()  # outputs a single context vector
            cur.close()
            one_context = one_context[0]  # takes the context vector
            dot = np.dot(word, one_context)    # taking the dot product between the word vector and context vector
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


####################################################################################################################
# Connecting to sql database
db_connection = psycopg2.connect(host="localhost", database="rampage", user="postgres", password="want2Thru-hike")

# Echo content first probe
# Get corresponding vectors for probe words
probe1 = ['JOBS', 'EPU_ECONOMY']
probe_vectors1 = get_probe_vectors(probe1, 'words_vectors_Sample_GKG_Export_tags')  # check name of word table in sql database
# Calculate activation of each word in the probe to each context
contextActAll1 = get_context_activation(probe_vectors1, 'context_vectors_Sample_GKG_Export_tags')  # check name of context table in sql db
# Get echo content for probe
echoContent1 = get_echo_content(contextActAll1, 'context_vectors_Sample_GKG_Export_tags')  # check name of context table in sql db

# Echo content second probe
# Get corresponding vectors for probe words
probe2 = ['JOBS', 'POVERTY']
probe_vectors2 = get_probe_vectors(probe2, 'words_vectors_Sample_GKG_Export_tags')  # check name of word table in sql db
# Calculate activation of each word in the probe to each context
contextActAll2 = get_context_activation(probe_vectors2, 'context_vectors_Sample_GKG_Export_tags')  # check name of context table in sql db
# Get echo content for probe
echoContent2 = get_echo_content(contextActAll2, 'context_vectors_Sample_GKG_Export_tags')  # check name of context table in sql db

# Calculate similarity of echo contents
echoSim = get_echoContent_sim(echoContent1, echoContent2)
print(echoSim)