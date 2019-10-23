import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from functools import reduce
import operator
from collections import OrderedDict
import psycopg2


# This code creates a sql table for contexts with both the words from the context
# and the context vector being in the sql table
# Requires the word sql table to already be created

# User defined functions
# function to read in raw GKG data and organize it into a list of words for each document
def clean_GKG_Data(datafile):
    # replaces missing data with 'NA'
    noMissingGKG = rawGKG.fillna('NA')

    # clean theme data
    justThemes = noMissingGKG['Themes']  # taking information from themes column
    cleanThemes = []
    for doc in justThemes:  # going through each document (row) in the GKG file
        if doc != 'NA':
            separateTheme = doc.split(';')  # making themes separated by ; into separate items
        else:
            separateTheme = ['']
        allThemeClean = []
        for themes in separateTheme:
            noDigits = ''.join([i for i in themes if not i.isdigit()])  # removing the numbers from themes
            noWB = noDigits.replace('WB__', '')  # removing the additional string from themes
            allThemeClean.append(noWB.upper())  # making themes be upper case
        cleanThemes.append(allThemeClean)

    # clean location data
    justLocations = noMissingGKG['Locations']  # taking information from locations column
    cleanLocations = []
    for doc in justLocations:  # going through each document (row) in the GKG file
        if doc != 'NA':
            separateLocations = doc.split(';')  # making locations separated by ; into separate items
        else:
            separateLocations = ['']
        simplifiedLocations = []
        for locations in separateLocations:  # going through individual locations
            if len(locations) >= 2:  # some locations have redundant information
                removeExtraLocationInfo = locations.split('#')[1]  # taking only information after the first #
            else:
                removeExtraLocationInfo = locations
            simplifiedLocations.append(removeExtraLocationInfo.upper()) # making locations be upper case
        cleanLocations.append(simplifiedLocations)

    # Clean person data
    justPersons = noMissingGKG['Persons']  # taking information from persons column
    cleanPersons = []
    for doc in justPersons:  # going through each document (row) in the GKG file
        if doc != 'NA':
            separatePersonsLower = doc.split(';')  # making persons separated by ; into separate items
        else:
            separatePersonsLower = ['']
        separatePersons = []
        for persons in separatePersonsLower:
            separatePersons.append(persons.upper())  # making persons be upper case
        cleanPersons.append(separatePersons)

    justOrganizations = noMissingGKG['Organizations']  # taking information from organizations column
    cleanOrganizations = []
    for doc in justOrganizations:  # going through each document (row) in the GKG file
        if doc != 'NA':
            separateOrganizationsLower = doc.split(';')  # making organizations separated by ; into separate items
        else:
            separateOrganizationsLower = ['']
        separateOrganizations = []
        for Organizations in separateOrganizationsLower:
            separateOrganizations.append(Organizations.upper())  # making organizations be upper case
        cleanOrganizations.append(separateOrganizations)

    # combine information for each document
    allInfo = []
    for i in list(range(0, len(cleanThemes))):
        combineInfo = [*cleanThemes[i], *cleanLocations[i], *cleanPersons[i], *cleanOrganizations[i]]
        allInfo.append(combineInfo)
    return allInfo


# read in GKG data
rawGKG = pd.read_csv('C:/Users/amlh3/Desktop/GDELT_GKG_Movement_3_16.csv')

# Setting up the vectors for words and contexts
contextCorpus = clean_GKG_Data(rawGKG)  # Organizing data to be run through model
length = 20000  # Setting vector length

#  Connecting to sql database
db_connection = psycopg2.connect(host="localhost", database="rampage", user="postgres", password="want2Thru-hike")
cur = db_connection.cursor()

# Deleting sql table if one already exists with that name, comment out if table doesn't exist
cur.execute("DROP TABLE context_vectors_GDELT_GKG_Movement_3_16")  # check table name before running
db_connection.commit()

#  Creating sql table that saves id, words in context, context vector
cur.execute("""
    CREATE TABLE context_vectors_GDELT_GKG_Movement_3_16 (
        id SERIAL PRIMARY KEY,
        themes VARCHAR(255) ARRAY NOT NULL,
        vector DOUBLE PRECISION ARRAY NOT NULL
    );""")
cur.close()
db_connection.commit()

# Populating sql table with vectors for each context/document in GKG file based on word vectors in another sql table
for document in contextCorpus:
    try:
        cur = db_connection.cursor()
        if isinstance(document, list):
            document = tuple(document)  # changing list to tuple to be added to search sql table with word vectors
            # querying word sql table to find all vectors for the words in the context/document
            cur.execute(f"SELECT vector FROM words_vectors_GDELT_GKG_Movement_3_16 WHERE word IN {document};")
        if isinstance(document, tuple):
            # querying word sql table to find all vectors for the words in the context/document
            cur.execute(f"SELECT vector FROM words_vectors_GDELT_GKG_Movement_3_16 WHERE word IN {document};")
        else:
            # querying word sql table to find the vectors for the word in the context/document
            cur.execute(f"SELECT vector FROM words_vectors_GDELT_GKG_Movement_3_16 WHERE word = '{document}';")
        probe_words = cur.fetchall()  # getting all word vectors from query
        context_vector = np.sum(probe_words, axis=0)[0]  # summing the word vectors to get context vectors
        # adding words/themes in context and context/document vector to context sql table
        cur.execute(f'INSERT INTO context_vectors_GDELT_GKG_Movement_3_16 (themes, vector) VALUES (ARRAY {list(document)},'
                    f' ARRAY {context_vector.tolist()});')
        cur.close()
        db_connection.commit()
    except: # Makes it so that the code doesn't crash when there is an error adding the context vector to the table
        print(f"An error has occurred for context {document}")
        cur.close()
        db_connection.commit()

# Test code
# cur = db_connection.cursor()
# cur.execute("SELECT * FROM context_vectors_Sample_GKG_Export WHERE 'LEADER' = ANY(themes);")
# test = cur.fetchall()
# cur.close()
# print(test)
