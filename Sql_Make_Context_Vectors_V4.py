import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from functools import reduce
import operator
from collections import OrderedDict
import psycopg2


# This code creates a sql table for contexts with only the context vector being in the sql table
# Requires the word sql table to already be created

# User defined functions
def text_cleaner(text):
    punctuation = ("*", ".", "!", "?", ",", ":", ";", "'", '"', "-", "--", "@", "#", '$', "%", "^", "&", "+", "=",
                   ")", "(", "/", "\\", "-rrb-", "-lrb-")
    for items in punctuation:
        text = text.replace(items, " ")
    text = text.replace("\n", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text


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
        allThemeVClean = []
        for themes in allThemeClean:
            vClean = text_cleaner(themes)
            allThemeVClean.append((vClean, 'theme'))
        cleanThemes.append(allThemeVClean)

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
            simplifiedLocations.append(removeExtraLocationInfo.upper())  # making locations be upper case
        allLocationsVClean = []
        for locations in simplifiedLocations:
            vClean = text_cleaner(locations)
            allLocationsVClean.append((vClean, 'location'))
        cleanLocations.append(allLocationsVClean)

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
        allPersonsVClean = []
        for persons in separatePersons:
            vClean = text_cleaner(persons)
            allPersonsVClean.append((vClean, 'person'))
        cleanPersons.append(allPersonsVClean)

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
        allOrganizationsVClean = []
        for organizations in separateOrganizations:
            vClean = text_cleaner(organizations)
            allOrganizationsVClean.append((vClean, 'organization'))
        cleanOrganizations.append(allOrganizationsVClean)

    # combine information for each document
    allInfo = []
    for i in list(range(0, len(cleanThemes))):
        combineInfo = [*cleanThemes[i], *cleanLocations[i], *cleanPersons[i], *cleanOrganizations[i]]
        allInfo.append(combineInfo)
    return allInfo


# function to create flat list of all unique words
def get_all_words(wordAndContextMatrix):
    flattenContexts = reduce(operator.add, wordAndContextMatrix)
    return flattenContexts


# read in GKG data
rawGKG = pd.read_csv('Sample_GKG_Export.csv')

# Setting up the vectors for words and contexts
contextCorpusTags = clean_GKG_Data(rawGKG)  # Organizing data to be run through model
contextCorpus = [np.transpose(contextCorpusTags[i])[0] for i in list(range(0, len(contextCorpusTags)))]
length = 20000  # Setting vector length

#  Connecting to sql database
db_connection = psycopg2.connect(host="localhost", database="rampage", user="postgres", password="want2Thru-hike")
cur = db_connection.cursor()

#  Deleting sql table if one already exists with that name, comment out if table doesn't exist
cur.execute("DROP TABLE context_vectors_Sample_GKG_Export_tags")  # check table name before running
db_connection.commit()

#  Creating sql table
cur.execute("""
    CREATE TABLE context_vectors_Sample_GKG_Export_tags (
        id SERIAL PRIMARY KEY,
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
            cur.execute(f"SELECT vector FROM words_vectors_Sample_GKG_Export_tags WHERE word IN {document};")
        if isinstance(document, np.ndarray):
            document = tuple(document)  # changing list to tuple to be added to search sql table with word vectors
            # querying word sql table to find all vectors for the words in the context/document
            cur.execute(f"SELECT vector FROM words_vectors_Sample_GKG_Export_tags WHERE word IN {document};")
        if isinstance(document, tuple):
            # querying word sql table to find all vectors for the words in the context/document
            cur.execute(f"SELECT vector FROM words_vectors_Sample_GKG_Export_tags WHERE word IN {document};")
        else:
            # querying word sql table to find the vectors for the word in the context/document
            cur.execute(f"SELECT vector FROM words_vectors_Sample_GKG_Export_tags WHERE word = '{document}';")
        probe_words = cur.fetchall()  # getting all word vectors from query
        context_vector = np.sum(probe_words, axis=0)[0]  # summing the word vectors to get context vectors
        # adding context/document vector to context sql table
        cur.execute(f'INSERT INTO context_vectors_Sample_GKG_Export_tags (vector) VALUES (ARRAY {context_vector.tolist()});')
        cur.close()
        db_connection.commit()
    except:  # Makes it so that the code doesn't crash when there is an error adding the context vector to the table
        print(f"An error has occurred for context {document}")
        cur.close()
        db_connection.commit()
