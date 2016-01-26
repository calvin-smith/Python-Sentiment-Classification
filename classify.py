#Calvin Smith-L'Ecuyer

#Float division
from __future__ import division

import re, collections, sys
from collections import Counter
from itertools import izip


#3.1 Most-frequent Class (MFC) Baseline
def MFC (TRAIN_CLASSES, TEST_DOC):
	with open (TRAIN_CLASSES, "r") as trainClasses:
		counterMFC = Counter()
		mostFrequentClass = ""
		counterMFC = Counter(trainClasses.read().split())
		#Debug
		#for item in counterMFC.items():
			#print item
	
	for word, count in counterMFC.most_common(1):
		mostFrequentClass = word

	with open ('output.classes', 'w') as outputFile:
		with open (TEST_DOC, "r") as testDocument:
			for line in testDocument:
				outputFile.write(mostFrequentClass + "\n")

		outputFile.close()
	

def trainLexicon ():
	#Set up a lexicon that holds a list of all positive and negative words
	#Then set up a key that reads "positive" or "negative" given the word sentiment
	#Key is left as a string to allow easier tweaking of positive/negative values

	lexicon = {}
	with open ('pos-words.txt', 'r') as positiveWords:
		for line in positiveWords:
			line = line.rstrip()
			lexicon[line] = "positive"

	with open ('neg-words.txt', 'r') as negativeWords:
		for line in negativeWords:
			line = line.rstrip()
			lexicon[line] = "negative"

	#Debug
	#for word in lexicon:
	#	print "Word: %s \tValue: %s\n" % (word, lexicon[word])
	
	return lexicon

#3.2 Sentiment Lexicon Baseline
def lexicon(TEST_DOC):
	#Take the lexicon built by trainLexicon() and return the most common class every time
	lexicon = trainLexicon()
	reducedLexicon = {}
	counterLexicon = Counter()
	mostFrequentClass = ""

	# Iterate through the Positive/Negative keys in lexicon and flatten the dictionary, 
	# removing the associated word into reducedLexicon
	for key in lexicon:
		reducedLexicon = lexicon[key]

	# Count all Negative/Positive in reducedLexicon
	counterLexicon = Counter(reducedLexicon.split())

	#For Positive/Negative, make mostFrequentClass equal to the most common
	for word, count in counterLexicon.most_common(1):
		mostFrequentClass = word

	#Debug	
	#for word in lexicon:
		#print "Word: %s \tValue: %s\n" % (word, lexicon[word])

	#Iterate through the TEST_DOC and output the most common class for each line
	with open ('output.classes', 'w') as outputFile:
		with open (TEST_DOC, "r") as testDocument:
			for line in testDocument:
				outputFile.write(mostFrequentClass + "\n")

		outputFile.close()


#3.3 Naive Bayes
def naiveBayes (TRAIN_DOCS, TRAIN_CLASSES, TEST_DOC):

	#Generate dictionaries that hold the count of each word in each class
	nbDictNegative = collections.defaultdict(lambda: 0)
	nbDictNeutral = collections.defaultdict(lambda: 0)
	nbDictPositive = collections.defaultdict(lambda: 0)

	for trainDocsLine, trainClassesLine in izip(open(TRAIN_DOCS), open(TRAIN_CLASSES)):
		trainClassesLine = trainClassesLine.rstrip()

		if trainClassesLine == "negative":
			word = re.findall('[a-zA-Z#@]+', trainDocsLine)
			for item in word:
				nbDictNegative[item] += 1

		elif trainClassesLine == "neutral":
			word = re.findall('[a-zA-Z#@]+', trainDocsLine)
			for item in word:
				nbDictNeutral[item] += 1

		elif trainClassesLine == "positive":
			word = re.findall('[a-zA-Z#@]+', trainDocsLine)
			for item in word:
				nbDictPositive[item] += 1

	sumDictionaries = dict(nbDictNegative.items() + nbDictNeutral.items() + nbDictPositive.items())

	#Create Counters to store values that will be re-refrenced many times
	#vocabularySize = sum(1 for line in nbDictNegative)
	vocabularySize = sum(1 for line in sumDictionaries)
	negativeClassSize = sum(1 for line in nbDictNegative)
	neutralClassSize = sum(1 for line in nbDictNeutral)
	positiveClassSize = sum(1 for line in nbDictPositive)

	#Calculate probabilies for each class
	probOfNegativeClass = negativeClassSize / vocabularySize
	probOfNeutralClass = neutralClassSize / vocabularySize
	probOfPositiveClass = positiveClassSize / vocabularySize

	with open ('output.classes', 'w') as outputFile:
		with open (TEST_DOC, "r") as testDocument:
			for line in testDocument:
				word = re.findall('[a-zA-Z#@]+', line)
				
				lineNegative = 1
				lineNeutral = 1
				linePositive = 1

				for item in word:

					lineNegative = lineNegative * ((nbDictNegative[item] + 1)/(negativeClassSize + vocabularySize))
					lineNeutral = lineNeutral * ((nbDictNeutral[item] + 1)/(neutralClassSize + vocabularySize))
					linePositive = linePositive * ((nbDictPositive[item] + 1)/(positiveClassSize + vocabularySize))
				
				lineNegative = lineNegative * probOfNegativeClass
				lineNeutral = lineNeutral * probOfNeutralClass
				linePositive = linePositive * probOfPositiveClass
				#Debug
				#print lineNegative
				#print lineNeutral
				#print linePositive
				#print "\n"

				sentimentDictionary = {"negative": lineNegative, "neutral": lineNeutral, "positive": linePositive}
				#print sentimentDictionary
				sentiment = max (sentimentDictionary, key = sentimentDictionary.get)
				
				
					#print "Line: %r\nSentiment: %r\n" % (line, sentiment)
					#Debug
					#print "Word: %s\tNum: %s" % (item, (nbDictNegative[item] + nbDictNeutral[item] + nbDictPositive[item]))
					#print "Negative: %r" % ((nbDictNegative[item] + 1)/(negativeClassSize + vocabularySize))
					#print "Neutral: %r" % ((nbDictNeutral[item] + 1)/(neutralClassSize + vocabularySize))
					#print "Positive: %r" % ((nbDictPositive[item] + 1)/(positiveClassSize + vocabularySize))
					#print "\n"
				
				outputFile.write(sentiment + "\n")

	outputFile.close()
	
	#Debug
	#for word in nbDictPositive:
	#	print "Word: %s\tNumber: %s" % (word, nbDictNegative[word])


#3.4 Naive Bayes with Binary Features
def naiveBayesBinaryFeatures (TRAIN_DOCS, TRAIN_CLASSES, TEST_DOC):

	#Generate dictionaries that hold the count of each word in each class
	nbDictNegative = collections.defaultdict(lambda: 0)
	nbDictNeutral = collections.defaultdict(lambda: 0)
	nbDictPositive = collections.defaultdict(lambda: 0)

	for trainDocsLine, trainClassesLine in izip(open(TRAIN_DOCS), open(TRAIN_CLASSES)):
		trainClassesLine = trainClassesLine.rstrip()

		if trainClassesLine == "negative":
			word = re.findall('[a-zA-Z#@]+', trainDocsLine)
			for item in word:
				nbDictNegative[item] = 1

		elif trainClassesLine == "neutral":
			word = re.findall('[a-zA-Z#@]+', trainDocsLine)
			for item in word:
				nbDictNeutral[item] = 1

		elif trainClassesLine == "positive":
			word = re.findall('[a-zA-Z#@]+', trainDocsLine)
			for item in word:
				nbDictPositive[item] = 1

	sumDictionaries = dict(nbDictNegative.items() + nbDictNeutral.items() + nbDictPositive.items())

	#Create Counters to store values that will be re-refrenced many times
	#vocabularySize = sum(1 for line in nbDictNegative)
	vocabularySize = sum(1 for line in sumDictionaries)
	negativeClassSize = sum(1 for line in nbDictNegative)
	neutralClassSize = sum(1 for line in nbDictNeutral)
	positiveClassSize = sum(1 for line in nbDictPositive)

	#Calculate probabilies for each class
	probOfNegativeClass = negativeClassSize / vocabularySize
	probOfNeutralClass = neutralClassSize / vocabularySize
	probOfPositiveClass = positiveClassSize / vocabularySize

	with open ('output.classes', 'w') as outputFile:
		with open (TEST_DOC, "r") as testDocument:
			for line in testDocument:
				word = re.findall('[a-zA-Z#@]+', line)
				
				lineNegative = 1
				lineNeutral = 1
				linePositive = 1

				for item in word:

					lineNegative = lineNegative * ((nbDictNegative[item] + 1)/(negativeClassSize + vocabularySize))
					lineNeutral = lineNeutral * ((nbDictNeutral[item] + 1)/(neutralClassSize + vocabularySize))
					linePositive = linePositive * ((nbDictPositive[item] + 1)/(positiveClassSize + vocabularySize))
				
				lineNegative = lineNegative * probOfNegativeClass
				lineNeutral = lineNeutral * probOfNeutralClass
				linePositive = linePositive * probOfPositiveClass
				#Debug
				#print lineNegative
				#print lineNeutral
				#print linePositive
				#print "\n"

				sentimentDictionary = {"negative": lineNegative, "neutral": lineNeutral, "positive": linePositive}
				#print sentimentDictionary
				sentiment = max (sentimentDictionary, key = sentimentDictionary.get)

				#print "Line: %r\nSentiment: %r\n" % (line, sentiment)
				#Debug
				#print "Word: %s\tNum: %s" % (item, (nbDictNegative[item] + nbDictNeutral[item] + nbDictPositive[item]))
				#print "Negative: %r" % ((nbDictNegative[item] + 1)/(negativeClassSize + vocabularySize))
				#print "Neutral: %r" % ((nbDictNeutral[item] + 1)/(neutralClassSize + vocabularySize))
				#print "Positive: %r" % ((nbDictPositive[item] + 1)/(positiveClassSize + vocabularySize))
				#print "\n"
				
				outputFile.write(sentiment + "\n")

	outputFile.close()
	
	#Debug
	#for word in nbDictPositive:
	#	print "Word: %s\tNumber: %s" % (word, nbDictNegative[word])



#python classify.py METHOD TRAIN_DOCS TRAIN_CLASSES TEST_DOC
def classify ():

	TRAIN_DOCS = sys.argv[2]
	TRAIN_CLASSES = sys.argv[3]
	TEST_DOC = sys.argv[4]

	#METHOD
	if sys.argv[1] == "baseline":
		MFC(TRAIN_CLASSES, TEST_DOC)
	elif sys.argv[1] == "lexicon":
		lexicon(TEST_DOC)
	elif sys.argv[1] == "nb": 
		naiveBayes(TRAIN_DOCS, TRAIN_CLASSES, TEST_DOC)
	elif sys.argv[1] == "nbbin":
		naiveBayesBinaryFeatures(TRAIN_DOCS, TRAIN_CLASSES, TEST_DOC)


#Start helper function classify()
classify()