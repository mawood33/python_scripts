import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

d1 = "detected at Det L(37.0, 127.0) Between the bearings of 293 and 294"
d2 = "DETECTED AT DET L(37.0, 127.0) BETWEEN THE BEARINGS OF 105 AND 110"
d1 = "I thought, I thought of thinking of thanking you for the gift"
d2 = "She was going to go and going to thinking of going to go and get you a GIFT!"

# Create dataframe
X_train = pd.DataFrame({'text': [d1, d2]})

def preprocess_text(text):
    # Tokenize words while ignoring punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Lowercase and lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token.lower(), pos='v') for token in tokens]

    # Remove stopwords
    keywords = [lemma for lemma in lemmas if lemma not in stopwords.words('english')]
    return keywords

### Vectorize the corpus to tf-idf using TfidfVectorizer
###   - put data in format acceptable for a machine learning model
# Create an instance of TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer=preprocess_text)

# Fit to the data and transform to feature matrix
X_train = vectorizer.fit_transform(X_train['text'])

# Convert sparse matrix to dataframe
X_train = pd.DataFrame.sparse.from_spmatrix(X_train)

# Save mapping on which index refers to which words
col_map = {v:k for k, v in vectorizer.vocabulary_.items()}

# Rename each column using the mapping
for col in X_train.columns:
    X_train.rename(columns={col: col_map[col]}, inplace=True)
print(X_train)

### See how test docs with or without unseen terms get transformed
d3 = "He thinks he will go!"
d4 = "They don't know what to buy!"
# When preprocess_text is applied, the test docs will transform into:
#   d3 = ['think', 'go']    # vectorizer is familiar with these terms
#   d4 = ['know', 'buy']    # vectorizer is not familiar with these terms
# 'go' is weighted up relative to 'think'
# Number of terms in matrix is dependent on training data much like any other sklearn transformers in general

# Create dataframe
X_test = pd.DataFrame({'text': [d3, d4]})

# Transform to feature matrix
X_test = vectorizer.transform(X_test['text'])

# Convert sparse matrix to dataframe
X_test = pd.DataFrame.sparse.from_spmatrix(X_test)

# Add column names to make it more readable
for col in X_test.columns:
    X_test.rename(columns={col: col_map[col]}, inplace=True)
print(X_test)
