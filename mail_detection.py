import pandas as pd
import string
import nltk 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


# Download the stopwords from NLTK
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('spam_ham_dataset.csv')

# Preprocess the text data
df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ' '))

# creates the stemmer object and an empty list to hold the processed text data
stemmer = PorterStemmer()
corpus = []

# creates a set of stop words from the NLTK library to filter out common words that may not be useful for classification
stop_words = set(stopwords.words('english'))

# iterates through each row of the DataFrame
for i in range(len(df)):
    text = df['text'].iloc[i].lower() # converts text to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).split() # removes punctuation and splits each word into an array
    text = [stemmer.stem(word) for word in text if word not in stop_words] # stems each word in the text so it is in its root form and filters out common words with little semantic meaning (stop words)
    text = ' '.join(text) # joins the words back together 
    corpus.append(text) # adds the processed text to the corpus list
    

# converts the labels in the DataFrame to numerical values (0 for 'ham' and 1 for 'spam') and stores them in a new column called 'label_num'
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
y = df.label_num

# splits the dataset into training and testing sets, with 80% of the data used for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# creates a Random Forest Classifier object and fits the model to the training data
clf = RandomForestClassifier(n_jobs = -1)

# fits the Random Forest Classifier model to the training data (X_train and y_train) n_jobs=-1 allows the model to use all available CPU cores for faster training.
clf.fit(X_train, y_train)

# evaluates the performance of the trained model on the test set (X_test and y_test) by calculating the accuracy score, which is the proportion of correctly classified instances in the test set. The score is printed to the console.
print(clf.score(X_test, y_test))

joblib.dump(clf, 'spam_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model and Vectorizer saved successfully!")