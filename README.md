# Language Detection Project

This project aims to create a model that can detect the language of a given text. The model is built using Python, and leverages Natural Language Processing (NLP) techniques and the Multinomial Naive Bayes algorithm.

## Project Structure

- `langdet_ap24.csv`: The dataset containing text samples and their corresponding languages.
- `language_detection.py`: The script for training the language detection model.
- `lang_detection_model.pkl`: The trained Naive Bayes model saved using pickle.
- `lang_detection_vector.pkl`: The TfidfVectorizer used to transform the text data, saved using pickle.
- `README.md`: Documentation for the project.

## Dependencies

To run this project, you need the following Python libraries:

- pandas
- nltk
- scikit-learn
- pickle

## You can install these dependencies using pip:


 pip install pandas nltk scikit-learn
## Data Preparation
1.Load the dataset from langdet_ap24.csv.
2.Check for and handle any null data.
3.Clean the text data by:
4.Converting text to lowercase.
Removing punctuation.
Tokenizing the text.
Add a new column Clean_Text to the dataset for the cleaned text.
## Text Vectorization
Transform the cleaned text data into numerical features using TfidfVectorizer.

## Model Training
1.Split the data into training and testing sets.
2.Train a Multinomial Naive Bayes model using the training set.
3.Evaluate the model using the testing set and print a classification report.
## Saving the Model
Save the trained model and the TfidfVectorizer using pickle for future use.

## How to Run the Project
1.Ensure all dependencies are installed.
2.Run the language_detection.py script to train the model and save the artifacts.
3.Use the saved lang_detection_model.pkl and lang_detection_vector.pkl for inference on new text data.
4.Example Usage
Here is an example of how to load the model and vectorizer for predicting the language of a new text:
import pickle
from nltk import word_tokenize
from string import punctuation

### Load the saved model and vectorizer
with open('lang_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('lang_detection_vector.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

### Function to clean and prepare text
def clean_data(txt):
    txt = txt.lower()
    txt = txt.replace('"', "")  
    txt = word_tokenize(txt)
    txt = [t for t in txt if t not in punctuation]  
    txt = "".join(txt)  
    return txt

### New text sample
new_text = "Your text here"
cleaned_text = clean_data(new_text)
vectorized_text = vectorizer.transform([cleaned_text])

### Predict the language
predicted_language = model.predict(vectorized_text)
print(predicted_language)
