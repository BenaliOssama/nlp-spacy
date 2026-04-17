import spacy

# Load the model
nlp = spacy.load('en_core_web_sm')

# Text to tokenize
text = "Tokenize this sentence. And this one too."

# Process the text
doc = nlp(text)

# Print each token
for token in doc:
    print(token.text)
