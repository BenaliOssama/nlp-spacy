import spacy

# Load the MEDIUM model (has word vectors)
nlp = spacy.load('en_core_web_md')

# Process the word "car"
doc = nlp("car")

# Get the embedding for the first token
car_vector = doc[0].vector

# Print shape
print("Shape:", car_vector.shape)

# Sum first 20 values
sum_first_20 = car_vector[:20].sum()
print("Sum of first 20:", sum_first_20.as_integer_ratio())
