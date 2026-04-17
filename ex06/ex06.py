import spacy

# Load model
nlp = spacy.load('en_core_web_sm')

# Read the file
with open('data/news_amazon.txt', 'r') as f:
    text = f.read()

# Process the text
doc = nlp(text)

# Find all sentences mentioning "Bezos" with POS tag "NNP"
print("=== Sentences mentioning Bezos (NNP) ===\n")

for sent in doc.sents:
    # Check if "Bezos" is in the sentence
    if any(token.text == "Bezos" for token in sent):
        # Find the Bezos token and check its POS tag
        for token in sent:
            if token.text == "Bezos" and token.tag_ == "NNP":
                print(f"INFO:  {token.text} {token.pos_} {token.tag_}")
                print(f"Sentence:  {sent.text}")
                print()
