import nltk
from textaugment import Wordnet

t = Wordnet()
output = t.augment('In the afternoon, John is going to town')
print(output)