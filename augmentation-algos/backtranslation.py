src = "en" # source language of the sentence
to = "es" # target language
from textaugment import Translate
t = Translate(src="en", to="fr")
output = t.augment('In the afternoon, John is going to town')
print(output)