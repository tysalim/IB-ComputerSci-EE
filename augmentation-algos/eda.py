from textaugment import EDA
t = EDA(random_state=1)
output = t.synonym_replacement("John is going to town", top_n=10)
print(output)