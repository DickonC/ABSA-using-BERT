import spacy

nlp = spacy.load('en_core_web_sm')

#applies the en_core_web_sm pipeline to "This T Shirt Sucks"
doc = nlp('These 100 T-Shirts sucks. Lebron James, born in 1980 in the US')

#Showing some of the tokens
#print(doc[0]," + ",doc[1])

# for token in doc:
#     print(token.text)

#creates a list of the tokens
tokens = [token.text for token in doc]
print(tokens)

#Lemmatizing tokens (taking words back to root form e.g. Has --> have or Birds --> Bird)
lemmas = [token.lemma_ for token in doc]
print(lemmas)

#Part of Speech, e.g. adjective, noun et.c
poses = [token.pos_ for token in doc]
print(poses)

#Named Entity Recognition e.g. recognising the name lebron james or the place US or the date 1980
print(doc.ents)

#Showing the entity with its text
for token in doc.ents:
    print(token.text, token.label_)


#better document visualisation
#run this part and go to http://127.0.0.1:5000
from spacy import displacy

#displacy.serve(doc, style='dep',port=5001)

#DEPENDENCY PARSING - understanding context between words

#displacy.render(doc, style='dep')

print("\n#########################\n")
for token in doc:
    print(token.dep_, token.text)