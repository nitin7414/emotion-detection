# import spacy
# from spacy.lang.en.stop_words import STOP_WORDS
# from nltk.stem.porter import *
# from nltk.stem.snowball import SnowballStemmer
# porter_stemmer = PorterStemmer()
# snowball_stemmer = SnowballStemmer("english")
# # print(porter_stemmer.stem("english"))
# # print(snowball_stemmer.stem("fastest"))
# nlp = spacy.load('en_core_web_md')
# doc = nlp(u'I am learning how to build chatbots')
# bui, cb = doc[5], doc[6]
# # print(list(bui.ancestors), list(cb.ancestors))
# # for token in doc:
# #   print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
# #  token.shape_, token.is_alpha, token.is_stop)

# # doc = nlp(u"What are some places to visit in Berlin and stay in Lubeck")
# # places = [doc[7], doc[11]]
# # actions = [doc[5], doc[9]]
# # for place in places:
# #     for tok in place.ancestors:
# #         if tok in actions:
# #             print(f"user is referring {place} to {tok}")
# #             break
# hello = nlp("Hello")
# hi = nlp("Hi")
# doc = nlp("How are you doing?")
# # print(hello.similarity(hi))
# # for token in doc:
# #     print(token.text, token.vector[:5])

# name = nlp("My name in Nitin Mishra")
# s_name= nlp("I love to play cricket")
# print(name.similarity(s_name))