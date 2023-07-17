import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
class Category:
    BOOKS = "BOOKS"
    CLOTHING = "CLOTHING"

train_X = ["i love the book", "this is a great book", "this fit is great", "i love the shoes"]
train_Y = [Category.BOOKS, Category.BOOKS, Category.CLOTHING,Category.CLOTHING]

vectorizer = CountVectorizer(binary = True) #중복 제거
train_x_vectors = vectorizer.fit_transform(train_X) 
print(vectorizer.get_feature_names_out())
print(train_x_vectors.toarray())

nlp = spacy.load("en_core_web_md")
print(train_X)
docs = [nlp(text) for text in train_X]
# print(docs[0].vector)
train_x_word_vectors = [x.vector for x in docs]

clf_svm_wv = svm.SVC(kernel='linear')
clf_svm_wv.fit(train_x_word_vectors, train_Y)

test_x = ["story"]
test_docs = [nlp(text) for text in test_x]
test_x_word_vectors =  [x.vector for x in test_docs]

print(clf_svm_wv.predict(test_x_word_vectors))
               