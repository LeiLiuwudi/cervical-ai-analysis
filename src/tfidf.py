import jieba.posseg as pseg
import codecs
from gensim import corpora, models, similarities

# 进行分词处理
def cutWord(sentence):
    result = []
    words = pseg.cut(sentence)
    stop_words = 'stop.txt'
    stopwords = codecs.open(stop_words, 'r', encoding='utf8').readlines()
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
    stopwords = [w.strip() for w in stopwords]
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result

#
def tfidf(docs, base):
    corpus = []
    for each in docs:
        corpus.append(cutWord(each))
    # 建立词袋模型
    dictionary = corpora.Dictionary(corpus)
    doc_vectors = [dictionary.doc2bow(text) for text in corpus]

    # 建立TF-IDF模型
    tfidf = models.TfidfModel(doc_vectors)
    tfidf_vectors = tfidf[doc_vectors]

    query_bow = dictionary.doc2bow(cutWord(base))
    index = similarities.MatrixSimilarity(tfidf_vectors)
    sims = index[query_bow]
    index = 0
    max = -1
    for each in list(enumerate(sims)):
        if each[1] > max:
            max = each[1]
            index = each[0]
    return index


if __name__ == '__main__':
    print(cutWord("颈部外伤后遗留四肢末梢麻木14年。无肝炎、结核等传染病病史及其密切接触史。无输血史及手术史。无特殊食物过敏史，无药物过敏史。无金属过敏史。否认糖尿病、冠心病、高血压病史。预防接种史不详"))
