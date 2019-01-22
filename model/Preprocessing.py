from gensim.models.wrappers import FastText
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import json

nltk.download('punkt')
data_path = "../data/"

class JsonPre:
    tag2id = {}
    id2tag = {}
    max_para = 0
    @staticmethod
    def getid(tag):
        if tag not in JsonPre.tag2id:
            cnt = len(JsonPre.tag2id)
            JsonPre.tag2id[tag] = cnt
            JsonPre.id2tag[cnt] = tag
            cnt += 1
        return JsonPre.tag2id[tag]
    @staticmethod
    def gettag(id):
        return JsonPre.id2tag[id]
    @staticmethod
    def section_split(data):
        for js in data:
            del js["simple"]
            new_full = []
            for i in js["full"]["sectionContents"]:
                if len(i["text"]) == 0 or i["isContentSection"] == False:
                    continue
                i["text"] = i["text"].split('\n')
                new_full.append(i)
            JsonPre.max_para = max(JsonPre.max_para, len(new_full))
            js["full"]["sectionContents"] = new_full
    @staticmethod
    def tagging(data):
        for js in data:
            last = None
            for i in js["full"]["sectionContents"]:
                i["tags"] = []
                for id, x in enumerate(i["text"]):
                    if last is None:
                        last = i["depth"]
                    th = i["depth"] - last
                    if id == len(i["text"]) - 1:
                        tag = "E"
                    else:
                        tag = "M"
                    if id == 0:
                        tag = "B%d"%(th)
                    i["tags"].append(JsonPre.getid(tag))
                last = i["depth"]
    @staticmethod
    def vectorize(data):
        for js in data:
            for i in js["full"]["sectionContents"]:
                i["vecs"] = []
                for x in i["text"]:
                    i["vecs"].append(JsonPre.str2vec(x))
    @staticmethod
    def str2vec(test_str):
        data = word_tokenize(test_str)
        rtn_vec = []
        for i in data:
            if not i in model.wv.vocab:
                print("The word cannot be found!")
                continue
            rtn_vec.append(model[i])
        return np.mean(np.array(rtn_vec), axis= 0)

if __name__ == '__main__':
    # load word_vec model
    model = FastText.load_fasttext_format(data_path+'wiki.en')

    with open(data_path+'1.json', 'r', encoding='utf-8') as f:
        text = ''.join(f.readlines())
        data = json.loads(text)
    JsonPre.section_split(data)
    JsonPre.tagging(data)
    JsonPre.vectorize(data)
    for sec in data[1]["full"]["sectionContents"]:
        print(sec)
        input()

