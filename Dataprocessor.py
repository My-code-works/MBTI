import json
import os


def get_json(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        text = ''.join(f.readlines())
    return json.loads(text)


class Dataprocessor:
    def __init__(self):
        self.id2tag = {0: 'PAD', 1: 'B0', 2: 'B1', 3: 'B2', 4: 'B3', 5: 'B4', 6: 'B-1', 
                       7: 'B-2', 8: 'B-3', 9: 'B-4', 10: 'M', 11: 'E'}
        self.tag2id = {}
        for i in self.id2tag:
            self.tag2id[self.id2tag[i]] = i
        self.max_paragraph = 0
    
    def getid(self, tag):
        return self.tag2id[tag]

    def gettag(self, id):
        return self.id2tag[id]

    def section_split(self, datalist):
        for js in datalist:
            del js["simple"]
            new_full = []
            for i in js["full"]["sectionContents"]:
                if len(i["text"]) == 0 or i["isContentSection"] == False:
                    continue
                i["text"] = i["text"].split('\n')
                new_full.append(i)
            self.max_paragraph = max(self.max_paragraph, len(new_full))
            js["full"]["sectionContents"] = new_full

    def tagging(self, data):
        for js in data:
            last = None
            for i in js["full"]["sectionContents"]:
                i["tags"] = []
                i["raw_tags"] = []
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
                    i["tags"].append(self.getid(tag))
                    i["raw_tags"].append(tag)
                last = i["depth"]
    
    def load_data(self, jsonfilelist):
        article_texts = []
        article_tags = []
        article_rawtags = []
        for f in jsonfilelist:
            print('loading %s...' % (os.path.basename(f)))
            data = get_json(f)
            self.section_split(data)
            self.tagging(data)
            for article in data:
                paragraphs = []
                tags = []
                rawtags = []
                for sec in article['full']['sectionContents']:
                    paragraphs.extend(sec['text'])
                    tags.extend(sec['tags'])
                    rawtags.extend(sec['raw_tags'])
                article_texts.append(paragraphs)
                article_tags.append(tags)
                article_rawtags.append(rawtags)
        
        print('Finish loading data! Total %d articles.' % (len(article_texts)))
        return article_texts, article_tags, article_rawtags
    

if __name__ == '__main__':
    filelist = [('../data/%d.json' % i) for i in range(500)]
    processor = Dataprocessor()
    train_texts, train_tags, train_rawtags = processor.load_data(filelist)
    print(len(train_texts), len(train_tags))
    print(train_texts[0], train_tags[0])
                    