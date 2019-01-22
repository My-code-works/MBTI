import json

class JsonPre:
    @staticmethod
    def section_split(data):
        for js in data:
            del js["simple"]
            for i in js["full"]["sectionContents"]:
                i["text"] = i["text"].split('\n')
    @staticmethod
    def tagging(data):
        pass
if __name__ == '__main__':
    with open('data/1.json', 'r', encoding='utf-8') as f:
        text = ''.join(f.readlines())
        data = json.loads(text)
    JsonPre.section_split(data)
    print(data[0]["full"]["sectionContents"][0])

