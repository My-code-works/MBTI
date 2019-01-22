from model.Preprocessing import JsonPre
import json

def get_json(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        text = ''.join(f.readlines())
    return json.loads(text)


if __name__ == '__main__':
    num_of_json = 500
    train = []
    for i in range(num_of_json):
        print('loading %d'%(i))
        data = get_json("data/%d.json"%(i))
        JsonPre.section_split(data)
        JsonPre.tagging(data)
        train += data
    print(len(train), len(JsonPre.tag2id), JsonPre.max_para)
    print(JsonPre.tag2id, JsonPre.id2tag)
