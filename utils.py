import pickle, os
from bert_utils import get_all_features


def save_train_data(train_texts, train_tags, train_rawtags):
    with open('save_model/train_texts.pk', 'wb') as f:
        pickle.dump(train_texts, f)

    with open('save_model/train_tags.pk', 'wb') as f:
        pickle.dump(train_tags, f)

    with open('save_model/train_rawtags.pk', 'wb') as f:
        pickle.dump(train_rawtags, f)
        

def load_train_data():
    with open('save_model/train_texts.pk', 'rb') as f:
        train_texts = pickle.load(f)

    with open('save_model/train_tags.pk', 'rb') as f:
        train_tags = pickle.load(f)

    with open('save_model/train_rawtags.pk', 'rb') as f:
        train_rawtags = pickle.load(f)

    return train_texts, train_tags, train_rawtags

        
def save_feature(feature):
    with open('save_model/feature.pk', 'rb') as f:
        pickle.dump(feature, f)


def load_feature():
    with open('save_model/feature.pk', 'rb') as f:
        return pickle.load(f)
    
'''
if __name__ == '__main__':
    train_texts, train_tags, train_rawtags = load_train_data()
    
    BERT_BASE = os.path.join(os.getcwd(), 'bert/bert_model/uncased_L-12_H-768_A-12')
    bert_config_file = os.path.join(BERT_BASE, 'bert_config.json')
    vocab_file = os.path.join(BERT_BASE, 'vocab.txt')
    bert_checkpoint = os.path.join(BERT_BASE, 'bert_model.ckpt')
    
    feature = load_feature()
    print(len(feature))
    
    feature += get_all_features(train_texts[6000:15000], bert_config_file, vocab_file, bert_checkpoint)
    print(len(feature))
    save_feature(feature)
    print('Saved')
'''