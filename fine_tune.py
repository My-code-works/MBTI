import csv
import bert.runDataProcessor


class MyProcessor(DataProcessor):
  """Processor for the move data set ."""

    def get_train_examples(self, data_dir):
    """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir='./data/', "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
    """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir='./data/', "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
    """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir='./data/', "test.tsv")), "test")

    def get_labels(self):
    """See base class."""
        return NULL

    def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[0])
                label = "0"
            else:
                text_a = tokenization.convert_to_unicode(line[1])
                label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class EditTsv:
    
    file_path = "./data/"
    train_rate = 0.8
    test_rate = 0.2

    @staticmethod
    def read_tsv():

        with open(file_path + 'train.tsv', 'r') as file:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                print(row)
                input()

    @staticmethod
    def save_train_tsv(texts, tags):
        
        with open(file_path + 'train.tsv', 'w') as file:
            tsv_writer = csv.writer(file, delimiter='\t')
            for text, tag in zip(texts, tags):
                for i in len(text):
                    tsv_writer.writerow([text[i], tag[i]])

    @staticmethod
    def save_dev_tsv(texts, tags):

        with open(file_path + 'dev.tsv', 'w') as file:
            tsv_writer = csv.writer(file, delimiter='\t')
            for text, tag in zip(texts, tags):
                for i in len(text):
                    tsv_writer.writerow([text[i], tag[i]])
    
    @staticmethod
    def save_test_tsv(texts, tags):    
        
        with open(file_path + 'test.tsv', 'w') as file:
            tsv_writer = csv.writer(file, delimiter='\t')
            for text, tag in zip(texts, tags):
                for i in len(text):
                    tsv_writer.writerow([text[i], tag[i]])