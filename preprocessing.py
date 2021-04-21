from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

class LabeledDataset(Dataset):
    def __init__(self, depression_data, neutral_data, window_size, batch_size, test, word2id):
        self.word2id = word2id
        self.word2id["[PAD]"] = 0
        self.next_id = max(1, len(word2id))
        self.posts = []
        self.post_features = [] # posts in their BoW matrix representation
        self.labels = []
        self.process_file(depression_data, 1, window_size)
        self.process_file(neutral_data, 0, window_size)

        # cut off data based on train/ test status
        if test:
            self.posts = self.posts[int(len(self.posts) * 0.8):]
            self.labels = self.labels[int(len(self.labels) * 0.8):]
        else:
            self.posts = self.posts[:int(len(self.posts) * 0.8)]
            self.labels = self.labels[:int(len(self.labels) * 0.8)]

        self.post_features = self.bag_of_words(self.posts)
        assert(len(self.post_features) == len(self.labels))

        num_batches = int(len(self.post_features) / batch_size)
        self.post_features = self.post_features[:num_batches * batch_size]
        self.post_features = pad_sequence(self.post_features, batch_first=True, padding_value=0).float()
        self.labels = pad_sequence(self.labels, batch_first=True, padding_value=0).float()

    def process_file(self, fn, label, window_size):
        with open(fn, 'r') as f:
            post_cnt = 0
            post = []
            for l in f:
                line = self.clean_line(l)
                words = line.split()

                for w in words:
                    w = w.strip('\n')
                    w = w.strip('\t')
                    w = w.strip()

                    if w == "[sep]":
                        post = post[:window_size]
                        self.posts.append(post)
                        post_cnt += 1
                        lab = []
                        lab.append(label)
                        self.labels.append(torch.tensor(lab))
                        post = []
                    else:
                        if w not in self.word2id:
                            self.word2id[w] = self.next_id
                            self.next_id += 1
                            post.append(self.word2id[w])
            print("file ", fn, " has ", post_cnt, " posts")


    def __len__(self):
        return len(self.post_features)

    def __getitem__(self, idx):
        to_return = (
            self.post_features[idx],
            self.labels[idx],
        )
        return to_return

    def clean_line(self, line):
        l = line
        l = l.strip('\n')
        l = l.strip('\t')
        to_remove = {"!", ".", ":", "-", "?", ",", ";", "/", "~", "<", ">", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "@", "#", "$", "*", "(", ")", "{", "}", "|", "%", "&"}
        for r in to_remove:
            l = l.replace(r, "")
        l = l.lower()
        return l

    def bag_of_words(self, posts):
        matrices = []
        for post in posts:
            matrix = torch.zeros(len(self.word2id))
            for w in post:
                matrix[int(w)] = 1
            matrices.append(matrix)
        return matrices

def load_autolabeler_dataset(depression_data, neutral_data, window_size, batch_size):
    """
    :param depression_data: filename for the posts indicating depression
    :param neutral_data: filename for the posts not indicating depression

    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for
    train and test
    """
    train_dataset = LabeledDataset(depression_data, neutral_data, window_size, batch_size, 0, {})
    test_dataset = LabeledDataset(depression_data, neutral_data, window_size, batch_size, 1, train_dataset.word2id)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    return train_loader, test_loader, len(test_dataset.word2id)
