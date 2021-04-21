from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

class LabeledDataset(Dataset):
    def __init__(self, depression_data, neutral_data, window_size, batch_size, test, word2id):
        self.word2id = word2id
        self.post_features = [] # posts in their BoW matrix representation
        self.depression_posts, self.depression_labels = process_file(depression_data, 1, window_size, word2id)
        self.neutral_posts, self.neutral_labels = process_file(neutral_data, 0, window_size, word2id)

        # cut off data based on train/ test status
        if test:
            self.depression_posts = self.depression_posts[int(len(self.depression_posts) * 0.8):]
            self.depression_labels = self.depression_labels[int(len(self.depression_labels) * 0.8):]
            self.neutral_posts = self.neutral_posts[int(len(self.neutral_posts) * 0.8):]
            self.neutral_labels = self.neutral_labels[int(len(self.neutral_labels) * 0.8):]
        else:
            self.depression_posts = self.depression_posts[:int(len(self.depression_posts) * 0.8)]
            self.depression_labels = self.depression_labels[:int(len(self.depression_labels) * 0.8)]
            self.neutral_posts = self.neutral_posts[:int(len(self.neutral_posts) * 0.8)]
            self.neutral_labels = self.neutral_labels[:int(len(self.neutral_labels) * 0.8)]

        self.post_features = bag_of_words(self.depression_posts + self.neutral_posts, len(word2id))
        self.labels = self.depression_labels + self.neutral_labels
        assert(len(self.post_features) == len(self.labels))

        num_batches = int(len(self.post_features) / batch_size)
        self.post_features = self.post_features[:num_batches * batch_size]
        self.post_features = pad_sequence(self.post_features, batch_first=True, padding_value=0).float()
        self.labels = pad_sequence(self.labels, batch_first=True, padding_value=0).float()


    def __len__(self):
        return len(self.post_features)

    def __getitem__(self, idx):
        to_return = (
            self.post_features[idx],
            self.labels[idx],
        )
        return to_return

def load_autolabeler_dataset(depression_data, neutral_data, window_size, batch_size, vocab_size):
    """
    :param depression_data: filename for the posts indicating depression
    :param neutral_data: filename for the posts not indicating depression

    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for
    train and test
    """
    word2id = create_vocabulary(depression_data, neutral_data, vocab_size)
    train_dataset = LabeledDataset(depression_data, neutral_data, window_size, batch_size, 0, word2id)
    test_dataset = LabeledDataset(depression_data, neutral_data, window_size, batch_size, 1, word2id)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    return train_loader, test_loader, len(word2id)

def load_main_dataset(depression_data, neutral_data, relationship_data, window_size, batch_size, vocab_size):
    word2id = create_vocabulary(depression_data, neutral_data, vocab_size)
    train_dataset = MainDataset(relationship_data, window_size, batch_size, 0, word2id)
    test_dataset = MainDataset(relationship_data, window_size, batch_size, 1, word2id)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    return train_loader, test_loader, len(word2id)

def create_vocabulary(depression_data, neutral_data, size):
    """
    Identify the <size> most frequently used words in the posts and return
    a word2id dictionary for them
    """
    word2freq = defaultdict(lambda: 0)
    word2id = {}
    word2id["[pad]"] = 0
    next_id = 1
    with open(depression_data, 'r') as f:
        for l in f:
            line = clean_line(l)
            words = line.split()

            for w in words:
                w = w.strip('\n')
                w = w.strip('\t')
                w = w.strip()
                word2freq[w] += 1

    with open(neutral_data, 'r') as f:
        for l in f:
            line = clean_line(line)
            words = line.split()

            for w in words:
                w = w.strip('\n')
                w = w.strip('\t')
                w = w.strip()
                word2freq[w] += 1

    sorted_dict = sorted(word2freq.items(), key=lambda x:x[1])
    sorted_dict = sorted_dict[-size:]
    for d in sorted_dict:
        word2id[d[0]] = next_id
        next_id += 1
    return word2id

def clean_line(line):
    l = line
    l = l.strip('\n')
    l = l.strip('\t')
    l = l.replace("/", " ")
    to_remove = {"!", ".", ":", "-", "?", ",", ";", "+", "=" , "\"", "~", "<", ">", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "@", "#", "$", "*", "(", ")", "{", "}", "|", "%", "&"}
    for r in to_remove:
        l = l.replace(r, "")
    l = l.lower()
    return l

def process_file(fn, label, window_size, word2id):
    posts = []
    labels = []
    with open(fn, 'r') as f:
        post_cnt = 0
        post = []
        for l in f:
            line = clean_line(l)
            words = line.split()

            for w in words:
                word = clean_word(w)

                if word == "[sep]":
                    post = post[:window_size]
                    posts.append(torch.tensor(post))
                    post_cnt += 1
                    lab = []
                    lab.append(label)
                    labels.append(torch.tensor(lab))
                    post = []
                else:
                    if word in word2id:
                        post.append(word2id[word])
        print("file ", fn, " has ", post_cnt, " posts")
        return posts, labels

def clean_word(w):
    """
    stem and strip word
    """
    word = w.strip('\t')
    word = word.strip()
    if word[-3:] == "ing":
        word = word[:-3]
    elif word[-3:] == "ed":
        word = word[:-2]
    elif word[-1:] == "s":
        word = word[:-1]
    return word

def bag_of_words(posts, vocab_size):
    matrices = []
    for post in posts:
        matrix = torch.zeros(vocab_size)
        for w in post:
            matrix[int(w)] = 1
        matrices.append(matrix)
    return matrices

class MainDataset(Dataset):
    def __init__(self, main_data, window_size, batch_size, test, word2id):
        self.word2id = word2id
        self.post_features = [] # posts in their BoW matrix representation
        self.posts, _ = process_file(main_data, 0, window_size, word2id)

        # cut off data based on train/ test status
        if test:
            self.posts = self.posts[int(len(self.posts) * 0.8):]
        else:
            self.posts = self.posts[:int(len(self.posts) * 0.8)]

        self.post_features = bag_of_words(self.posts, len(word2id))

        self.post_features = pad_sequence(self.post_features, batch_first=True, padding_value=0).float()
        num_batches = int(len(self.post_features) / batch_size)
        self.post_features = self.post_features[:num_batches * batch_size]

        self.posts = pad_sequence(self.posts, batch_first=True, padding_value=0)
        self.posts = self.posts[:num_batches * batch_size]


    def __len__(self):
        return len(self.post_features)

    def __getitem__(self, idx):
        to_return = (
            self.posts[idx],
            self.post_features[idx],
        )
        return to_return
