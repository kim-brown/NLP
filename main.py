from comet_ml import Experiment
import torch
import torch.nn
from torch import optim
import argparse
import math
import numpy as np
from preprocessing import *
from autolabeler import AutoLabeler
from main_model import MainModel
from tqdm import tqdm

hyper_params = {
    "batch_size": 4,
    "embedding_size": 375,
    "num_epochs": 30,
    "learning_rate": 0.001,
    "window_size": 128,
    "hidden_layer_size": 512,
    "num_layers": 2
 }

autolabeler_hyper_params = {
    "batch_size" : 2,
    "num_epochs" : 150,
    "learning_rate": 0.0001,
    "window_size": 128,
    "hidden_layer_size": 512,
    "vocab_size": 2000
}

def autolabeler_train(model, train_loader, optimizer, experiment):
    """
    Trains the autolabeler.
    """
    loss_fn = torch.nn.MSELoss()
    with experiment.train():
        for i in range(autolabeler_hyper_params["num_epochs"]):
            for inps, labs in tqdm(train_loader):
                optimizer.zero_grad()
                logits = model(inps)
                loss = loss_fn(logits, labs)
                loss.backward()
                optimizer.step()

def autolabeler_test(model, test_loader, optimizer, experiment):
    """
    Tests the autolabeler.
    """
    loss_fn = torch.nn.MSELoss()
    num_correct = 0.0
    total_posts = 0.0
    total_loss = 0.0
    with experiment.test():
        for inps, labs in tqdm(test_loader):
            logits = model(inps)
            total_loss += loss_fn(logits, labs).item() * len(labs)
            num_correct += torch.sum(((logits > 0.5) == labs)).item()
            total_posts += len(labs)

        accuracy = num_correct / total_posts
        perplexity = torch.exp(torch.tensor(total_loss / total_posts)).item()
        experiment.log_metric("perplexity", perplexity)
        experiment.log_metric("accuracy", accuracy)
        print("PERPLEXITY: ", perplexity)
        print("ACCURACY: ", accuracy)


def train(model, train_loader, optimizer, experiment):
    """
    Trains the model.
    :param model: the model to train
    :param train_loader: train data dataloader
    :param optimizer: optimization function
    :param experiment: comet.ml experiment
    """
    loss_fn = torch.nn.MSELoss()
    flagged_cnt = 0
    total_posts = 0
    with experiment.train():
        for i in range(hyper_params["num_epochs"]):
            for posts, feats in tqdm(train_loader):
                optimizer.zero_grad()
                auto_labels = autolabeler_model(feats)
                labs = (auto_labels > 0.5).float()
                flagged_cnt += torch.sum(labs).item()
                total_posts += len(labs)
                logits = model(posts)
                loss = loss_fn(logits, labs)
                loss.backward()
                optimizer.step()
    print("flagged: ", flagged_cnt, " /", total_posts)


def test(model, test_loader, experiment):
    """
    Validates model performance
    :param model: the trained model
    :param test_loader: Testing data dataloader
    :param experiment: comet.ml experiment
    """
    model = model.eval()
    loss_fn = torch.nn.MSELoss()
    total_loss = 0.0
    num_correct = 0.0
    total_posts = 0.0
    flagged_cnt = 0
    with experiment.validate():
        for i in range(hyper_params["num_epochs"]):
            for posts, feats in tqdm(test_loader):
                auto_labels = autolabeler_model(feats)
                labs = (auto_labels > 0.5).float()
                logits = model(posts)
                total_loss += loss_fn(logits, labs).item()
                num_correct += torch.sum(((logits > 0.5) == labs)).item()
                total_posts += len(labs)
                flagged_cnt += torch.sum(labs).item()
    accuracy = num_correct / total_posts
    perplexity = torch.exp(torch.tensor(total_loss / total_posts)).item()
    experiment.log_metric("perplexity", perplexity)
    experiment.log_metric("accuracy", accuracy)
    print("flagged: ", flagged_cnt, " /", total_posts)
    print("PERPLEXITY: ", perplexity)
    print("ACCURACY: ", accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("depression_data")
    parser.add_argument("neutral_data")
    parser.add_argument("relationship_data")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-a", "--autolabeler", action="store_true",
                        help="train + test autolabeler")
    args = parser.parse_args()

    experiment = Experiment(log_code=True)
    experiment.log_parameters(hyper_params)

    main_train_loader, main_test_loader, vocab_size = load_main_dataset(args.depression_data,
        args.neutral_data, args.relationship_data, hyper_params["window_size"],
        hyper_params["batch_size"], autolabeler_hyper_params["vocab_size"])
    main_model = MainModel(vocab_size, hyper_params["hidden_layer_size"],
    hyper_params["embedding_size"], hyper_params["window_size"], hyper_params["num_layers"])

    autolabeler_model = AutoLabeler(vocab_size, autolabeler_hyper_params["hidden_layer_size"])

    if args.autolabeler:
        print("training/ testing the autolabeler")
        train_loader, test_loader, vocab_size = load_autolabeler_dataset(args.depression_data, args.neutral_data,
            autolabeler_hyper_params["window_size"], autolabeler_hyper_params["batch_size"],
            autolabeler_hyper_params["vocab_size"])

        autolabeler_optimizer = optim.Adam(autolabeler_model.parameters(), autolabeler_hyper_params["learning_rate"])
        autolabeler_train(autolabeler_model, train_loader, autolabeler_optimizer, experiment)
        autolabeler_test(autolabeler_model, test_loader, autolabeler_optimizer, experiment)
        torch.save(autolabeler_model.state_dict(), 'autolabeler.pt')
    else:
        autolabeler_model.load_state_dict(torch.load('autolabeler.pt'))
    if args.load:
        main_model.load_state_dict(torch.load('main_model.pt'))
    if args.train:
        print("running training loop...")
        optimizer = optim.Adam(main_model.parameters(), hyper_params["learning_rate"])
        train(main_model, main_train_loader, optimizer, experiment)
    if args.save:
        torch.save(main_model.state_dict(), 'main_model.pt')
    if args.test:
        print("running testing loop...")
        test(main_model, main_test_loader, experiment)
