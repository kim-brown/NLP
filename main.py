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

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyper_params = {
    "batch_size": 1,
    "embedding_size": 64,
    "num_epochs": 1,
    "learning_rate": 0.01,
    "window_size": 128,
    "num_heads": 1,
 }

autolabeler_hyper_params = {
    "batch_size" : 2,
    "num_epochs" : 50,
    "learning_rate": 0.0001,
    "window_size": 256,
    "hidden_layer_size": 128,
    "vocab_size": 750
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
                # print("LOSS: ", loss)
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
            # print("LOGITS: ", logits)
            total_loss += loss_fn(logits, labs).item() * len(labs)
            num_correct += torch.sum(((logits > 0.5) == labs)).item()
            total_posts += len(labs)

        accuracy = num_correct / total_posts
        # print("num correct: ", num_correct, " total posts: ", total_posts)
        perplexity = torch.exp(torch.tensor(total_loss / total_posts)).item()
        experiment.log_metric("perplexity", perplexity)
        print("PERPLEXITY: ", perplexity)
        print("ACCURACY: ", accuracy)


def train(model, train_loader, optimizer, experiment):
    """
    Trains the model.
    :param model: the initilized model to use for forward and backward pass
    :param train_loader: Dataloader of training data
    :param optimizer: the initilized optimizer
    :param experiment: comet.ml experiment object
    """
    loss_fn = torch.nn.MSELoss()
    with experiment.train():
        for i in range(hyper_params["num_epochs"]):
            for inps in tqdm(train_loader):
                print("INPS: ", inps)
                optimizer.zero_grad()
                labs = (autolabeler_model(inps) > 0.5)
                logits = model(inps)
                loss = loss_fn(logits, labs)
                print("LOSS: ", loss)
                loss.backward()
                optimizer.step()


def test(model, test_loader, experiment):
    """
    Validates the model performance as LM on never-seen data using perplexity.
    :param model: the trained model to use for testing
    :param test_loader: Dataloader of testing data
    :param experiment: comet.ml experiment object
    """
    # TODO: Write the testing loop and calculate perplexity
    model = model.eval() # in stencil
    perplexity = 0.0
    total_loss = 0.0
    word_cnt = 0.0
    with experiment.validate():
        for i in range(hyper_params["num_epochs"]):
            for inps in tqdm(test_loader):
                logits = model(inps)
                total_loss += loss.item() * torch.sum(inp_lens).item()
                word_cnt += torch.sum(inp_lens).item()


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

    if args.autolabeler:
        print("training/ testing the autolabeler")
        train_loader, test_loader, vocab_size = load_autolabeler_dataset(args.depression_data, args.neutral_data,
        autolabeler_hyper_params["window_size"], autolabeler_hyper_params["batch_size"],
        autolabeler_hyper_params["vocab_size"])
        print("VOCAB SIZE: ", vocab_size)
        autolabeler_model = AutoLabeler(vocab_size, autolabeler_hyper_params["hidden_layer_size"])
        autolabeler_optimizer = optim.Adam(autolabeler_model.parameters(), autolabeler_hyper_params["learning_rate"])
        autolabeler_train(autolabeler_model, train_loader, autolabeler_optimizer, experiment)
        autolabeler_test(autolabeler_model, test_loader, autolabeler_optimizer, experiment)
        torch.save(autolabeler_model.state_dict, 'autolabeler.pt')
    if args.load:
        main_model.load_state_dict(torch.load('main_model.pt'))
    if args.train:
        # run train loop here
        print("running training loop...")
        train_loader, test_loader, vocab_size = load_main_dataset(args.depression_data,
        args.neutral_data, args.relationship_data,
        hyper_params["window_size"], hyper_params["batch_size"], autolabeler_hyper_params["vocab_size"])
        main_model = MainModel(vocab_size)
        optimizer = optim.Adam(main_model.parameters(), hyper_params["learning_rate"])
        train(main_model, train_loader, optimizer, experiment)
    if args.save:
        torch.save(main_model.state_dict(), 'main_model.pt')
    if args.test:
        # run test loop here
        print("running testing loop...")
        test(model, test_loader, experiment)
