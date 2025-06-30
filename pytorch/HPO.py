import sys
sys.path.append('/home/aislab/Documents/Tellaroli/ProgettoSML')
from mymodels import squeezenet1_1, mobilenet_v3_small, AlexNet
from dataUtils import load_data, make_dataset, make_train_set, make_val_set
from torch import nn,stack,no_grad
import torch, numpy as np
import tempfile
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset, random_split
from pathlib import Path
from functools import partial
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

def train_slam(config, model, model_params:dict, trainset, valset, data_dir=None, checkpoint=True):
    device = "cuda"
    net = model(**{key:config[key] for key in model_params})
    net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0
    
    trainloader = DataLoader(
        trainset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )
    valloader = DataLoader(
        valset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )
    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        model.train(True)
        for i, data in enumerate(trainloader, 0):
            x, area, ate, are = data
            x = x.to(device)
            area = area.to(device)
            ate = ate.to(device)
            are = are.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(x,area)
            labels = stack([ate,are],dim=1).to(device).float()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        model.eval()
        for i, data in enumerate(valloader, 0):
            with no_grad():
                x, area, ate, are = data
                x = x.to(device)
                area = area.to(device)
                ate = ate.to(device)
                are = are.to(device)

                outputs = net(x,area)

                labels = stack([ate,are],dim=1).to(device).float()
                loss = loss_fn(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        if checkpoint:
            checkpoint_data = {
                "epoch": epoch,
                "net_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                train.report(
                    {"loss": val_loss / val_steps},
                    checkpoint=checkpoint,
                )
        else: train.report({"loss": val_loss / val_steps})

    print("Finished Training")

def test_error(net, device="cpu", loss_fn=nn.MSELoss()):
    _, testset = load_data()
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )
    test_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            x, area, ate, are = data
            x = x.to(device)
            area = area.to(device)
            ate = ate.to(device)
            are = are.to(device)
            outputs = net(x,area)
            labels = stack([ate,are],dim=1).to(device).float()
            loss = loss_fn(outputs, labels)
            test_loss += loss.cpu().numpy()
    return test_loss / len(testloader)

config = {
    "dropout" : tune.grid_search(np.linspace(0,0.7,35)),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.grid_search([2, 4, 8, 16, 32])
}
max_num_epochs = 10
num_samples = 5
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=max_num_epochs,
    grace_period=3,
    reduction_factor=2,
)
trainset = make_train_set()
valset = make_val_set()
result = tune.run(
    partial(
        train_slam,
        model=squeezenet1_1,
        model_params={'dropout':config["dropout"]},
        data_dir='/home/aislab/Documents/Tellaroli/ProgettoSML/rayTune',
        checkpoint=False,
        trainset = trainset,
        valset = valset
    ),
    checkpoint_score_attr=None,     #
    checkpoint_freq=0,              # no checkpoint
    checkpoint_at_end= False,       #
    keep_checkpoints_num=1,         #
    resources_per_trial={"cpu": 10, "gpu": 1},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler)
best_trial = result.get_best_trial("loss", "min", "last")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final validation loss: {best_trial.last_result['loss']}")


#best_trained_model = SqueezeNet(best_trial.config["dropout"])
#device = "cpu"
#best_trained_model.to(device)
#best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="loss", mode="min")
#with best_checkpoint.as_directory() as checkpoint_dir:
#    data_path = Path(checkpoint_dir) / "data.pkl"
#    with open(data_path, "rb") as fp:
#        best_checkpoint_data = pickle.load(fp)
#    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
#    test_acc = test_error(best_trained_model, device)
#    print("Best trial test set accuracy: {}".format(test_acc))

# SQUEEZE NET
# Best trial config: {'dropout': 0, 'lr': 0.02848174232146005, 'batch_size': 32}
# Best trial final validation loss: 0.03440441271024091
# MOBILE NET
# Best trial config: {'dropout': 0.3088235294117647, 'lr': 0.00025266266504172765, 'batch_size': 32}  2h
# Best trial final validation loss: 0.012229122731479861
# AlexNet
# Best trial config: {'dropout': 0.041176470588235294, 'lr': 0.000260626067537044, 'batch_size': 8}  1.5h
# Best trial final validation loss: 0.027499174020816513