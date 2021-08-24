import numpy as np
import torch as torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.autograd import Variable
from plot import *
import _pickle as cPickle
import os
import signal
from timeit import timeit


# -----------------------------------------------------------

class CfData(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_batch = self.X[idx]
        y_batch = self.Y[idx]
        sample = {'X': x_batch, 'Y': y_batch}
        return sample


# -----------------------------------------------------------

class LinearRegression(torch.nn.Module):
    def __init__(self, input_size: int, poly_n: int):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, poly_n)
        self.oupt = torch.nn.Linear(poly_n, 1)

    def forward(self, x):
        out = self.oupt(torch.nn.ReLU()(self.linear(x)))
        return out


# -----------------------------------------------------------

def accuracy(model, X, Y):
    # Average L2 loss / mean(ground truth)
    with torch.no_grad():
        oupt = 100 * (1 - torch.abs(torch.squeeze(model(X)).mean() - Y.mean()) / Y.mean())
    return oupt


def compare_data(model, poly_n: int, device="cuda", angles=32, fn='model.pdf'):
    # Get mean quantities
    with open('fos.pickle', "rb") as f:
        fos = cPickle.load(f)

    p_data = np.load('data.npy').astype(np.float32)
    split = len(p_data[:, 0]) // (len(fos['t']))
    gt = p_data[:, -1]

    with torch.no_grad():
        cd_hat = (torch.squeeze(model(torch.tensor(p_data[:, 0:-1],
                                                   device=device)))
                  .cpu().detach().numpy())
    cd_hat = np.array([cd_hat[i * len(fos['t']):(i + 1) * len(fos['t'])] for i in range(split)])
    gt = np.array([gt[i * len(fos['t']):(i + 1) * len(fos['t'])] for i in range(split)])

    plot_model(np.mean(cd_hat, axis=0), fos, np.mean(gt, axis=0), fn=fn)


def compare_model(model, poly_n: int, device="cuda", angles=32, fn='model_gt.pdf'):
    # Get mean quantities
    with open('fos.pickle', "rb") as f:
        fos = cPickle.load(f)

    chunk = angles * len(fos['t'])

    p_data = np.load('data.npy').astype(np.float32)
    gt = p_data[0:chunk, -1]

    with torch.no_grad():
        cd_hat = (torch.squeeze(model(torch.tensor(p_data[0:chunk, 0:-1],
                                                   device=device)))
                  .cpu().detach().numpy())
    cd_hat = np.array([cd_hat[i * len(fos['t']):(i + 1) * len(fos['t'])] for i in range(angles)])
    gt = np.array([gt[i * len(fos['t']):(i + 1) * len(fos['t'])] for i in range(angles)])

    plot_model(np.mean(cd_hat, axis=0), fos, np.mean(gt, axis=0), fn=fn)


# -----------------------------------------------------------
def handler(signum, frame):
    raise RuntimeError


def main(wd):
    # 0. get started, seed for reproducibility
    print("\nStart multivariate regression \n")
    torch.manual_seed(1)
    np.random.seed(1)

    # 1. Split data and DataLoader objects
    data = np.load('data.npy').astype(np.float32)

    poly_n = 10
    x_train, x_test, y_train, y_test = \
        train_test_split(data[:, 0:-1], data[:, -1],
                         test_size=0.2, shuffle=False)
    print("Create data generator ")
    ds_train = CfData(x_train, y_train)  # all 200 rows

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu:0")

    params = {'batch_size': int(1028),
              'shuffle': True,
              'num_workers': 16,
              'pin_memory': True}
    train_ldr = torch.utils.data.DataLoader(ds_train, **params)

    # 2. create network
    model = LinearRegression(np.shape(x_train)[-1], poly_n)
    if torch.cuda.device_count() > 1:
        # model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    else:
        print(f"Using {torch.cuda.device_count()} GPU")
    model.to(device)

    # 3. train model
    max_epochs = 70
    ep_log_interval = 10
    lrn_rate = 0.00001
    wd = wd

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrn_rate, weight_decay=wd)

    print("\nbat_size = %3d " % params['batch_size'])
    print("loss = " + str(loss_func))
    print("optimizer = Adam")
    print("max_epochs = %3d " % max_epochs)
    print("lrn_rate = %0.3f " % lrn_rate)

    print("\nStarting training")
    'models/nn_regression.pth'
    try:
        model.load_state_dict(torch.load('models/'+str(wd) + '_nn_regression.pth'))
    except FileNotFoundError:
        print('No model saved, starting from scratch')
    except RuntimeError:
        print('Trying different model')
    finally:
        pass

    cum_loss = []
    try:
        for epoch in tqdm(range(0, int(max_epochs)), desc='Training net'):
            epoch_loss = 0  # for one full epoch
            for batch_idx, batch in enumerate(train_ldr):
                X = (batch['X']).to(device)
                Y = (batch['Y']).to(device)

                optimizer.zero_grad()
                oupt = model(X)  # predicted income
                loss_val = loss_func(torch.squeeze(oupt), Y)
                epoch_loss += loss_val.item()
                loss_val.backward()
                optimizer.step()
            signal.signal(signal.SIGINT, handler)
            if epoch % ep_log_interval == 0:
                cum_loss.append(epoch_loss)
                print("epoch = %4d   loss = %0.4f" % (epoch, cum_loss[-1]))

    except RuntimeError:
        print('Outta time bitch!')
    print("\nDone ")
    print("epochs = %4d :   total loss = %0.7f" % (epoch, epoch_loss))
    plot_loss(max_epochs, cum_loss, fn='figures/cost_wd_' + str(wd) + '.pdf')

    # 4. evaluate model accuracy
    print("\nComputing model accuracy")
    model = model.eval()
    acc_train = accuracy(model, X, Y)  # item-by-item
    print(f"Accuracy on training data = {acc_train:.4f} %")

    X_test = torch.from_numpy(x_train).to(device)
    Y_test = torch.from_numpy(y_train).to(device)

    acc_test = accuracy(model, X_test, Y_test)  # item-by-item
    print(f"Accuracy on test data = {acc_test:.4f} %")

    # 5. save model (state_dict approach)
    print("\nSaving trained model state")
    torch.save(model.state_dict(), 'models/'+str(wd) + '_nn_regression.pth')

    # 6. Test accuracy on test data and plot results
    print("\nCompare model to original data")

    compare_model(model, poly_n, fn='figures/model_gt_wd_'+str(wd)+'.pdf')
    try:
        compare_data(model, poly_n, fn='figures/model_data_wd_' + str(wd) + '.pdf')
    finally:
        print('CD func didnt work')

    print("\nBetter?")


if __name__ == "__main__":
    main(0.0)
    for w in [-4, -3]:
        main(10**w)


