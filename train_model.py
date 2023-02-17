import torch
import torch.nn as nn
import numpy as np


def train_model(model, batch_size, train_loader, val_loader, device, clip,
                domain, weighted, dataamount, seed):
    print("Start training...")
    epochs = 3
    valid_loss_min = np.Inf
    criterion = nn.BCELoss().to(device)
    lr = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(epochs):
        model.train()
        counter = 0
        # we train on the train data
        h = model.init_hidden(batch_size)
        for inputs, labels in train_loader:
            counter += 1
            h = tuple([e.data for e in h])
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1)
            model.zero_grad()
            output = model(inputs, h)
            loss = criterion(output, labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            print("Step: {} ".format(counter), "Loss: {:.6f}".format(loss))

        # we check the loss on the validation data
        val_h = model.init_hidden(batch_size)
        val_losses = []
        model.eval()
        for inp, lab in val_loader:
            val_h = tuple([each.data for each in val_h])
            inp, lab = inp.to(device), lab.to(device)
            lab = lab.unsqueeze(1)
            out = model(inp, val_h)
            val_loss = criterion(out, lab.float())
            val_losses.append(val_loss.item())

        print("Epoch: {}/{}...".format(i + 1, epochs),
              "Step: {} ".format(counter),
              "Val Loss: {:.6f}".format(np.mean(val_losses)))

        if np.mean(val_losses) < valid_loss_min:
            torch.save(model.state_dict(),
                       'models/' + domain + '/awsa_' + weighted + '_' + dataamount + '_' + str(seed) + '.pt')
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                            np.mean(val_losses)))
            valid_loss_min = np.mean(val_losses)
