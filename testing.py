import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn


def test_model(model, batch_size, test_loader, device, domain, dataamount, seed):
    print("Testing...")
    test_losses = []
    num_correct = 0
    pred = -1
    h = model.init_hidden(batch_size)
    criterion = nn.BCELoss().to(device)

    model.eval()

    total_labels = torch.LongTensor()
    total_preds = torch.LongTensor()

    for inputs, labels in test_loader:
        h = tuple([each.data for each in h])
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.unsqueeze(1)
        output = model(inputs, h)
        test_loss = criterion(output, labels.float())
        test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze())  # Rounds the output to 0/1
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)
        labels = labels.to("cpu").data.numpy()
        pred = pred.to("cpu").data.numpy()
        total_labels = torch.cat((total_labels, torch.LongTensor(labels)))
        total_preds = torch.cat((total_preds, torch.LongTensor(pred)))

    print("Printing results::: ")
    print(pred)
    labels = total_labels.data.numpy()
    preds = total_preds.data.numpy()

    print("weighted precision_recall_fscore_support:")
    print(precision_recall_fscore_support(labels, preds, average='weighted'))
    print("============================================")

    print(precision_recall_fscore_support(labels, preds, average=None))
    print("============================================")

    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}%".format(test_acc * 100))
    print(domain + " - " + dataamount + " - " + str(seed))
