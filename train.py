import kagglehub
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import albumentations as A
from triplet_loader import PictureDataset
import torch
from torch import nn
from tqdm import tqdm
from model import FaceRecognitionModel
from sklearn.metrics import f1_score
from pytorch_metric_learning.losses import ArcFaceLoss

batch_size = 64
max_epochs = 40
embedding_size = 512
h, w = 128, 128

# Download dataset
path = kagglehub.dataset_download("wannad1e/celeba-500-label-folders")
path_data  = path + '/celeba/celebA_anno.txt'
path_split = path + '/celeba/celebA_train_split.txt'

# Train / validation / test split
all_df   = pd.read_csv(path_data, sep=" ", header=None, names=["Filename", "Label"])
split_df = pd.read_csv(path_split, sep=" ", header=None, names=["Filename", "Label"])
train_df = all_df[split_df['Label']==0]
val_df   = all_df[split_df['Label']==1]
test_df  = all_df[split_df['Label']==2]
train_df = pd.concat([train_df, val_df])
train_df.index = np.arange(len(train_df))
test_df.index = np.arange(len(test_df))
uniq_list = list(pd.unique(all_df['Label']))

# for ArcFace Loss
def pred(logits):
    return torch.argmax(logits, dim=1)

# Augmentations
transform_train = A.Compose([
    A.Resize(h, w),
    A.Rotate(p=0.5, limit=20),
    A.pytorch.ToTensorV2()
])

transform_test = A.Compose([
    A.Resize(h, w),
    A.pytorch.ToTensorV2()
])

picture_folder = path + '/celeba/celebA_imgs'
train_loader = DataLoader(PictureDataset(picture_folder, train_df, transform_train), batch_size = batch_size, shuffle= True, num_workers = 4, pin_memory=True, prefetch_factor=2)
test_loader  = DataLoader(PictureDataset(picture_folder, test_df, transform_test), batch_size = batch_size, shuffle= False, num_workers = 4, pin_memory=True, prefetch_factor=2)

print("Load model...")
model = FaceRecognitionModel()
model.init(ArcFaceLoss(
        num_classes=500,
        embedding_size=embedding_size,
        margin= 57 * 0.5,
        scale= 64
    ),
    embedding_size)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion_triplet = nn.TripletMarginLoss(margin=1, eps=1e-7)
criterion_arc = model.loss_fn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(test_loader = None, train_loader = None):
    train_loss, train_acc, train_f1, test_loss, test_acc, test_f1, loss_fn = train_loop(model, train_loader, optimizer, criterion_arc, criterion_triplet, device)
    _, _, test_f1_tmp = test(model, test_loader, criterion_arc, criterion_triplet, device)
    print(f"Test f1: {100 * round(sum(test_f1_tmp) / len(test_f1_tmp), 4)}%")

    if not os.path.isdir("model"):
        os.mkdir("model")
    torch.save({
        'model': model.state_dict()
    }, 'model/model.pth')
    graphics(train_acc, test_acc, train_loss, test_loss, train_f1, test_f1)

def train_loop(model, train_loader, optimizer, criterion_arc, criterion_triplet, device):
    print("Training...")
    cnt_overfitting, max_cnt_overfitting = 0, 5
    train_loss, test_loss, train_acc = [], [], []
    test_acc, test_f1, train_f1 = [], [0], []
    model.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.8)
    progress_bar = tqdm(range(max_epochs))
    for _ in progress_bar:
        tmp_loss, tmp_acc, tmp_f1 = [], [], []
        model.train()
        for orig, pos, neg, label in train_loader:
            optimizer.zero_grad()
            orig, pos, neg = orig.to(device), pos.to(device), neg.to(device)
            orig, pos, neg = model.encode(orig), model.encode(pos), model.encode(neg)
            loss = 0.7 * criterion_arc(orig, label.to(device)) + 0.3 * criterion_triplet(orig, pos, neg)
            loss.backward()
            optimizer.step()
            tmp_loss.append(loss.detach().item())
            tmp_acc.append(triplet_accuracy(orig, pos, neg))
            tmp_f1.append(f1_score(label.cpu(), pred(criterion_arc.get_logits(orig).cpu()), average='micro'))

        with torch.no_grad():
            model.eval()
            tmp_max = max(test_f1)
            test_loss_tmp, test_acc_tmp, test_ce_f1 = test(model, test_loader, criterion_arc, criterion_triplet, device)
            test_loss.append(sum(test_loss_tmp) / len(test_loss_tmp))
            test_acc.append(sum(test_acc_tmp) / len(test_acc_tmp))
            test_f1.append(sum(test_ce_f1) / len(test_ce_f1))

        train_loss.append(sum(tmp_loss) / len(tmp_loss))
        train_acc.append(sum(tmp_acc) / len(tmp_acc))
        train_f1.append(sum(tmp_f1) / len(tmp_f1))
        scheduler.step()
        progress_bar.set_postfix({
            "train_loss": f"{train_loss[-1]:.4f}",
            "test_loss": f"{test_loss[-1]:.4f}",
            "train_triplet_acc": f"{train_acc[-1]:.2%}",
            "test_triplet_acc": f"{test_acc[-1]:.2%}",
            "train_f1": f"{train_f1[-1]:.2%}",
            "test_f1": f"{test_f1[-1]:.2%}"
        })

        if(test_f1[-1] + 0.001 < tmp_max):
            cnt_overfitting+=1
            print(cnt_overfitting)
            if (cnt_overfitting > max_cnt_overfitting or train_f1[-1] - test_f1[-1] > 0.3):
                print("Early stopping")
                break
    test_f1.pop(0)
    return train_loss, train_acc, train_f1, test_loss, test_acc, test_f1, criterion_arc

def test(model, test_loader, criterion_arc, criterion_triplet, device):
    test_loss, test_triplet_acc, test_f1 = [], [], []
    with torch.no_grad():
        for orig, pos, neg, label in test_loader:
            orig, pos, neg = orig.to(device), pos.to(device), neg.to(device)
            orig, pos, neg = model.encode(orig), model.encode(pos), model.encode(neg)
            loss = 0.7 * criterion_arc(orig, label.to(device)) + 0.3 * criterion_triplet(orig, pos, neg)
            test_loss.append(loss.detach().item())
            test_triplet_acc.append(triplet_accuracy(orig, pos, neg))
            test_f1.append(f1_score(label.cpu(), pred(criterion_arc.get_logits(orig)).cpu(), average='micro'))
    return test_loss, test_triplet_acc, test_f1

def triplet_accuracy(anchor, positive, negative):
    pos_dist = torch.norm(anchor - positive, dim=1)
    neg_dist = torch.norm(anchor - negative, dim=1)
    return torch.mean((neg_dist - pos_dist + 1).float()).item() / 2

def graphics(train_acc, test_acc, train_loss, test_loss, train_f1, test_f1):
    if not os.path.isdir("logs"):
        os.mkdir("logs")
    plt.plot(train_acc, label='train acc')
    plt.plot(test_acc,  label='test acc')
    plt.title('Train and test triplet accuracy')
    plt.legend()
    plt.savefig('logs/triplet_accuracy.png')
    plt.clf()
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.title('Train and test loss')
    plt.legend()
    plt.savefig('logs/loss.png')
    plt.clf()
    plt.plot(train_f1, label='train F1')
    plt.plot(test_f1, label='test F1')
    plt.title('Train and test F1')
    plt.legend()
    plt.savefig('logs/F1_score.png')

if __name__ == "__main__":
    print("Tr model...")
    main(test_loader, train_loader)
    print("End model")