""" 
    File Name:          RSNA_ICH/make_prediction.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/2/19
    Python Version:     3.5.4
    File Description:   

"""
from apex import amp
from albumentations import Compose, ShiftScaleRotate, Resize, \
    HorizontalFlip, RandomBrightnessContrast, Normalize
from albumentations.pytorch import ToTensor

from utilities.ich_dataset import *

NUM_CARDINALITY = 8
NUM_EPOCHS = 10
IMG_DIM = 256
DEVICE = 'cuda'

trn_lbl_df = load_trn_lbl_df()
trn_hdr_df = pd.read_pickle(TRN_HDR_DF_PATH)
tst_hdr_df = pd.read_pickle(TST_HDR_DF_PATH)

trn_df = pd.concat([trn_hdr_df, trn_lbl_df], axis=1, join='inner')

trn_outlier_mask, tst_outlier_mask = get_outlier(trn_df, tst_hdr_df)
valid_trn_df = trn_df[~trn_outlier_mask]

trn_dset_kwargs = {
    'training': True,
    'dataframe': valid_trn_df,
    'window_ranges': DEFAULT_WINDOW_RANGES,
    'equalization': False,
    'regularize_dim': 512,
    'low_memory': True,
}
tst_dset_kwargs = {
    'training': False,
    'dataframe': tst_hdr_df,
    'window_ranges': DEFAULT_WINDOW_RANGES,
    'equalization': False,
    'regularize_dim': 512,
    'low_memory': True,
}

# channel_avgs, channel_stds, nan_sample_ids = \
#     normalize_dset(trn_dset_kwargs, num_workers=-1, batch_size=128)
#
# valid_trn_df = valid_trn_df.drop(nan_sample_ids)
valid_trn_df = valid_trn_df[: (len(valid_trn_df) // 10)]

trn_transform = Compose([
    Resize(IMG_DIM, IMG_DIM),
    # Normalize(mean=channel_avgs, std=channel_stds),
    HorizontalFlip(),
    RandomBrightnessContrast(),
    ShiftScaleRotate(),
    ToTensor()
])
tst_transform = Compose([
    Resize(IMG_DIM, IMG_DIM),
    # Normalize(mean=channel_avgs, std=channel_stds),
    ToTensor()
])

trn_dset = ICHDataset(**trn_dset_kwargs, transform=trn_transform)
tst_dset = ICHDataset(**tst_dset_kwargs, transform=tst_transform)

dldr_kwargs = {
    'batch_size': 32,
    'num_workers': 32,
    'pin_memory': True,
    'timeout': 1000,
}

trn_dldr = DataLoader(trn_dset, **dldr_kwargs)
tst_dldr = DataLoader(tst_dset, **dldr_kwargs)

model = torch.hub.load('facebookresearch/WSL-Images',
                       f'resnext101_32x{NUM_CARDINALITY}d_wsl',).to(DEVICE)

model.fc = torch.nn.Linear(2048, len(DIAGNOSIS)).to(DEVICE)

criterion = torch.nn.BCEWithLogitsLoss()

optim = torch.optim.Adam([{'params': model.parameters(), 'lr': 2e-5}], lr=2e-5)

amp.initialize(model, optim, opt_level='O1')

for epoch in range(NUM_EPOCHS):

    print('Epoch {}/{}'.format(epoch, NUM_EPOCHS - 1))
    print('-' * 10)

    model.train()
    tr_loss = 0

    tk0 = tqdm(trn_dldr, desc="Iteration")

    for step, batch in enumerate(tk0):

        inputs = batch["image"]
        labels = batch["labels"]

        inputs = inputs.to(DEVICE, dtype=torch.float)
        labels = labels.to(DEVICE, dtype=torch.float)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()

        tr_loss += loss.item()

        optim.step()
        optim.zero_grad()

        if epoch == 1 and step > 6000:
            epoch_loss = tr_loss / 6000
            print('Training Loss: {:.4f}'.format(epoch_loss))
            break

        input("Press Enter to continue...")

    epoch_loss = tr_loss / len(trn_dldr)
    print('Training Loss: {:.4f}'.format(epoch_loss))
