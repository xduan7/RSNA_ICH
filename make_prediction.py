""" 
    File Name:          RSNA_ICH/make_prediction.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/2/19
    Python Version:     3.5.4
    File Description:   

"""
import click
from apex import amp
from albumentations import Compose, ShiftScaleRotate, Resize, \
    HorizontalFlip, RandomBrightnessContrast, Normalize
from albumentations.pytorch import ToTensor

from utilities.csv_processing import tst_lbl_df_to_submission_csv
from utilities.ich_dataset import *

NUM_CARDINALITY = 8
NUM_EPOCHS = 10
IMG_DIM = 256



@click.command()
@click.argument('device_num', type=int)
@click.option('--equalization', '-e', is_flag=True)
@click.option('--normalization', '-n', is_flag=True)
def make_prediction(
        device_num: int,
        equalization: bool = False,
        normalization: bool = False,
):
    device = torch.device(f'cuda:{device_num}')

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
        'equalization': equalization,
        'regularize_dim': 512,
        'low_memory': True,
    }
    tst_dset_kwargs = {
        'training': False,
        'dataframe': tst_hdr_df,
        'window_ranges': DEFAULT_WINDOW_RANGES,
        'equalization': equalization,
        'regularize_dim': 512,
        'low_memory': True,
    }

    if normalization:
        channel_avgs, channel_stds, nan_sample_ids = \
            normalize_dset(trn_dset_kwargs, num_workers=-1, batch_size=128)

        valid_trn_df.drop(nan_sample_ids, inplace=True)

        trn_transform = Compose([
            Resize(IMG_DIM, IMG_DIM),
            Normalize(mean=channel_avgs, std=channel_stds),
            HorizontalFlip(),
            RandomBrightnessContrast(),
            ShiftScaleRotate(),
            ToTensor()
        ])
        tst_transform = Compose([
            Resize(IMG_DIM, IMG_DIM),
            Normalize(mean=channel_avgs, std=channel_stds),
            ToTensor()
        ])
    else:
        trn_transform = Compose([
            Resize(IMG_DIM, IMG_DIM),
            HorizontalFlip(),
            RandomBrightnessContrast(),
            ShiftScaleRotate(),
            ToTensor()
        ])
        tst_transform = Compose([
            Resize(IMG_DIM, IMG_DIM),
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
                           f'resnext101_32x{NUM_CARDINALITY}d_wsl',).to(device)

    model.fc = torch.nn.Linear(2048, len(DIAGNOSIS)).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(
        weight=torch.FloatTensor([2, 1, 1, 1, 1, 1]).to(device))

    optim = torch.optim.Adam(params=model.parameters(),
                             lr=2e-5)

    amp.initialize(model, optim, opt_level='O1')

    for epoch in range(NUM_EPOCHS):

        print('Epoch {}/{}'.format(epoch, NUM_EPOCHS - 1))
        print('-' * 10)

        model.train()
        tr_loss = 0

        trn_iter = tqdm(trn_dldr, desc="Iteration")

        for step, batch in enumerate(trn_iter):

            inputs = batch["image"]
            labels = batch["labels"]

            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

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

        epoch_loss = tr_loss / len(trn_dldr)
        print('Training Loss: {:.4f}'.format(epoch_loss))

        with torch.no_grad():

            ids = []
            predictions = []
            tst_iter = tqdm(tst_dldr, desc="Iteration")
            for step, batch in enumerate(tst_iter):

                inputs = batch['image']
                ids.extend(batch['id'])

                inputs = inputs.to(device, dtype=torch.float)
                # labels = labels.to(device, dtype=torch.float)

                predictions.extend(model(inputs).tolist())

            pred_df = pd.DataFrame(predictions, columns=DIAGNOSIS, index=ids)

            masked_pred_df = pred_df.copy(deep=True)
            masked_pred_df.loc[tst_outlier_mask] = [0, 0, 0, 0, 0, 0]

            tst_lbl_df_to_submission_csv(
                pred_df,
                f'./e{equalization}n{normalization}_{epoch+1}.csv')
            tst_lbl_df_to_submission_csv(
                masked_pred_df,
                f'./masked_e{equalization}n{normalization}_{epoch + 1}.csv')


if __name__ == '__main__':
    make_prediction()
