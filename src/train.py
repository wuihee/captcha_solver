import glob
import os

import numpy as np
import torch
from sklearn import metrics, model_selection, preprocessing

import config
import dataset
import engine
from model import CaptchaModel


def run_training():
    # image_files = [file for file in os.listdir(config.DATA_DIR)]
    os.chdir('/Users/wuihee/Desktop/Programming/Scripts/CAPTCHA Solver/src')
    image_files = glob.glob(os.path.abspath(os.path.join(config.DATA_DIR, "*.png")))

    # "/.../.../abcd.png" -> Grabbing the answers of the captcha.
    targets_orig = [x.split('\\')[-1][:-4] for x in image_files]

    # "abcd" -> ['a', 'b', 'c', 'd']
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]

    # Label encoder.
    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    targets_enc = [lbl_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc) + 1

    # print(targets_enc)
    # print(len(lbl_enc.classes_))

    (
        train_imgs,
        test_imgs,
        train_targets,
        test_targets,
        train_orig_targets,
        test_orig_targets
    ) = model_selection.train_test_split(image_files, targets_enc, targets_orig, test_size=0.1, random_state=42)

    train_dataset = dataset.ClassificationDataset(image_paths=train_imgs, targets=train_targets, resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True)

    test_dataset = dataset.ClassificationDataset(image_paths=test_imgs, targets=test_targets, resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False)

    model = CaptchaModel(num_chars=len(lbl_enc.classes_))
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=-.8, patience=5, verbose=True)
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer)
        valid_preds, valid_loss = engine.eval_fn(model, train_loader)


if __name__ == "__main__":
    run_training()
