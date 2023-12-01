import os
import numpy as np
from tqdm import tqdm


def split_dataset(data_path, label_path, prefix, target_dir):
    X_train = np.load(data_path)
    y_train = np.load(label_path)
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with tqdm(total=len(X_train)) as tbar:
        for i in range(len(X_train)):
            np.save(
                os.path.join(target_dir, prefix + "_data_{}".format(i)),
                np.concatenate((X_train[i], y_train[i]), axis=2)
            )
            # np.save(
            #     os.path.join(target_dir, prefix + "_label_{}".format(i)),
            #     np.asarray(y_train[i])
            # )
            tbar.update(1)


split_dataset("../data/5000_01_travel_data_32.npy", "../data/5000_01_travel_label_32.npy", prefix="travel01", target_dir="../data/travel01")