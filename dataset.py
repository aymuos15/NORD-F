import os
import scipy.io
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# DATA_DIR = "/home/localssk23/Downloads/ishika/data/DOGS"
# FLOWERS_DIR = "/home/localssk23/Downloads/ishika/data/Oxford_Flowers_102/"

class StanfordDogsDataset(Dataset):
    def __init__(self, root_dir, dataset_type='train', transform=None, random_seed= 14, subset_fraction=0.2, flowers_dir=None):
        self.root_dir = root_dir
        self.transform = transform
        self.flowers_dir = flowers_dir
        self.random_seed = random_seed

        images_dir = os.path.join(root_dir, 'Images')
        split_file = os.path.join(root_dir, 'train_list.mat' if dataset_type != 'test' else 'test_list.mat')

        split_data = scipy.io.loadmat(split_file)
        images_files = [os.path.join(images_dir, f[0][0]) for f in split_data['file_list']]
        labels = [int(f[0]) - 1 for f in split_data['labels']]  # Convert labels to zero-indexed

        img_df = pd.DataFrame({'Image': images_files, 'Label': labels})

        top_classes = img_df['Label'].value_counts().nlargest(5).index.tolist()
        df_top_classes = img_df[img_df['Label'].isin(top_classes)].copy()

        train_df, temp_df = train_test_split(
            df_top_classes,
            test_size=0.2,
            stratify=df_top_classes['Label'],
            random_state=random_seed
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df['Label'],
            random_state=random_seed
        )

        if dataset_type == 'train':
            df = train_df
        elif dataset_type == 'val':
            df = val_df
        elif dataset_type == 'test':
            df = test_df
        elif dataset_type == 'train_subset':
            df_subset = train_df.sample(n=100, random_state=random_seed).reset_index(drop=True).copy()
            df = df_subset
        elif dataset_type in ['ood', 'ood_subset']:
            df_ood_classes = img_df[~img_df['Label'].isin(top_classes)].copy()
            ood_classes = df_ood_classes['Label'].unique().tolist()
            label_map = {old_label: new_label + 5 for new_label, old_label in enumerate(ood_classes)}
            df_ood_classes['Label'] = df_ood_classes['Label'].map(label_map)
            if dataset_type == 'ood':
                df = df_ood_classes.sample(frac=1, random_state=random_seed).reset_index(drop=True).copy()
            elif dataset_type == 'ood_subset':
                df = df_ood_classes.sample(n=100, random_state=random_seed).reset_index(drop=True).copy()
        elif dataset_type == 'val_test':
            df = pd.concat([val_df, test_df]).reset_index(drop=True).copy()
        elif dataset_type in ['far_ood', 'far_ood_subset']:
            flowers_images_dir = os.path.join(self.flowers_dir, 'jpg')
            imagelabels = scipy.io.loadmat(os.path.join(self.flowers_dir, 'imagelabels.mat'))['labels'][0]
            setid = scipy.io.loadmat(os.path.join(self.flowers_dir, 'setid.mat'))
            test_set = setid['tstid'][0]
            val_set = setid['valid'][0]
            train_set = setid['trnid'][0]

            # Combine all sets into a single dataframe
            all_set = np.concatenate([test_set, val_set, train_set])
            flowers_files = [os.path.join(flowers_images_dir, f'image_{i:05d}.jpg') for i in all_set]
            flowers_labels = [imagelabels[i - 1] for i in all_set]  # Convert to zero-indexed

            flowers_df = pd.DataFrame({'Image': flowers_files, 'Label': flowers_labels})

            # Relabel so that it doesn't clash with any existing labels
            flowers_df['Label'] = flowers_df['Label'] + 200  # Start far_ood labels from 200

            self.flowers_images = flowers_files  # Store flower images

            if dataset_type == 'far_ood':
                df = flowers_df.reset_index(drop=True).copy()
            elif dataset_type == 'far_ood_subset':
                df = flowers_df.sample(n=100, random_state=random_seed).reset_index(drop=True).copy()
        else:
            raise ValueError("Invalid dataset_type. Supported types: 'train', 'val', 'ood', 'test', 'train_subset', 'ood_subset', 'val_test', 'far_ood', 'far_ood_subset'.")

        if dataset_type not in ['ood', 'ood_subset', 'far_ood', 'far_ood_subset']:
            label_map = {old_label: new_label for new_label, old_label in enumerate(top_classes)}
            df['Label'] = df['Label'].map(label_map)

        self.img_name_list = df['Image'].tolist()
        self.label_list = df['Label'].tolist()

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        np.random.seed(idx + self.random_seed)  # Set seed based on index and random_seed

        img_path_1 = self.img_name_list[idx]
        image_1 = Image.open(img_path_1).convert('RGB')

        idx_2 = np.random.randint(len(self.img_name_list))
        img_path_2 = self.img_name_list[idx_2]
        image_2 = Image.open(img_path_2).convert('RGB')

        target_1 = self.label_list[idx]
        target_2 = self.label_list[idx_2]

        same_target = 0 if target_1 == target_2 else 1

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2, target_1, target_2, same_target