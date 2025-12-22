from dataclasses import dataclass, field
from typing import Callable, Optional, Any
import os
import torch
import torchaudio.transforms as transforms
import torchaudio
# import librosa
from joblib import Parallel, delayed
import ast
import numpy as np
import pandas as pd
from torch.utils.data import Sampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class AudioConfig:
    """
    Central configuration for Audio Processing.
    """
    sr: int = 32000
    n_fft: int = 2048
    win_length: int = n_fft
    hop_length: int = win_length // 2
    n_mels: int = 256
    top_db: int = 80

    # Slicing/Striping params
    stripe_width: int = sr // (n_fft // 4)
    stripe_overlap: int = stripe_width // 5

    # Helper to inject external slicing logic
    slicing_func: Any = None

    prior_cut_sec: int = 5


class BirdDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            label_encoder,
            meta_df: pd.DataFrame,
            config: AudioConfig,
            group_mode: bool = True,
            return_id: bool = True,
            prediction_as_target: bool = False,
            root_dir: str = '/kaggle/input/birdclef-2025/train_audio',
            multitarget: bool = True,
    ):
        """
        Stores waves and ohe labels (possible storage of predictions)

        Args:
            label_encoder: encoder text -> id & id -> text, should implement .fit .transform methods
            meta_df: dataframe with audio file path / primary label / secondary label
            config: audio processing config
            group_mode: flag to create targets file wise or chunk wise
            return_id: flag to return file path with waves and targets
            prediction_as_target: flag to return precalculated predictions instead of true targets,
            root_dir: audio files directory,
            multitarget: flag to include secondary labels into targets,
        """
        super().__init__()
        # sklearn labelencoder
        self.label_encoder = label_encoder

        self.cfg = config
        self.root_dir = root_dir
        self.group_mode = group_mode
        self.return_id = return_id
        self.multitarget = multitarget
        self.prediction_as_target = prediction_as_target

        # will serve as pseudo labels storage later
        self.predictions = None

        # creating spectrograms and preparing labels
        results_with_index = self._parallel_prepare(meta_df)

        # extract spectrograms and labels
        self.spectrs, self.labels, self.idx = zip(*results_with_index)

        if not group_mode:
            self.labels = list(self.labels)

            # expand labels with respect to token_num in each sample
            for i, tokenized_tensor in enumerate(self.spectrs):
                expand_size = tokenized_tensor.shape[-1]
                self.labels[i] = np.tile(self.labels[i], (expand_size, 1))

            # concatenate all tensor into 1 huge brick, because we don't care about their relations
            self.spectrs = torch.cat(self.spectrs, dim=-1)
            self.labels = [arr for sublist in self.labels for arr in sublist]

        # label tensor init with n_classes
        label_tensor = torch.zeros(len(self.labels), len(self.label_encoder.classes_))  # B, L
        # ohe assigning
        # todo it seems very slow, but I don't know better approach for multi label ohe assigning
        # because once again, self.labels isn't a size=1 array
        for i, label in enumerate(self.labels):
            label_tensor[i, label] = 1

        self.labels = label_tensor
        self.idx = np.array(self.idx, dtype=object)

        self.token_h, self.token_w = config.n_mels, config.stripe_width

    def preprocess_label_file(
            self,
            file_path,
            label,
            secondary_labels,
            root_dir=None,
    ):
        """
        Reads audio -> STFT -> Mel/Mel**2/Linear Features -> Stacks them.
        Reads id(str) -> id from label_encoder
        """
        if root_dir is None:
            root_dir = self.root_dir
        path = os.path.join(root_dir, file_path)
        waveform, sr = torchaudio.load(path, num_frames=self.cfg.sr * self.cfg.prior_cut_sec)

        if sr != self.cfg.sr:
            waveform = transforms.Resample(sr, self.cfg.sr)(waveform)

        waveform = waveform[0]

        label_id = self.label_encoder.transform([label])

        # secondary label processing
        # str -> list
        secondary_labels = ast.literal_eval(secondary_labels)
        # if secondary labels actually exist:
        if (secondary_labels != ['']) and self.multitarget:
            secondary_labels = self.label_encoder.transform(secondary_labels)
            label_id = np.concatenate((label_id, secondary_labels))

        slices = self.cfg.slicing_func(  # [C, H, W] -> [C, H, W, T]
            ar=waveform.unsqueeze(0).unsqueeze(0),
            stripe=self.cfg.stripe_width * self.cfg.hop_length,
            overlap=self.cfg.stripe_overlap * self.cfg.hop_length,
            pad_value=0,
        )

        return slices, label_id, file_path

    def _parallel_prepare(self, df):
        """
        parallel file processing using joblib
        """

        columns_of_interest = ['filename', 'primary_label', 'secondary_labels']

        results_with_index = Parallel(n_jobs=-1)(
            delayed(self.preprocess_label_file)(
                file_path=row.filename,
                label=row.primary_label,
                secondary_labels=row.secondary_labels,
            )
            for _, row in df[columns_of_interest].iterrows()
        )

        return results_with_index

    def update_pseudo_labels(self, obj: pd.Series):
        """
        appends new pseudolabels into existing pseudolabels storage
        """
        if self.predictions is None:
            self.predictions = pd.Series()

        self.predictions = pd.concat([self.predictions, obj])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        returns batch with ohe target
        """
        if self.group_mode:
            # it is the only option to fix dataloader select random samples via list problem
            # we can't select tuple[list], so instead we have to iterate
            # yes, it's the ONLY PLACE, we can't even use collate_fn
            if isinstance(index, list) or isinstance(index, np.ndarray):
                batch_tensor = [self.spectrs[i] for i in index]
            else:
                batch_tensor = self.spectrs[index]

        else:
            # because in not group mode we iterate through tokens and they are in the last dim
            batch_tensor = self.spectrs[..., index]

        str_id = self.idx[index]
        # pseudo labels
        if self.prediction_as_target:
            label_tensor = self.predictions[str_id].values
        else:
            label_tensor = self.labels[index]

        # in predict mode, I also want to retrieve a file name
        if self.return_id:
            return batch_tensor, label_tensor, str_id
        else:
            return batch_tensor, label_tensor


class OnlineDataset(torch.utils.data.Dataset):
    """
    Essentially, just loads only needed part of default BirdDataset
    """
    def __init__(
            self,
            meta_df,
            label_encoder,
            config: AudioConfig,
            group_mode: bool = True,
            return_id: bool = True,
            prediction_as_target: bool = False,
            root_dir: str = '/kaggle/input/birdclef-2025/train_audio/',
            multitarget: bool = True,
    ):
        super().__init__()
        # sklearn labelencoder
        self.label_encoder = label_encoder

        self.cfg = config
        self.root_dir = root_dir
        self.group_mode = group_mode
        self.return_id = return_id
        self.multitarget = multitarget
        self.prediction_as_target = prediction_as_target

        # df that contains info about each file
        self.meta_df = meta_df
        self.predictions = None

    def update_pseudo_labels(self, obj: pd.Series):
        """
        appends new pseudolabels into existing pseudolabels storage
        """
        if self.predictions is None:
            self.predictions = pd.Series()

        self.predictions = pd.concat([self.predictions, obj])

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, index):
        temp_ds = BirdDataset(
            meta_df=self.meta_df.iloc[index],
            label_encoder=self.label_encoder,
            config=self.cfg,
            group_mode=self.group_mode,
            return_id=self.return_id,
            prediction_as_target=self.prediction_as_target,
            root_dir=self.root_dir,
            multitarget=self.multitarget,
        )
        if self.prediction_as_target:
            temp_ds.predictions = self.predictions[index]

        return temp_ds[:]


# AI generated (not slope)
class IndexBatchSampler(Sampler):
    def __init__(self, data_source, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        """
        Yields a list of indices at each iteration instead of a single index.

        Args:
            data_source: The dataset (used to determine length).
            batch_size: Number of indices to yield per iteration.
            shuffle: Whether to shuffle indices before batching.
            drop_last: Whether to drop the last incomplete batch.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        # Create a list of all indices
        indices = list(range(len(self.data_source)))

        if self.shuffle:
            np.random.shuffle(indices)

        # Yield chunks (batches) of indices
        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # Handle the remaining items
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # Calculate how many batches this sampler will produce
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size


class PseudoOnlineDatasets(torch.utils.data.Dataset):
    """
    concatenates onlinedatasets with random sampling
    """
    def __init__(self, datasets: list[OnlineDataset], group_mode: bool = True):
        super().__init__()
        self.datasets = datasets
        self.lens = np.array([len(dataset.meta_df) for dataset in self.datasets])
        self.lens_sum = np.sum(self.lens)
        self.sample_p = self.lens / np.sum(self.lens)
        self.group_mode = group_mode

    def __len__(self):
        return self.lens_sum

    def __getitem__(self, index):
        ds_id = np.random.choice(len(self.datasets), p=self.sample_p)
        ds = self.datasets[ds_id]
        rescaled_index = np.floor(index / self.lens_sum * self.lens[ds_id]).astype(int)
        return ds[rescaled_index]
