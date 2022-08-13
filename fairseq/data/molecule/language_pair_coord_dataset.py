#! /usr/bin/python
# -*- coding: utf-8 -*-

from ..language_pair_dataset import LanguagePairDataset, collate
from ..data_utils import collate_tokens_multidim


def collate_coordinates(
        samples, batch, sort_order, coord_mode,
        left_pad_source=True):
    if len(samples) == 0:
        return batch

    if coord_mode != 'raw':
        raise NotImplementedError

    def merge(key, left_pad):
        return collate_tokens_multidim(
            [s[key] for s in samples],
            pad_value=0.0, left_pad=left_pad, move_eos_to_beginning=False,
        )

    if samples[0].get('src_coord', None) is not None:
        feature = merge('src_coord', left_pad=left_pad_source)
        feature = feature.index_select(0, sort_order)
        batch['net_input']['src_coord'] = feature

    return batch

def collate_target(samples, batch, input_target=False):
    if len(samples) == 0:
        return batch
    if input_target:
        batch['net_input']['tgt_tokens'] = batch['target']
    return batch


class LanguagePairCoordinatesDataset(LanguagePairDataset):
    """LanguagePairDataset + 3D coordinates.

    Args:
        src_coord (torch.utils.data.Dataset): source coordinates dataset to wrap
            Each item is a float tensor in shape `(src_len, 3)`.
    """

    AVAILABLE_COORD_MODES = ('flatten', 'raw')

    def __init__(self, *args, **kwargs):
        src_coord = kwargs.pop('src_coord', None)
        coord_mode = kwargs.pop('coord_mode', 'raw')
        input_target = kwargs.pop('input_target', False)

        if coord_mode not in self.AVAILABLE_COORD_MODES:
            raise ValueError(f'Unknown coordinate mode {coord_mode}')

        super().__init__(*args, **kwargs)

        self.src_coord = src_coord
        self.coord_mode = coord_mode
        self.input_target = input_target

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        src_coord_item = self.src_coord[index] if self.src_coord is not None else None
        sample['src_coord'] = src_coord_item
        return sample

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        See `LanguagePairDataset.collater` for more details.

        Returns:
            dict: a mini-batch with the keys in `LanguagePairDataset.collater` and following *extra* keys:

                - `src_coord` (FloatTensor): an 3D Tensor of coordinates of source tokens.
        """
        try:
            batch, sort_order = collate(
                samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
                left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
                input_feeding=self.input_feeding, return_sort_order=True,
            )
        except ValueError:
            return {}
            
        collate_coordinates(
            samples, batch, sort_order, self.coord_mode,
            left_pad_source=self.left_pad_source,
        )
        collate_target(samples, batch, input_target=self.input_target)
        return batch

    @property
    def supports_prefetch(self):
        return (
            super().supports_prefetch
            and (getattr(self.src_coord, 'supports_prefetch', False) or self.src_coord is None)
        )

    def prefetch(self, indices):
        super().prefetch(indices)
        if self.src_coord is not None:
            self.src_coord.prefetch(indices)

    @classmethod
    def from_base_dataset(cls,
                          base,
                          src_coord=None,
                          coord_mode='raw',
                          input_target=False):
        """Create dataset from base dataset.

        Args:
            base (LanguagePairDataset): the original dataset
            src_coord (torch.utils.data.Dataset): source coordinates dataset to wrap
            coord_mode (str): coordinates representation mode

        Returns:
            LanguagePairCoordinatesDataset:
        """

        return cls(base.src,
                   base.src_sizes,
                   base.src_dict,
                   tgt=base.tgt,
                   tgt_sizes=base.tgt_sizes,
                   tgt_dict=base.tgt_dict,
                   left_pad_source=base.left_pad_source,
                   left_pad_target=base.left_pad_target,
                   max_source_positions=base.max_source_positions,
                   max_target_positions=base.max_target_positions,
                   shuffle=base.shuffle,
                   input_feeding=base.input_feeding,
                   remove_eos_from_source=base.remove_eos_from_source,
                   append_eos_to_target=base.append_eos_to_target,
                   src_coord=src_coord,
                   coord_mode=coord_mode,
                   input_target=input_target)
