#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch
from ..language_pair_dataset import LanguagePairDataset, collate
from ..data_utils import collate_tokens_multidim, collate_tokens


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


def collate_multi_ligands(samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=False, return_sort_order=False, split_token=None):
    if len(samples) == 0:
        return {}
    
    def merge(key, left_pad, move_eos_to_beginning=False):
        return collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )
    
    def split_tensor(tensor, split_token):
        split_indices = torch.nonzero(tensor == torch.ones_like(tensor) * split_token).squeeze(1)
        split_indices = torch.cat([torch.LongTensor([-1]), split_indices, torch.LongTensor([tensor.size(0)])])
        split_tensors = []
        for i in range(split_indices.size(0) - 1):
            split_tensors.append(tensor[split_indices[i]+1:split_indices[i+1]])
        return split_tensors

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    targets = None
    sample_ratios = None

    if samples[0].get('target', None) is not None:
        if split_token is None:
            targets = merge('target', left_pad=left_pad_target)
        else:
            for s in samples:
                # split target into multiple ligands
                s['target'] = split_tensor(s['target'], split_token)
                
            max_ligand_num = max(len(s['target']) for s in samples)
            targets = []
            ntokens = 0
            
            for i in range(max_ligand_num):
                target_i = collate_tokens([s['target'][i] if i < len(s['target']) else (torch.ones(1)*pad_idx).long() for s in samples], pad_idx, eos_idx, left_pad_target, move_eos_to_beginning=False)
                target_i = target_i.index_select(0, sort_order)
                for s in samples:
                    ntokens += len(s['target'][i]) if i < len(s['target']) else 0
                targets.append(target_i)
            
            if samples[0].get('sample_ratios', None) is not None:
                sample_ratios = []
                for i in range(max_ligand_num):
                    sample_ratios.append(torch.FloatTensor([s['sample_ratios'][i] if i < len(s['sample_ratios']) else torch.zeros(1) for s in samples]))
                sample_ratios = torch.stack(sample_ratios, dim=1)
            else:
                sample_ratios = torch.ones(len(samples), max_ligand_num)
                for i in range(len(samples)):
                    for j in range(max_ligand_num):
                        if j >= len(samples[i]['target']):
                            sample_ratios[i][j] = 0
            
            sample_ratios = sample_ratios/torch.sum(sample_ratios, dim=1, keepdim=True)
            sample_ratios = sample_ratios.index_select(0, sort_order)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
            
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'tgt_tokens': targets,
            'sample_ratios': sample_ratios,
        },
        'target': targets,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    if return_sort_order:
        return batch, sort_order
    else:
        return batch

class LanguagePairCoordinatesMultiLigandsDataset(LanguagePairDataset):
    """LanguagePairDataset + 3D coordinates + multi-ligands with sample ratio.
    """
    AVAILABLE_COORD_MODES = ('flatten', 'raw')

    def __init__(self, src, src_sizes, src_dict, src_coord, coord_mode='raw', input_target=True,
                 tgt=None, tgt_sizes=None, tgt_dict=None, sample_ratios=None,
                 left_pad_source=True, left_pad_target=False,
                 max_source_positions=1024, max_target_positions=1024,
                 shuffle=True, input_feeding=False, remove_eos_from_source=True, append_eos_to_target=False,):
        super().__init__(src, src_sizes, src_dict, tgt, tgt_sizes, tgt_dict, left_pad_source, left_pad_target,
                         max_source_positions, max_target_positions, shuffle, input_feeding, remove_eos_from_source,
                         append_eos_to_target)
        self.src_coord = src_coord
        self.coord_mode = coord_mode
        self.input_target = input_target
        
        if coord_mode not in self.AVAILABLE_COORD_MODES:
            raise ValueError(f'Unknown coordinate mode {coord_mode}')
        
        self.sample_ratios = sample_ratios

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        src_coord_item = self.src_coord[index] if self.src_coord is not None else None
        sample['src_coord'] = src_coord_item
        sample['sample_ratios'] = self.sample_ratios[index] if self.sample_ratios is not None else None
        return sample
    
    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        See `LanguagePairDataset.collater` for more details.

        Returns:
            dict: a mini-batch with the keys in `LanguagePairDataset.collater` and following *extra* keys:

                - `src_coord` (FloatTensor): an 3D Tensor of coordinates of source tokens.
        """
        split_token = 962
        batch, sort_order = collate_multi_ligands(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding, return_sort_order=True, split_token=split_token,
        )
            
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
