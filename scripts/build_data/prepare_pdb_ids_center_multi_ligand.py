#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Build dataset from PDB ID list using the center coordinates of the binding site of each pdb.

PDB ID list format: CSV format with columns ([] means optional):
pdb_id, center_x, center_y, center_z, [uniprot_id]

For customized pdb strcuture files, you can put your structure files to the `--pdb-path` folder, and in the csv file, put the filenames in the `pdb_id` column. 
"""

import argparse
import logging
import random
from pathlib import Path
from os.path import dirname as parent
import copy

import numpy as np
import torch
from fairseq.molecule_utils import config
from fairseq.molecule_utils.database.caching_utils import get_cache
from fairseq.molecule_utils.external.fairseq_dataset_build_utils import process_one_pdb_given_center_coord, dump_center_data_multi_ligand
from fy_common_ext.io import csv_reader
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Build dataset from PDB ID list using the center coordinates of the binding site of each pdb.')
    parser.add_argument('pdb_id_list', type=Path, help='PDB ID list csv file with binding site center')
    parser.add_argument('name', help='Dataset name')
    parser.add_argument('-t', '--threshold', type=float, default=10.0,
                        help='Near center threshold, default is %(default)s')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Output data folder')
    parser.add_argument('-pp', '--pdb-path', type=Path, default=config.pdb_cache_path(), help='PDB file path')
    parser.add_argument('-c', '--cache-size', type=int, default=200, help='PDB file cache size')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--ligand-smiles-file', type=str, default=None, help='path to the ligand smiles path')
    parser.add_argument('--sample-weight-file', type=str, default=None, help='path to the sample weight file')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s:%(filename)s:%(message)s', level=logging.DEBUG if args.verbose else logging.INFO)
    get_cache('af2_mmcif').max_size = args.cache_size
    np.random.seed(1234)
    random.seed(1234)

    with csv_reader(args.pdb_id_list, dict_reader=True) as reader:
        input_list = list(reader)

    all_data = []
    for index, input_row in enumerate(tqdm(input_list)):
        print(f'Processing {input_row["pdb_id"]}')
        data = process_one_pdb_given_center_coord(
            index, input_row, threshold=args.threshold, pdb_mmcif_path=args.pdb_path,
        )
        if data is not None:
            all_data.append(data)

    with open(args.ligand_smiles_file) as fr:
        # use ' ' as the delimiter
        all_ligands = [e.strip().split(' ') for e in fr]
    
    if args.sample_weight_file is not None:
        sample_weights = []
        with open(args.sample_weight_file) as fr:
            for line in fr:
                weights = line.strip().split(' ')
                sample_weights.append(torch.tensor([float(e) for e in weights]))
    else: 
        sample_weights = None
        
    dump_center_data_multi_ligand(all_data, args.name, output_dir=args.output,
              fairseq_root=Path(parent(parent(parent(__file__)))), 
              pre_dicts_root=Path(parent(parent(parent(__file__)))) / 'dict', 
              max_len=1023, ligand_list=all_ligands, sample_weights=sample_weights)


if __name__ == '__main__':
    main()
