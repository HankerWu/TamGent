# TamGent
Tailoring Molecules for Protein Pockets: a Transformer-based Generative Solution for Structured-based Drug Design

# Introduction

Code base: [fairseq-v0.8.0](https://github.com/facebookresearch/fairseq)

Fairseq(-py) is a sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling and other text generation tasks.

# Installation

```bash
conda create -n TamGent python=3.8
conda activate TamGent

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
conda install pytorch-cluster -c pyg
pip install scipy
pip install rdkit-pypi

git clone https://github.com/xyc1207/TamGent.git
cd TamGent
git checkout main
python -m pip install -e .[chem]
```

# Dataset

The dataset is available at [data](https://microsoftapc-my.sharepoint.com/:f:/g/personal/v-kehanwu_microsoft_com/EmcBPtAwq1JNvgWCRkTsVzwB3vKWh12GXucGA8wtZL0Lnw?e=a3uuiy).

## Build customized dataset

You can build your customized dataset through the following methods:

1. Build customized dataset based on pdb ids, the script will automatically find the binding sites according to the ligands in the structure file.

   ```bash
   python scripts/build_data/prepare_pdb_ids.py ${PDB_ID_LIST} ${DATASET_NAME} -o ${OUTPUT_PATH} -t ${threshold}
   ```

   `PDB_ID_LIST` format: CSV format with columns ([] means optional):

   `pdb_id,[ligand_inchi,uniprot_id]`
2. Build customized dataset based on pdb ids using the center coordinates of the binding site of each pdb.

   ```bash
   python scripts/build_data/prepare_pdb_ids_center.py ${PDB_ID_LIST} ${DATASET_NAME} -o ${OUTPUT_PATH} -t ${threshold}
   ```

   `PDB_ID_LIST` format: CSV format with columns ([] means optional):

   `pdb_id, center_x, center_y, center_z, [uniprot_id]`
3. Build customized dataset based on pdb ids using the center coordinates of the binding site of each pdb, and add the provided scaffold to each center

   ```bash
   python scripts/build_data/prepare_pdb_ids_center_scaffold.py ${PDB_ID_LIST} ${DATASET_NAME} -o ${OUTPUT_PATH} -t ${threshold} --scaffold-file ${SCAFFOLD_FILE}
   ```

   `PDB_ID_LIST` format: CSV format with columns ([] means optional):

   `pdb_id, center_x, center_y, center_z, [uniprot_id]`
   
4. Build dataset from PDB ID list using the residue ids(indexes) of the binding site of each pdb.

   ```bash
   python scripts/build_data/prepare_pdb_ids_res_ids.py ${PDB_ID_LIST} ${DATASET_NAME} -o ${OUTPUT_PATH} --res-ids-fn ${RES_IDS_FN}
   ```

   `PDB_ID_LIST` format: CSV format with columns ([] means optional):

   `pdb_id,[uniprot_id]`

   `RES_IDS_FN` format: residue ids filename, a dict like:

   ```python
   {
     0:
       {
         chain_id_A: Array[res_id_A1, res_id_A2, ...],
         chain_id_B: Array[res_id_B1, res_id_B2, ...],
         ...
       },
     1:
       {
         ...
       },
     ...
   }  
   ```

   stored as pickle file. The order is the same as `PDB_ID_LIST`.

   For customized pdb strcuture files, you can put your structure files to the `--pdb-path` folder, and in the `PDB_ID_LIST` csv file, put the filenames in the `pdb_id` column.

# Model

The pretrained model is available at [model](https://microsoftapc-my.sharepoint.com/:f:/g/personal/v-kehanwu_microsoft_com/EipAXgQfu6lPm1y2OP1ZUyEBsqQbPZ7aukhJ8_hgUej0yw?e=fE9G6h).

# Run scripts

```bash
# train a new model
bash scripts/train.sh -D ${DATA_PATH} --savedir ${SAVED_MODEL_PATH}

# generate molecules
bash scripts/generate.sh -b ${BEAM_SIZE} -s ${SEED} -D ${DATA_PATH} --dataset ${TESTSET_NAME} --ckpt ${MODEL_PATH} --savedir ${OUTPUT_PATH}

```

# Citation

Please cite as:

```bibtex
@inproceedings{TamGent,
  title = {Tailoring Molecules for Protein Pockets: A Transformer-based Generative Solution for Structured-based Drug Design},
  author = {Kehan Wu, Yingce Xia, Yang Fan, Pan Deng, Lijun Wu, Shufang Xie, Tong Wang, Haiguang Liu, Tao Qin and Tie-Yan Liu},
  year = {2022},
}
```

