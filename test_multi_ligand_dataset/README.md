# Build Multi-ligand Dataset

Build customized dataset based on pdb ids using the center coordinates of the binding site of each pdb. 

```bash
python scripts/build_data/prepare_pdb_ids_center_multi_ligand.py ${PDB_ID_LIST} ${DATASET_NAME} -o ${OUTPUT_PATH} -t ${threshold}--ligand-smiles-file ${LIGAND_SMILES_FILE} --sample-weight-file ${SAMPLE_WEIGHT_FILE}
```

`PDB_ID_LIST` format: CSV format with columns ([] means optional):

   `pdb_id, center_x, center_y, center_z, [uniprot_id]`

`LIGAND_SMILES_FILE` and `SAMPLE_WEIGHT_FILE` format: Each row corresponds to a pocket, and multiple ligands(sample weights) within each row are separated by spaces.

If `SAMPLE_WEIGHT_FILE` is not provided, the script will automatically set the sample weights of each ligand to equal.

# Run Inference with multi-ligand
Add `--vae --gen-vae --multi-ligand` arguments:
```bash
bash scripts/generate.sh -b ${BEAM_SIZE} -s ${SEED} -D ${DATA_PATH} --dataset ${TESTSET_NAME} --ckpt ${MODEL_PATH} --savedir ${OUTPUT_PATH} --vae --gen-vae --multi-ligand
```