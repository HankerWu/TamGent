#%%
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem

import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_folder', type=str,
                    help='result folder')
parser.add_argument('output_file', type=str, 
                    help='output file', default='gpcr_unconditional_output')

args = parser.parse_args()
results = defaultdict(set)

def one_file(fname, results):
    with open(fname, "r", encoding="utf8") as fr:
        all_lines = [e.strip() for e in fr]
    for e in all_lines:
        segs = e.strip().split("\t")
        smi = segs[1]
        if smi.count("(") != smi.count(")"):
            continue
        if "c" not in smi and "n" not in smi:
            continue
        idx = int(segs[0].replace("H-", ""))
        results[idx].add(smi)

FF = glob.glob(f"{args.input_folder}/*/output.txt")
for ff in tqdm(FF,total=len(FF)):
    one_file(ff, results)

all_results = set()
for v in tqdm(results.values(),total=len(results)):
    all_results = all_results.union(v)



can_smiles = set()
for e in all_results:
    m = Chem.MolFromSmiles(e)
    if m is None:
        continue
    ssr = Chem.GetSymmSSSR(m)
    if len(ssr) < 1:
        continue
    smi = Chem.MolToSmiles(m)
    can_smiles.add(smi)
    
with open(f"{args.output_file}", "w", encoding="utf8") as fw:
    for e in can_smiles:
        print(e,file=fw)
