export CUDA_VISIBLE_DEVICES=2

for seed in {1..2}; do
for beta in "0.1" "1"; do

bash ../scripts/generate.sh --beam 20 --seed $seed -D gpcr-bin-cond --dataset GPCR --ckpt checkpoints/crossdock_pdb_A10/checkpoint_best.pt --savedir ./gpcr-outs-condition --beta $beta --gen-vae

done
done
