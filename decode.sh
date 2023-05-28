export CUDA_VISIBLE_DEVICES=2

for seed in {1..1}; do
for beta in "0.1" "1"; do
# for A in "10" "15" "20"; do
bash scripts/generate.sh --beam 20 --seed $seed -D /home/yinxia/workdir/MSR-EWHA/Tamgent/GPCR-8CU6-t10 --dataset GPCR --ckpt /home/yinxia/workdir/MSR-EWHA/Tamgent/checkpoints/crossdock_pdb_A10/checkpoint_best.pt --savedir ./GPCR-8cu6-decode-t10-v1 --beta $beta --gen-vae

bash scripts/generate.sh --beam 20 --seed $seed -D /home/yinxia/workdir/MSR-EWHA/Tamgent/GPCR-8CU6-t15 --dataset GPCR --ckpt /home/yinxia/workdir/MSR-EWHA/Tamgent/checkpoints/crossdock_pdb_A10/checkpoint_best.pt --savedir ./GPCR-8cu6-decode-t15-v1 --beta $beta --gen-vae

# bash scripts/generate.sh --beam 20 --seed $seed -D dataset/GPCR-${A}A --dataset GPCR-${A}A --ckpt checkpoints/A${A}/checkpoint_best.pt --savedir ./outputs --gen-vae --beta $beta
# done
done
done
