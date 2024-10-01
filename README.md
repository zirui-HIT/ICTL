# ICTL

## Preprocess
1. Create conda environment with requirements.txt
2. Download [Super-NI](https://github.com/allenai/natural-instructions) and put it into ./dataset/Super-NI

## Source Sampling
1. Embed the dataset with ./embed/scripts/embed.sh
2. Sample the source demonstrations with ./sample/slurm/sample.sh
3. Embed the sampled demonstrations with ./sample/slurm/embed.sh

## Target Transfer
1. Transfer the sampled demonstrations with ./transfer/slurm/transfer.slurm
2. Verify the transferred demonstrations with ./transfer/slurm/verify.slurm
3. Embed the verified demonstrations for the following sampling with ./transfer/slurm/embed.slurm
4. Sample the verified demonstrations with ./transfer/slurm/sample.slurm

## Inference
1. Inference the answer with ./generate/slurm/generate.sh
2. Evaluate the results with ./generate/slurm/generate.sh