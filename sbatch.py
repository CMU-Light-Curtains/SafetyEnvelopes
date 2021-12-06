"""
Example command on Seuss:

python sbatch.py launch --cmd="python sl_train.py with epochs=1000 -p -m sacred --eval_every=20"

"""

import fire
import os

# SBATCH --exclude=compute-0-[5,9]
# SBATCH --exclude=compute-0-[5]


def launch(cmd):
    sbatch_header = f"""#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --exclude=compute-0-[5]
#SBATCH --ntasks-per-node=1
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -o /home/sancha/sout.txt
#SBATCH -e /home/sancha/sout.txt

srun hostname

module load singularity
"""

    singularity_cmd = \
f"""
singularity exec --nv --bind /opt:/opt --bind /opt/cuda/10.0:/usr/local/cuda /home/sancha/sid_16_04.sif bash -c "source ~/.bashrc && {cmd}"
"""

    sbatch_cmd = sbatch_header + singularity_cmd

    # Create temporary file.
    tmp_fname = f".tmp.sbatch.sh"
    with open(tmp_fname, 'w') as f:
        print(sbatch_cmd, file=f)

    os.system(f'cat {tmp_fname}')
    os.system(f'sbatch {tmp_fname}')


if __name__ == '__main__':
    fire.Fire()

