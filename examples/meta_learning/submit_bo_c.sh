#!/bin/bash
#
#SBATCH -J bo-c
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task 4
#SBATCH --time=200:00:00
#SBATCH --partition=cpu
#SBATCH --qos=nopreemption
#SBATCH --export=ALL
#SBATCH --output=bo-c.log
#SBATCH --gres=gpu:0

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/h/rhickman/sw/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/h/rhickman/sw/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/h/rhickman/sw/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/rhickman//sw/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate hermes

date >> bo-c.log
echo "" >> bo-c.log
python run_bo.py c
echo "" >> bo-c.log
date >> bo-c.log
