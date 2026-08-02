source /opt/ros/noetic/setup.bash
source "$HOME/YOPO/Controller/devel/setup.bash"
source "$HOME/YOPO/Simulator/devel/setup.bash" --extend

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yopo

export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:${PYTHONPATH:-}"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
