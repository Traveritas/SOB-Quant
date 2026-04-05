conda create -n yaqa_env python=3.10 -y
conda activate yaqa_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets glog tqdm accelerate
cd yaqa-quantization
pip install -r requirements.txt
$env:CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
pip install fast-hadamard-transform --no-build-isolation
cd qtip-kernels
python setup.py install
cd ..
