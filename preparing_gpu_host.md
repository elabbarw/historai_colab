## Steps to prepare a Whisper or Faster-Whipser Nvidia Host on Ubuntu 20.04 LTS

sudo apt update && apt upgrade -y
sudo apt -y install build-essential htop gcc make && sudo apt autoremove -y
sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
sudo touch /etc/modprobe.d/blacklist-nouveau.conf
sudo echo 'blacklist nouveau' >> /etc/modprobe.d/blacklist-nouveau.conf
sudo echo 'options nouveau modeset=0' >> /etc/modprobe.d/blacklist-nouveau.conf
sudo reboot

****

wget -N https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run && sudo sh cuda_11.7.0_515.43.04_linux.run


wget -N https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.1/local_installers/11.8/cudnn-local-repo-ubuntu2004-8.9.1.23_1.0-1_arm64.deb && sudo dpkg -i cudnn-local-repo-ubuntu2004-8.9.1.23_1.0-1_arm64.deb

sudo apt update && sudo apt install libcudnn8 libcudnn8-dev


echo 'PATH=$(echo "$PATH:/home/***YOURUBUNTUUSERNAME***/.local/bin")' >> ~/.bashrc
echo 'PATH=$(echo "$PATH:/usr/local/cuda-11.7/bin")' >> ~/.bashrc
echo 'LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64"' >> ~/.bashrc
source ~/.bashrc

sudo apt-get update && sudo apt -y install python-is-python3 ffmpeg libaio-dev python3-pip python3-venv curl git git-lfs

python -m pip install --upgrade pip setuptools wheel testresources
pip install -r requirements.txt
python -m spacy download en_core_web_trf



