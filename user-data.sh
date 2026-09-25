#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "Starting EC2 User Data Script..."

# Create a 16GB swap file to prevent OOM
fallocate -l 16G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

# Update and install dependencies
yum update -y
yum install -y jq htop tmux

# Install uv (blazing fast python package manager)
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Find out our bucket name from tags (or we can just hardcode it for now via sed later)
# We will replace amazon-ml-challenge-data-303164493301-us-east-1 below

cd /home/ec2-user
aws s3 cp s3://amazon-ml-challenge-data-303164493301-us-east-1/code.zip .
unzip code.zip

# Download datasets
mkdir -p dataset
aws s3 sync s3://amazon-ml-challenge-data-303164493301-us-east-1/dataset/ dataset/

# Setup environment
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Run the parallel prediction script
echo "Running ML Pipeline..."
export PYTHONPATH=/home/ec2-user/code/business_entity_resolution
python code/business_entity_resolution/src/predict_parallel.py

# Upload outputs
echo "Uploading results..."
aws s3 sync outputs/ s3://amazon-ml-challenge-data-303164493301-us-east-1/outputs/

# Terminate instance to save costs
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
