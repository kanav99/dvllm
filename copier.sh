#!/bin/bash

# Check if the correct number of arguments was provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <remote_username> <server_ip_or_hostname> <model_name>"
    echo "Example: $0 ubuntu 192.168.1.50 meta-llama/Llama-2-7b-hf"
    exit 1
fi

USERNAME=$1
SERVER=$2
MODEL_NAME=$3

# Define the local and remote paths based on your project structure
LOCAL_DIR="$HOME/.dvllm/$MODEL_NAME"
REMOTE_DIR="~/.dvllm/$MODEL_NAME"

# Ensure the local base directory exists before copying
echo "Preparing local directory: $LOCAL_DIR"
mkdir -p "$LOCAL_DIR"

# Copy the client.bin file
echo "Copying client.bin from $SERVER..."
scp "$USERNAME@$SERVER:$REMOTE_DIR/client.bin" "$LOCAL_DIR/"

# Copy the commits directory recursively
echo "Copying commitments directory from $SERVER..."
scp -r "$USERNAME@$SERVER:$REMOTE_DIR/commits" "$LOCAL_DIR/"

echo "Transfer complete! Client files for $MODEL_NAME are ready."
