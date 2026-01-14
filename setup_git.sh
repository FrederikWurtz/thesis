#!/bin/bash

# # Get the directory where this script is located
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# # Change to the script's directory
# cd "$SCRIPT_DIR" || { echo "Failed to change to script directory"; exit 1; }


eval "$(ssh-agent -s)"
ssh-add /work/FrederikWürtzSørensen#7865/keys/ssh/id_ed25519
ssh -T git@github.com # Test connection

git config --global user.email "Frederik-610@hotmail.com"
git config --global user.name "FrederikWurtz"
echo "Git configuration set successfully"

