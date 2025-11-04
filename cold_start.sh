# Install uv
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install pip
python3 get-pip.py

# Install dependencies
pip3 install -r requirements.txt
