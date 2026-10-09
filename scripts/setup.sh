set -ex

cd openrlhf/

pip install .
pip uninstall flash-attn -y
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn