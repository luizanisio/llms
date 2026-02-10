export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#if [ $# -eq 0 ]; then
#    python ../../src/treinar_unsloth.py treinar_qwen15.yaml --info
#else
python ../../src/treinar_unsloth.py treinar_qwen7.yaml "$@"

#fi