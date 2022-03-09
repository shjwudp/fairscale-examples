# FairScale Pipe RPC + DDP Example

```bash
pip install -r requirements.txt
torchrun \
    --standalone \
    --nproc_per_node 6 \
    mnist.py \
        --pipeline-length 3
```
