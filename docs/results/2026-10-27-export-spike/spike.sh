#!/bin/bash
# Export spike: every candidate through every path. Failures are results, so each step logs its exit code.
# Run from the repo root. V defaults to the pipeline venv; X follows RS_DATA_DIR like the exporters do.
V=${V:-~/Projects/record-scan/pipeline/.venv/bin/python}
X=$($V -c "from pipeline import config; print(config.EXPORTS_DIR)")
step() { echo "=== $*"; "$@"; echo "=== exit $?"; }
echo "=== start $(date) $(git rev-parse --short HEAD) $(uptime)"
for m in dinov3-vits16 dinov3-vitsplus16 dinov3-convnext-tiny pe-core-s16; do
  px=224; [ $m = pe-core-s16 ] && px=384
  for prec in fp32 fp16 fp16-attn32; do
    step $V -m pipeline.export.to_coreml --model $m --px $px --precision $prec
    step $V -m pipeline.export.parity --model $m --exported $X/$m-$px-$prec.mlpackage
  done
  step $V -m pipeline.export.to_executorch --model $m --px $px --backend xnnpack
  step $V -m pipeline.export.parity --model $m --exported $X/$m-$px-xnnpack.pte
done
echo "=== all done $(date) $(uptime)"
