

# trace collection

python jdocs/scripts/investigation/tools/trace_inference.py --task "Use left arm to pick up the yogurt bottle and place it in the bin" --notes "hallucinate due to other object on the table" --duration 20

python jdocs/scripts/investigation/tools/trace_inference.py --task "Use left arm to pick up the yogurt bottle and place it in the bin" --notes "no hallucinate due to banana on the plate and no other object on the table" --duration 20

python jdocs/scripts/investigation/tools/trace_inference.py --task "Use left arm to pick up the yogurt bottle and place it in the bin" --notes "no hallucinate due to no other object on the table" --duration 20