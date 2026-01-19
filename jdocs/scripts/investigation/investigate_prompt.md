

# trace collection

python jdocs/scripts/investigation/tools/trace_inference.py --task "Use left arm to pick up the yogurt bottle and place it in the bin" --notes "hallucinate due to other object on the table" --duration 20

python jdocs/scripts/investigation/tools/trace_inference.py --task "Use left arm to pick up the yogurt bottle and place it in the bin" --notes "no hallucinate due to banana on the plate and no other object on the table" --duration 20

python jdocs/scripts/investigation/tools/trace_inference.py --task "Use left arm to pick up the yogurt bottle and place it in the bin" --notes "no hallucinate due to no other object on the table" --duration 20

# model introspection
- attention visualization

python jdocs/scripts/investigation/tools/visualize_attention.py --compare --checkpoint                    
  outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model --case1                      
  logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table --case2                                     
  logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj --output-dir                           
  logs/yogurt_banana_leftarm/attention_analysis 


## multi-camera 

python jdocs/scripts/investigation/tools/cross_attention_capture.py --case-dir logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table  --output-dir logs/yogurt_banana_leftarm/cross_attention_per_camera/halluc

python jdocs/scripts/investigation/tools/cross_attention_capture.py --case-dir logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj --output-dir logs/yogurt_banana_leftarm/cross_attention_per_camera/normal