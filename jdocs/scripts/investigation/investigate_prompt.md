
# new vla investigatin 20260121
jdocs/bimanual/bin/first_principles_research_implementation_plan.md

 Commits Made (5 total)                                                                                        
                                                                                                                
  1. 6546dce3 - Implementation plan document                                                                    
  2. 2f3ad8ee - Phase A tools (vision_feature_comparison.py, prefix_embedding_analysis.py)                      
  3. 98f148ec - Phase B tools (kv_cache_content_analysis.py, denoising_step_analysis.py)                        
  4. 2eebfccb - Phase C tool (trajectory_distribution_visualization.py)                                         
  5. 1cf795fb - Findings document templates                                                                     
                                                                                                                
  Tools Implemented                                                                                             
  Tool: vision_feature_comparison.py                                                                            
  Location: jdocs/scripts/investigation/tools/                                                                  
  Purpose: Compare SigLIP visual features between cases                                                         
  ────────────────────────────────────────                                                                      
  Tool: prefix_embedding_analysis.py                                                                            
  Location: jdocs/scripts/investigation/tools/                                                                  
  Purpose: Analyze prefix embedding (vision+language+state) differences                                         
  ────────────────────────────────────────                                                                      
  Tool: kv_cache_content_analysis.py                                                                            
  Location: jdocs/scripts/investigation/tools/                                                                  
  Purpose: Analyze KV cache content by layer and token region                                                   
  ────────────────────────────────────────                                                                      
  Tool: denoising_step_analysis.py                                                                              
  Location: jdocs/scripts/investigation/tools/                                                                  
  Purpose: Trace trajectory emergence through 10 denoising steps                                                
  ────────────────────────────────────────                                                                      
  Tool: trajectory_distribution_visualization.py                                                                
  Location: jdocs/scripts/investigation/tools/                                                                  
  Purpose: 3D PCA visualization of training trajectory distribution                                             
  Documents Created                                                                                             
  ┌──────────────────────────────────────────────────┬─────────────────────┬─────────────────────────────────┐  
  │                     Document                     │      Location       │             Purpose             │  
  ├──────────────────────────────────────────────────┼─────────────────────┼─────────────────────────────────┤  
  │ first_principles_research_implementation_plan.md │ jdocs/bimanual/bin/ │ Detailed implementation plan    │  
  ├──────────────────────────────────────────────────┼─────────────────────┼─────────────────────────────────┤  
  │ hallucination_mechanism_findings.md              │ jdocs/bimanual/bin/ │ Template for Phase A/B findings │  
  ├──────────────────────────────────────────────────┼─────────────────────┼─────────────────────────────────┤  
  │ trajectory_distribution_analysis.md              │ jdocs/bimanual/bin/ │ Template for Phase C findings   │  
  ├──────────────────────────────────────────────────┼─────────────────────┼─────────────────────────────────┤  
  │ hallucination_root_cause_synthesis.md            │ jdocs/bimanual/bin/ │ Template for final synthesis    │  
  └──────────────────────────────────────────────────┴─────────────────────┴─────────────────────────────────┘  
  Next Steps                                                                                                    
                                                                                                                
  Run the experiments with the provided commands in the synthesis document:                                     
                                                                                                                
  cd jdocs/scripts/investigation/tools                                                                          
                                                                                                                
  # Phase A                                                                                                     
  python jdocs/scripts/investigation/tools/vision_feature_comparison.py                        
  python jdocs/scripts/investigation/tools/prefix_embedding_analysis.py 
                                                                                                                
  # Phase B                                                                                                     
  python jdocs/scripts/investigation/tools/kv_cache_content_analysis.py                        
  python jdocs/scripts/investigation/tools/denoising_step_analysis.py 
                                                                                                                
  # Phase C                                                                                                     
  python jdocs/scripts/investigation/tools/trajectory_distribution_visualization.py --task-filter "yogurt" ... 


  Step 200 (pre-divergence):                                                                                    
  - Halluc: Action Delta = 4.56                                                                                 
  - Normal: Action Delta = 2.19                                                                                 
  - All cases cluster together in trajectory PCA space                                                          
                                                                                                                
  Step 250 (during hallucination):                                                                              
  - Halluc: Action Delta = 11.77 (arm moving!)                                                                  
  - Normal: Action Delta = 2.34 (staying still)                                                                 
  - In left wrist camera, you can see the halluc arm has moved position                                         
                                                                                                                
  You raise an excellent point about causality visualization. The current visualization shows WHERE cases fall  
  in trajectory space, but not the causal mechanism: P(trajectory | KV_cache).                                  
                                                                                                                
  Let me create a better visualization that shows:                                                              
  1. Action distribution given visual context - multiple inference samples to show probability                  
  2. Dataset scene-action linkage - what trajectories exist in training for similar scenes 

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


now we are encountering our smolvla hallucination situation under certain situation(you can reference "## 1. Problem Statement" of jdocs/bimanual/bin/smolvla_hallucination_investigation_design.md for detailed information). claude has did some research into this issue, and generated report you can reference: jdocs/bimanual/bin/smolvla_hallucination_investigation_design.md. also claude created some tools which generated some analysis using tools under jdocs/scripts/investigation/tools, the analysis folder is logs/analysis. the 3 cases I collected are under logs/yogurt_banana_leftarm. now need you to do analysis, research, then bring the best expertise to help me investigate and propose tools, methods, experiments to do to systematically identify the potential root causes of the hallucination issues. the process is not necessary to be in one shot, it can be designed in interative way, based on current evidence to generate reasonable hypothesises, then propose methods/experiments/tools to diagnosis/analyze, then based on feedback data to further investigate until find the root cause, which can direct us how to correct the issue in data collection or other aspects. don't make any changes, only write your analysis/comments to doc: jdocs/bimanual/bin/gemini_hallucinate_investigation.md 


## dataset analysis tool
<!-- 20260120 -->
we are in the process of investigating and research on the hallucination behavior we encountered for our trained smolvla model outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model which finetuned based on our collected dataset datasets_bimanuel/multitasks. 

1) first, some typical hallucination behavior has been described in "## 1. Problem Statement" of jdocs/bimanual/bin/smolvla_hallucination_investigation_design.md, which you can check. the previous research and tests are summarized in jdocs/bimanual/bin/smolvla_hallucination_investigation_design.md, which by analyzing the collected counterparts cases, 1) identify the moments where behavior diverge between cases(hallucinated action flows vs stay still) 2) tried cross attention analysis, and use spatial attention to visualize how action state(q) is cross attended with visual & language (kv), the output then further generate embedding context for diffusion to generate action flow(NOTE: this is my current understanding of how spatial attention visualization works, correct me if I am wrong) 3) but so far we still didn't find the causality evidence or mechanism to explain how the cross attention trigger the different diffusion flow generation. some hypothesis and proposed methods are mentioned in jdocs/bimanual/bin/smolvla_hallucination_investigation_design.md, like masking right wrist camera during inference(which to me we actually already have a normal case where there is no other object on table or plate which kind of doing the same thing as masking does); the other proposal is to further understand how diffusion flow generation/denoising process works and how they are affected by the kv cache values or output of qkv(NOTE: correct me if I am wrong: diffusion flow generation is the same thing as denoising process). I think you should plan, research and think how do investigation/test along this direction to further find evidence/mechanism to explain the hallucination root cause.

2) as we know, the vla model is trained/finetuned on the dataset we collected, it could be possible that the bias/defects/issues in the collected dataset which caused the hallucination issue we encountered. also to fix the hallucination issue we have with the current model checkpoint, at the end of the day, we need to collect more data to address the issues we had experienced or haven't experienced. but our current vla data collection mostly based on empirical experiences, which gave us dilemma like: a) we don't have unlimited resource or time to collect a huge dataset which can cover different possible holes or aspects which required to train/finetune a good smolvla model b) the current smolvla checkpoint's issues are hard to related to the problems in current dataset, we need some tools or methods which can systematically help us understand the drawbacks and issues of current datasets, which can be translated to directions to help us collect new data demos which help improve the dataset quality and coverage, so improve the future finetuned model on the new dataset. so you could need to do more research on what are best practical methods which can help us diagnose dataset, pointing out its weakness(statisically, with number, with visual effect), which then can help translate to directions on what new dataset we need to collect. this will be very useful, if we can visualize the defects of the dataset, link defects to existing issue of the model, link to directions to collect new data, even better if can relate the defects(ie: traj distribution for certain case in current dataset) to the cross attention and diffusion flow generation issues we identified above

this is a big research plan, you can make above 1) and 2) first as separate researches, write down research docs into jdocs/bimanual/bin, then systematically think these 2 parts together to see if we can reach the ideal point: "we can visualize the defects of the dataset, link defects to existing issue of the model, link to directions to collect new data, even better if can relate the defects(ie: traj distribution for certain case in current dataset) to the cross attention and diffusion flow generation issues we identified above"


push back: 1) I don't agree with current "Current Status Summary": first I don't agree with "Missing post-completion idle training data", if you check 3  
  cases I collected as counterparts, why the other normal cases learned to stay still after tasks completed, but the hallucination case didn't. also if     
  vla model cannot tell the difference between if there is target object on table or not, what is the usage of vlm part, which to me is to generate task    
  related context which used to condition on to generate corresponding action flow, if there is no object on the table, the condition embedding should      
  clearly direct action flow generation to stay still instead of moving to empty space. and this is one of the key point you need to investigate in the     
  action flow generation mechanism part, and how kv or conditioned embedding triggered the difference of hallucinating action flow vs stay still flow in    
  normal cases. for "position replay pattern", it is also wrong, the pattern is to demo vla to learn how to do the task if there is clear target object,    
  but for our hallucination case, this doesn't apply, as there is no object on the table, and in training dataset, there is no demo data teach model to     
  approaching empty space. you should not be distracted by those premature conclusions, you should focus on using first principle mindset to use collected  
  3 cases to walk through the fundamental mechanism how vision/language/state trigger the flow generation, and what are the difference during the process   
  could explain the different action flow generation after target object was put in the bin 2) for dataset analysis, I remember I read some paper or tools which can generate 3D traj distribution visual diagram for certain context, for example "yogurt bottle pick and place cases", which is similar to diffusion flow generation process(vla condition on the qkv generated embedding to sample from the action flow distribution space? correct me if I am wrong). if we can analyze like this, we can link the current dataset demoed action flow distribution of this case(yogurt bottle) to the 3 cases we collected and their action flow generation process, and hopefully we can find some links between them, which will give us a solid research findings, which can explain the hallucination phenomenon, direct new data demo collection and research paper publish.

  you should redo the research and planning as your current plan is based on many false conclusions or incorrect directions. after that as i said before, no rush to implementation, always write down your reseach plan or findings, which we will review, then implement or test step by step iteratively 


  Next Steps                                                                                                                                                
                                                                                                                                                            
  Run the experiments with the provided commands in the synthesis document:                                                                                 
                                                                                                                                                            
  cd jdocs/scripts/investigation/tools                                                                                                                      
                                                                                                                                                            
  # Phase A                                                                                                                                                 
  python vision_feature_comparison.py --checkpoint <path> --case-dirs <dirs> --step 200                                                                     
  python prefix_embedding_analysis.py --checkpoint <path> --case-dirs <dirs> --step 200                                                                     
                                                                                                                                                            
  # Phase B                                                                                                                                                 
  python kv_cache_content_analysis.py --checkpoint <path> --case-dirs <dirs> --step 200                                                                     
  python denoising_step_analysis.py --checkpoint <path> --case-dirs <dirs> --step 200                                                                       
                                                                                                                                                            
  # Phase C                                                                                                                                                 
  python trajectory_distribution_visualization.py --dataset <path> --task-filter "yogurt" ...  