
# coding

- use exceptions than random defaults
- fail fast or throw exceptions to verify key cfgs loading

# do verification not just by reading but by running real runtime experiments to verify the pipeline works
- after cc verify there is no issue for training pipeline(as it is very expensive if there is any error which costs the whole training failed)
- we need to ask cc to do runtime verification
  - jdocs/scripts/bimanual/verify_multitasks_dataset.py
  - jdocs/scripts/bimanual/verify_training_pipeline.py

# verify ideas ito experiements
- we don't assume vla can generalize from given tasks, we need setup experiements in dataset task names(only place orange to plate, and later test place to bin to test its generalization capability)


# use layer of files to abstract to handle complexity

- use yml cfg file to handle hardware complexity & changes, make hardware changes transparent to upper scripts