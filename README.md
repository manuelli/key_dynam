# Code Overview

Our approach requires

1. Generating a dataset
2. Training the visual models (dense correspondence, transporter, etc.)
3. Training the dynamics models
4. Evaluating closed-loop MPC performance

Each experiment/task has it's own folder, e.g. ``key_dynam/experiments/drake_pusher_slider``. This folder contains scripts that perform each of the four steps listed above.

## Environment setup
Our code is setup to run inside a docker environment. To build the docker image use

```angular2
cd key_dynam/docker && ./docker_build.py
```
and to run the docker container use
```angular2
cd key_dynam/docker && ./docker_run.py
```

 
### Generating a dataset
To generate a dataset use for example ``key_dynam/experiments/drake_pusher_slider/collect_episodes.py``

### Training visual model
Use the `train_dense_descriptor_vision` function of the `key_dynam/experiments/drake_pusher_slider/train_and_evaluate.py` file.

### Training and Evaluating the dynamics model
Use the methods `DD_3D, GT_3D, transporter_3D, train_conv_autoencoder_dynamics` in the `key_dynam/experiments/drake_pusher_slider/train_and_evaluate.py` file.
