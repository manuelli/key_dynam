## Dynamics

Learning the dynamics model using deep neural networks


### PusherSlider

Go into the experiment folder, the configuration information is stored in `config.yaml`
```
cd experiments/01
```

Generate training data
```
python create_pushing_dataset.py
```

Train the dynamics model
```
python train_dynamics_model.py
```

A tensorboard directory will be created in the same place where your model checkpoints are being saved.
For example 
```
key_dynam/experiments/01/dump/2019-10-17-19-58-16-827416/tensorboard
```
If you launch tensorboard in any directory above this using

```
tensorboard --logdir .
```
then you will see your training curves.

It will tell you to navigate to 

Evaluate the trained dynamics model
```
python eval_dynamics_model.py
```
