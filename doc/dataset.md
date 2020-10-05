

## Recording/Logging Episodes

The basic class for logging an episode is ``episode_container.py``

### Drake Simulation

#### Single Episode
A single episode is recorded using the ``EpisodeContainer`` class. The main data structure of this class is a dict stored in ``episode.data`` with three keys

* ``name``: The name of the episode (should be unique)
* ``config``: Any config related to the setup of this episode
* ``trajectory``: This is the main place where observations and actions are stored. This is a list of dictionaries, there should be as many elements in this list as timesteps in the episode. Each element of this list is itself a dict. Usually these dicts would have keys like 
  * ``observation``: any observation data like positions, velocities, images
  * ``action``: The action that was taken. Note this is the ``observation`` is the observation **before** this action was taken.
  
  
The dict ``episode.data`` can stored to disk using ``pickle``.

##### Images
Images are not stored in the `pickle` file but rather get saved to `hdf5` files. See the `save_images_to_hdf5` function.

#### Multiple Episodes
These get organized in ``MultiEpisodeContainer``. This saves a dict to disk using ``pickle``. The keys are the names of the individual episodes. The values are just the data dicts from above associated with each episode.
  

### Robot Hardware Imitation Learning Dataset Structure (SSCV)

#### Directory structure for logs

```
data_dir/
  2019-01-20-78/
    raw/
      2019-01-20-78.rosbag
      camera_config.yaml
    processed/
      states.json
      images_camera_0/
        000001_depth.png
        000001_rgn.png
        ...
        image_masks/
          000001_mask.png
          000001_mask_visible.png
      images_camera_0
```
- **camera_config.yaml**: contains information about camera extrinsics
- **states.json**: contains information about robot state (e.g. end-effector position, gripper position etc.). Also contains information on which images correspond to the time that this robot command was recorded.


## Dataset

The basic unit of a dataset is an **Episode**. This is similar to how the `pdc` repo is organized, see [here](https://github.com/RobotLocomotion/pytorch-dense-correspondence-private/blob/master-python3/doc/python3_overview.md). The basic class that reads an episode is `EpisodeReader`. 

### Episode Reader
There is an abstract base class [EpisodeReader](../dataset/episode_reader.py) that all other episode readers are derived from. Base classes must implement several methods, the most important of which is 

```angular2
get_data(self, idx)
```

which retrieves the data from that episode at a particular idx. Exactly what data is fetched (e.g. images, joint states, etc.) will depend on the configuration file.

So far we support two types of episodes
- **DrakeSimEpisodeReader**: For data generated using the `key_dynam` Drake simulation setup.
- **DynamicSpartanEpisodeReader**: For the type of data used in the SSCV paper.

#### Image Data
Note that the basic `EpisodeReader` class doesn't handle the visual data. This is taken care of another type of `EpisodeReader`, namely what we will call an `image_episode_reader`. Typically this is of type `dense_correspondence.dataset.episode_reader.EpisodeReader`, the code for which is in the pdc repo. In order to add vision capabilities to a `key_dynam.dataset.episode_reader.EpisodeReader` class you simply pass in the appropriate `dense_correspondence.EpisodeReader`. This can then be accessed by using the `@property` 

```angular2
episode.image_episode
```


### MultiEpisodeDataset

This is the main `torch.utils.data.Dataset` type class, see [MultiEpisodeDataset](../dataset/episode_dataset.py). Essentially this class contains many `EpisodeReader` objects and samples from them appropriately.

#### Action and Observation Functions
This is where most of the work is actually done. These functions, examples of which are found in [function_factory.py](../dataset/function_factory.py) process the raw data returned by the EpisodeReader classes and extract what is needed to actually run the training (e.g. gripper position, gripper velocity action, keypoint correspondences, etc.). See the 

```
def _getitem(self,
                 episode,
                 idx,
                 rollout_length,
                 n_history=None,
                 visual_observation=None,
                 ):
```

method for details on how this is performed. 
 
# Other Useful READMEs

- Dense Correspondence README [here](https://github.com/RobotLocomotion/pytorch-dense-correspondence/blob/master/doc/data_organization.md)
