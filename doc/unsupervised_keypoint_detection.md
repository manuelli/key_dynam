## Unsupervised Keypoint Detection

Generate image list for building the dataset

```
find . -regextype sed -regex ".*/logs/push_box/2019-07-3.*/processed/images_camera_1/.*_rgb.png" > config/push_box_kp.txt
```

Train & evaluation

```
cd modules/unsupervised
bash scripts/train_BoxPush_kp.sh
bash scripts/eval_BoxPush_kp.sh
```
