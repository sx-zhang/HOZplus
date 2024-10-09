## Setup
- Create conda environment. `pip install -r requirements.txt`
- Download the [dataset](https://drive.google.com/file/d/1kvYvutjqc6SLEO65yQjo8AuU85voT5sC/view), which refers to [ECCV-VN](https://github.com/xiaobaishu0097/ECCV-VN). The offline data is discretized from [AI2-Thor](https://ai2thor.allenai.org/) simulator.
- Download the [pretrain dataset](https://drive.google.com/file/d/1dFQV10i4IixaSUxN2Dtc6EGEayr661ce/view), which refers to [VTNet](https://github.com/xiaobaishu0097/ICLR_VTNet).
- You can also use the [DETR object detection features](https://drive.google.com/file/d/1d761VxrwctupzOat4qxsLCm5ndC4wA-M/view?usp=sharing).
The `data` folder should look like this
```python
data/ 
    └── Scene_Data/
        ├── FloorPlan1/
        │   ├── resnet18_featuremap.hdf5
        │   ├── graph.json
        │   ├── visible_object_map_1.5.json
        │   ├── detr_features_22cls.hdf5
        │   ├── grid.json
        │   └── optimal_action.json
        ├── FloorPlan2/
        └── ...
    └── AI2Thor_VisTrans_Pretrain_Data/
        ├── data/
        ├── annotation_train.json
        ├── annotation_val.json
        └── annotation_test.json
``` 
## Training and Evaluation

### Pre-train the search thinking network of our HOZ++ model

`python main_pretraining.py --title ST_Pretrain --model ST_Pretrain --workers 9 --gpu-ids 0 --epochs 20 --log-dir runs/pretrain --save-model-dir trained_models/pretrain`
### Train our HOZ++ model
`python main.py --title hozplus --model HOZplus --workers 9 --gpu-ids 0 --max-ep 3000000 --log-dir runs --save-model-dir trained_model --pretrained-trans trained_models/pretrain/checkpoint0004.pth` 
### Evaluate our HOZ++ model
`python full_eval.py --title hozplus --model HOZplus --results-json eval_best_results/debug.json --gpu-ids 0 --log-dir runs --save-model-dir trained_model`  
