# Deep Sort with PyTorch 
### Detectron2 version of [DeepSORT PyTorch](https://github.com/ZQPei/deep_sort_pytorch)

### Demo

1. Clone this repository: `git clone --recurse-submodules https://github.com/sayef/detectron2-deepsort-pytorch.git`

2. Install: 

    *Tested on Python 3.6 Only*

    ```
    cd detectron2-deepsort-pytorch
    pip install -r requirements.txt
    pip install -e detectron2
    pip install git+git://github.com/facebookresearch/fvcore.git@1f3825f82b622409ea4145d192dbd36a64e91d49
    pip install cython
    pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
    ```

4. Download checkpoint `ckpt.t7` from here: [Google Drive](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)
and place into `deep_sort/deep/checkpoint/`.

5. Run the demo: `ython demo_detectron2_deepsort.py demo-input.avi --ignore_display --save_path demo-output.avi --use_cuda False`

### Notes:
1. Try out different detectron2 models: Change the configs in `___init__` of `detectron2_detection.py`
2. Regarding any issues of detectron2, please refer to  [Detectron2](https://github.com/facebookresearch/detectron2) repository.

### References and Credits:
1. [Pytorch implementation of deepsort with Yolo3](https://github.com/ZQPei/deep_sort_pytorch)
2. [Facebook Detectron2](https://github.com/facebookresearch/detectron2)
3. [Deepsort](https://github.com/nwojke/deep_sort)
4. [SORT](https://github.com/abewley/sort)
