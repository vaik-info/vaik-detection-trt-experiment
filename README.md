# vaik-detection-trt-experiment

Create Pascal VOC xml file by inference model. Calc mAP and draw a box with score.

## Example

![vaik-detection-trt-experiment](https://user-images.githubusercontent.com/116471878/199637208-76753193-a391-4b5b-a84f-5418502a8d2a.png)


## Install

- amd64(g4dn.xlarge)

```shell
pip install -r requirements.txt
```

- arm64(JetsonXavierNX)

```shell
cat requirements.txt | xargs -n 1 pip install
```

## Usage

### Create Pascal VOC xml file

```shell
python inference.py --input_saved_model_path '~/output_trt_model/model.fp16.trt' \
                --input_classes_path '~/.vaik-mnist-detection-dataset/classes.txt' \
                --input_image_dir_path '~/.vaik-mnist-detection-dataset/valid' \
                --output_xml_dir_path '~/.vaik-mnist-detection-dataset/valid_inference' \
                --score_th 0.2 \
                --nms_th 0.5
```

#### Output

![vaik-detection-trt-experiment-output](https://user-images.githubusercontent.com/116471878/199637324-dae09efc-abb1-4c76-ba6d-e7fe1846bd22.png)

-----
