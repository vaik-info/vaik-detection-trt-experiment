# vaik-detection-trt-experiment

Create Pascal VOC xml file by Tensor RT inference model. Calc mAP and draw a box with score.

## Example

![vaik-detection-trt-experiment](https://user-images.githubusercontent.com/116471878/199637208-76753193-a391-4b5b-a84f-5418502a8d2a.png)


## Install

- amd64(g4dn.xlarge)

```shell
docker build -t g4dnxl_ed_experiment -f ./Dockerfile.g4dn.xlarge .
sudo docker run --runtime=nvidia \
           --name g4dnxl_ed_experiment_container \
           --rm \
           -v ~/.vaik-mnist-detection-dataset:/workspace/vaik-mnist-detection-dataset \
           -v ~/output_trt_model:/workspace/output_trt_model \
           -v $(pwd):/workspace/source \
           -it g4dnxl_ed_experiment /bin/bash
```

- arm64(JetsonXavierNX)

```shell
sudo docker build -t jxnj502_experiment -f ./Dockerfile.jetson_xavier_nx_jp_502 .
sudo docker run --runtime=nvidia \
           --name jxnj502_experiment_container \
           --rm \
           -v ~/.vaik-mnist-detection-dataset:/workspace/vaik-mnist-detection-dataset \
           -v ~/output_trt_model:/workspace/output_trt_model \
           -v $(pwd):/workspace/source \
           -it jxnj502_experiment /bin/bash
```

## Usage

### Create Pascal VOC xml file

```shell
cd /workspace/source
python3 inference.py --input_saved_model_path '/workspace/output_trt_model/model.fp16.trt' \
                --input_classes_path '/workspace/vaik-mnist-detection-dataset/classes.txt' \
                --input_image_dir_path '/workspace/vaik-mnist-detection-dataset/valid' \
                --output_xml_dir_path '/workspace/vaik-mnist-detection-dataset/valid_inference' \
                --score_th 0.2 \
                --nms_th 0.5
```

#### Output

![vaik-detection-trt-experiment-output](https://user-images.githubusercontent.com/116471878/199637324-dae09efc-abb1-4c76-ba6d-e7fe1846bd22.png)

-----


### Calc mAP

- only amd64(g4dn.xlarge)

```shell
python3 calc_map.py --answer_label_dir_path '/workspace/vaik-mnist-detection-dataset/valid' \
                --inference_label_dir_path '/workspace/vaik-mnist-detection-dataset/valid_inference' \
                --classes_txt_path '/workspace/vaik-mnist-detection-dataset/classes.txt'
```

#### Output

``` text
## CSV Format
"class", "iou_th", "ap",  "precision",  "recall", "num" 
"mAP(ALL)", "0.9112",  "",  "", ""
"zero", "0.5", "0.9327", "0.9890", "0.4998", "112", 
"one", "0.5", "0.9672", "0.9958", "0.5082", "82", 
"two", "0.5", "0.9638", "0.9921", "0.4983", "107", 
"three", "0.5", "0.9501", "0.9995", "0.4854", "106", 
"four", "0.5", "0.9685", "0.9843", "0.5260", "76", 
"five", "0.5", "0.9604", "0.9995", "0.4995", "79", 
"six", "0.5", "0.9206", "0.9868", "0.4889", "86", 
"seven", "0.5", "0.9503", "0.9994", "0.4942", "95", 
"eight", "0.5", "0.9595", "0.9986", "0.4987", "111", 
"nine", "0.5", "0.9662", "0.9940", "0.5093", "69", 
```

----

### Draw box

```shell
python3 draw_box.py --input_image_dir_path '/workspace/.vaik-mnist-detection-dataset/valid' \
                --input_label_dir_path '/workspace/.vaik-mnist-detection-dataset/valid_inference' \
                --input_classes_path '/workspace/.vaik-mnist-detection-dataset/classes.txt' \
                --output_image_dir_path '/workspace/.vaik-mnist-detection-dataset/valid_inference_draw'
```

#### Output

![valid1](https://user-images.githubusercontent.com/116471878/199640085-ce7773c3-f4c7-4b85-aa61-d85649bd4f31.png)
![valid2](https://user-images.githubusercontent.com/116471878/199640088-0c5d1baf-ef14-44f4-a06f-b0eb9951df9a.png)
