import argparse
import os
import glob
from tqdm import tqdm
import numpy as np
from mean_average_precision import MetricBuilder
from vaik_pascal_voc_rw_ex import pascal_voc_rw_ex


def dump_map(result_metric, classes, answer_box_num_list):
    dump_string = ""
    dump_string += "## CSV Format\n"
    dump_string += "\"class\", \"iou_th\", \"ap\",  \"precision\",  \"recall\", \"num\" \n"
    dump_string += f"\"mAP(ALL)\", \"{result_metric['mAP']:.4f}\",  \"\",  \"\", \"\"\n"

    for class_index in range(len(classes)):
        dump_string += f"\"{classes[class_index]}\", "
        dump_string += f"\"{list(result_metric.keys())[0]}\", "
        dump_string += f"\"{result_metric[list(result_metric.keys())[0]][class_index]['ap']:.4f}\", "
        dump_string += f"\"{np.average(result_metric[list(result_metric.keys())[0]][class_index]['precision']):.4f}\", "
        dump_string += f"\"{np.average(result_metric[list(result_metric.keys())[0]][class_index]['recall']):.4f}\", "
        dump_string += f"\"{answer_box_num_list[class_index]}\", \n"
    print(dump_string)


def calc_map(answer_inference_label_path_list, classes):
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=len(classes))
    answer_box_num_list = [0, ] * len(classes)
    for answer_label_path, inference_label_path in tqdm(answer_inference_label_path_list, desc='calc_map'):
        answer_xml_dict = pascal_voc_rw_ex.read_pascal_voc_xml(answer_label_path)
        if inference_label_path is None:
            inference_xml_dict = {'annotation': {'object': []}}
        else:
            inference_xml_dict = pascal_voc_rw_ex.read_pascal_voc_xml(inference_label_path)
        if 'object' not in answer_xml_dict['annotation'].keys():
            answer_xml_dict['annotation']['object'] = []
        if 'object' not in inference_xml_dict['annotation'].keys():
            inference_xml_dict['annotation']['object'] = []
        if not isinstance(answer_xml_dict['annotation']['object'], list):
            answer_xml_dict['annotation']['object'] = [answer_xml_dict['annotation']['object']]
        if not isinstance(inference_xml_dict['annotation']['object'], list):
            inference_xml_dict['annotation']['object'] = [inference_xml_dict['annotation']['object']]
        gt_list = []
        for answer_objects_dict in answer_xml_dict['annotation']['object']:
            if answer_objects_dict['name'] in classes:
                gt_list.append([int(answer_objects_dict['bndbox']['xmin']), int(answer_objects_dict['bndbox']['ymin']),
                                    int(answer_objects_dict['bndbox']['xmax']), int(answer_objects_dict['bndbox']['ymax']),
                                classes.index(answer_objects_dict['name']), 0, 0])
            answer_box_num_list[classes.index(answer_objects_dict['name'])] += 1
        gt = np.array(gt_list)

        pr_list = []
        for inference_objects_dict in inference_xml_dict['annotation']['object']:
            if inference_objects_dict['name'] in classes:
                pr_list.append([int(inference_objects_dict['bndbox']['xmin']), int(inference_objects_dict['bndbox']['ymin']),
                                    int(inference_objects_dict['bndbox']['xmax']), int(inference_objects_dict['bndbox']['ymax']),
                                classes.index(inference_objects_dict['name']), float(inference_objects_dict['score'])])
        pr = np.array(pr_list)
        metric_fn.add(pr, gt)

    all_metric = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05),
                                 recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')
    return all_metric, answer_box_num_list


def main(answer_label_dir_path, inference_label_dir_path, classes_txt_path):
    with open(classes_txt_path, 'r') as f:
        classes = f.readlines()
        classes = [label.strip() for label in classes]

    answer_label_path_list = glob.glob(os.path.join(answer_label_dir_path, '**/*.xml'), recursive=True)
    answer_inference_label_path_list = []
    for answer_label_path in answer_label_path_list:
        inference_label_path = os.path.join(inference_label_dir_path, os.path.basename(answer_label_path))
        if not os.path.exists(inference_label_path):
            inference_label_path = None
        answer_inference_label_path_list.append((answer_label_path, inference_label_path))
    all_metric, answer_box_num_list = calc_map(answer_inference_label_path_list, classes)
    dump_map(all_metric, classes, answer_box_num_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--answer_label_dir_path', type=str, default='~/.vaik-mnist-detection-dataset/valid')
    parser.add_argument('--inference_label_dir_path', type=str,
                        default='~/.vaik-mnist-detection-dataset/valid_inference')
    parser.add_argument('--classes_txt_path', type=str, default='~/.vaik-mnist-detection-dataset/classes.txt')
    args = parser.parse_args()

    args.answer_label_dir_path = os.path.expanduser(args.answer_label_dir_path)
    args.inference_label_dir_path = os.path.expanduser(args.inference_label_dir_path)
    args.classes_txt_path = os.path.expanduser(args.classes_txt_path)

    main(**args.__dict__)
