import argparse
import os
import glob
import re
import random
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from vaik_pascal_voc_rw_ex import pascal_voc_rw_ex

random.seed(777)


def draw_box(image_path, label_path, classes, color_list, font_path=os.path.join(os.path.dirname(__file__), 'arial.ttf'),
             min_font_size=16):
    org_image = Image.open(open(image_path, 'rb')).convert('RGBA')
    
    xml_dict = pascal_voc_rw_ex.read_pascal_voc_xml(label_path)
    if 'object' not in xml_dict['annotation'].keys():
        xml_dict['annotation']['object'] = []
    if not isinstance(xml_dict['annotation']['object'], list):
        xml_dict['annotation']['object'] = [xml_dict['annotation']['object']]

    canvas = Image.new("RGBA", org_image.size, (255, 255, 255, 0))
    draw_canvas = ImageDraw.Draw(canvas)

    if 'score' in xml_dict['annotation'].keys():
        xml_dict['annotation']['object'] = sorted(xml_dict['annotation']['object'], key=lambda x:x['score'])[::-1]
    for objects_dict in xml_dict['annotation']['object'][::-1]:
        xmin = float(objects_dict['bndbox']['xmin'])
        xmax = float(objects_dict['bndbox']['xmax'])
        ymin = float(objects_dict['bndbox']['ymin'])
        ymax = float(objects_dict['bndbox']['ymax'])
        score = 1.0 if 'score' not in objects_dict.keys() else float(objects_dict['score'])
        font_size = max(min_font_size, int(0.1 * min(xmax - xmin, ymax - ymin)))
        font = ImageFont.truetype(font_path, font_size)
        if objects_dict['name'] in classes:
            color = color_list[classes.index(objects_dict['name'])]
            color = (color[0], color[1], color[2], min(255, max(0, min(255, int(score * 2 * 255)))))
            draw_canvas.rectangle(((max(0, xmin), min(org_image.size[1], ymin)),
                                   (max(0, xmax), min(org_image.size[1], ymax))),
                                  outline=color,
                                  width=4)
            draw_canvas.text((xmin + font_size * 0.2, ymin),
                             f'{objects_dict["name"]}\n{score * 100:.1f}%',
                             fill=color, font=font)
    draw_image = Image.alpha_composite(org_image, canvas)
    return draw_image


def draw(label_image_path_list, classes, output_dir_path):
    os.makedirs(output_dir_path, exist_ok=True)
    color_list = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(classes))]
    for image_path, label_path in tqdm(label_image_path_list):
        draw_image = draw_box(image_path, label_path, classes, color_list)
        draw_image.save(f'{os.path.join(output_dir_path, os.path.basename(image_path))}.png')


def main(input_image_dir_path, input_label_dir_path, input_classes_path, output_image_dir_path):
    with open(input_classes_path, 'r') as f:
        classes = f.readlines()
        classes = [label.strip() for label in classes]
    image_path_list = sorted(
        [file_path for file_path in glob.glob(os.path.join(input_image_dir_path, '**/*.*'), recursive=True) if
         re.search('.*\.(png|jpg|bmp)$', file_path)])
    label_image_path_list = []
    for image_path in image_path_list:
        label_path = os.path.join(input_label_dir_path, f'{os.path.splitext(os.path.basename(image_path))[0]}.xml')
        if not os.path.exists(label_path):
            continue
        label_image_path_list.append([image_path, label_path])

    draw(label_image_path_list, classes, output_image_dir_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='draw box')
    parser.add_argument('--input_image_dir_path', type=str, default='~/.vaik-mnist-detection-dataset/valid')
    parser.add_argument('--input_label_dir_path', type=str, default='~/.vaik-mnist-detection-dataset/valid')
    parser.add_argument('--input_classes_path', type=str, default='~/.vaik-mnist-detection-dataset/classes.txt')
    parser.add_argument('--output_image_dir_path', type=str,
                        default='~/.vaik-mnist-detection-dataset/valid_inference_draw')
    args = parser.parse_args()

    args.input_image_dir_path = os.path.expanduser(args.input_image_dir_path)
    args.input_label_dir_path = os.path.expanduser(args.input_label_dir_path)
    args.input_classes_path = os.path.expanduser(args.input_classes_path)
    args.output_image_dir_path = os.path.expanduser(args.output_image_dir_path)

    main(**args.__dict__)
