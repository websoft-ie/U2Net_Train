#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys

import imgviz
import numpy as np

import labelme


def LMe2Voc(inputDir, outputDir, labelfile):
    noViz = False
    if osp.exists(outputDir):
        print("Output directory already exists:", outputDir)
        sys.exit(1)
    os.makedirs(outputDir)
    os.makedirs(osp.join(outputDir, "JPEGImages"))
    os.makedirs(osp.join(outputDir, "SegmentationClass"))
    os.makedirs(osp.join(outputDir, "SegmentationClassPNG"))
    if not noViz:
        os.makedirs(
            osp.join(outputDir, "SegmentationClassVisualization")
        )
    os.makedirs(osp.join(outputDir, "SegmentationObject"))
    os.makedirs(osp.join(outputDir, "SegmentationObjectPNG"))
    if not noViz:
        os.makedirs(
            osp.join(outputDir, "SegmentationObjectVisualization")
        )
    print("Creating dataset:", outputDir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(labelfile).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(outputDir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    for filename in glob.glob(osp.join(inputDir, "*.json")):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(outputDir, "JPEGImages", base + ".jpg")
        out_cls_file = osp.join(
            outputDir, "SegmentationClass", base + ".npy"
        )
        out_clsp_file = osp.join(
            outputDir, "SegmentationClassPNG", base + ".png"
        )
        if not noViz:
            out_clsv_file = osp.join(
                outputDir,
                "SegmentationClassVisualization",
                base + ".jpg",
            )
        out_ins_file = osp.join(
            outputDir, "SegmentationObject", base + ".npy"
        )
        out_insp_file = osp.join(
            outputDir, "SegmentationObjectPNG", base + ".png"
        )
        if not noViz:
            out_insv_file = osp.join(
                outputDir,
                "SegmentationObjectVisualization",
                base + ".jpg",
            )

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)

        cls, ins = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        ins[cls == -1] = 0  # ignore it.

        # class label
        labelme.utils.lblsave(out_clsp_file, cls)
        np.save(out_cls_file, cls)
        if not noViz:
            clsv = imgviz.label2rgb(
                label=cls,
                img=imgviz.rgb2gray(img),
                label_names=class_names,
                font_size=15,
                loc="rb",
            )
            imgviz.io.imsave(out_clsv_file, clsv)

        # instance label
        labelme.utils.lblsave(out_insp_file, ins)
        np.save(out_ins_file, ins)
        if not noViz:
            instance_ids = np.unique(ins)
            instance_names = [str(i) for i in range(max(instance_ids) + 1)]
            insv = imgviz.label2rgb(
                label=ins,
                img=imgviz.rgb2gray(img),
                label_names=instance_names,
                font_size=15,
                loc="rb",
            )
            imgviz.io.imsave(out_insv_file, insv)



def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument(
        "--noviz", help="no visualization", action="store_true"
    )
    args = parser.parse_args()
    LMe2Voc(args.input_dir, args.output_dir, args.labels)


if __name__ == "__main__":
    main()
