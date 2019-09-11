# semiauto_annotate
Semi-autocomplete object detection annotations using semi-trained models

Currently only works with the Pascal VOC XML format using [LabelImg](https://github.com/tzutalin/labelImg) and [Keras RetinaNet](https://github.com/fizyr/keras-retinanet/blob/master/README.md).

Once annotations have been predicted, extraneous or innacurate annotations can be adjusted using LabelImg, which can then be used to get higher quality annotations at a much faster speed to continue training of original, semi-trained model.
