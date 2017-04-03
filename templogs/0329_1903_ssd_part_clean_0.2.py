./build/tools/caffe: /home/siyu/anaconda2/lib/libtiff.so.5: no version information available (required by /usr/local/lib/libopencv_highgui.so.2.4)
I0329 19:07:48.911365  2693 caffe.cpp:217] Using GPUs 1
I0329 19:07:50.076565  2693 caffe.cpp:222] GPU 1: GeForce GTX TITAN X
I0329 19:07:50.373936  2693 solver.cpp:63] Initializing solver from parameters: 
train_net: "models/VGGNet/ssd_coco_part_clean/train.prototxt"
test_net: "models/VGGNet/ssd_coco_part_clean/test.prototxt"
test_iter: 1000
test_interval: 5000
base_lr: 0.0005
display: 100
max_iter: 250000
lr_policy: "step"
gamma: 0.1
momentum: 0.9
weight_decay: 5e-05
stepsize: 100000
snapshot: 5000
snapshot_prefix: "models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean"
solver_mode: GPU
device_id: 1
debug_info: false
train_state {
  level: 0
  stage: ""
}
snapshot_after_train: true
test_initialization: false
average_loss: 100
iter_size: 1
type: "SGD"
eval_type: "detection"
ap_version: "11point"
I0329 19:07:50.374111  2693 solver.cpp:96] Creating training net from train_net file: models/VGGNet/ssd_coco_part_clean/train.prototxt
I0329 19:07:50.377276  2693 net.cpp:58] Initializing net from parameters: 
name: "VGG_ssd_coco_part_clean_train"
state {
  phase: TRAIN
  level: 0
  stage: ""
}
layer {
  name: "data"
  type: "AnnotatedData"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    mean_value: 104
    mean_value: 117
    mean_value: 123
    force_color: true
    resize_param {
      prob: 1
      resize_mode: WARP
      height: 512
      width: 512
      interp_mode: LINEAR
      interp_mode: AREA
      interp_mode: NEAREST
      interp_mode: CUBIC
      interp_mode: LANCZOS4
    }
    emit_constraint {
      emit_type: MIN_OVERLAP
      emit_overlap: 0.1
    }
    distort_param {
      brightness_prob: 0.5
      brightness_delta: 32
      contrast_prob: 0.5
      contrast_lower: 0.5
      contrast_upper: 1.5
      hue_prob: 0.5
      hue_delta: 18
      saturation_prob: 0.5
      saturation_lower: 0.5
      saturation_upper: 1.5
      random_order_prob: 0
    }
    expand_param {
      prob: 0.5
      max_expand_ratio: 4
    }
  }
  data_param {
    source: "/data/siyu/dataset/coco/lmdb/COCO_Train2014_person_lmdb"
    batch_size: 8
    backend: LMDB
  }
  annotated_data_param {
    batch_sampler {
      sampler {
        min_scale: 0.2
        max_scale: 0.5
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2
      }
      sample_constraint {
        min_object_coverage: 0.1
        max_object_coverage: 0.5
      }
      max_sample: 1
      max_trials: 50
      sampler_type: PART
    }
    batch_sampler {
      max_sample: 1
      max_trials: 1
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2
      }
      sample_constraint {
        min_jaccard_overlap: 0.1
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2
      }
      sample_constraint {
        min_jaccard_overlap: 0.3
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2
      }
      sample_constraint {
        min_jaccard_overlap: 0.5
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2
      }
      sample_constraint {
        min_jaccard_overlap: 0.7
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2
      }
      sample_constraint {
        min_jaccard_overlap: 0.9
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2
      }
      sample_constraint {
        max_jaccard_overlap: 1
      }
      max_sample: 1
      max_trials: 50
    }
    label_map_file: "/data/siyu/dataset/coco/labelmap_coco-person.prototxt"
    part_sampler_prob: 0.2
  }
}
layer {
  name: "conv1_1"
  type: "Convolution"
  bottom: "data"
  top: "conv1_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu1_1"
  type: "ReLU"
  bottom: "conv1_1"
  top: "conv1_1"
}
layer {
  name: "conv1_2"
  type: "Convolution"
  bottom: "conv1_1"
  top: "conv1_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu1_2"
  type: "ReLU"
  bottom: "conv1_2"
  top: "conv1_2"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1_2"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2_1"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu2_1"
  type: "ReLU"
  bottom: "conv2_1"
  top: "conv2_1"
}
layer {
  name: "conv2_2"
  type: "Convolution"
  bottom: "conv2_1"
  top: "conv2_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu2_2"
  type: "ReLU"
  bottom: "conv2_2"
  top: "conv2_2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2_2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv3_1"
  type: "Convolution"
  bottom: "pool2"
  top: "conv3_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu3_1"
  type: "ReLU"
  bottom: "conv3_1"
  top: "conv3_1"
}
layer {
  name: "conv3_2"
  type: "Convolution"
  bottom: "conv3_1"
  top: "conv3_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu3_2"
  type: "ReLU"
  bottom: "conv3_2"
  top: "conv3_2"
}
layer {
  name: "conv3_3"
  type: "Convolution"
  bottom: "conv3_2"
  top: "conv3_3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu3_3"
  type: "ReLU"
  bottom: "conv3_3"
  top: "conv3_3"
}
layer {
  name: "pool3"
  type: "Pooling"
  bottom: "conv3_3"
  top: "pool3"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv4_1"
  type: "Convolution"
  bottom: "pool3"
  top: "conv4_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu4_1"
  type: "ReLU"
  bottom: "conv4_1"
  top: "conv4_1"
}
layer {
  name: "conv4_2"
  type: "Convolution"
  bottom: "conv4_1"
  top: "conv4_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu4_2"
  type: "ReLU"
  bottom: "conv4_2"
  top: "conv4_2"
}
layer {
  name: "conv4_3"
  type: "Convolution"
  bottom: "conv4_2"
  top: "conv4_3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu4_3"
  type: "ReLU"
  bottom: "conv4_3"
  top: "conv4_3"
}
layer {
  name: "pool4"
  type: "Pooling"
  bottom: "conv4_3"
  top: "pool4"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv5_1"
  type: "Convolution"
  bottom: "pool4"
  top: "conv5_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    dilation: 1
  }
}
layer {
  name: "relu5_1"
  type: "ReLU"
  bottom: "conv5_1"
  top: "conv5_1"
}
layer {
  name: "conv5_2"
  type: "Convolution"
  bottom: "conv5_1"
  top: "conv5_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    dilation: 1
  }
}
layer {
  name: "relu5_2"
  type: "ReLU"
  bottom: "conv5_2"
  top: "conv5_2"
}
layer {
  name: "conv5_3"
  type: "Convolution"
  bottom: "conv5_2"
  top: "conv5_3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    dilation: 1
  }
}
layer {
  name: "relu5_3"
  type: "ReLU"
  bottom: "conv5_3"
  top: "conv5_3"
}
layer {
  name: "pool5"
  type: "Pooling"
  bottom: "conv5_3"
  top: "pool5"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
  name: "fc6"
  type: "Convolution"
  bottom: "pool5"
  top: "fc6"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 1024
    pad: 6
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    dilation: 6
  }
}
layer {
  name: "relu6"
  type: "ReLU"
  bottom: "fc6"
  top: "fc6"
}
layer {
  name: "fc7"
  type: "Convolution"
  bottom: "fc6"
  top: "fc7"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 1024
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu7"
  type: "ReLU"
  bottom: "fc7"
  top: "fc7"
}
layer {
  name: "conv6_1"
  type: "Convolution"
  bottom: "fc7"
  top: "conv6_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv6_1_relu"
  type: "ReLU"
  bottom: "conv6_1"
  top: "conv6_1"
}
layer {
  name: "conv6_2"
  type: "Convolution"
  bottom: "conv6_1"
  top: "conv6_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv6_2_relu"
  type: "ReLU"
  bottom: "conv6_2"
  top: "conv6_2"
}
layer {
  name: "conv7_1"
  type: "Convolution"
  bottom: "conv6_2"
  top: "conv7_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv7_1_relu"
  type: "ReLU"
  bottom: "conv7_1"
  top: "conv7_1"
}
layer {
  name: "conv7_2"
  type: "Convolution"
  bottom: "conv7_1"
  top: "conv7_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv7_2_relu"
  type: "ReLU"
  bottom: "conv7_2"
  top: "conv7_2"
}
layer {
  name: "conv8_1"
  type: "Convolution"
  bottom: "conv7_2"
  top: "conv8_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv8_1_relu"
  type: "ReLU"
  bottom: "conv8_1"
  top: "conv8_1"
}
layer {
  name: "conv8_2"
  type: "Convolution"
  bottom: "conv8_1"
  top: "conv8_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 0
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv8_2_relu"
  type: "ReLU"
  bottom: "conv8_2"
  top: "conv8_2"
}
layer {
  name: "conv9_1"
  type: "Convolution"
  bottom: "conv8_2"
  top: "conv9_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv9_1_relu"
  type: "ReLU"
  bottom: "conv9_1"
  top: "conv9_1"
}
layer {
  name: "conv9_2"
  type: "Convolution"
  bottom: "conv9_1"
  top: "conv9_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 0
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv9_2_relu"
  type: "ReLU"
  bottom: "conv9_2"
  top: "conv9_2"
}
layer {
  name: "conv4_3_norm"
  type: "Normalize"
  bottom: "conv4_3"
  top: "conv4_3_norm"
  norm_param {
    across_spatial: false
    scale_filler {
      type: "constant"
      value: 20
    }
    channel_shared: false
  }
}
layer {
  name: "conv4_3_norm_mbox_loc"
  type: "Convolution"
  bottom: "conv4_3_norm"
  top: "conv4_3_norm_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 16
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv4_3_norm_mbox_loc_perm"
  type: "Permute"
  bottom: "conv4_3_norm_mbox_loc"
  top: "conv4_3_norm_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv4_3_norm_mbox_loc_perm"
  top: "conv4_3_norm_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_conf"
  type: "Convolution"
  bottom: "conv4_3_norm"
  top: "conv4_3_norm_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 8
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv4_3_norm_mbox_conf_perm"
  type: "Permute"
  bottom: "conv4_3_norm_mbox_conf"
  top: "conv4_3_norm_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv4_3_norm_mbox_conf_perm"
  top: "conv4_3_norm_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_clean"
  type: "Convolution"
  bottom: "conv4_3_norm"
  top: "conv4_3_norm_mbox_clean"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 4
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv4_3_norm_mbox_clean_perm"
  type: "Permute"
  bottom: "conv4_3_norm_mbox_clean"
  top: "conv4_3_norm_mbox_clean_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_clean_flat"
  type: "Flatten"
  bottom: "conv4_3_norm_mbox_clean_perm"
  top: "conv4_3_norm_mbox_clean_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv4_3_norm"
  bottom: "data"
  top: "conv4_3_norm_mbox_priorbox"
  prior_box_param {
    min_size: 35.84
    max_size: 76.8
    aspect_ratio: 2
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 8
    offset: 0.5
  }
}
layer {
  name: "fc7_mbox_loc"
  type: "Convolution"
  bottom: "fc7"
  top: "fc7_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 24
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "fc7_mbox_loc_perm"
  type: "Permute"
  bottom: "fc7_mbox_loc"
  top: "fc7_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "fc7_mbox_loc_flat"
  type: "Flatten"
  bottom: "fc7_mbox_loc_perm"
  top: "fc7_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "fc7_mbox_conf"
  type: "Convolution"
  bottom: "fc7"
  top: "fc7_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 12
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "fc7_mbox_conf_perm"
  type: "Permute"
  bottom: "fc7_mbox_conf"
  top: "fc7_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "fc7_mbox_conf_flat"
  type: "Flatten"
  bottom: "fc7_mbox_conf_perm"
  top: "fc7_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "fc7_mbox_clean"
  type: "Convolution"
  bottom: "fc7"
  top: "fc7_mbox_clean"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 6
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "fc7_mbox_clean_perm"
  type: "Permute"
  bottom: "fc7_mbox_clean"
  top: "fc7_mbox_clean_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "fc7_mbox_clean_flat"
  type: "Flatten"
  bottom: "fc7_mbox_clean_perm"
  top: "fc7_mbox_clean_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "fc7_mbox_priorbox"
  type: "PriorBox"
  bottom: "fc7"
  bottom: "data"
  top: "fc7_mbox_priorbox"
  prior_box_param {
    min_size: 76.8
    max_size: 168.96
    aspect_ratio: 2
    aspect_ratio: 3
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 16
    offset: 0.5
  }
}
layer {
  name: "conv6_2_mbox_loc"
  type: "Convolution"
  bottom: "conv6_2"
  top: "conv6_2_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 24
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv6_2_mbox_loc_perm"
  type: "Permute"
  bottom: "conv6_2_mbox_loc"
  top: "conv6_2_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv6_2_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv6_2_mbox_loc_perm"
  top: "conv6_2_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv6_2_mbox_conf"
  type: "Convolution"
  bottom: "conv6_2"
  top: "conv6_2_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 12
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv6_2_mbox_conf_perm"
  type: "Permute"
  bottom: "conv6_2_mbox_conf"
  top: "conv6_2_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv6_2_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv6_2_mbox_conf_perm"
  top: "conv6_2_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv6_2_mbox_clean"
  type: "Convolution"
  bottom: "conv6_2"
  top: "conv6_2_mbox_clean"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 6
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv6_2_mbox_clean_perm"
  type: "Permute"
  bottom: "conv6_2_mbox_clean"
  top: "conv6_2_mbox_clean_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv6_2_mbox_clean_flat"
  type: "Flatten"
  bottom: "conv6_2_mbox_clean_perm"
  top: "conv6_2_mbox_clean_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv6_2_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv6_2"
  bottom: "data"
  top: "conv6_2_mbox_priorbox"
  prior_box_param {
    min_size: 168.96
    max_size: 261.12
    aspect_ratio: 2
    aspect_ratio: 3
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 32
    offset: 0.5
  }
}
layer {
  name: "conv7_2_mbox_loc"
  type: "Convolution"
  bottom: "conv7_2"
  top: "conv7_2_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 24
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv7_2_mbox_loc_perm"
  type: "Permute"
  bottom: "conv7_2_mbox_loc"
  top: "conv7_2_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv7_2_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv7_2_mbox_loc_perm"
  top: "conv7_2_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv7_2_mbox_conf"
  type: "Convolution"
  bottom: "conv7_2"
  top: "conv7_2_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 12
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv7_2_mbox_conf_perm"
  type: "Permute"
  bottom: "conv7_2_mbox_conf"
  top: "conv7_2_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv7_2_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv7_2_mbox_conf_perm"
  top: "conv7_2_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv7_2_mbox_clean"
  type: "Convolution"
  bottom: "conv7_2"
  top: "conv7_2_mbox_clean"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 6
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv7_2_mbox_clean_perm"
  type: "Permute"
  bottom: "conv7_2_mbox_clean"
  top: "conv7_2_mbox_clean_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv7_2_mbox_clean_flat"
  type: "Flatten"
  bottom: "conv7_2_mbox_clean_perm"
  top: "conv7_2_mbox_clean_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv7_2_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv7_2"
  bottom: "data"
  top: "conv7_2_mbox_priorbox"
  prior_box_param {
    min_size: 261.12
    max_size: 353.28
    aspect_ratio: 2
    aspect_ratio: 3
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 64
    offset: 0.5
  }
}
layer {
  name: "conv8_2_mbox_loc"
  type: "Convolution"
  bottom: "conv8_2"
  top: "conv8_2_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 16
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv8_2_mbox_loc_perm"
  type: "Permute"
  bottom: "conv8_2_mbox_loc"
  top: "conv8_2_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv8_2_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv8_2_mbox_loc_perm"
  top: "conv8_2_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv8_2_mbox_conf"
  type: "Convolution"
  bottom: "conv8_2"
  top: "conv8_2_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 8
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv8_2_mbox_conf_perm"
  type: "Permute"
  bottom: "conv8_2_mbox_conf"
  top: "conv8_2_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv8_2_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv8_2_mbox_conf_perm"
  top: "conv8_2_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv8_2_mbox_clean"
  type: "Convolution"
  bottom: "conv8_2"
  top: "conv8_2_mbox_clean"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 4
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv8_2_mbox_clean_perm"
  type: "Permute"
  bottom: "conv8_2_mbox_clean"
  top: "conv8_2_mbox_clean_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv8_2_mbox_clean_flat"
  type: "Flatten"
  bottom: "conv8_2_mbox_clean_perm"
  top: "conv8_2_mbox_clean_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv8_2_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv8_2"
  bottom: "data"
  top: "conv8_2_mbox_priorbox"
  prior_box_param {
    min_size: 353.28
    max_size: 445.44
    aspect_ratio: 2
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 100
    offset: 0.5
  }
}
layer {
  name: "conv9_2_mbox_loc"
  type: "Convolution"
  bottom: "conv9_2"
  top: "conv9_2_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 16
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv9_2_mbox_loc_perm"
  type: "Permute"
  bottom: "conv9_2_mbox_loc"
  top: "conv9_2_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv9_2_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv9_2_mbox_loc_perm"
  top: "conv9_2_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv9_2_mbox_conf"
  type: "Convolution"
  bottom: "conv9_2"
  top: "conv9_2_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 8
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv9_2_mbox_conf_perm"
  type: "Permute"
  bottom: "conv9_2_mbox_conf"
  top: "conv9_2_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv9_2_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv9_2_mbox_conf_perm"
  top: "conv9_2_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv9_2_mbox_clean"
  type: "Convolution"
  bottom: "conv9_2"
  top: "conv9_2_mbox_clean"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 4
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv9_2_mbox_clean_perm"
  type: "Permute"
  bottom: "conv9_2_mbox_clean"
  top: "conv9_2_mbox_clean_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv9_2_mbox_clean_flat"
I0329 19:07:50.377820  2693 layer_factory.hpp:77] Creating layer data
I0329 19:07:50.378252  2693 net.cpp:100] Creating Layer data
I0329 19:07:50.378268  2693 net.cpp:408] data -> data
I0329 19:07:50.378312  2693 net.cpp:408] data -> label
I0329 19:07:50.379772  2731 db_lmdb.cpp:35] Opened lmdb /data/siyu/dataset/coco/lmdb/COCO_Train2014_person_lmdb
I0329 19:07:50.397542  2693 annotated_data_layer.cpp:62] output data size: 8,3,512,512
I0329 19:07:50.441294  2693 net.cpp:150] Setting up data
I0329 19:07:50.441347  2693 net.cpp:157] Top shape: 8 3 512 512 (6291456)
I0329 19:07:50.441354  2693 net.cpp:157] Top shape: 1 1 1 8 (8)
I0329 19:07:50.441361  2693 net.cpp:165] Memory required for data: 25165856
I0329 19:07:50.441373  2693 layer_factory.hpp:77] Creating layer data_data_0_split
I0329 19:07:50.441395  2693 net.cpp:100] Creating Layer data_data_0_split
I0329 19:07:50.441404  2693 net.cpp:434] data_data_0_split <- data
I0329 19:07:50.441421  2693 net.cpp:408] data_data_0_split -> data_data_0_split_0
I0329 19:07:50.441435  2693 net.cpp:408] data_data_0_split -> data_data_0_split_1
I0329 19:07:50.441445  2693 net.cpp:408] data_data_0_split -> data_data_0_split_2
I0329 19:07:50.441453  2693 net.cpp:408] data_data_0_split -> data_data_0_split_3
I0329 19:07:50.441462  2693 net.cpp:408] data_data_0_split -> data_data_0_split_4
I0329 19:07:50.441469  2693 net.cpp:408] data_data_0_split -> data_data_0_split_5
I0329 19:07:50.441476  2693 net.cpp:408] data_data_0_split -> data_data_0_split_6
I0329 19:07:50.441606  2693 net.cpp:150] Setting up data_data_0_split
I0329 19:07:50.441614  2693 net.cpp:157] Top shape: 8 3 512 512 (6291456)
I0329 19:07:50.441619  2693 net.cpp:157] Top shape: 8 3 512 512 (6291456)
I0329 19:07:50.441627  2693 net.cpp:157] Top shape: 8 3 512 512 (6291456)
I0329 19:07:50.441630  2693 net.cpp:157] Top shape: 8 3 512 512 (6291456)
I0329 19:07:50.441635  2693 net.cpp:157] Top shape: 8 3 512 512 (6291456)
I0329 19:07:50.441656  2693 net.cpp:157] Top shape: 8 3 512 512 (6291456)
I0329 19:07:50.441661  2693 net.cpp:157] Top shape: 8 3 512 512 (6291456)
I0329 19:07:50.441665  2693 net.cpp:165] Memory required for data: 201326624
I0329 19:07:50.441670  2693 layer_factory.hpp:77] Creating layer conv1_1
I0329 19:07:50.441699  2693 net.cpp:100] Creating Layer conv1_1
I0329 19:07:50.441706  2693 net.cpp:434] conv1_1 <- data_data_0_split_0
I0329 19:07:50.441714  2693 net.cpp:408] conv1_1 -> conv1_1
I0329 19:07:50.724930  2693 net.cpp:150] Setting up conv1_1
I0329 19:07:50.724975  2693 net.cpp:157] Top shape: 8 64 512 512 (134217728)
I0329 19:07:50.724982  2693 net.cpp:165] Memory required for data: 738197536
I0329 19:07:50.725004  2693 layer_factory.hpp:77] Creating layer relu1_1
I0329 19:07:50.725021  2693 net.cpp:100] Creating Layer relu1_1
I0329 19:07:50.725028  2693 net.cpp:434] relu1_1 <- conv1_1
I0329 19:07:50.725038  2693 net.cpp:395] relu1_1 -> conv1_1 (in-place)
I0329 19:07:50.725373  2693 net.cpp:150] Setting up relu1_1
I0329 19:07:50.725390  2693 net.cpp:157] Top shape: 8 64 512 512 (134217728)
I0329 19:07:50.725396  2693 net.cpp:165] Memory required for data: 1275068448
I0329 19:07:50.725402  2693 layer_factory.hpp:77] Creating layer conv1_2
I0329 19:07:50.725421  2693 net.cpp:100] Creating Layer conv1_2
I0329 19:07:50.725427  2693 net.cpp:434] conv1_2 <- conv1_1
I0329 19:07:50.725435  2693 net.cpp:408] conv1_2 -> conv1_2
I0329 19:07:50.728260  2693 net.cpp:150] Setting up conv1_2
I0329 19:07:50.728281  2693 net.cpp:157] Top shape: 8 64 512 512 (134217728)
I0329 19:07:50.728286  2693 net.cpp:165] Memory required for data: 1811939360
I0329 19:07:50.728299  2693 layer_factory.hpp:77] Creating layer relu1_2
I0329 19:07:50.728308  2693 net.cpp:100] Creating Layer relu1_2
I0329 19:07:50.728315  2693 net.cpp:434] relu1_2 <- conv1_2
I0329 19:07:50.728322  2693 net.cpp:395] relu1_2 -> conv1_2 (in-place)
I0329 19:07:50.728507  2693 net.cpp:150] Setting up relu1_2
I0329 19:07:50.728520  2693 net.cpp:157] Top shape: 8 64 512 512 (134217728)
I0329 19:07:50.728526  2693 net.cpp:165] Memory required for data: 2348810272
I0329 19:07:50.728533  2693 layer_factory.hpp:77] Creating layer pool1
I0329 19:07:50.728543  2693 net.cpp:100] Creating Layer pool1
I0329 19:07:50.728551  2693 net.cpp:434] pool1 <- conv1_2
I0329 19:07:50.728559  2693 net.cpp:408] pool1 -> pool1
I0329 19:07:50.728618  2693 net.cpp:150] Setting up pool1
I0329 19:07:50.728629  2693 net.cpp:157] Top shape: 8 64 256 256 (33554432)
I0329 19:07:50.728634  2693 net.cpp:165] Memory required for data: 2483028000
I0329 19:07:50.728639  2693 layer_factory.hpp:77] Creating layer conv2_1
I0329 19:07:50.728652  2693 net.cpp:100] Creating Layer conv2_1
I0329 19:07:50.728659  2693 net.cpp:434] conv2_1 <- pool1
I0329 19:07:50.728667  2693 net.cpp:408] conv2_1 -> conv2_1
I0329 19:07:50.731015  2693 net.cpp:150] Setting up conv2_1
I0329 19:07:50.731034  2693 net.cpp:157] Top shape: 8 128 256 256 (67108864)
I0329 19:07:50.731040  2693 net.cpp:165] Memory required for data: 2751463456
I0329 19:07:50.731053  2693 layer_factory.hpp:77] Creating layer relu2_1
I0329 19:07:50.731063  2693 net.cpp:100] Creating Layer relu2_1
I0329 19:07:50.731070  2693 net.cpp:434] relu2_1 <- conv2_1
I0329 19:07:50.731076  2693 net.cpp:395] relu2_1 -> conv2_1 (in-place)
I0329 19:07:50.731263  2693 net.cpp:150] Setting up relu2_1
I0329 19:07:50.731276  2693 net.cpp:157] Top shape: 8 128 256 256 (67108864)
I0329 19:07:50.731281  2693 net.cpp:165] Memory required for data: 3019898912
I0329 19:07:50.731287  2693 layer_factory.hpp:77] Creating layer conv2_2
I0329 19:07:50.731300  2693 net.cpp:100] Creating Layer conv2_2
I0329 19:07:50.731308  2693 net.cpp:434] conv2_2 <- conv2_1
I0329 19:07:50.731317  2693 net.cpp:408] conv2_2 -> conv2_2
I0329 19:07:50.733649  2693 net.cpp:150] Setting up conv2_2
I0329 19:07:50.733667  2693 net.cpp:157] Top shape: 8 128 256 256 (67108864)
I0329 19:07:50.733674  2693 net.cpp:165] Memory required for data: 3288334368
I0329 19:07:50.733706  2693 layer_factory.hpp:77] Creating layer relu2_2
I0329 19:07:50.733716  2693 net.cpp:100] Creating Layer relu2_2
I0329 19:07:50.733722  2693 net.cpp:434] relu2_2 <- conv2_2
I0329 19:07:50.733729  2693 net.cpp:395] relu2_2 -> conv2_2 (in-place)
I0329 19:07:50.734040  2693 net.cpp:150] Setting up relu2_2
I0329 19:07:50.734057  2693 net.cpp:157] Top shape: 8 128 256 256 (67108864)
I0329 19:07:50.734062  2693 net.cpp:165] Memory required for data: 3556769824
I0329 19:07:50.734068  2693 layer_factory.hpp:77] Creating layer pool2
I0329 19:07:50.734078  2693 net.cpp:100] Creating Layer pool2
I0329 19:07:50.734084  2693 net.cpp:434] pool2 <- conv2_2
I0329 19:07:50.734091  2693 net.cpp:408] pool2 -> pool2
I0329 19:07:50.734139  2693 net.cpp:150] Setting up pool2
I0329 19:07:50.734149  2693 net.cpp:157] Top shape: 8 128 128 128 (16777216)
I0329 19:07:50.734154  2693 net.cpp:165] Memory required for data: 3623878688
I0329 19:07:50.734159  2693 layer_factory.hpp:77] Creating layer conv3_1
I0329 19:07:50.734170  2693 net.cpp:100] Creating Layer conv3_1
I0329 19:07:50.734179  2693 net.cpp:434] conv3_1 <- pool2
I0329 19:07:50.734185  2693 net.cpp:408] conv3_1 -> conv3_1
I0329 19:07:50.739344  2693 net.cpp:150] Setting up conv3_1
I0329 19:07:50.739364  2693 net.cpp:157] Top shape: 8 256 128 128 (33554432)
I0329 19:07:50.739372  2693 net.cpp:165] Memory required for data: 3758096416
I0329 19:07:50.739384  2693 layer_factory.hpp:77] Creating layer relu3_1
I0329 19:07:50.739394  2693 net.cpp:100] Creating Layer relu3_1
I0329 19:07:50.739399  2693 net.cpp:434] relu3_1 <- conv3_1
I0329 19:07:50.739408  2693 net.cpp:395] relu3_1 -> conv3_1 (in-place)
I0329 19:07:50.739603  2693 net.cpp:150] Setting up relu3_1
I0329 19:07:50.739617  2693 net.cpp:157] Top shape: 8 256 128 128 (33554432)
I0329 19:07:50.739622  2693 net.cpp:165] Memory required for data: 3892314144
I0329 19:07:50.739629  2693 layer_factory.hpp:77] Creating layer conv3_2
I0329 19:07:50.739642  2693 net.cpp:100] Creating Layer conv3_2
I0329 19:07:50.739650  2693 net.cpp:434] conv3_2 <- conv3_1
I0329 19:07:50.739660  2693 net.cpp:408] conv3_2 -> conv3_2
I0329 19:07:50.746950  2693 net.cpp:150] Setting up conv3_2
I0329 19:07:50.746976  2693 net.cpp:157] Top shape: 8 256 128 128 (33554432)
I0329 19:07:50.746986  2693 net.cpp:165] Memory required for data: 4026531872
I0329 19:07:50.746996  2693 layer_factory.hpp:77] Creating layer relu3_2
I0329 19:07:50.747005  2693 net.cpp:100] Creating Layer relu3_2
I0329 19:07:50.747011  2693 net.cpp:434] relu3_2 <- conv3_2
I0329 19:07:50.747018  2693 net.cpp:395] relu3_2 -> conv3_2 (in-place)
I0329 19:07:50.747205  2693 net.cpp:150] Setting up relu3_2
I0329 19:07:50.747220  2693 net.cpp:157] Top shape: 8 256 128 128 (33554432)
I0329 19:07:50.747225  2693 net.cpp:165] Memory required for data: 4160749600
I0329 19:07:50.747231  2693 layer_factory.hpp:77] Creating layer conv3_3
I0329 19:07:50.747246  2693 net.cpp:100] Creating Layer conv3_3
I0329 19:07:50.747254  2693 net.cpp:434] conv3_3 <- conv3_2
I0329 19:07:50.747263  2693 net.cpp:408] conv3_3 -> conv3_3
I0329 19:07:50.754720  2693 net.cpp:150] Setting up conv3_3
I0329 19:07:50.754740  2693 net.cpp:157] Top shape: 8 256 128 128 (33554432)
I0329 19:07:50.754746  2693 net.cpp:165] Memory required for data: 4294967328
I0329 19:07:50.754756  2693 layer_factory.hpp:77] Creating layer relu3_3
I0329 19:07:50.754765  2693 net.cpp:100] Creating Layer relu3_3
I0329 19:07:50.754771  2693 net.cpp:434] relu3_3 <- conv3_3
I0329 19:07:50.754779  2693 net.cpp:395] relu3_3 -> conv3_3 (in-place)
I0329 19:07:50.755091  2693 net.cpp:150] Setting up relu3_3
I0329 19:07:50.755108  2693 net.cpp:157] Top shape: 8 256 128 128 (33554432)
I0329 19:07:50.755115  2693 net.cpp:165] Memory required for data: 4429185056
I0329 19:07:50.755120  2693 layer_factory.hpp:77] Creating layer pool3
I0329 19:07:50.755129  2693 net.cpp:100] Creating Layer pool3
I0329 19:07:50.755136  2693 net.cpp:434] pool3 <- conv3_3
I0329 19:07:50.755143  2693 net.cpp:408] pool3 -> pool3
I0329 19:07:50.755194  2693 net.cpp:150] Setting up pool3
I0329 19:07:50.755221  2693 net.cpp:157] Top shape: 8 256 64 64 (8388608)
I0329 19:07:50.755228  2693 net.cpp:165] Memory required for data: 4462739488
I0329 19:07:50.755233  2693 layer_factory.hpp:77] Creating layer conv4_1
I0329 19:07:50.755245  2693 net.cpp:100] Creating Layer conv4_1
I0329 19:07:50.755252  2693 net.cpp:434] conv4_1 <- pool3
I0329 19:07:50.755261  2693 net.cpp:408] conv4_1 -> conv4_1
I0329 19:07:50.768599  2693 net.cpp:150] Setting up conv4_1
I0329 19:07:50.768620  2693 net.cpp:157] Top shape: 8 512 64 64 (16777216)
I0329 19:07:50.768625  2693 net.cpp:165] Memory required for data: 4529848352
I0329 19:07:50.768635  2693 layer_factory.hpp:77] Creating layer relu4_1
I0329 19:07:50.768646  2693 net.cpp:100] Creating Layer relu4_1
I0329 19:07:50.768651  2693 net.cpp:434] relu4_1 <- conv4_1
I0329 19:07:50.768659  2693 net.cpp:395] relu4_1 -> conv4_1 (in-place)
I0329 19:07:50.768847  2693 net.cpp:150] Setting up relu4_1
I0329 19:07:50.768862  2693 net.cpp:157] Top shape: 8 512 64 64 (16777216)
I0329 19:07:50.768867  2693 net.cpp:165] Memory required for data: 4596957216
I0329 19:07:50.768872  2693 layer_factory.hpp:77] Creating layer conv4_2
I0329 19:07:50.768885  2693 net.cpp:100] Creating Layer conv4_2
I0329 19:07:50.768893  2693 net.cpp:434] conv4_2 <- conv4_1
I0329 19:07:50.768901  2693 net.cpp:408] conv4_2 -> conv4_2
I0329 19:07:50.792934  2693 net.cpp:150] Setting up conv4_2
I0329 19:07:50.792959  2693 net.cpp:157] Top shape: 8 512 64 64 (16777216)
I0329 19:07:50.792965  2693 net.cpp:165] Memory required for data: 4664066080
I0329 19:07:50.792982  2693 layer_factory.hpp:77] Creating layer relu4_2
I0329 19:07:50.792995  2693 net.cpp:100] Creating Layer relu4_2
I0329 19:07:50.793001  2693 net.cpp:434] relu4_2 <- conv4_2
I0329 19:07:50.793009  2693 net.cpp:395] relu4_2 -> conv4_2 (in-place)
I0329 19:07:50.793210  2693 net.cpp:150] Setting up relu4_2
I0329 19:07:50.793223  2693 net.cpp:157] Top shape: 8 512 64 64 (16777216)
I0329 19:07:50.793228  2693 net.cpp:165] Memory required for data: 4731174944
I0329 19:07:50.793234  2693 layer_factory.hpp:77] Creating layer conv4_3
I0329 19:07:50.793249  2693 net.cpp:100] Creating Layer conv4_3
I0329 19:07:50.793257  2693 net.cpp:434] conv4_3 <- conv4_2
I0329 19:07:50.793265  2693 net.cpp:408] conv4_3 -> conv4_3
I0329 19:07:50.818205  2693 net.cpp:150] Setting up conv4_3
I0329 19:07:50.818233  2693 net.cpp:157] Top shape: 8 512 64 64 (16777216)
I0329 19:07:50.818239  2693 net.cpp:165] Memory required for data: 4798283808
I0329 19:07:50.818248  2693 layer_factory.hpp:77] Creating layer relu4_3
I0329 19:07:50.818259  2693 net.cpp:100] Creating Layer relu4_3
I0329 19:07:50.818265  2693 net.cpp:434] relu4_3 <- conv4_3
I0329 19:07:50.818272  2693 net.cpp:395] relu4_3 -> conv4_3 (in-place)
I0329 19:07:50.818791  2693 net.cpp:150] Setting up relu4_3
I0329 19:07:50.818810  2693 net.cpp:157] Top shape: 8 512 64 64 (16777216)
I0329 19:07:50.818815  2693 net.cpp:165] Memory required for data: 4865392672
I0329 19:07:50.818821  2693 layer_factory.hpp:77] Creating layer conv4_3_relu4_3_0_split
I0329 19:07:50.818830  2693 net.cpp:100] Creating Layer conv4_3_relu4_3_0_split
I0329 19:07:50.818835  2693 net.cpp:434] conv4_3_relu4_3_0_split <- conv4_3
I0329 19:07:50.818843  2693 net.cpp:408] conv4_3_relu4_3_0_split -> conv4_3_relu4_3_0_split_0
I0329 19:07:50.818853  2693 net.cpp:408] conv4_3_relu4_3_0_split -> conv4_3_relu4_3_0_split_1
I0329 19:07:50.818910  2693 net.cpp:150] Setting up conv4_3_relu4_3_0_split
I0329 19:07:50.818922  2693 net.cpp:157] Top shape: 8 512 64 64 (16777216)
I0329 19:07:50.818928  2693 net.cpp:157] Top shape: 8 512 64 64 (16777216)
I0329 19:07:50.818931  2693 net.cpp:165] Memory required for data: 4999610400
I0329 19:07:50.818938  2693 layer_factory.hpp:77] Creating layer pool4
I0329 19:07:50.818949  2693 net.cpp:100] Creating Layer pool4
I0329 19:07:50.818958  2693 net.cpp:434] pool4 <- conv4_3_relu4_3_0_split_0
I0329 19:07:50.818964  2693 net.cpp:408] pool4 -> pool4
I0329 19:07:50.819011  2693 net.cpp:150] Setting up pool4
I0329 19:07:50.819021  2693 net.cpp:157] Top shape: 8 512 32 32 (4194304)
I0329 19:07:50.819044  2693 net.cpp:165] Memory required for data: 5016387616
I0329 19:07:50.819051  2693 layer_factory.hpp:77] Creating layer conv5_1
I0329 19:07:50.819066  2693 net.cpp:100] Creating Layer conv5_1
I0329 19:07:50.819074  2693 net.cpp:434] conv5_1 <- pool4
I0329 19:07:50.819083  2693 net.cpp:408] conv5_1 -> conv5_1
I0329 19:07:50.843195  2693 net.cpp:150] Setting up conv5_1
I0329 19:07:50.843220  2693 net.cpp:157] Top shape: 8 512 32 32 (4194304)
I0329 19:07:50.843226  2693 net.cpp:165] Memory required for data: 5033164832
I0329 19:07:50.843235  2693 layer_factory.hpp:77] Creating layer relu5_1
I0329 19:07:50.843245  2693 net.cpp:100] Creating Layer relu5_1
I0329 19:07:50.843251  2693 net.cpp:434] relu5_1 <- conv5_1
I0329 19:07:50.843261  2693 net.cpp:395] relu5_1 -> conv5_1 (in-place)
I0329 19:07:50.843461  2693 net.cpp:150] Setting up relu5_1
I0329 19:07:50.843477  2693 net.cpp:157] Top shape: 8 512 32 32 (4194304)
I0329 19:07:50.843482  2693 net.cpp:165] Memory required for data: 5049942048
I0329 19:07:50.843487  2693 layer_factory.hpp:77] Creating layer conv5_2
I0329 19:07:50.843502  2693 net.cpp:100] Creating Layer conv5_2
I0329 19:07:50.843509  2693 net.cpp:434] conv5_2 <- conv5_1
I0329 19:07:50.843518  2693 net.cpp:408] conv5_2 -> conv5_2
I0329 19:07:50.867422  2693 net.cpp:150] Setting up conv5_2
I0329 19:07:50.867445  2693 net.cpp:157] Top shape: 8 512 32 32 (4194304)
I0329 19:07:50.867451  2693 net.cpp:165] Memory required for data: 5066719264
I0329 19:07:50.867461  2693 layer_factory.hpp:77] Creating layer relu5_2
I0329 19:07:50.867471  2693 net.cpp:100] Creating Layer relu5_2
I0329 19:07:50.867480  2693 net.cpp:434] relu5_2 <- conv5_2
I0329 19:07:50.867489  2693 net.cpp:395] relu5_2 -> conv5_2 (in-place)
I0329 19:07:50.867702  2693 net.cpp:150] Setting up relu5_2
I0329 19:07:50.867718  2693 net.cpp:157] Top shape: 8 512 32 32 (4194304)
I0329 19:07:50.867723  2693 net.cpp:165] Memory required for data: 5083496480
I0329 19:07:50.867729  2693 layer_factory.hpp:77] Creating layer conv5_3
I0329 19:07:50.867744  2693 net.cpp:100] Creating Layer conv5_3
I0329 19:07:50.867753  2693 net.cpp:434] conv5_3 <- conv5_2
I0329 19:07:50.867763  2693 net.cpp:408] conv5_3 -> conv5_3
I0329 19:07:50.891664  2693 net.cpp:150] Setting up conv5_3
I0329 19:07:50.891686  2693 net.cpp:157] Top shape: 8 512 32 32 (4194304)
I0329 19:07:50.891692  2693 net.cpp:165] Memory required for data: 5100273696
I0329 19:07:50.891702  2693 layer_factory.hpp:77] Creating layer relu5_3
I0329 19:07:50.891721  2693 net.cpp:100] Creating Layer relu5_3
I0329 19:07:50.891727  2693 net.cpp:434] relu5_3 <- conv5_3
I0329 19:07:50.891736  2693 net.cpp:395] relu5_3 -> conv5_3 (in-place)
I0329 19:07:50.892073  2693 net.cpp:150] Setting up relu5_3
I0329 19:07:50.892091  2693 net.cpp:157] Top shape: 8 512 32 32 (4194304)
I0329 19:07:50.892096  2693 net.cpp:165] Memory required for data: 5117050912
I0329 19:07:50.892102  2693 layer_factory.hpp:77] Creating layer pool5
I0329 19:07:50.892114  2693 net.cpp:100] Creating Layer pool5
I0329 19:07:50.892120  2693 net.cpp:434] pool5 <- conv5_3
I0329 19:07:50.892128  2693 net.cpp:408] pool5 -> pool5
I0329 19:07:50.892186  2693 net.cpp:150] Setting up pool5
I0329 19:07:50.892197  2693 net.cpp:157] Top shape: 8 512 32 32 (4194304)
I0329 19:07:50.892202  2693 net.cpp:165] Memory required for data: 5133828128
I0329 19:07:50.892207  2693 layer_factory.hpp:77] Creating layer fc6
I0329 19:07:50.892222  2693 net.cpp:100] Creating Layer fc6
I0329 19:07:50.892230  2693 net.cpp:434] fc6 <- pool5
I0329 19:07:50.892241  2693 net.cpp:408] fc6 -> fc6
I0329 19:07:50.939756  2693 net.cpp:150] Setting up fc6
I0329 19:07:50.939805  2693 net.cpp:157] Top shape: 8 1024 32 32 (8388608)
I0329 19:07:50.939811  2693 net.cpp:165] Memory required for data: 5167382560
I0329 19:07:50.939823  2693 layer_factory.hpp:77] Creating layer relu6
I0329 19:07:50.939842  2693 net.cpp:100] Creating Layer relu6
I0329 19:07:50.939851  2693 net.cpp:434] relu6 <- fc6
I0329 19:07:50.939860  2693 net.cpp:395] relu6 -> fc6 (in-place)
I0329 19:07:50.940163  2693 net.cpp:150] Setting up relu6
I0329 19:07:50.940177  2693 net.cpp:157] Top shape: 8 1024 32 32 (8388608)
I0329 19:07:50.940182  2693 net.cpp:165] Memory required for data: 5200936992
I0329 19:07:50.940188  2693 layer_factory.hpp:77] Creating layer fc7
I0329 19:07:50.940207  2693 net.cpp:100] Creating Layer fc7
I0329 19:07:50.940212  2693 net.cpp:434] fc7 <- fc6
I0329 19:07:50.940222  2693 net.cpp:408] fc7 -> fc7
I0329 19:07:50.951663  2693 net.cpp:150] Setting up fc7
I0329 19:07:50.951683  2693 net.cpp:157] Top shape: 8 1024 32 32 (8388608)
I0329 19:07:50.951689  2693 net.cpp:165] Memory required for data: 5234491424
I0329 19:07:50.951699  2693 layer_factory.hpp:77] Creating layer relu7
I0329 19:07:50.951710  2693 net.cpp:100] Creating Layer relu7
I0329 19:07:50.951717  2693 net.cpp:434] relu7 <- fc7
I0329 19:07:50.951725  2693 net.cpp:395] relu7 -> fc7 (in-place)
I0329 19:07:50.952061  2693 net.cpp:150] Setting up relu7
I0329 19:07:50.952080  2693 net.cpp:157] Top shape: 8 1024 32 32 (8388608)
I0329 19:07:50.952085  2693 net.cpp:165] Memory required for data: 5268045856
I0329 19:07:50.952090  2693 layer_factory.hpp:77] Creating layer fc7_relu7_0_split
I0329 19:07:50.952103  2693 net.cpp:100] Creating Layer fc7_relu7_0_split
I0329 19:07:50.952108  2693 net.cpp:434] fc7_relu7_0_split <- fc7
I0329 19:07:50.952118  2693 net.cpp:408] fc7_relu7_0_split -> fc7_relu7_0_split_0
I0329 19:07:50.952128  2693 net.cpp:408] fc7_relu7_0_split -> fc7_relu7_0_split_1
I0329 19:07:50.952145  2693 net.cpp:408] fc7_relu7_0_split -> fc7_relu7_0_split_2
I0329 19:07:50.952157  2693 net.cpp:408] fc7_relu7_0_split -> fc7_relu7_0_split_3
I0329 19:07:50.952164  2693 net.cpp:408] fc7_relu7_0_split -> fc7_relu7_0_split_4
I0329 19:07:50.952266  2693 net.cpp:150] Setting up fc7_relu7_0_split
I0329 19:07:50.952277  2693 net.cpp:157] Top shape: 8 1024 32 32 (8388608)
I0329 19:07:50.952285  2693 net.cpp:157] Top shape: 8 1024 32 32 (8388608)
I0329 19:07:50.952289  2693 net.cpp:157] Top shape: 8 1024 32 32 (8388608)
I0329 19:07:50.952296  2693 net.cpp:157] Top shape: 8 1024 32 32 (8388608)
I0329 19:07:50.952301  2693 net.cpp:157] Top shape: 8 1024 32 32 (8388608)
I0329 19:07:50.952306  2693 net.cpp:165] Memory required for data: 5435818016
I0329 19:07:50.952311  2693 layer_factory.hpp:77] Creating layer conv6_1
I0329 19:07:50.952325  2693 net.cpp:100] Creating Layer conv6_1
I0329 19:07:50.952333  2693 net.cpp:434] conv6_1 <- fc7_relu7_0_split_0
I0329 19:07:50.952344  2693 net.cpp:408] conv6_1 -> conv6_1
I0329 19:07:50.956163  2693 net.cpp:150] Setting up conv6_1
I0329 19:07:50.956182  2693 net.cpp:157] Top shape: 8 256 32 32 (2097152)
I0329 19:07:50.956188  2693 net.cpp:165] Memory required for data: 5444206624
I0329 19:07:50.956197  2693 layer_factory.hpp:77] Creating layer conv6_1_relu
I0329 19:07:50.956207  2693 net.cpp:100] Creating Layer conv6_1_relu
I0329 19:07:50.956213  2693 net.cpp:434] conv6_1_relu <- conv6_1
I0329 19:07:50.956223  2693 net.cpp:395] conv6_1_relu -> conv6_1 (in-place)
I0329 19:07:50.956562  2693 net.cpp:150] Setting up conv6_1_relu
I0329 19:07:50.956578  2693 net.cpp:157] Top shape: 8 256 32 32 (2097152)
I0329 19:07:50.956583  2693 net.cpp:165] Memory required for data: 5452595232
I0329 19:07:50.956589  2693 layer_factory.hpp:77] Creating layer conv6_2
I0329 19:07:50.956605  2693 net.cpp:100] Creating Layer conv6_2
I0329 19:07:50.956611  2693 net.cpp:434] conv6_2 <- conv6_1
I0329 19:07:50.956620  2693 net.cpp:408] conv6_2 -> conv6_2
I0329 19:07:50.969329  2693 net.cpp:150] Setting up conv6_2
I0329 19:07:50.969352  2693 net.cpp:157] Top shape: 8 512 16 16 (1048576)
I0329 19:07:50.969357  2693 net.cpp:165] Memory required for data: 5456789536
I0329 19:07:50.969378  2693 layer_factory.hpp:77] Creating layer conv6_2_relu
I0329 19:07:50.969388  2693 net.cpp:100] Creating Layer conv6_2_relu
I0329 19:07:50.969393  2693 net.cpp:434] conv6_2_relu <- conv6_2
I0329 19:07:50.969401  2693 net.cpp:395] conv6_2_relu -> conv6_2 (in-place)
I0329 19:07:50.969612  2693 net.cpp:150] Setting up conv6_2_relu
I0329 19:07:50.969642  2693 net.cpp:157] Top shape: 8 512 16 16 (1048576)
I0329 19:07:50.969648  2693 net.cpp:165] Memory required for data: 5460983840
I0329 19:07:50.969653  2693 layer_factory.hpp:77] Creating layer conv6_2_conv6_2_relu_0_split
I0329 19:07:50.969662  2693 net.cpp:100] Creating Layer conv6_2_conv6_2_relu_0_split
I0329 19:07:50.969671  2693 net.cpp:434] conv6_2_conv6_2_relu_0_split <- conv6_2
I0329 19:07:50.969681  2693 net.cpp:408] conv6_2_conv6_2_relu_0_split -> conv6_2_conv6_2_relu_0_split_0
I0329 19:07:50.969691  2693 net.cpp:408] conv6_2_conv6_2_relu_0_split -> conv6_2_conv6_2_relu_0_split_1
I0329 19:07:50.969701  2693 net.cpp:408] conv6_2_conv6_2_relu_0_split -> conv6_2_conv6_2_relu_0_split_2
I0329 19:07:50.969708  2693 net.cpp:408] conv6_2_conv6_2_relu_0_split -> conv6_2_conv6_2_relu_0_split_3
I0329 19:07:50.969717  2693 net.cpp:408] conv6_2_conv6_2_relu_0_split -> conv6_2_conv6_2_relu_0_split_4
I0329 19:07:50.969820  2693 net.cpp:150] Setting up conv6_2_conv6_2_relu_0_split
I0329 19:07:50.969830  2693 net.cpp:157] Top shape: 8 512 16 16 (1048576)
I0329 19:07:50.969836  2693 net.cpp:157] Top shape: 8 512 16 16 (1048576)
I0329 19:07:50.969842  2693 net.cpp:157] Top shape: 8 512 16 16 (1048576)
I0329 19:07:50.969847  2693 net.cpp:157] Top shape: 8 512 16 16 (1048576)
I0329 19:07:50.969852  2693 net.cpp:157] Top shape: 8 512 16 16 (1048576)
I0329 19:07:50.969857  2693 net.cpp:165] Memory required for data: 5481955360
I0329 19:07:50.969863  2693 layer_factory.hpp:77] Creating layer conv7_1
I0329 19:07:50.969879  2693 net.cpp:100] Creating Layer conv7_1
I0329 19:07:50.969887  2693 net.cpp:434] conv7_1 <- conv6_2_conv6_2_relu_0_split_0
I0329 19:07:50.969897  2693 net.cpp:408] conv7_1 -> conv7_1
I0329 19:07:50.971619  2693 net.cpp:150] Setting up conv7_1
I0329 19:07:50.971638  2693 net.cpp:157] Top shape: 8 128 16 16 (262144)
I0329 19:07:50.971643  2693 net.cpp:165] Memory required for data: 5483003936
I0329 19:07:50.971652  2693 layer_factory.hpp:77] Creating layer conv7_1_relu
I0329 19:07:50.971663  2693 net.cpp:100] Creating Layer conv7_1_relu
I0329 19:07:50.971669  2693 net.cpp:434] conv7_1_relu <- conv7_1
I0329 19:07:50.971678  2693 net.cpp:395] conv7_1_relu -> conv7_1 (in-place)
I0329 19:07:50.972012  2693 net.cpp:150] Setting up conv7_1_relu
I0329 19:07:50.972029  2693 net.cpp:157] Top shape: 8 128 16 16 (262144)
I0329 19:07:50.972035  2693 net.cpp:165] Memory required for data: 5484052512
I0329 19:07:50.972041  2693 layer_factory.hpp:77] Creating layer conv7_2
I0329 19:07:50.972055  2693 net.cpp:100] Creating Layer conv7_2
I0329 19:07:50.972061  2693 net.cpp:434] conv7_2 <- conv7_1
I0329 19:07:50.972072  2693 net.cpp:408] conv7_2 -> conv7_2
I0329 19:07:50.976078  2693 net.cpp:150] Setting up conv7_2
I0329 19:07:50.976097  2693 net.cpp:157] Top shape: 8 256 8 8 (131072)
I0329 19:07:50.976104  2693 net.cpp:165] Memory required for data: 5484576800
I0329 19:07:50.976111  2693 layer_factory.hpp:77] Creating layer conv7_2_relu
I0329 19:07:50.976122  2693 net.cpp:100] Creating Layer conv7_2_relu
I0329 19:07:50.976130  2693 net.cpp:434] conv7_2_relu <- conv7_2
I0329 19:07:50.976136  2693 net.cpp:395] conv7_2_relu -> conv7_2 (in-place)
I0329 19:07:50.976474  2693 net.cpp:150] Setting up conv7_2_relu
I0329 19:07:50.976490  2693 net.cpp:157] Top shape: 8 256 8 8 (131072)
I0329 19:07:50.976495  2693 net.cpp:165] Memory required for data: 5485101088
I0329 19:07:50.976501  2693 layer_factory.hpp:77] Creating layer conv7_2_conv7_2_relu_0_split
I0329 19:07:50.976512  2693 net.cpp:100] Creating Layer conv7_2_conv7_2_relu_0_split
I0329 19:07:50.976517  2693 net.cpp:434] conv7_2_conv7_2_relu_0_split <- conv7_2
I0329 19:07:50.976526  2693 net.cpp:408] conv7_2_conv7_2_relu_0_split -> conv7_2_conv7_2_relu_0_split_0
I0329 19:07:50.976536  2693 net.cpp:408] conv7_2_conv7_2_relu_0_split -> conv7_2_conv7_2_relu_0_split_1
I0329 19:07:50.976543  2693 net.cpp:408] conv7_2_conv7_2_relu_0_split -> conv7_2_conv7_2_relu_0_split_2
I0329 19:07:50.976553  2693 net.cpp:408] conv7_2_conv7_2_relu_0_split -> conv7_2_conv7_2_relu_0_split_3
I0329 19:07:50.976577  2693 net.cpp:408] conv7_2_conv7_2_relu_0_split -> conv7_2_conv7_2_relu_0_split_4
I0329 19:07:50.976675  2693 net.cpp:150] Setting up conv7_2_conv7_2_relu_0_split
I0329 19:07:50.976686  2693 net.cpp:157] Top shape: 8 256 8 8 (131072)
I0329 19:07:50.976691  2693 net.cpp:157] Top shape: 8 256 8 8 (131072)
I0329 19:07:50.976696  2693 net.cpp:157] Top shape: 8 256 8 8 (131072)
I0329 19:07:50.976701  2693 net.cpp:157] Top shape: 8 256 8 8 (131072)
I0329 19:07:50.976707  2693 net.cpp:157] Top shape: 8 256 8 8 (131072)
I0329 19:07:50.976711  2693 net.cpp:165] Memory required for data: 5487722528
I0329 19:07:50.976716  2693 layer_factory.hpp:77] Creating layer conv8_1
I0329 19:07:50.976730  2693 net.cpp:100] Creating Layer conv8_1
I0329 19:07:50.976738  2693 net.cpp:434] conv8_1 <- conv7_2_conv7_2_relu_0_split_0
I0329 19:07:50.976749  2693 net.cpp:408] conv8_1 -> conv8_1
I0329 19:07:50.978750  2693 net.cpp:150] Setting up conv8_1
I0329 19:07:50.978770  2693 net.cpp:157] Top shape: 8 128 8 8 (65536)
I0329 19:07:50.978775  2693 net.cpp:165] Memory required for data: 5487984672
I0329 19:07:50.978783  2693 layer_factory.hpp:77] Creating layer conv8_1_relu
I0329 19:07:50.978795  2693 net.cpp:100] Creating Layer conv8_1_relu
I0329 19:07:50.978801  2693 net.cpp:434] conv8_1_relu <- conv8_1
I0329 19:07:50.978808  2693 net.cpp:395] conv8_1_relu -> conv8_1 (in-place)
I0329 19:07:50.979010  2693 net.cpp:150] Setting up conv8_1_relu
I0329 19:07:50.979024  2693 net.cpp:157] Top shape: 8 128 8 8 (65536)
I0329 19:07:50.979030  2693 net.cpp:165] Memory required for data: 5488246816
I0329 19:07:50.979035  2693 layer_factory.hpp:77] Creating layer conv8_2
I0329 19:07:50.979049  2693 net.cpp:100] Creating Layer conv8_2
I0329 19:07:50.979058  2693 net.cpp:434] conv8_2 <- conv8_1
I0329 19:07:50.979068  2693 net.cpp:408] conv8_2 -> conv8_2
I0329 19:07:50.983253  2693 net.cpp:150] Setting up conv8_2
I0329 19:07:50.983273  2693 net.cpp:157] Top shape: 8 256 6 6 (73728)
I0329 19:07:50.983279  2693 net.cpp:165] Memory required for data: 5488541728
I0329 19:07:50.983288  2693 layer_factory.hpp:77] Creating layer conv8_2_relu
I0329 19:07:50.983299  2693 net.cpp:100] Creating Layer conv8_2_relu
I0329 19:07:50.983305  2693 net.cpp:434] conv8_2_relu <- conv8_2
I0329 19:07:50.983312  2693 net.cpp:395] conv8_2_relu -> conv8_2 (in-place)
I0329 19:07:50.983664  2693 net.cpp:150] Setting up conv8_2_relu
I0329 19:07:50.983681  2693 net.cpp:157] Top shape: 8 256 6 6 (73728)
I0329 19:07:50.983686  2693 net.cpp:165] Memory required for data: 5488836640
I0329 19:07:50.983691  2693 layer_factory.hpp:77] Creating layer conv8_2_conv8_2_relu_0_split
I0329 19:07:50.983703  2693 net.cpp:100] Creating Layer conv8_2_conv8_2_relu_0_split
I0329 19:07:50.983711  2693 net.cpp:434] conv8_2_conv8_2_relu_0_split <- conv8_2
I0329 19:07:50.983717  2693 net.cpp:408] conv8_2_conv8_2_relu_0_split -> conv8_2_conv8_2_relu_0_split_0
I0329 19:07:50.983727  2693 net.cpp:408] conv8_2_conv8_2_relu_0_split -> conv8_2_conv8_2_relu_0_split_1
I0329 19:07:50.983736  2693 net.cpp:408] conv8_2_conv8_2_relu_0_split -> conv8_2_conv8_2_relu_0_split_2
I0329 19:07:50.983747  2693 net.cpp:408] conv8_2_conv8_2_relu_0_split -> conv8_2_conv8_2_relu_0_split_3
I0329 19:07:50.983757  2693 net.cpp:408] conv8_2_conv8_2_relu_0_split -> conv8_2_conv8_2_relu_0_split_4
I0329 19:07:50.983855  2693 net.cpp:150] Setting up conv8_2_conv8_2_relu_0_split
I0329 19:07:50.983865  2693 net.cpp:157] Top shape: 8 256 6 6 (73728)
I0329 19:07:50.983872  2693 net.cpp:157] Top shape: 8 256 6 6 (73728)
I0329 19:07:50.983877  2693 net.cpp:157] Top shape: 8 256 6 6 (73728)
I0329 19:07:50.983882  2693 net.cpp:157] Top shape: 8 256 6 6 (73728)
I0329 19:07:50.983888  2693 net.cpp:157] Top shape: 8 256 6 6 (73728)
I0329 19:07:50.983892  2693 net.cpp:165] Memory required for data: 5490311200
I0329 19:07:50.983897  2693 layer_factory.hpp:77] Creating layer conv9_1
I0329 19:07:50.983911  2693 net.cpp:100] Creating Layer conv9_1
I0329 19:07:50.983917  2693 net.cpp:434] conv9_1 <- conv8_2_conv8_2_relu_0_split_0
I0329 19:07:50.983942  2693 net.cpp:408] conv9_1 -> conv9_1
I0329 19:07:50.985247  2693 net.cpp:150] Setting up conv9_1
I0329 19:07:50.985266  2693 net.cpp:157] Top shape: 8 128 6 6 (36864)
I0329 19:07:50.985271  2693 net.cpp:165] Memory required for data: 5490458656
I0329 19:07:50.985280  2693 layer_factory.hpp:77] Creating layer conv9_1_relu
I0329 19:07:50.985291  2693 net.cpp:100] Creating Layer conv9_1_relu
I0329 19:07:50.985298  2693 net.cpp:434] conv9_1_relu <- conv9_1
I0329 19:07:50.985306  2693 net.cpp:395] conv9_1_relu -> conv9_1 (in-place)
I0329 19:07:50.985641  2693 net.cpp:150] Setting up conv9_1_relu
I0329 19:07:50.985659  2693 net.cpp:157] Top shape: 8 128 6 6 (36864)
I0329 19:07:50.985664  2693 net.cpp:165] Memory required for data: 5490606112
I0329 19:07:50.985669  2693 layer_factory.hpp:77] Creating layer conv9_2
I0329 19:07:50.985687  2693 net.cpp:100] Creating Layer conv9_2
I0329 19:07:50.985693  2693 net.cpp:434] conv9_2 <- conv9_1
I0329 19:07:50.985702  2693 net.cpp:408] conv9_2 -> conv9_2
I0329 19:07:50.989855  2693 net.cpp:150] Setting up conv9_2
I0329 19:07:50.989874  2693 net.cpp:157] Top shape: 8 256 4 4 (32768)
I0329 19:07:50.989881  2693 net.cpp:165] Memory required for data: 5490737184
I0329 19:07:50.989888  2693 layer_factory.hpp:77] Creating layer conv9_2_relu
I0329 19:07:50.989899  2693 net.cpp:100] Creating Layer conv9_2_relu
I0329 19:07:50.989907  2693 net.cpp:434] conv9_2_relu <- conv9_2
I0329 19:07:50.989913  2693 net.cpp:395] conv9_2_relu -> conv9_2 (in-place)
I0329 19:07:50.990248  2693 net.cpp:150] Setting up conv9_2_relu
I0329 19:07:50.990265  2693 net.cpp:157] Top shape: 8 256 4 4 (32768)
I0329 19:07:50.990272  2693 net.cpp:165] Memory required for data: 5490868256
I0329 19:07:50.990276  2693 layer_factory.hpp:77] Creating layer conv9_2_conv9_2_relu_0_split
I0329 19:07:50.990288  2693 net.cpp:100] Creating Layer conv9_2_conv9_2_relu_0_split
I0329 19:07:50.990294  2693 net.cpp:434] conv9_2_conv9_2_relu_0_split <- conv9_2
I0329 19:07:50.990304  2693 net.cpp:408] conv9_2_conv9_2_relu_0_split -> conv9_2_conv9_2_relu_0_split_0
I0329 19:07:50.990314  2693 net.cpp:408] conv9_2_conv9_2_relu_0_split -> conv9_2_conv9_2_relu_0_split_1
I0329 19:07:50.990324  2693 net.cpp:408] conv9_2_conv9_2_relu_0_split -> conv9_2_conv9_2_relu_0_split_2
I0329 19:07:50.990332  2693 net.cpp:408] conv9_2_conv9_2_relu_0_split -> conv9_2_conv9_2_relu_0_split_3
I0329 19:07:50.990420  2693 net.cpp:150] Setting up conv9_2_conv9_2_relu_0_split
I0329 19:07:50.990430  2693 net.cpp:157] Top shape: 8 256 4 4 (32768)
I0329 19:07:50.990437  2693 net.cpp:157] Top shape: 8 256 4 4 (32768)
I0329 19:07:50.990442  2693 net.cpp:157] Top shape: 8 256 4 4 (32768)
I0329 19:07:50.990447  2693 net.cpp:157] Top shape: 8 256 4 4 (32768)
I0329 19:07:50.990452  2693 net.cpp:165] Memory required for data: 5491392544
I0329 19:07:50.990456  2693 layer_factory.hpp:77] Creating layer conv4_3_norm
I0329 19:07:50.990468  2693 net.cpp:100] Creating Layer conv4_3_norm
I0329 19:07:50.990476  2693 net.cpp:434] conv4_3_norm <- conv4_3_relu4_3_0_split_1
I0329 19:07:50.990483  2693 net.cpp:408] conv4_3_norm -> conv4_3_norm
I0329 19:07:50.990670  2693 net.cpp:150] Setting up conv4_3_norm
I0329 19:07:50.990681  2693 net.cpp:157] Top shape: 8 512 64 64 (16777216)
I0329 19:07:50.990686  2693 net.cpp:165] Memory required for data: 5558501408
I0329 19:07:50.990694  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_conv4_3_norm_0_split
I0329 19:07:50.990702  2693 net.cpp:100] Creating Layer conv4_3_norm_conv4_3_norm_0_split
I0329 19:07:50.990710  2693 net.cpp:434] conv4_3_norm_conv4_3_norm_0_split <- conv4_3_norm
I0329 19:07:50.990717  2693 net.cpp:408] conv4_3_norm_conv4_3_norm_0_split -> conv4_3_norm_conv4_3_norm_0_split_0
I0329 19:07:50.990734  2693 net.cpp:408] conv4_3_norm_conv4_3_norm_0_split -> conv4_3_norm_conv4_3_norm_0_split_1
I0329 19:07:50.990746  2693 net.cpp:408] conv4_3_norm_conv4_3_norm_0_split -> conv4_3_norm_conv4_3_norm_0_split_2
I0329 19:07:50.990753  2693 net.cpp:408] conv4_3_norm_conv4_3_norm_0_split -> conv4_3_norm_conv4_3_norm_0_split_3
I0329 19:07:50.990841  2693 net.cpp:150] Setting up conv4_3_norm_conv4_3_norm_0_split
I0329 19:07:50.990855  2693 net.cpp:157] Top shape: 8 512 64 64 (16777216)
I0329 19:07:50.990861  2693 net.cpp:157] Top shape: 8 512 64 64 (16777216)
I0329 19:07:50.990866  2693 net.cpp:157] Top shape: 8 512 64 64 (16777216)
I0329 19:07:50.990872  2693 net.cpp:157] Top shape: 8 512 64 64 (16777216)
I0329 19:07:50.990876  2693 net.cpp:165] Memory required for data: 5826936864
I0329 19:07:50.990882  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_loc
I0329 19:07:50.990895  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_loc
I0329 19:07:50.990903  2693 net.cpp:434] conv4_3_norm_mbox_loc <- conv4_3_norm_conv4_3_norm_0_split_0
I0329 19:07:50.990913  2693 net.cpp:408] conv4_3_norm_mbox_loc -> conv4_3_norm_mbox_loc
I0329 19:07:50.993407  2693 net.cpp:150] Setting up conv4_3_norm_mbox_loc
I0329 19:07:50.993427  2693 net.cpp:157] Top shape: 8 16 64 64 (524288)
I0329 19:07:50.993432  2693 net.cpp:165] Memory required for data: 5829034016
I0329 19:07:50.993441  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_loc_perm
I0329 19:07:50.993460  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_loc_perm
I0329 19:07:50.993469  2693 net.cpp:434] conv4_3_norm_mbox_loc_perm <- conv4_3_norm_mbox_loc
I0329 19:07:50.993476  2693 net.cpp:408] conv4_3_norm_mbox_loc_perm -> conv4_3_norm_mbox_loc_perm
I0329 19:07:50.993619  2693 net.cpp:150] Setting up conv4_3_norm_mbox_loc_perm
I0329 19:07:50.993630  2693 net.cpp:157] Top shape: 8 64 64 16 (524288)
I0329 19:07:50.993635  2693 net.cpp:165] Memory required for data: 5831131168
I0329 19:07:50.993640  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_loc_flat
I0329 19:07:50.993654  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_loc_flat
I0329 19:07:50.993661  2693 net.cpp:434] conv4_3_norm_mbox_loc_flat <- conv4_3_norm_mbox_loc_perm
I0329 19:07:50.993669  2693 net.cpp:408] conv4_3_norm_mbox_loc_flat -> conv4_3_norm_mbox_loc_flat
I0329 19:07:50.993708  2693 net.cpp:150] Setting up conv4_3_norm_mbox_loc_flat
I0329 19:07:50.993718  2693 net.cpp:157] Top shape: 8 65536 (524288)
I0329 19:07:50.993723  2693 net.cpp:165] Memory required for data: 5833228320
I0329 19:07:50.993728  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_conf
I0329 19:07:50.993754  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_conf
I0329 19:07:50.993762  2693 net.cpp:434] conv4_3_norm_mbox_conf <- conv4_3_norm_conv4_3_norm_0_split_1
I0329 19:07:50.993772  2693 net.cpp:408] conv4_3_norm_mbox_conf -> conv4_3_norm_mbox_conf
I0329 19:07:50.995383  2693 net.cpp:150] Setting up conv4_3_norm_mbox_conf
I0329 19:07:50.995401  2693 net.cpp:157] Top shape: 8 8 64 64 (262144)
I0329 19:07:50.995407  2693 net.cpp:165] Memory required for data: 5834276896
I0329 19:07:50.995417  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_conf_perm
I0329 19:07:50.995429  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_conf_perm
I0329 19:07:50.995435  2693 net.cpp:434] conv4_3_norm_mbox_conf_perm <- conv4_3_norm_mbox_conf
I0329 19:07:50.995442  2693 net.cpp:408] conv4_3_norm_mbox_conf_perm -> conv4_3_norm_mbox_conf_perm
I0329 19:07:50.995584  2693 net.cpp:150] Setting up conv4_3_norm_mbox_conf_perm
I0329 19:07:50.995597  2693 net.cpp:157] Top shape: 8 64 64 8 (262144)
I0329 19:07:50.995602  2693 net.cpp:165] Memory required for data: 5835325472
I0329 19:07:50.995607  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_conf_flat
I0329 19:07:50.995615  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_conf_flat
I0329 19:07:50.995621  2693 net.cpp:434] conv4_3_norm_mbox_conf_flat <- conv4_3_norm_mbox_conf_perm
I0329 19:07:50.995631  2693 net.cpp:408] conv4_3_norm_mbox_conf_flat -> conv4_3_norm_mbox_conf_flat
I0329 19:07:50.995667  2693 net.cpp:150] Setting up conv4_3_norm_mbox_conf_flat
I0329 19:07:50.995677  2693 net.cpp:157] Top shape: 8 32768 (262144)
I0329 19:07:50.995682  2693 net.cpp:165] Memory required for data: 5836374048
I0329 19:07:50.995687  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_clean
I0329 19:07:50.995712  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_clean
I0329 19:07:50.995721  2693 net.cpp:434] conv4_3_norm_mbox_clean <- conv4_3_norm_conv4_3_norm_0_split_2
I0329 19:07:50.995733  2693 net.cpp:408] conv4_3_norm_mbox_clean -> conv4_3_norm_mbox_clean
I0329 19:07:50.997202  2693 net.cpp:150] Setting up conv4_3_norm_mbox_clean
I0329 19:07:50.997221  2693 net.cpp:157] Top shape: 8 4 64 64 (131072)
I0329 19:07:50.997226  2693 net.cpp:165] Memory required for data: 5836898336
I0329 19:07:50.997236  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_clean_perm
I0329 19:07:50.997249  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_clean_perm
I0329 19:07:50.997256  2693 net.cpp:434] conv4_3_norm_mbox_clean_perm <- conv4_3_norm_mbox_clean
I0329 19:07:50.997264  2693 net.cpp:408] conv4_3_norm_mbox_clean_perm -> conv4_3_norm_mbox_clean_perm
I0329 19:07:50.997402  2693 net.cpp:150] Setting up conv4_3_norm_mbox_clean_perm
I0329 19:07:50.997414  2693 net.cpp:157] Top shape: 8 64 64 4 (131072)
I0329 19:07:50.997419  2693 net.cpp:165] Memory required for data: 5837422624
I0329 19:07:50.997424  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_clean_flat
I0329 19:07:50.997434  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_clean_flat
I0329 19:07:50.997440  2693 net.cpp:434] conv4_3_norm_mbox_clean_flat <- conv4_3_norm_mbox_clean_perm
I0329 19:07:50.997448  2693 net.cpp:408] conv4_3_norm_mbox_clean_flat -> conv4_3_norm_mbox_clean_flat
I0329 19:07:50.997484  2693 net.cpp:150] Setting up conv4_3_norm_mbox_clean_flat
I0329 19:07:50.997494  2693 net.cpp:157] Top shape: 8 16384 (131072)
I0329 19:07:50.997499  2693 net.cpp:165] Memory required for data: 5837946912
I0329 19:07:50.997504  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_priorbox
I0329 19:07:50.997514  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_priorbox
I0329 19:07:50.997520  2693 net.cpp:434] conv4_3_norm_mbox_priorbox <- conv4_3_norm_conv4_3_norm_0_split_3
I0329 19:07:50.997527  2693 net.cpp:434] conv4_3_norm_mbox_priorbox <- data_data_0_split_1
I0329 19:07:50.997537  2693 net.cpp:408] conv4_3_norm_mbox_priorbox -> conv4_3_norm_mbox_priorbox
I0329 19:07:50.997575  2693 net.cpp:150] Setting up conv4_3_norm_mbox_priorbox
I0329 19:07:50.997586  2693 net.cpp:157] Top shape: 1 2 65536 (131072)
I0329 19:07:50.997591  2693 net.cpp:165] Memory required for data: 5838471200
I0329 19:07:50.997596  2693 layer_factory.hpp:77] Creating layer fc7_mbox_loc
I0329 19:07:50.997612  2693 net.cpp:100] Creating Layer fc7_mbox_loc
I0329 19:07:50.997620  2693 net.cpp:434] fc7_mbox_loc <- fc7_relu7_0_split_1
I0329 19:07:50.997629  2693 net.cpp:408] fc7_mbox_loc -> fc7_mbox_loc
I0329 19:07:51.001173  2693 net.cpp:150] Setting up fc7_mbox_loc
I0329 19:07:51.001194  2693 net.cpp:157] Top shape: 8 24 32 32 (196608)
I0329 19:07:51.001199  2693 net.cpp:165] Memory required for data: 5839257632
I0329 19:07:51.001209  2693 layer_factory.hpp:77] Creating layer fc7_mbox_loc_perm
I0329 19:07:51.001217  2693 net.cpp:100] Creating Layer fc7_mbox_loc_perm
I0329 19:07:51.001224  2693 net.cpp:434] fc7_mbox_loc_perm <- fc7_mbox_loc
I0329 19:07:51.001233  2693 net.cpp:408] fc7_mbox_loc_perm -> fc7_mbox_loc_perm
I0329 19:07:51.001374  2693 net.cpp:150] Setting up fc7_mbox_loc_perm
I0329 19:07:51.001385  2693 net.cpp:157] Top shape: 8 32 32 24 (196608)
I0329 19:07:51.001390  2693 net.cpp:165] Memory required for data: 5840044064
I0329 19:07:51.001396  2693 layer_factory.hpp:77] Creating layer fc7_mbox_loc_flat
I0329 19:07:51.001405  2693 net.cpp:100] Creating Layer fc7_mbox_loc_flat
I0329 19:07:51.001410  2693 net.cpp:434] fc7_mbox_loc_flat <- fc7_mbox_loc_perm
I0329 19:07:51.001420  2693 net.cpp:408] fc7_mbox_loc_flat -> fc7_mbox_loc_flat
I0329 19:07:51.001453  2693 net.cpp:150] Setting up fc7_mbox_loc_flat
I0329 19:07:51.001463  2693 net.cpp:157] Top shape: 8 24576 (196608)
I0329 19:07:51.001468  2693 net.cpp:165] Memory required for data: 5840830496
I0329 19:07:51.001473  2693 layer_factory.hpp:77] Creating layer fc7_mbox_conf
I0329 19:07:51.001503  2693 net.cpp:100] Creating Layer fc7_mbox_conf
I0329 19:07:51.001513  2693 net.cpp:434] fc7_mbox_conf <- fc7_relu7_0_split_2
I0329 19:07:51.001520  2693 net.cpp:408] fc7_mbox_conf -> fc7_mbox_conf
I0329 19:07:51.003716  2693 net.cpp:150] Setting up fc7_mbox_conf
I0329 19:07:51.003737  2693 net.cpp:157] Top shape: 8 12 32 32 (98304)
I0329 19:07:51.003743  2693 net.cpp:165] Memory required for data: 5841223712
I0329 19:07:51.003753  2693 layer_factory.hpp:77] Creating layer fc7_mbox_conf_perm
I0329 19:07:51.003762  2693 net.cpp:100] Creating Layer fc7_mbox_conf_perm
I0329 19:07:51.003768  2693 net.cpp:434] fc7_mbox_conf_perm <- fc7_mbox_conf
I0329 19:07:51.003777  2693 net.cpp:408] fc7_mbox_conf_perm -> fc7_mbox_conf_perm
I0329 19:07:51.003917  2693 net.cpp:150] Setting up fc7_mbox_conf_perm
I0329 19:07:51.003929  2693 net.cpp:157] Top shape: 8 32 32 12 (98304)
I0329 19:07:51.003934  2693 net.cpp:165] Memory required for data: 5841616928
I0329 19:07:51.003939  2693 layer_factory.hpp:77] Creating layer fc7_mbox_conf_flat
I0329 19:07:51.003949  2693 net.cpp:100] Creating Layer fc7_mbox_conf_flat
I0329 19:07:51.003957  2693 net.cpp:434] fc7_mbox_conf_flat <- fc7_mbox_conf_perm
I0329 19:07:51.003964  2693 net.cpp:408] fc7_mbox_conf_flat -> fc7_mbox_conf_flat
I0329 19:07:51.003998  2693 net.cpp:150] Setting up fc7_mbox_conf_flat
I0329 19:07:51.004007  2693 net.cpp:157] Top shape: 8 12288 (98304)
I0329 19:07:51.004012  2693 net.cpp:165] Memory required for data: 5842010144
I0329 19:07:51.004017  2693 layer_factory.hpp:77] Creating layer fc7_mbox_clean
I0329 19:07:51.004031  2693 net.cpp:100] Creating Layer fc7_mbox_clean
I0329 19:07:51.004040  2693 net.cpp:434] fc7_mbox_clean <- fc7_relu7_0_split_3
I0329 19:07:51.004050  2693 net.cpp:408] fc7_mbox_clean -> fc7_mbox_clean
I0329 19:07:51.005823  2693 net.cpp:150] Setting up fc7_mbox_clean
I0329 19:07:51.005842  2693 net.cpp:157] Top shape: 8 6 32 32 (49152)
I0329 19:07:51.005847  2693 net.cpp:165] Memory required for data: 5842206752
I0329 19:07:51.005859  2693 layer_factory.hpp:77] Creating layer fc7_mbox_clean_perm
I0329 19:07:51.005869  2693 net.cpp:100] Creating Layer fc7_mbox_clean_perm
I0329 19:07:51.005875  2693 net.cpp:434] fc7_mbox_clean_perm <- fc7_mbox_clean
I0329 19:07:51.005885  2693 net.cpp:408] fc7_mbox_clean_perm -> fc7_mbox_clean_perm
I0329 19:07:51.006022  2693 net.cpp:150] Setting up fc7_mbox_clean_perm
I0329 19:07:51.006036  2693 net.cpp:157] Top shape: 8 32 32 6 (49152)
I0329 19:07:51.006041  2693 net.cpp:165] Memory required for data: 5842403360
I0329 19:07:51.006045  2693 layer_factory.hpp:77] Creating layer fc7_mbox_clean_flat
I0329 19:07:51.006053  2693 net.cpp:100] Creating Layer fc7_mbox_clean_flat
I0329 19:07:51.006059  2693 net.cpp:434] fc7_mbox_clean_flat <- fc7_mbox_clean_perm
I0329 19:07:51.006067  2693 net.cpp:408] fc7_mbox_clean_flat -> fc7_mbox_clean_flat
I0329 19:07:51.006098  2693 net.cpp:150] Setting up fc7_mbox_clean_flat
I0329 19:07:51.006108  2693 net.cpp:157] Top shape: 8 6144 (49152)
I0329 19:07:51.006114  2693 net.cpp:165] Memory required for data: 5842599968
I0329 19:07:51.006119  2693 layer_factory.hpp:77] Creating layer fc7_mbox_priorbox
I0329 19:07:51.006129  2693 net.cpp:100] Creating Layer fc7_mbox_priorbox
I0329 19:07:51.006135  2693 net.cpp:434] fc7_mbox_priorbox <- fc7_relu7_0_split_4
I0329 19:07:51.006141  2693 net.cpp:434] fc7_mbox_priorbox <- data_data_0_split_2
I0329 19:07:51.006151  2693 net.cpp:408] fc7_mbox_priorbox -> fc7_mbox_priorbox
I0329 19:07:51.006187  2693 net.cpp:150] Setting up fc7_mbox_priorbox
I0329 19:07:51.006197  2693 net.cpp:157] Top shape: 1 2 24576 (49152)
I0329 19:07:51.006202  2693 net.cpp:165] Memory required for data: 5842796576
I0329 19:07:51.006207  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_loc
I0329 19:07:51.006222  2693 net.cpp:100] Creating Layer conv6_2_mbox_loc
I0329 19:07:51.006229  2693 net.cpp:434] conv6_2_mbox_loc <- conv6_2_conv6_2_relu_0_split_1
I0329 19:07:51.006239  2693 net.cpp:408] conv6_2_mbox_loc -> conv6_2_mbox_loc
I0329 19:07:51.009050  2693 net.cpp:150] Setting up conv6_2_mbox_loc
I0329 19:07:51.009081  2693 net.cpp:157] Top shape: 8 24 16 16 (49152)
I0329 19:07:51.009088  2693 net.cpp:165] Memory required for data: 5842993184
I0329 19:07:51.009099  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_loc_perm
I0329 19:07:51.009109  2693 net.cpp:100] Creating Layer conv6_2_mbox_loc_perm
I0329 19:07:51.009114  2693 net.cpp:434] conv6_2_mbox_loc_perm <- conv6_2_mbox_loc
I0329 19:07:51.009122  2693 net.cpp:408] conv6_2_mbox_loc_perm -> conv6_2_mbox_loc_perm
I0329 19:07:51.009260  2693 net.cpp:150] Setting up conv6_2_mbox_loc_perm
I0329 19:07:51.009272  2693 net.cpp:157] Top shape: 8 16 16 24 (49152)
I0329 19:07:51.009277  2693 net.cpp:165] Memory required for data: 5843189792
I0329 19:07:51.009284  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_loc_flat
I0329 19:07:51.009291  2693 net.cpp:100] Creating Layer conv6_2_mbox_loc_flat
I0329 19:07:51.009299  2693 net.cpp:434] conv6_2_mbox_loc_flat <- conv6_2_mbox_loc_perm
I0329 19:07:51.009307  2693 net.cpp:408] conv6_2_mbox_loc_flat -> conv6_2_mbox_loc_flat
I0329 19:07:51.009341  2693 net.cpp:150] Setting up conv6_2_mbox_loc_flat
I0329 19:07:51.009353  2693 net.cpp:157] Top shape: 8 6144 (49152)
I0329 19:07:51.009358  2693 net.cpp:165] Memory required for data: 5843386400
I0329 19:07:51.009363  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_conf
I0329 19:07:51.009377  2693 net.cpp:100] Creating Layer conv6_2_mbox_conf
I0329 19:07:51.009387  2693 net.cpp:434] conv6_2_mbox_conf <- conv6_2_conv6_2_relu_0_split_2
I0329 19:07:51.009394  2693 net.cpp:408] conv6_2_mbox_conf -> conv6_2_mbox_conf
I0329 19:07:51.011435  2693 net.cpp:150] Setting up conv6_2_mbox_conf
I0329 19:07:51.011453  2693 net.cpp:157] Top shape: 8 12 16 16 (24576)
I0329 19:07:51.011459  2693 net.cpp:165] Memory required for data: 5843484704
I0329 19:07:51.011468  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_conf_perm
I0329 19:07:51.011482  2693 net.cpp:100] Creating Layer conv6_2_mbox_conf_perm
I0329 19:07:51.011492  2693 net.cpp:434] conv6_2_mbox_conf_perm <- conv6_2_mbox_conf
I0329 19:07:51.011502  2693 net.cpp:408] conv6_2_mbox_conf_perm -> conv6_2_mbox_conf_perm
I0329 19:07:51.011649  2693 net.cpp:150] Setting up conv6_2_mbox_conf_perm
I0329 19:07:51.011662  2693 net.cpp:157] Top shape: 8 16 16 12 (24576)
I0329 19:07:51.011667  2693 net.cpp:165] Memory required for data: 5843583008
I0329 19:07:51.011672  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_conf_flat
I0329 19:07:51.011679  2693 net.cpp:100] Creating Layer conv6_2_mbox_conf_flat
I0329 19:07:51.011688  2693 net.cpp:434] conv6_2_mbox_conf_flat <- conv6_2_mbox_conf_perm
I0329 19:07:51.011698  2693 net.cpp:408] conv6_2_mbox_conf_flat -> conv6_2_mbox_conf_flat
I0329 19:07:51.011735  2693 net.cpp:150] Setting up conv6_2_mbox_conf_flat
I0329 19:07:51.011745  2693 net.cpp:157] Top shape: 8 3072 (24576)
I0329 19:07:51.011750  2693 net.cpp:165] Memory required for data: 5843681312
I0329 19:07:51.011754  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_clean
I0329 19:07:51.011770  2693 net.cpp:100] Creating Layer conv6_2_mbox_clean
I0329 19:07:51.011778  2693 net.cpp:434] conv6_2_mbox_clean <- conv6_2_conv6_2_relu_0_split_3
I0329 19:07:51.011786  2693 net.cpp:408] conv6_2_mbox_clean -> conv6_2_mbox_clean
I0329 19:07:51.013274  2693 net.cpp:150] Setting up conv6_2_mbox_clean
I0329 19:07:51.013293  2693 net.cpp:157] Top shape: 8 6 16 16 (12288)
I0329 19:07:51.013299  2693 net.cpp:165] Memory required for data: 5843730464
I0329 19:07:51.013324  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_clean_perm
I0329 19:07:51.013336  2693 net.cpp:100] Creating Layer conv6_2_mbox_clean_perm
I0329 19:07:51.013344  2693 net.cpp:434] conv6_2_mbox_clean_perm <- conv6_2_mbox_clean
I0329 19:07:51.013351  2693 net.cpp:408] conv6_2_mbox_clean_perm -> conv6_2_mbox_clean_perm
I0329 19:07:51.013489  2693 net.cpp:150] Setting up conv6_2_mbox_clean_perm
I0329 19:07:51.013499  2693 net.cpp:157] Top shape: 8 16 16 6 (12288)
I0329 19:07:51.013504  2693 net.cpp:165] Memory required for data: 5843779616
I0329 19:07:51.013525  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_clean_flat
I0329 19:07:51.013533  2693 net.cpp:100] Creating Layer conv6_2_mbox_clean_flat
I0329 19:07:51.013540  2693 net.cpp:434] conv6_2_mbox_clean_flat <- conv6_2_mbox_clean_perm
I0329 19:07:51.013546  2693 net.cpp:408] conv6_2_mbox_clean_flat -> conv6_2_mbox_clean_flat
I0329 19:07:51.013584  2693 net.cpp:150] Setting up conv6_2_mbox_clean_flat
I0329 19:07:51.013595  2693 net.cpp:157] Top shape: 8 1536 (12288)
I0329 19:07:51.013600  2693 net.cpp:165] Memory required for data: 5843828768
I0329 19:07:51.013605  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_priorbox
I0329 19:07:51.013615  2693 net.cpp:100] Creating Layer conv6_2_mbox_priorbox
I0329 19:07:51.013622  2693 net.cpp:434] conv6_2_mbox_priorbox <- conv6_2_conv6_2_relu_0_split_4
I0329 19:07:51.013629  2693 net.cpp:434] conv6_2_mbox_priorbox <- data_data_0_split_3
I0329 19:07:51.013638  2693 net.cpp:408] conv6_2_mbox_priorbox -> conv6_2_mbox_priorbox
I0329 19:07:51.013675  2693 net.cpp:150] Setting up conv6_2_mbox_priorbox
I0329 19:07:51.013685  2693 net.cpp:157] Top shape: 1 2 6144 (12288)
I0329 19:07:51.013690  2693 net.cpp:165] Memory required for data: 5843877920
I0329 19:07:51.013695  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_loc
I0329 19:07:51.013710  2693 net.cpp:100] Creating Layer conv7_2_mbox_loc
I0329 19:07:51.013717  2693 net.cpp:434] conv7_2_mbox_loc <- conv7_2_conv7_2_relu_0_split_1
I0329 19:07:51.013727  2693 net.cpp:408] conv7_2_mbox_loc -> conv7_2_mbox_loc
I0329 19:07:51.016145  2693 net.cpp:150] Setting up conv7_2_mbox_loc
I0329 19:07:51.016167  2693 net.cpp:157] Top shape: 8 24 8 8 (12288)
I0329 19:07:51.016175  2693 net.cpp:165] Memory required for data: 5843927072
I0329 19:07:51.016183  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_loc_perm
I0329 19:07:51.016194  2693 net.cpp:100] Creating Layer conv7_2_mbox_loc_perm
I0329 19:07:51.016201  2693 net.cpp:434] conv7_2_mbox_loc_perm <- conv7_2_mbox_loc
I0329 19:07:51.016209  2693 net.cpp:408] conv7_2_mbox_loc_perm -> conv7_2_mbox_loc_perm
I0329 19:07:51.016355  2693 net.cpp:150] Setting up conv7_2_mbox_loc_perm
I0329 19:07:51.016366  2693 net.cpp:157] Top shape: 8 8 8 24 (12288)
I0329 19:07:51.016371  2693 net.cpp:165] Memory required for data: 5843976224
I0329 19:07:51.016376  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_loc_flat
I0329 19:07:51.016386  2693 net.cpp:100] Creating Layer conv7_2_mbox_loc_flat
I0329 19:07:51.016391  2693 net.cpp:434] conv7_2_mbox_loc_flat <- conv7_2_mbox_loc_perm
I0329 19:07:51.016398  2693 net.cpp:408] conv7_2_mbox_loc_flat -> conv7_2_mbox_loc_flat
I0329 19:07:51.016435  2693 net.cpp:150] Setting up conv7_2_mbox_loc_flat
I0329 19:07:51.016445  2693 net.cpp:157] Top shape: 8 1536 (12288)
I0329 19:07:51.016450  2693 net.cpp:165] Memory required for data: 5844025376
I0329 19:07:51.016455  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_conf
I0329 19:07:51.016469  2693 net.cpp:100] Creating Layer conv7_2_mbox_conf
I0329 19:07:51.016477  2693 net.cpp:434] conv7_2_mbox_conf <- conv7_2_conv7_2_relu_0_split_2
I0329 19:07:51.016489  2693 net.cpp:408] conv7_2_mbox_conf -> conv7_2_mbox_conf
I0329 19:07:51.018182  2693 net.cpp:150] Setting up conv7_2_mbox_conf
I0329 19:07:51.018201  2693 net.cpp:157] Top shape: 8 12 8 8 (6144)
I0329 19:07:51.018209  2693 net.cpp:165] Memory required for data: 5844049952
I0329 19:07:51.018218  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_conf_perm
I0329 19:07:51.018229  2693 net.cpp:100] Creating Layer conv7_2_mbox_conf_perm
I0329 19:07:51.018234  2693 net.cpp:434] conv7_2_mbox_conf_perm <- conv7_2_mbox_conf
I0329 19:07:51.018242  2693 net.cpp:408] conv7_2_mbox_conf_perm -> conv7_2_mbox_conf_perm
I0329 19:07:51.018386  2693 net.cpp:150] Setting up conv7_2_mbox_conf_perm
I0329 19:07:51.018399  2693 net.cpp:157] Top shape: 8 8 8 12 (6144)
I0329 19:07:51.018404  2693 net.cpp:165] Memory required for data: 5844074528
I0329 19:07:51.018409  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_conf_flat
I0329 19:07:51.018430  2693 net.cpp:100] Creating Layer conv7_2_mbox_conf_flat
I0329 19:07:51.018440  2693 net.cpp:434] conv7_2_mbox_conf_flat <- conv7_2_mbox_conf_perm
I0329 19:07:51.018447  2693 net.cpp:408] conv7_2_mbox_conf_flat -> conv7_2_mbox_conf_flat
I0329 19:07:51.018486  2693 net.cpp:150] Setting up conv7_2_mbox_conf_flat
I0329 19:07:51.018496  2693 net.cpp:157] Top shape: 8 768 (6144)
I0329 19:07:51.018501  2693 net.cpp:165] Memory required for data: 5844099104
I0329 19:07:51.018506  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_clean
I0329 19:07:51.018519  2693 net.cpp:100] Creating Layer conv7_2_mbox_clean
I0329 19:07:51.018527  2693 net.cpp:434] conv7_2_mbox_clean <- conv7_2_conv7_2_relu_0_split_3
I0329 19:07:51.018538  2693 net.cpp:408] conv7_2_mbox_clean -> conv7_2_mbox_clean
I0329 19:07:51.020126  2693 net.cpp:150] Setting up conv7_2_mbox_clean
I0329 19:07:51.020145  2693 net.cpp:157] Top shape: 8 6 8 8 (3072)
I0329 19:07:51.020151  2693 net.cpp:165] Memory required for data: 5844111392
I0329 19:07:51.020162  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_clean_perm
I0329 19:07:51.020172  2693 net.cpp:100] Creating Layer conv7_2_mbox_clean_perm
I0329 19:07:51.020179  2693 net.cpp:434] conv7_2_mbox_clean_perm <- conv7_2_mbox_clean
I0329 19:07:51.020186  2693 net.cpp:408] conv7_2_mbox_clean_perm -> conv7_2_mbox_clean_perm
I0329 19:07:51.020339  2693 net.cpp:150] Setting up conv7_2_mbox_clean_perm
I0329 19:07:51.020350  2693 net.cpp:157] Top shape: 8 8 8 6 (3072)
I0329 19:07:51.020355  2693 net.cpp:165] Memory required for data: 5844123680
I0329 19:07:51.020361  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_clean_flat
I0329 19:07:51.020370  2693 net.cpp:100] Creating Layer conv7_2_mbox_clean_flat
I0329 19:07:51.020375  2693 net.cpp:434] conv7_2_mbox_clean_flat <- conv7_2_mbox_clean_perm
I0329 19:07:51.020385  2693 net.cpp:408] conv7_2_mbox_clean_flat -> conv7_2_mbox_clean_flat
I0329 19:07:51.020424  2693 net.cpp:150] Setting up conv7_2_mbox_clean_flat
I0329 19:07:51.020434  2693 net.cpp:157] Top shape: 8 384 (3072)
I0329 19:07:51.020439  2693 net.cpp:165] Memory required for data: 5844135968
I0329 19:07:51.020444  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_priorbox
I0329 19:07:51.020453  2693 net.cpp:100] Creating Layer conv7_2_mbox_priorbox
I0329 19:07:51.020459  2693 net.cpp:434] conv7_2_mbox_priorbox <- conv7_2_conv7_2_relu_0_split_4
I0329 19:07:51.020465  2693 net.cpp:434] conv7_2_mbox_priorbox <- data_data_0_split_4
I0329 19:07:51.020475  2693 net.cpp:408] conv7_2_mbox_priorbox -> conv7_2_mbox_priorbox
I0329 19:07:51.020515  2693 net.cpp:150] Setting up conv7_2_mbox_priorbox
I0329 19:07:51.020526  2693 net.cpp:157] Top shape: 1 2 1536 (3072)
I0329 19:07:51.020531  2693 net.cpp:165] Memory required for data: 5844148256
I0329 19:07:51.020536  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_loc
I0329 19:07:51.020551  2693 net.cpp:100] Creating Layer conv8_2_mbox_loc
I0329 19:07:51.020560  2693 net.cpp:434] conv8_2_mbox_loc <- conv8_2_conv8_2_relu_0_split_1
I0329 19:07:51.020570  2693 net.cpp:408] conv8_2_mbox_loc -> conv8_2_mbox_loc
I0329 19:07:51.022250  2693 net.cpp:150] Setting up conv8_2_mbox_loc
I0329 19:07:51.022269  2693 net.cpp:157] Top shape: 8 16 6 6 (4608)
I0329 19:07:51.022275  2693 net.cpp:165] Memory required for data: 5844166688
I0329 19:07:51.022284  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_loc_perm
I0329 19:07:51.022296  2693 net.cpp:100] Creating Layer conv8_2_mbox_loc_perm
I0329 19:07:51.022303  2693 net.cpp:434] conv8_2_mbox_loc_perm <- conv8_2_mbox_loc
I0329 19:07:51.022311  2693 net.cpp:408] conv8_2_mbox_loc_perm -> conv8_2_mbox_loc_perm
I0329 19:07:51.022459  2693 net.cpp:150] Setting up conv8_2_mbox_loc_perm
I0329 19:07:51.022470  2693 net.cpp:157] Top shape: 8 6 6 16 (4608)
I0329 19:07:51.022475  2693 net.cpp:165] Memory required for data: 5844185120
I0329 19:07:51.022480  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_loc_flat
I0329 19:07:51.022488  2693 net.cpp:100] Creating Layer conv8_2_mbox_loc_flat
I0329 19:07:51.022506  2693 net.cpp:434] conv8_2_mbox_loc_flat <- conv8_2_mbox_loc_perm
I0329 19:07:51.022516  2693 net.cpp:408] conv8_2_mbox_loc_flat -> conv8_2_mbox_loc_flat
I0329 19:07:51.022554  2693 net.cpp:150] Setting up conv8_2_mbox_loc_flat
I0329 19:07:51.022567  2693 net.cpp:157] Top shape: 8 576 (4608)
I0329 19:07:51.022572  2693 net.cpp:165] Memory required for data: 5844203552
I0329 19:07:51.022578  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_conf
I0329 19:07:51.022591  2693 net.cpp:100] Creating Layer conv8_2_mbox_conf
I0329 19:07:51.022599  2693 net.cpp:434] conv8_2_mbox_conf <- conv8_2_conv8_2_relu_0_split_2
I0329 19:07:51.022608  2693 net.cpp:408] conv8_2_mbox_conf -> conv8_2_mbox_conf
I0329 19:07:51.024024  2693 net.cpp:150] Setting up conv8_2_mbox_conf
I0329 19:07:51.024044  2693 net.cpp:157] Top shape: 8 8 6 6 (2304)
I0329 19:07:51.024050  2693 net.cpp:165] Memory required for data: 5844212768
I0329 19:07:51.024060  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_conf_perm
I0329 19:07:51.024070  2693 net.cpp:100] Creating Layer conv8_2_mbox_conf_perm
I0329 19:07:51.024077  2693 net.cpp:434] conv8_2_mbox_conf_perm <- conv8_2_mbox_conf
I0329 19:07:51.024088  2693 net.cpp:408] conv8_2_mbox_conf_perm -> conv8_2_mbox_conf_perm
I0329 19:07:51.024240  2693 net.cpp:150] Setting up conv8_2_mbox_conf_perm
I0329 19:07:51.024251  2693 net.cpp:157] Top shape: 8 6 6 8 (2304)
I0329 19:07:51.024256  2693 net.cpp:165] Memory required for data: 5844221984
I0329 19:07:51.024261  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_conf_flat
I0329 19:07:51.024271  2693 net.cpp:100] Creating Layer conv8_2_mbox_conf_flat
I0329 19:07:51.024277  2693 net.cpp:434] conv8_2_mbox_conf_flat <- conv8_2_mbox_conf_perm
I0329 19:07:51.024284  2693 net.cpp:408] conv8_2_mbox_conf_flat -> conv8_2_mbox_conf_flat
I0329 19:07:51.024322  2693 net.cpp:150] Setting up conv8_2_mbox_conf_flat
I0329 19:07:51.024333  2693 net.cpp:157] Top shape: 8 288 (2304)
I0329 19:07:51.024338  2693 net.cpp:165] Memory required for data: 5844231200
I0329 19:07:51.024341  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_clean
I0329 19:07:51.024358  2693 net.cpp:100] Creating Layer conv8_2_mbox_clean
I0329 19:07:51.024366  2693 net.cpp:434] conv8_2_mbox_clean <- conv8_2_conv8_2_relu_0_split_3
I0329 19:07:51.024375  2693 net.cpp:408] conv8_2_mbox_clean -> conv8_2_mbox_clean
I0329 19:07:51.025815  2693 net.cpp:150] Setting up conv8_2_mbox_clean
I0329 19:07:51.025836  2693 net.cpp:157] Top shape: 8 4 6 6 (1152)
I0329 19:07:51.025842  2693 net.cpp:165] Memory required for data: 5844235808
I0329 19:07:51.025851  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_clean_perm
I0329 19:07:51.025861  2693 net.cpp:100] Creating Layer conv8_2_mbox_clean_perm
I0329 19:07:51.025868  2693 net.cpp:434] conv8_2_mbox_clean_perm <- conv8_2_mbox_clean
I0329 19:07:51.025876  2693 net.cpp:408] conv8_2_mbox_clean_perm -> conv8_2_mbox_clean_perm
I0329 19:07:51.026027  2693 net.cpp:150] Setting up conv8_2_mbox_clean_perm
I0329 19:07:51.026039  2693 net.cpp:157] Top shape: 8 6 6 4 (1152)
I0329 19:07:51.026044  2693 net.cpp:165] Memory required for data: 5844240416
I0329 19:07:51.026049  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_clean_flat
I0329 19:07:51.026059  2693 net.cpp:100] Creating Layer conv8_2_mbox_clean_flat
I0329 19:07:51.026065  2693 net.cpp:434] conv8_2_mbox_clean_flat <- conv8_2_mbox_clean_perm
I0329 19:07:51.026072  2693 net.cpp:408] conv8_2_mbox_clean_flat -> conv8_2_mbox_clean_flat
I0329 19:07:51.026110  2693 net.cpp:150] Setting up conv8_2_mbox_clean_flat
I0329 19:07:51.026120  2693 net.cpp:157] Top shape: 8 144 (1152)
I0329 19:07:51.026125  2693 net.cpp:165] Memory required for data: 5844245024
I0329 19:07:51.026129  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_priorbox
I0329 19:07:51.026141  2693 net.cpp:100] Creating Layer conv8_2_mbox_priorbox
I0329 19:07:51.026150  2693 net.cpp:434] conv8_2_mbox_priorbox <- conv8_2_conv8_2_relu_0_split_4
I0329 19:07:51.026157  2693 net.cpp:434] conv8_2_mbox_priorbox <- data_data_0_split_5
I0329 19:07:51.026176  2693 net.cpp:408] conv8_2_mbox_priorbox -> conv8_2_mbox_priorbox
I0329 19:07:51.026218  2693 net.cpp:150] Setting up conv8_2_mbox_priorbox
I0329 19:07:51.026228  2693 net.cpp:157] Top shape: 1 2 576 (1152)
I0329 19:07:51.026233  2693 net.cpp:165] Memory required for data: 5844249632
I0329 19:07:51.026238  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_loc
I0329 19:07:51.026253  2693 net.cpp:100] Creating Layer conv9_2_mbox_loc
I0329 19:07:51.026262  2693 net.cpp:434] conv9_2_mbox_loc <- conv9_2_conv9_2_relu_0_split_0
I0329 19:07:51.026273  2693 net.cpp:408] conv9_2_mbox_loc -> conv9_2_mbox_loc
I0329 19:07:51.027987  2693 net.cpp:150] Setting up conv9_2_mbox_loc
I0329 19:07:51.028005  2693 net.cpp:157] Top shape: 8 16 4 4 (2048)
I0329 19:07:51.028012  2693 net.cpp:165] Memory required for data: 5844257824
I0329 19:07:51.028020  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_loc_perm
I0329 19:07:51.028030  2693 net.cpp:100] Creating Layer conv9_2_mbox_loc_perm
I0329 19:07:51.028041  2693 net.cpp:434] conv9_2_mbox_loc_perm <- conv9_2_mbox_loc
I0329 19:07:51.028050  2693 net.cpp:408] conv9_2_mbox_loc_perm -> conv9_2_mbox_loc_perm
I0329 19:07:51.028198  2693 net.cpp:150] Setting up conv9_2_mbox_loc_perm
I0329 19:07:51.028209  2693 net.cpp:157] Top shape: 8 4 4 16 (2048)
I0329 19:07:51.028215  2693 net.cpp:165] Memory required for data: 5844266016
I0329 19:07:51.028220  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_loc_flat
I0329 19:07:51.028228  2693 net.cpp:100] Creating Layer conv9_2_mbox_loc_flat
I0329 19:07:51.028234  2693 net.cpp:434] conv9_2_mbox_loc_flat <- conv9_2_mbox_loc_perm
I0329 19:07:51.028241  2693 net.cpp:408] conv9_2_mbox_loc_flat -> conv9_2_mbox_loc_flat
I0329 19:07:51.028276  2693 net.cpp:150] Setting up conv9_2_mbox_loc_flat
I0329 19:07:51.028286  2693 net.cpp:157] Top shape: 8 256 (2048)
I0329 19:07:51.028291  2693 net.cpp:165] Memory required for data: 5844274208
I0329 19:07:51.028296  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_conf
I0329 19:07:51.028311  2693 net.cpp:100] Creating Layer conv9_2_mbox_conf
I0329 19:07:51.028318  2693 net.cpp:434] conv9_2_mbox_conf <- conv9_2_conv9_2_relu_0_split_1
I0329 19:07:51.028329  2693 net.cpp:408] conv9_2_mbox_conf -> conv9_2_mbox_conf
I0329 19:07:51.029906  2693 net.cpp:150] Setting up conv9_2_mbox_conf
I0329 19:07:51.029927  2693 net.cpp:157] Top shape: 8 8 4 4 (1024)
I0329 19:07:51.029932  2693 net.cpp:165] Memory required for data: 5844278304
I0329 19:07:51.029943  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_conf_perm
I0329 19:07:51.029955  2693 net.cpp:100] Creating Layer conv9_2_mbox_conf_perm
I0329 19:07:51.029963  2693 net.cpp:434] conv9_2_mbox_conf_perm <- conv9_2_mbox_conf
I0329 19:07:51.029973  2693 net.cpp:408] conv9_2_mbox_conf_perm -> conv9_2_mbox_conf_perm
I0329 19:07:51.030127  2693 net.cpp:150] Setting up conv9_2_mbox_conf_perm
I0329 19:07:51.030139  2693 net.cpp:157] Top shape: 8 4 4 8 (1024)
I0329 19:07:51.030144  2693 net.cpp:165] Memory required for data: 5844282400
I0329 19:07:51.030149  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_conf_flat
I0329 19:07:51.030158  2693 net.cpp:100] Creating Layer conv9_2_mbox_conf_flat
I0329 19:07:51.030164  2693 net.cpp:434] conv9_2_mbox_conf_flat <- conv9_2_mbox_conf_perm
I0329 19:07:51.030170  2693 net.cpp:408] conv9_2_mbox_conf_flat -> conv9_2_mbox_conf_flat
I0329 19:07:51.030208  2693 net.cpp:150] Setting up conv9_2_mbox_conf_flat
I0329 19:07:51.030218  2693 net.cpp:157] Top shape: 8 128 (1024)
I0329 19:07:51.030223  2693 net.cpp:165] Memory required for data: 5844286496
I0329 19:07:51.030228  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_clean
I0329 19:07:51.030241  2693 net.cpp:100] Creating Layer conv9_2_mbox_clean
I0329 19:07:51.030249  2693 net.cpp:434] conv9_2_mbox_clean <- conv9_2_conv9_2_relu_0_split_2
I0329 19:07:51.030261  2693 net.cpp:408] conv9_2_mbox_clean -> conv9_2_mbox_clean
I0329 19:07:51.031749  2693 net.cpp:150] Setting up conv9_2_mbox_clean
I0329 19:07:51.031769  2693 net.cpp:157] Top shape: 8 4 4 4 (512)
I0329 19:07:51.031788  2693 net.cpp:165] Memory required for data: 5844288544
I0329 19:07:51.031798  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_clean_perm
I0329 19:07:51.031816  2693 net.cpp:100] Creating Layer conv9_2_mbox_clean_perm
I0329 19:07:51.031826  2693 net.cpp:434] conv9_2_mbox_clean_perm <- conv9_2_mbox_clean
I0329 19:07:51.031836  2693 net.cpp:408] conv9_2_mbox_clean_perm -> conv9_2_mbox_clean_perm
I0329 19:07:51.031991  2693 net.cpp:150] Setting up conv9_2_mbox_clean_perm
I0329 19:07:51.032001  2693 net.cpp:157] Top shape: 8 4 4 4 (512)
I0329 19:07:51.032008  2693 net.cpp:165] Memory required for data: 5844290592
I0329 19:07:51.032013  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_clean_flat
I0329 19:07:51.032021  2693 net.cpp:100] Creating Layer conv9_2_mbox_clean_flat
I0329 19:07:51.032027  2693 net.cpp:434] conv9_2_mbox_clean_flat <- conv9_2_mbox_clean_perm
I0329 19:07:51.032034  2693 net.cpp:408] conv9_2_mbox_clean_flat -> conv9_2_mbox_clean_flat
I0329 19:07:51.032073  2693 net.cpp:150] Setting up conv9_2_mbox_clean_flat
I0329 19:07:51.032083  2693 net.cpp:157] Top shape: 8 64 (512)
I0329 19:07:51.032088  2693 net.cpp:165] Memory required for data: 5844292640
I0329 19:07:51.032094  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_priorbox
I0329 19:07:51.032104  2693 net.cpp:100] Creating Layer conv9_2_mbox_priorbox
I0329 19:07:51.032109  2693 net.cpp:434] conv9_2_mbox_priorbox <- conv9_2_conv9_2_relu_0_split_3
I0329 19:07:51.032116  2693 net.cpp:434] conv9_2_mbox_priorbox <- data_data_0_split_6
I0329 19:07:51.032125  2693 net.cpp:408] conv9_2_mbox_priorbox -> conv9_2_mbox_priorbox
I0329 19:07:51.032165  2693 net.cpp:150] Setting up conv9_2_mbox_priorbox
I0329 19:07:51.032174  2693 net.cpp:157] Top shape: 1 2 256 (512)
I0329 19:07:51.032179  2693 net.cpp:165] Memory required for data: 5844294688
I0329 19:07:51.032184  2693 layer_factory.hpp:77] Creating layer mbox_loc
I0329 19:07:51.032193  2693 net.cpp:100] Creating Layer mbox_loc
I0329 19:07:51.032199  2693 net.cpp:434] mbox_loc <- conv4_3_norm_mbox_loc_flat
I0329 19:07:51.032207  2693 net.cpp:434] mbox_loc <- fc7_mbox_loc_flat
I0329 19:07:51.032213  2693 net.cpp:434] mbox_loc <- conv6_2_mbox_loc_flat
I0329 19:07:51.032222  2693 net.cpp:434] mbox_loc <- conv7_2_mbox_loc_flat
I0329 19:07:51.032227  2693 net.cpp:434] mbox_loc <- conv8_2_mbox_loc_flat
I0329 19:07:51.032232  2693 net.cpp:434] mbox_loc <- conv9_2_mbox_loc_flat
I0329 19:07:51.032244  2693 net.cpp:408] mbox_loc -> mbox_loc
I0329 19:07:51.032304  2693 net.cpp:150] Setting up mbox_loc
I0329 19:07:51.032315  2693 net.cpp:157] Top shape: 8 98624 (788992)
I0329 19:07:51.032320  2693 net.cpp:165] Memory required for data: 5847450656
I0329 19:07:51.032325  2693 layer_factory.hpp:77] Creating layer mbox_conf
I0329 19:07:51.032335  2693 net.cpp:100] Creating Layer mbox_conf
I0329 19:07:51.032342  2693 net.cpp:434] mbox_conf <- conv4_3_norm_mbox_conf_flat
I0329 19:07:51.032349  2693 net.cpp:434] mbox_conf <- fc7_mbox_conf_flat
I0329 19:07:51.032356  2693 net.cpp:434] mbox_conf <- conv6_2_mbox_conf_flat
I0329 19:07:51.032366  2693 net.cpp:434] mbox_conf <- conv7_2_mbox_conf_flat
I0329 19:07:51.032371  2693 net.cpp:434] mbox_conf <- conv8_2_mbox_conf_flat
I0329 19:07:51.032377  2693 net.cpp:434] mbox_conf <- conv9_2_mbox_conf_flat
I0329 19:07:51.032384  2693 net.cpp:408] mbox_conf -> mbox_conf
I0329 19:07:51.032420  2693 net.cpp:150] Setting up mbox_conf
I0329 19:07:51.032429  2693 net.cpp:157] Top shape: 8 49312 (394496)
I0329 19:07:51.032434  2693 net.cpp:165] Memory required for data: 5849028640
I0329 19:07:51.032439  2693 layer_factory.hpp:77] Creating layer mbox_priorbox
I0329 19:07:51.032447  2693 net.cpp:100] Creating Layer mbox_priorbox
I0329 19:07:51.032454  2693 net.cpp:434] mbox_priorbox <- conv4_3_norm_mbox_priorbox
I0329 19:07:51.032459  2693 net.cpp:434] mbox_priorbox <- fc7_mbox_priorbox
I0329 19:07:51.032465  2693 net.cpp:434] mbox_priorbox <- conv6_2_mbox_priorbox
I0329 19:07:51.032471  2693 net.cpp:434] mbox_priorbox <- conv7_2_mbox_priorbox
I0329 19:07:51.032487  2693 net.cpp:434] mbox_priorbox <- conv8_2_mbox_priorbox
I0329 19:07:51.032493  2693 net.cpp:434] mbox_priorbox <- conv9_2_mbox_priorbox
I0329 19:07:51.032502  2693 net.cpp:408] mbox_priorbox -> mbox_priorbox
I0329 19:07:51.032543  2693 net.cpp:150] Setting up mbox_priorbox
I0329 19:07:51.032553  2693 net.cpp:157] Top shape: 1 2 98624 (197248)
I0329 19:07:51.032558  2693 net.cpp:165] Memory required for data: 5849817632
I0329 19:07:51.032563  2693 layer_factory.hpp:77] Creating layer mbox_clean
I0329 19:07:51.032572  2693 net.cpp:100] Creating Layer mbox_clean
I0329 19:07:51.032577  2693 net.cpp:434] mbox_clean <- conv4_3_norm_mbox_clean_flat
I0329 19:07:51.032584  2693 net.cpp:434] mbox_clean <- fc7_mbox_clean_flat
I0329 19:07:51.032590  2693 net.cpp:434] mbox_clean <- conv6_2_mbox_clean_flat
I0329 19:07:51.032596  2693 net.cpp:434] mbox_clean <- conv7_2_mbox_clean_flat
I0329 19:07:51.032603  2693 net.cpp:434] mbox_clean <- conv8_2_mbox_clean_flat
I0329 19:07:51.032608  2693 net.cpp:434] mbox_clean <- conv9_2_mbox_clean_flat
I0329 19:07:51.032616  2693 net.cpp:408] mbox_clean -> mbox_clean
I0329 19:07:51.032652  2693 net.cpp:150] Setting up mbox_clean
I0329 19:07:51.032665  2693 net.cpp:157] Top shape: 8 24656 (197248)
I0329 19:07:51.032668  2693 net.cpp:165] Memory required for data: 5850606624
I0329 19:07:51.032673  2693 layer_factory.hpp:77] Creating layer mbox_loss
I0329 19:07:51.032687  2693 net.cpp:100] Creating Layer mbox_loss
I0329 19:07:51.032696  2693 net.cpp:434] mbox_loss <- mbox_loc
I0329 19:07:51.032701  2693 net.cpp:434] mbox_loss <- mbox_conf
I0329 19:07:51.032707  2693 net.cpp:434] mbox_loss <- mbox_priorbox
I0329 19:07:51.032712  2693 net.cpp:434] mbox_loss <- label
I0329 19:07:51.032718  2693 net.cpp:434] mbox_loss <- mbox_clean
I0329 19:07:51.032728  2693 net.cpp:408] mbox_loss -> mbox_loss
I0329 19:07:51.032811  2693 layer_factory.hpp:77] Creating layer mbox_loss_smooth_L1_loc
I0329 19:07:51.032943  2693 layer_factory.hpp:77] Creating layer mbox_loss_softmax_conf
I0329 19:07:51.032958  2693 layer_factory.hpp:77] Creating layer mbox_loss_softmax_conf
I0329 19:07:51.033380  2693 layer_factory.hpp:77] Creating layer mbox_loss_clean
I0329 19:07:51.033488  2693 net.cpp:150] Setting up mbox_loss
I0329 19:07:51.033500  2693 net.cpp:157] Top shape: (1)
I0329 19:07:51.033505  2693 net.cpp:160]     with loss weight 1
I0329 19:07:51.033540  2693 net.cpp:165] Memory required for data: 5850606628
I0329 19:07:51.033545  2693 net.cpp:226] mbox_loss needs backward computation.
I0329 19:07:51.033552  2693 net.cpp:226] mbox_clean needs backward computation.
I0329 19:07:51.033561  2693 net.cpp:228] mbox_priorbox does not need backward computation.
I0329 19:07:51.033572  2693 net.cpp:226] mbox_conf needs backward computation.
I0329 19:07:51.033582  2693 net.cpp:226] mbox_loc needs backward computation.
I0329 19:07:51.033589  2693 net.cpp:228] conv9_2_mbox_priorbox does not need backward computation.
I0329 19:07:51.033596  2693 net.cpp:226] conv9_2_mbox_clean_flat needs backward computation.
I0329 19:07:51.033601  2693 net.cpp:226] conv9_2_mbox_clean_perm needs backward computation.
I0329 19:07:51.033607  2693 net.cpp:226] conv9_2_mbox_clean needs backward computation.
I0329 19:07:51.033612  2693 net.cpp:226] conv9_2_mbox_conf_flat needs backward computation.
I0329 19:07:51.033617  2693 net.cpp:226] conv9_2_mbox_conf_perm needs backward computation.
I0329 19:07:51.033622  2693 net.cpp:226] conv9_2_mbox_conf needs backward computation.
I0329 19:07:51.033627  2693 net.cpp:226] conv9_2_mbox_loc_flat needs backward computation.
I0329 19:07:51.033632  2693 net.cpp:226] conv9_2_mbox_loc_perm needs backward computation.
I0329 19:07:51.033638  2693 net.cpp:226] conv9_2_mbox_loc needs backward computation.
I0329 19:07:51.033643  2693 net.cpp:228] conv8_2_mbox_priorbox does not need backward computation.
I0329 19:07:51.033649  2693 net.cpp:226] conv8_2_mbox_clean_flat needs backward computation.
I0329 19:07:51.033654  2693 net.cpp:226] conv8_2_mbox_clean_perm needs backward computation.
I0329 19:07:51.033659  2693 net.cpp:226] conv8_2_mbox_clean needs backward computation.
I0329 19:07:51.033679  2693 net.cpp:226] conv8_2_mbox_conf_flat needs backward computation.
I0329 19:07:51.033684  2693 net.cpp:226] conv8_2_mbox_conf_perm needs backward computation.
I0329 19:07:51.033691  2693 net.cpp:226] conv8_2_mbox_conf needs backward computation.
I0329 19:07:51.033696  2693 net.cpp:226] conv8_2_mbox_loc_flat needs backward computation.
I0329 19:07:51.033701  2693 net.cpp:226] conv8_2_mbox_loc_perm needs backward computation.
I0329 19:07:51.033706  2693 net.cpp:226] conv8_2_mbox_loc needs backward computation.
I0329 19:07:51.033711  2693 net.cpp:228] conv7_2_mbox_priorbox does not need backward computation.
I0329 19:07:51.033718  2693 net.cpp:226] conv7_2_mbox_clean_flat needs backward computation.
I0329 19:07:51.033723  2693 net.cpp:226] conv7_2_mbox_clean_perm needs backward computation.
I0329 19:07:51.033728  2693 net.cpp:226] conv7_2_mbox_clean needs backward computation.
I0329 19:07:51.033733  2693 net.cpp:226] conv7_2_mbox_conf_flat needs backward computation.
I0329 19:07:51.033740  2693 net.cpp:226] conv7_2_mbox_conf_perm needs backward computation.
I0329 19:07:51.033745  2693 net.cpp:226] conv7_2_mbox_conf needs backward computation.
I0329 19:07:51.033751  2693 net.cpp:226] conv7_2_mbox_loc_flat needs backward computation.
I0329 19:07:51.033756  2693 net.cpp:226] conv7_2_mbox_loc_perm needs backward computation.
I0329 19:07:51.033761  2693 net.cpp:226] conv7_2_mbox_loc needs backward computation.
I0329 19:07:51.033766  2693 net.cpp:228] conv6_2_mbox_priorbox does not need backward computation.
I0329 19:07:51.033772  2693 net.cpp:226] conv6_2_mbox_clean_flat needs backward computation.
I0329 19:07:51.033777  2693 net.cpp:226] conv6_2_mbox_clean_perm needs backward computation.
I0329 19:07:51.033782  2693 net.cpp:226] conv6_2_mbox_clean needs backward computation.
I0329 19:07:51.033787  2693 net.cpp:226] conv6_2_mbox_conf_flat needs backward computation.
I0329 19:07:51.033792  2693 net.cpp:226] conv6_2_mbox_conf_perm needs backward computation.
I0329 19:07:51.033797  2693 net.cpp:226] conv6_2_mbox_conf needs backward computation.
I0329 19:07:51.033802  2693 net.cpp:226] conv6_2_mbox_loc_flat needs backward computation.
I0329 19:07:51.033807  2693 net.cpp:226] conv6_2_mbox_loc_perm needs backward computation.
I0329 19:07:51.033812  2693 net.cpp:226] conv6_2_mbox_loc needs backward computation.
I0329 19:07:51.033818  2693 net.cpp:228] fc7_mbox_priorbox does not need backward computation.
I0329 19:07:51.033825  2693 net.cpp:226] fc7_mbox_clean_flat needs backward computation.
I0329 19:07:51.033830  2693 net.cpp:226] fc7_mbox_clean_perm needs backward computation.
I0329 19:07:51.033835  2693 net.cpp:226] fc7_mbox_clean needs backward computation.
I0329 19:07:51.033841  2693 net.cpp:226] fc7_mbox_conf_flat needs backward computation.
I0329 19:07:51.033846  2693 net.cpp:226] fc7_mbox_conf_perm needs backward computation.
I0329 19:07:51.033851  2693 net.cpp:226] fc7_mbox_conf needs backward computation.
I0329 19:07:51.033856  2693 net.cpp:226] fc7_mbox_loc_flat needs backward computation.
I0329 19:07:51.033862  2693 net.cpp:226] fc7_mbox_loc_perm needs backward computation.
I0329 19:07:51.033867  2693 net.cpp:226] fc7_mbox_loc needs backward computation.
I0329 19:07:51.033872  2693 net.cpp:228] conv4_3_norm_mbox_priorbox does not need backward computation.
I0329 19:07:51.033879  2693 net.cpp:226] conv4_3_norm_mbox_clean_flat needs backward computation.
I0329 19:07:51.033885  2693 net.cpp:226] conv4_3_norm_mbox_clean_perm needs backward computation.
I0329 19:07:51.033890  2693 net.cpp:226] conv4_3_norm_mbox_clean needs backward computation.
I0329 19:07:51.033895  2693 net.cpp:226] conv4_3_norm_mbox_conf_flat needs backward computation.
I0329 19:07:51.033903  2693 net.cpp:226] conv4_3_norm_mbox_conf_perm needs backward computation.
I0329 19:07:51.033908  2693 net.cpp:226] conv4_3_norm_mbox_conf needs backward computation.
I0329 19:07:51.033913  2693 net.cpp:226] conv4_3_norm_mbox_loc_flat needs backward computation.
I0329 19:07:51.033918  2693 net.cpp:226] conv4_3_norm_mbox_loc_perm needs backward computation.
I0329 19:07:51.033932  2693 net.cpp:226] conv4_3_norm_mbox_loc needs backward computation.
I0329 19:07:51.033938  2693 net.cpp:226] conv4_3_norm_conv4_3_norm_0_split needs backward computation.
I0329 19:07:51.033943  2693 net.cpp:226] conv4_3_norm needs backward computation.
I0329 19:07:51.033949  2693 net.cpp:226] conv9_2_conv9_2_relu_0_split needs backward computation.
I0329 19:07:51.033956  2693 net.cpp:226] conv9_2_relu needs backward computation.
I0329 19:07:51.033959  2693 net.cpp:226] conv9_2 needs backward computation.
I0329 19:07:51.033965  2693 net.cpp:226] conv9_1_relu needs backward computation.
I0329 19:07:51.033970  2693 net.cpp:226] conv9_1 needs backward computation.
I0329 19:07:51.033975  2693 net.cpp:226] conv8_2_conv8_2_relu_0_split needs backward computation.
I0329 19:07:51.033982  2693 net.cpp:226] conv8_2_relu needs backward computation.
I0329 19:07:51.033987  2693 net.cpp:226] conv8_2 needs backward computation.
I0329 19:07:51.033991  2693 net.cpp:226] conv8_1_relu needs backward computation.
I0329 19:07:51.033996  2693 net.cpp:226] conv8_1 needs backward computation.
I0329 19:07:51.034001  2693 net.cpp:226] conv7_2_conv7_2_relu_0_split needs backward computation.
I0329 19:07:51.034006  2693 net.cpp:226] conv7_2_relu needs backward computation.
I0329 19:07:51.034011  2693 net.cpp:226] conv7_2 needs backward computation.
I0329 19:07:51.034016  2693 net.cpp:226] conv7_1_relu needs backward computation.
I0329 19:07:51.034021  2693 net.cpp:226] conv7_1 needs backward computation.
I0329 19:07:51.034026  2693 net.cpp:226] conv6_2_conv6_2_relu_0_split needs backward computation.
I0329 19:07:51.034032  2693 net.cpp:226] conv6_2_relu needs backward computation.
I0329 19:07:51.034040  2693 net.cpp:226] conv6_2 needs backward computation.
I0329 19:07:51.034045  2693 net.cpp:226] conv6_1_relu needs backward computation.
I0329 19:07:51.034050  2693 net.cpp:226] conv6_1 needs backward computation.
I0329 19:07:51.034056  2693 net.cpp:226] fc7_relu7_0_split needs backward computation.
I0329 19:07:51.034061  2693 net.cpp:226] relu7 needs backward computation.
I0329 19:07:51.034066  2693 net.cpp:226] fc7 needs backward computation.
I0329 19:07:51.034072  2693 net.cpp:226] relu6 needs backward computation.
I0329 19:07:51.034077  2693 net.cpp:226] fc6 needs backward computation.
I0329 19:07:51.034082  2693 net.cpp:226] pool5 needs backward computation.
I0329 19:07:51.034088  2693 net.cpp:226] relu5_3 needs backward computation.
I0329 19:07:51.034093  2693 net.cpp:226] conv5_3 needs backward computation.
I0329 19:07:51.034098  2693 net.cpp:226] relu5_2 needs backward computation.
I0329 19:07:51.034104  2693 net.cpp:226] conv5_2 needs backward computation.
I0329 19:07:51.034109  2693 net.cpp:226] relu5_1 needs backward computation.
I0329 19:07:51.034114  2693 net.cpp:226] conv5_1 needs backward computation.
I0329 19:07:51.034119  2693 net.cpp:226] pool4 needs backward computation.
I0329 19:07:51.034126  2693 net.cpp:226] conv4_3_relu4_3_0_split needs backward computation.
I0329 19:07:51.034131  2693 net.cpp:226] relu4_3 needs backward computation.
I0329 19:07:51.034135  2693 net.cpp:226] conv4_3 needs backward computation.
I0329 19:07:51.034142  2693 net.cpp:226] relu4_2 needs backward computation.
I0329 19:07:51.034145  2693 net.cpp:226] conv4_2 needs backward computation.
I0329 19:07:51.034152  2693 net.cpp:226] relu4_1 needs backward computation.
I0329 19:07:51.034155  2693 net.cpp:226] conv4_1 needs backward computation.
I0329 19:07:51.034162  2693 net.cpp:226] pool3 needs backward computation.
I0329 19:07:51.034167  2693 net.cpp:226] relu3_3 needs backward computation.
I0329 19:07:51.034171  2693 net.cpp:226] conv3_3 needs backward computation.
I0329 19:07:51.034176  2693 net.cpp:226] relu3_2 needs backward computation.
I0329 19:07:51.034181  2693 net.cpp:226] conv3_2 needs backward computation.
I0329 19:07:51.034186  2693 net.cpp:226] relu3_1 needs backward computation.
I0329 19:07:51.034195  2693 net.cpp:226] conv3_1 needs backward computation.
I0329 19:07:51.034206  2693 net.cpp:226] pool2 needs backward computation.
I0329 19:07:51.034212  2693 net.cpp:226] relu2_2 needs backward computation.
I0329 19:07:51.034217  2693 net.cpp:226] conv2_2 needs backward computation.
I0329 19:07:51.034222  2693 net.cpp:226] relu2_1 needs backward computation.
I0329 19:07:51.034227  2693 net.cpp:226] conv2_1 needs backward computation.
I0329 19:07:51.034232  2693 net.cpp:226] pool1 needs backward computation.
I0329 19:07:51.034237  2693 net.cpp:226] relu1_2 needs backward computation.
I0329 19:07:51.034242  2693 net.cpp:226] conv1_2 needs backward computation.
I0329 19:07:51.034247  2693 net.cpp:226] relu1_1 needs backward computation.
I0329 19:07:51.034252  2693 net.cpp:226] conv1_1 needs backward computation.
I0329 19:07:51.034260  2693 net.cpp:228] data_data_0_split does not need backward computation.
I0329 19:07:51.034268  2693 net.cpp:228] data does not need backward computation.
I0329 19:07:51.034273  2693 net.cpp:270] This network produces output mbox_loss
I0329 19:07:51.034375  2693 net.cpp:283] Network initialization done.
I0329 19:07:51.037641  2693 solver.cpp:196] Creating test net (#0) specified by test_net file: models/VGGNet/ssd_coco_part_clean/test.prototxt
I0329 19:07:51.038595  2693 net.cpp:58] Initializing net from parameters: 
name: "VGG_ssd_coco_part_clean_test"
state {
  phase: TEST
}
layer {
  name: "data"
  type: "AnnotatedData"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mean_value: 104
    mean_value: 117
    mean_value: 123
    force_color: true
    resize_param {
      prob: 1
      resize_mode: WARP
      height: 512
      width: 512
      interp_mode: LINEAR
    }
  }
  data_param {
    source: "/data/siyu/dataset/coco/lmdb/COCO_Val2014_person_lmdb"
    batch_size: 2
    backend: LMDB
  }
  annotated_data_param {
    batch_sampler {
    }
    label_map_file: "/data/siyu/dataset/coco/labelmap_coco-person.prototxt"
    part_sampler_prob: 0
  }
}
layer {
  name: "conv1_1"
  type: "Convolution"
  bottom: "data"
  top: "conv1_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu1_1"
  type: "ReLU"
  bottom: "conv1_1"
  top: "conv1_1"
}
layer {
  name: "conv1_2"
  type: "Convolution"
  bottom: "conv1_1"
  top: "conv1_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu1_2"
  type: "ReLU"
  bottom: "conv1_2"
  top: "conv1_2"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1_2"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2_1"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu2_1"
  type: "ReLU"
  bottom: "conv2_1"
  top: "conv2_1"
}
layer {
  name: "conv2_2"
  type: "Convolution"
  bottom: "conv2_1"
  top: "conv2_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu2_2"
  type: "ReLU"
  bottom: "conv2_2"
  top: "conv2_2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2_2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv3_1"
  type: "Convolution"
  bottom: "pool2"
  top: "conv3_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu3_1"
  type: "ReLU"
  bottom: "conv3_1"
  top: "conv3_1"
}
layer {
  name: "conv3_2"
  type: "Convolution"
  bottom: "conv3_1"
  top: "conv3_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu3_2"
  type: "ReLU"
  bottom: "conv3_2"
  top: "conv3_2"
}
layer {
  name: "conv3_3"
  type: "Convolution"
  bottom: "conv3_2"
  top: "conv3_3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu3_3"
  type: "ReLU"
  bottom: "conv3_3"
  top: "conv3_3"
}
layer {
  name: "pool3"
  type: "Pooling"
  bottom: "conv3_3"
  top: "pool3"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv4_1"
  type: "Convolution"
  bottom: "pool3"
  top: "conv4_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu4_1"
  type: "ReLU"
  bottom: "conv4_1"
  top: "conv4_1"
}
layer {
  name: "conv4_2"
  type: "Convolution"
  bottom: "conv4_1"
  top: "conv4_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu4_2"
  type: "ReLU"
  bottom: "conv4_2"
  top: "conv4_2"
}
layer {
  name: "conv4_3"
  type: "Convolution"
  bottom: "conv4_2"
  top: "conv4_3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu4_3"
  type: "ReLU"
  bottom: "conv4_3"
  top: "conv4_3"
}
layer {
  name: "pool4"
  type: "Pooling"
  bottom: "conv4_3"
  top: "pool4"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv5_1"
  type: "Convolution"
  bottom: "pool4"
  top: "conv5_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    dilation: 1
  }
}
layer {
  name: "relu5_1"
  type: "ReLU"
  bottom: "conv5_1"
  top: "conv5_1"
}
layer {
  name: "conv5_2"
  type: "Convolution"
  bottom: "conv5_1"
  top: "conv5_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    dilation: 1
  }
}
layer {
  name: "relu5_2"
  type: "ReLU"
  bottom: "conv5_2"
  top: "conv5_2"
}
layer {
  name: "conv5_3"
  type: "Convolution"
  bottom: "conv5_2"
  top: "conv5_3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    dilation: 1
  }
}
layer {
  name: "relu5_3"
  type: "ReLU"
  bottom: "conv5_3"
  top: "conv5_3"
}
layer {
  name: "pool5"
  type: "Pooling"
  bottom: "conv5_3"
  top: "pool5"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
  name: "fc6"
  type: "Convolution"
  bottom: "pool5"
  top: "fc6"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 1024
    pad: 6
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    dilation: 6
  }
}
layer {
  name: "relu6"
  type: "ReLU"
  bottom: "fc6"
  top: "fc6"
}
layer {
  name: "fc7"
  type: "Convolution"
  bottom: "fc6"
  top: "fc7"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 1024
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu7"
  type: "ReLU"
  bottom: "fc7"
  top: "fc7"
}
layer {
  name: "conv6_1"
  type: "Convolution"
  bottom: "fc7"
  top: "conv6_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv6_1_relu"
  type: "ReLU"
  bottom: "conv6_1"
  top: "conv6_1"
}
layer {
  name: "conv6_2"
  type: "Convolution"
  bottom: "conv6_1"
  top: "conv6_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv6_2_relu"
  type: "ReLU"
  bottom: "conv6_2"
  top: "conv6_2"
}
layer {
  name: "conv7_1"
  type: "Convolution"
  bottom: "conv6_2"
  top: "conv7_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv7_1_relu"
  type: "ReLU"
  bottom: "conv7_1"
  top: "conv7_1"
}
layer {
  name: "conv7_2"
  type: "Convolution"
  bottom: "conv7_1"
  top: "conv7_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv7_2_relu"
  type: "ReLU"
  bottom: "conv7_2"
  top: "conv7_2"
}
layer {
  name: "conv8_1"
  type: "Convolution"
  bottom: "conv7_2"
  top: "conv8_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv8_1_relu"
  type: "ReLU"
  bottom: "conv8_1"
  top: "conv8_1"
}
layer {
  name: "conv8_2"
  type: "Convolution"
  bottom: "conv8_1"
  top: "conv8_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 0
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv8_2_relu"
  type: "ReLU"
  bottom: "conv8_2"
  top: "conv8_2"
}
layer {
  name: "conv9_1"
  type: "Convolution"
  bottom: "conv8_2"
  top: "conv9_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv9_1_relu"
  type: "ReLU"
  bottom: "conv9_1"
  top: "conv9_1"
}
layer {
  name: "conv9_2"
  type: "Convolution"
  bottom: "conv9_1"
  top: "conv9_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 0
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv9_2_relu"
  type: "ReLU"
  bottom: "conv9_2"
  top: "conv9_2"
}
layer {
  name: "conv4_3_norm"
  type: "Normalize"
  bottom: "conv4_3"
  top: "conv4_3_norm"
  norm_param {
    across_spatial: false
    scale_filler {
      type: "constant"
      value: 20
    }
    channel_shared: false
  }
}
layer {
  name: "conv4_3_norm_mbox_loc"
  type: "Convolution"
  bottom: "conv4_3_norm"
  top: "conv4_3_norm_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 16
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv4_3_norm_mbox_loc_perm"
  type: "Permute"
  bottom: "conv4_3_norm_mbox_loc"
  top: "conv4_3_norm_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv4_3_norm_mbox_loc_perm"
  top: "conv4_3_norm_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_conf"
  type: "Convolution"
  bottom: "conv4_3_norm"
  top: "conv4_3_norm_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 8
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv4_3_norm_mbox_conf_perm"
  type: "Permute"
  bottom: "conv4_3_norm_mbox_conf"
  top: "conv4_3_norm_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv4_3_norm_mbox_conf_perm"
  top: "conv4_3_norm_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_clean"
  type: "Convolution"
  bottom: "conv4_3_norm"
  top: "conv4_3_norm_mbox_clean"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 4
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv4_3_norm_mbox_clean_perm"
  type: "Permute"
  bottom: "conv4_3_norm_mbox_clean"
  top: "conv4_3_norm_mbox_clean_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_clean_flat"
  type: "Flatten"
  bottom: "conv4_3_norm_mbox_clean_perm"
  top: "conv4_3_norm_mbox_clean_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv4_3_norm"
  bottom: "data"
  top: "conv4_3_norm_mbox_priorbox"
  prior_box_param {
    min_size: 35.84
    max_size: 76.8
    aspect_ratio: 2
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 8
    offset: 0.5
  }
}
layer {
  name: "fc7_mbox_loc"
  type: "Convolution"
  bottom: "fc7"
  top: "fc7_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 24
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "fc7_mbox_loc_perm"
  type: "Permute"
  bottom: "fc7_mbox_loc"
  top: "fc7_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "fc7_mbox_loc_flat"
  type: "Flatten"
  bottom: "fc7_mbox_loc_perm"
  top: "fc7_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "fc7_mbox_conf"
  type: "Convolution"
  bottom: "fc7"
  top: "fc7_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 12
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "fc7_mbox_conf_perm"
  type: "Permute"
  bottom: "fc7_mbox_conf"
  top: "fc7_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "fc7_mbox_conf_flat"
  type: "Flatten"
  bottom: "fc7_mbox_conf_perm"
  top: "fc7_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "fc7_mbox_clean"
  type: "Convolution"
  bottom: "fc7"
  top: "fc7_mbox_clean"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 6
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "fc7_mbox_clean_perm"
  type: "Permute"
  bottom: "fc7_mbox_clean"
  top: "fc7_mbox_clean_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "fc7_mbox_clean_flat"
  type: "Flatten"
  bottom: "fc7_mbox_clean_perm"
  top: "fc7_mbox_clean_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "fc7_mbox_priorbox"
  type: "PriorBox"
  bottom: "fc7"
  bottom: "data"
  top: "fc7_mbox_priorbox"
  prior_box_param {
    min_size: 76.8
    max_size: 168.96
    aspect_ratio: 2
    aspect_ratio: 3
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 16
    offset: 0.5
  }
}
layer {
  name: "conv6_2_mbox_loc"
  type: "Convolution"
  bottom: "conv6_2"
  top: "conv6_2_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 24
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv6_2_mbox_loc_perm"
  type: "Permute"
  bottom: "conv6_2_mbox_loc"
  top: "conv6_2_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv6_2_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv6_2_mbox_loc_perm"
  top: "conv6_2_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv6_2_mbox_conf"
  type: "Convolution"
  bottom: "conv6_2"
  top: "conv6_2_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 12
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv6_2_mbox_conf_perm"
  type: "Permute"
  bottom: "conv6_2_mbox_conf"
  top: "conv6_2_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv6_2_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv6_2_mbox_conf_perm"
  top: "conv6_2_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv6_2_mbox_clean"
  type: "Convolution"
  bottom: "conv6_2"
  top: "conv6_2_mbox_clean"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 6
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv6_2_mbox_clean_perm"
  type: "Permute"
  bottom: "conv6_2_mbox_clean"
  top: "conv6_2_mbox_clean_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv6_2_mbox_clean_flat"
  type: "Flatten"
  bottom: "conv6_2_mbox_clean_perm"
  top: "conv6_2_mbox_clean_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv6_2_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv6_2"
  bottom: "data"
  top: "conv6_2_mbox_priorbox"
  prior_box_param {
    min_size: 168.96
    max_size: 261.12
    aspect_ratio: 2
    aspect_ratio: 3
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 32
    offset: 0.5
  }
}
layer {
  name: "conv7_2_mbox_loc"
  type: "Convolution"
  bottom: "conv7_2"
  top: "conv7_2_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 24
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv7_2_mbox_loc_perm"
  type: "Permute"
  bottom: "conv7_2_mbox_loc"
  top: "conv7_2_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv7_2_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv7_2_mbox_loc_perm"
  top: "conv7_2_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv7_2_mbox_conf"
  type: "Convolution"
  bottom: "conv7_2"
  top: "conv7_2_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 12
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv7_2_mbox_conf_perm"
  type: "Permute"
  bottom: "conv7_2_mbox_conf"
  top: "conv7_2_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv7_2_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv7_2_mbox_conf_perm"
  top: "conv7_2_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv7_2_mbox_clean"
  type: "Convolution"
  bottom: "conv7_2"
  top: "conv7_2_mbox_clean"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 6
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv7_2_mbox_clean_perm"
  type: "Permute"
  bottom: "conv7_2_mbox_clean"
  top: "conv7_2_mbox_clean_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv7_2_mbox_clean_flat"
  type: "Flatten"
  bottom: "conv7_2_mbox_clean_perm"
  top: "conv7_2_mbox_clean_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv7_2_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv7_2"
  bottom: "data"
  top: "conv7_2_mbox_priorbox"
  prior_box_param {
    min_size: 261.12
    max_size: 353.28
    aspect_ratio: 2
    aspect_ratio: 3
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 64
    offset: 0.5
  }
}
layer {
  name: "conv8_2_mbox_loc"
  type: "Convolution"
  bottom: "conv8_2"
  top: "conv8_2_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 16
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv8_2_mbox_loc_perm"
  type: "Permute"
  bottom: "conv8_2_mbox_loc"
  top: "conv8_2_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv8_2_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv8_2_mbox_loc_perm"
  top: "conv8_2_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv8_2_mbox_conf"
  type: "Convolution"
  bottom: "conv8_2"
  top: "conv8_2_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 8
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv8_2_mbox_conf_perm"
  type: "Permute"
  bottom: "conv8_2_mbox_conf"
  top: "conv8_2_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv8_2_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv8_2_mbox_conf_perm"
  top: "conv8_2_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv8_2_mbox_clean"
  type: "Convolution"
  bottom: "conv8_2"
  top: "conv8_2_mbox_clean"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 4
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv8_2_mbox_clean_perm"
  type: "Permute"
  bottom: "conv8_2_mbox_clean"
  top: "conv8_2_mbox_clean_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv8_2_mbox_clean_flat"
  type: "Flatten"
  bottom: "conv8_2_mbox_clean_perm"
  top: "conv8_2_mbox_clean_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv8_2_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv8_2"
  bottom: "data"
  top: "conv8_2_mbox_priorbox"
  prior_box_param {
    min_size: 353.28
    max_size: 445.44
    aspect_ratio: 2
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 100
    offset: 0.5
  }
}
layer {
  name: "conv9_2_mbox_loc"
  type: "Convolution"
  bottom: "conv9_2"
  top: "conv9_2_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 16
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv9_2_mbox_loc_perm"
  type: "Permute"
  bottom: "conv9_2_mbox_loc"
  top: "conv9_2_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv9_2_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv9_2_mbox_loc_perm"
  top: "conv9_2_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv9_2_mbox_conf"
  type: "Convolution"
  bottom: "conv9_2"
  top: "conv9_2_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 8
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv9_2_mbox_conf_perm"
  type: "Permute"
  bottom: "conv9_2_mbox_conf"
  top: "conv9_2_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv9_2_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv9_2_mbox_conf_perm"
  top: "conv9_2_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv9_2_mbox_clean"
  type: "Convolution"
  bottom: "conv9_2"
  top: "conv9_2_mbox_clean"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 4
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv9_2_mbox_clean_perm"
  type: "Permute"
  bottom: "conv9_2_mbox_clean"
  top: "conv9_2_mbox_clean_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv9_2_mbox_clean_flat"
  type: "Flatten"
  bottom: "conv9_2_mbox_clean_perm"
  top: "conv9_2_mbox_clean_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv9_2_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv9_2"
  bottom: "data"
  top: "conv9_2_mbox_priorbox"
  prior_box_param {
    min_size: 445.44
    max_size: 537.6
    aspect_ratio: 2
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 300
    offset: 0.5
  }
}
layer {
  name: "mbox_loc"
  type: "Concat"
  bottom: "conv4_3_norm_mbox_loc_flat"
  bottom: "fc7_mbox_loc_flat"
  bottom: "conv6_2_mbox_loc_flat"
  bottom: "conv7_2_mbox_loc_flat"
  bottom: "conv8_2_mbox_loc_flat"
  bottom: "conv9_2_mbox_loc_flat"
  top: "mbox_loc"
  concat_param {
    axis: 1
  }
}
layer {
  name: "mbox_conf"
  type: "Concat"
  bottom: "conv4_3_norm_mbox_conf_flat"
  bottom: "fc7_mbox_conf_flat"
  bottom: "conv6_2_mbox_conf_flat"
  bottom: "conv7_2_mbox_conf_flat"
  bottom: "conv8_2_mbox_conf_flat"
  bottom: "conv9_2_mbox_conf_flat"
  top: "mbox_conf"
  concat_param {
    axis: 1
  }
}
layer {
  name: "mbox_priorbox"
  type: "Concat"
  bottom: "conv4_3_norm_mbox_priorbox"
  bottom: "fc7_mbox_priorbox"
  bottom: "conv6_2_mbox_priorbox"
  bottom: "conv7_2_mbox_priorbox"
  bottom: "conv8_2_mbox_priorbox"
  bottom: "conv9_2_mbox_priorbox"
  top: "mbox_priorbox"
  concat_param {
    axis: 2
  }
}
layer {
  name: "mbox_clean"
  type: "Concat"
  bottom: "conv4_3_norm_mbox_clean_flat"
  bottom: "fc7_mbox_clean_flat"
  bottom: "conv6_2_mbox_clean_flat"
  bottom: "conv7_2_mbox_clean_flat"
  bottom: "conv8_2_mbox_clean_flat"
  bottom: "conv9_2_mbox_clean_flat"
  top: "mbox_clean"
  concat_param {
    axis: 1
  }
}
layer {
  name: "mbox_conf_reshape"
  type: "Reshape"
  bottom: "mbox_conf"
  top: "mbox_conf_reshape"
  reshape_param {
    shape {
      dim: 0
      dim: -1
      dim: 2
    }
  }
}
layer {
  name: "mbox_conf_softmax"
  type: "Softmax"
  bottom: "mbox_conf_reshape"
  top: "mbox_conf_softmax"
  softmax_param {
    axis: 2
  }
}
layer {
  name: "mbox_conf_flatten"
  type: "Flatten"
  bottom: "mbox_conf_softmax"
  top: "mbox_conf_flatten"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "mbox_clean_sigmoid"
  type: "Sigmoid"
  bottom: "mbox_clean"
  top: "mbox_clean_sigmoid"
}
layer {
  name: "detection_out"
  type: "DetectionOutput"
  bottom: "mbox_loc"
  bottom: "mbox_conf_flatten"
  bottom: "mbox_priorbox"
  bottom: "mbox_clean_sigmoid"
  top: "detection_out"
  include {
    phase: TEST
  }
  de
I0329 19:07:51.039207  2693 layer_factory.hpp:77] Creating layer data
I0329 19:07:51.039283  2693 net.cpp:100] Creating Layer data
I0329 19:07:51.039296  2693 net.cpp:408] data -> data
I0329 19:07:51.039310  2693 net.cpp:408] data -> label
I0329 19:07:51.040769  2764 db_lmdb.cpp:35] Opened lmdb /data/siyu/dataset/coco/lmdb/COCO_Val2014_person_lmdb
I0329 19:07:51.044848  2693 annotated_data_layer.cpp:62] output data size: 2,3,512,512
I0329 19:07:51.057054  2693 net.cpp:150] Setting up data
I0329 19:07:51.057087  2693 net.cpp:157] Top shape: 2 3 512 512 (1572864)
I0329 19:07:51.057096  2693 net.cpp:157] Top shape: 1 1 6 8 (48)
I0329 19:07:51.057102  2693 net.cpp:165] Memory required for data: 6291648
I0329 19:07:51.057111  2693 layer_factory.hpp:77] Creating layer data_data_0_split
I0329 19:07:51.057128  2693 net.cpp:100] Creating Layer data_data_0_split
I0329 19:07:51.057133  2693 net.cpp:434] data_data_0_split <- data
I0329 19:07:51.057144  2693 net.cpp:408] data_data_0_split -> data_data_0_split_0
I0329 19:07:51.057158  2693 net.cpp:408] data_data_0_split -> data_data_0_split_1
I0329 19:07:51.057169  2693 net.cpp:408] data_data_0_split -> data_data_0_split_2
I0329 19:07:51.057179  2693 net.cpp:408] data_data_0_split -> data_data_0_split_3
I0329 19:07:51.057188  2693 net.cpp:408] data_data_0_split -> data_data_0_split_4
I0329 19:07:51.057198  2693 net.cpp:408] data_data_0_split -> data_data_0_split_5
I0329 19:07:51.057204  2693 net.cpp:408] data_data_0_split -> data_data_0_split_6
I0329 19:07:51.057382  2693 net.cpp:150] Setting up data_data_0_split
I0329 19:07:51.057410  2693 net.cpp:157] Top shape: 2 3 512 512 (1572864)
I0329 19:07:51.057417  2693 net.cpp:157] Top shape: 2 3 512 512 (1572864)
I0329 19:07:51.057423  2693 net.cpp:157] Top shape: 2 3 512 512 (1572864)
I0329 19:07:51.057430  2693 net.cpp:157] Top shape: 2 3 512 512 (1572864)
I0329 19:07:51.057435  2693 net.cpp:157] Top shape: 2 3 512 512 (1572864)
I0329 19:07:51.057440  2693 net.cpp:157] Top shape: 2 3 512 512 (1572864)
I0329 19:07:51.057446  2693 net.cpp:157] Top shape: 2 3 512 512 (1572864)
I0329 19:07:51.057451  2693 net.cpp:165] Memory required for data: 50331840
I0329 19:07:51.057457  2693 layer_factory.hpp:77] Creating layer conv1_1
I0329 19:07:51.057476  2693 net.cpp:100] Creating Layer conv1_1
I0329 19:07:51.057483  2693 net.cpp:434] conv1_1 <- data_data_0_split_0
I0329 19:07:51.057492  2693 net.cpp:408] conv1_1 -> conv1_1
I0329 19:07:51.059969  2693 net.cpp:150] Setting up conv1_1
I0329 19:07:51.059990  2693 net.cpp:157] Top shape: 2 64 512 512 (33554432)
I0329 19:07:51.059996  2693 net.cpp:165] Memory required for data: 184549568
I0329 19:07:51.060011  2693 layer_factory.hpp:77] Creating layer relu1_1
I0329 19:07:51.060022  2693 net.cpp:100] Creating Layer relu1_1
I0329 19:07:51.060029  2693 net.cpp:434] relu1_1 <- conv1_1
I0329 19:07:51.060035  2693 net.cpp:395] relu1_1 -> conv1_1 (in-place)
I0329 19:07:51.060405  2693 net.cpp:150] Setting up relu1_1
I0329 19:07:51.060423  2693 net.cpp:157] Top shape: 2 64 512 512 (33554432)
I0329 19:07:51.060430  2693 net.cpp:165] Memory required for data: 318767296
I0329 19:07:51.060434  2693 layer_factory.hpp:77] Creating layer conv1_2
I0329 19:07:51.060449  2693 net.cpp:100] Creating Layer conv1_2
I0329 19:07:51.060456  2693 net.cpp:434] conv1_2 <- conv1_1
I0329 19:07:51.060467  2693 net.cpp:408] conv1_2 -> conv1_2
I0329 19:07:51.062909  2693 net.cpp:150] Setting up conv1_2
I0329 19:07:51.062929  2693 net.cpp:157] Top shape: 2 64 512 512 (33554432)
I0329 19:07:51.062935  2693 net.cpp:165] Memory required for data: 452985024
I0329 19:07:51.062948  2693 layer_factory.hpp:77] Creating layer relu1_2
I0329 19:07:51.062958  2693 net.cpp:100] Creating Layer relu1_2
I0329 19:07:51.062963  2693 net.cpp:434] relu1_2 <- conv1_2
I0329 19:07:51.062971  2693 net.cpp:395] relu1_2 -> conv1_2 (in-place)
I0329 19:07:51.063326  2693 net.cpp:150] Setting up relu1_2
I0329 19:07:51.063344  2693 net.cpp:157] Top shape: 2 64 512 512 (33554432)
I0329 19:07:51.063350  2693 net.cpp:165] Memory required for data: 587202752
I0329 19:07:51.063355  2693 layer_factory.hpp:77] Creating layer pool1
I0329 19:07:51.063367  2693 net.cpp:100] Creating Layer pool1
I0329 19:07:51.063374  2693 net.cpp:434] pool1 <- conv1_2
I0329 19:07:51.063381  2693 net.cpp:408] pool1 -> pool1
I0329 19:07:51.063454  2693 net.cpp:150] Setting up pool1
I0329 19:07:51.063464  2693 net.cpp:157] Top shape: 2 64 256 256 (8388608)
I0329 19:07:51.063469  2693 net.cpp:165] Memory required for data: 620757184
I0329 19:07:51.063474  2693 layer_factory.hpp:77] Creating layer conv2_1
I0329 19:07:51.063488  2693 net.cpp:100] Creating Layer conv2_1
I0329 19:07:51.063495  2693 net.cpp:434] conv2_1 <- pool1
I0329 19:07:51.063505  2693 net.cpp:408] conv2_1 -> conv2_1
I0329 19:07:51.066124  2693 net.cpp:150] Setting up conv2_1
I0329 19:07:51.066144  2693 net.cpp:157] Top shape: 2 128 256 256 (16777216)
I0329 19:07:51.066150  2693 net.cpp:165] Memory required for data: 687866048
I0329 19:07:51.066162  2693 layer_factory.hpp:77] Creating layer relu2_1
I0329 19:07:51.066174  2693 net.cpp:100] Creating Layer relu2_1
I0329 19:07:51.066179  2693 net.cpp:434] relu2_1 <- conv2_1
I0329 19:07:51.066187  2693 net.cpp:395] relu2_1 -> conv2_1 (in-place)
I0329 19:07:51.066424  2693 net.cpp:150] Setting up relu2_1
I0329 19:07:51.066437  2693 net.cpp:157] Top shape: 2 128 256 256 (16777216)
I0329 19:07:51.066443  2693 net.cpp:165] Memory required for data: 754974912
I0329 19:07:51.066450  2693 layer_factory.hpp:77] Creating layer conv2_2
I0329 19:07:51.066463  2693 net.cpp:100] Creating Layer conv2_2
I0329 19:07:51.066485  2693 net.cpp:434] conv2_2 <- conv2_1
I0329 19:07:51.066498  2693 net.cpp:408] conv2_2 -> conv2_2
I0329 19:07:51.070397  2693 net.cpp:150] Setting up conv2_2
I0329 19:07:51.070420  2693 net.cpp:157] Top shape: 2 128 256 256 (16777216)
I0329 19:07:51.070425  2693 net.cpp:165] Memory required for data: 822083776
I0329 19:07:51.070436  2693 layer_factory.hpp:77] Creating layer relu2_2
I0329 19:07:51.070446  2693 net.cpp:100] Creating Layer relu2_2
I0329 19:07:51.070452  2693 net.cpp:434] relu2_2 <- conv2_2
I0329 19:07:51.070461  2693 net.cpp:395] relu2_2 -> conv2_2 (in-place)
I0329 19:07:51.070821  2693 net.cpp:150] Setting up relu2_2
I0329 19:07:51.070838  2693 net.cpp:157] Top shape: 2 128 256 256 (16777216)
I0329 19:07:51.070843  2693 net.cpp:165] Memory required for data: 889192640
I0329 19:07:51.070849  2693 layer_factory.hpp:77] Creating layer pool2
I0329 19:07:51.070859  2693 net.cpp:100] Creating Layer pool2
I0329 19:07:51.070865  2693 net.cpp:434] pool2 <- conv2_2
I0329 19:07:51.070873  2693 net.cpp:408] pool2 -> pool2
I0329 19:07:51.070943  2693 net.cpp:150] Setting up pool2
I0329 19:07:51.070955  2693 net.cpp:157] Top shape: 2 128 128 128 (4194304)
I0329 19:07:51.070960  2693 net.cpp:165] Memory required for data: 905969856
I0329 19:07:51.070965  2693 layer_factory.hpp:77] Creating layer conv3_1
I0329 19:07:51.070977  2693 net.cpp:100] Creating Layer conv3_1
I0329 19:07:51.070984  2693 net.cpp:434] conv3_1 <- pool2
I0329 19:07:51.070993  2693 net.cpp:408] conv3_1 -> conv3_1
I0329 19:07:51.075371  2693 net.cpp:150] Setting up conv3_1
I0329 19:07:51.075390  2693 net.cpp:157] Top shape: 2 256 128 128 (8388608)
I0329 19:07:51.075397  2693 net.cpp:165] Memory required for data: 939524288
I0329 19:07:51.075409  2693 layer_factory.hpp:77] Creating layer relu3_1
I0329 19:07:51.075418  2693 net.cpp:100] Creating Layer relu3_1
I0329 19:07:51.075424  2693 net.cpp:434] relu3_1 <- conv3_1
I0329 19:07:51.075434  2693 net.cpp:395] relu3_1 -> conv3_1 (in-place)
I0329 19:07:51.075809  2693 net.cpp:150] Setting up relu3_1
I0329 19:07:51.075827  2693 net.cpp:157] Top shape: 2 256 128 128 (8388608)
I0329 19:07:51.075832  2693 net.cpp:165] Memory required for data: 973078720
I0329 19:07:51.075839  2693 layer_factory.hpp:77] Creating layer conv3_2
I0329 19:07:51.075852  2693 net.cpp:100] Creating Layer conv3_2
I0329 19:07:51.075858  2693 net.cpp:434] conv3_2 <- conv3_1
I0329 19:07:51.075868  2693 net.cpp:408] conv3_2 -> conv3_2
I0329 19:07:51.083173  2693 net.cpp:150] Setting up conv3_2
I0329 19:07:51.083192  2693 net.cpp:157] Top shape: 2 256 128 128 (8388608)
I0329 19:07:51.083199  2693 net.cpp:165] Memory required for data: 1006633152
I0329 19:07:51.083207  2693 layer_factory.hpp:77] Creating layer relu3_2
I0329 19:07:51.083219  2693 net.cpp:100] Creating Layer relu3_2
I0329 19:07:51.083225  2693 net.cpp:434] relu3_2 <- conv3_2
I0329 19:07:51.083231  2693 net.cpp:395] relu3_2 -> conv3_2 (in-place)
I0329 19:07:51.083462  2693 net.cpp:150] Setting up relu3_2
I0329 19:07:51.083477  2693 net.cpp:157] Top shape: 2 256 128 128 (8388608)
I0329 19:07:51.083482  2693 net.cpp:165] Memory required for data: 1040187584
I0329 19:07:51.083487  2693 layer_factory.hpp:77] Creating layer conv3_3
I0329 19:07:51.083504  2693 net.cpp:100] Creating Layer conv3_3
I0329 19:07:51.083513  2693 net.cpp:434] conv3_3 <- conv3_2
I0329 19:07:51.083523  2693 net.cpp:408] conv3_3 -> conv3_3
I0329 19:07:51.091081  2693 net.cpp:150] Setting up conv3_3
I0329 19:07:51.091101  2693 net.cpp:157] Top shape: 2 256 128 128 (8388608)
I0329 19:07:51.091107  2693 net.cpp:165] Memory required for data: 1073742016
I0329 19:07:51.091116  2693 layer_factory.hpp:77] Creating layer relu3_3
I0329 19:07:51.091127  2693 net.cpp:100] Creating Layer relu3_3
I0329 19:07:51.091133  2693 net.cpp:434] relu3_3 <- conv3_3
I0329 19:07:51.091140  2693 net.cpp:395] relu3_3 -> conv3_3 (in-place)
I0329 19:07:51.091511  2693 net.cpp:150] Setting up relu3_3
I0329 19:07:51.091527  2693 net.cpp:157] Top shape: 2 256 128 128 (8388608)
I0329 19:07:51.091532  2693 net.cpp:165] Memory required for data: 1107296448
I0329 19:07:51.091552  2693 layer_factory.hpp:77] Creating layer pool3
I0329 19:07:51.091564  2693 net.cpp:100] Creating Layer pool3
I0329 19:07:51.091570  2693 net.cpp:434] pool3 <- conv3_3
I0329 19:07:51.091598  2693 net.cpp:408] pool3 -> pool3
I0329 19:07:51.091672  2693 net.cpp:150] Setting up pool3
I0329 19:07:51.091684  2693 net.cpp:157] Top shape: 2 256 64 64 (2097152)
I0329 19:07:51.091689  2693 net.cpp:165] Memory required for data: 1115685056
I0329 19:07:51.091694  2693 layer_factory.hpp:77] Creating layer conv4_1
I0329 19:07:51.091708  2693 net.cpp:100] Creating Layer conv4_1
I0329 19:07:51.091717  2693 net.cpp:434] conv4_1 <- pool3
I0329 19:07:51.091727  2693 net.cpp:408] conv4_1 -> conv4_1
I0329 19:07:51.104542  2693 net.cpp:150] Setting up conv4_1
I0329 19:07:51.104562  2693 net.cpp:157] Top shape: 2 512 64 64 (4194304)
I0329 19:07:51.104568  2693 net.cpp:165] Memory required for data: 1132462272
I0329 19:07:51.104578  2693 layer_factory.hpp:77] Creating layer relu4_1
I0329 19:07:51.104586  2693 net.cpp:100] Creating Layer relu4_1
I0329 19:07:51.104593  2693 net.cpp:434] relu4_1 <- conv4_1
I0329 19:07:51.104602  2693 net.cpp:395] relu4_1 -> conv4_1 (in-place)
I0329 19:07:51.104969  2693 net.cpp:150] Setting up relu4_1
I0329 19:07:51.104986  2693 net.cpp:157] Top shape: 2 512 64 64 (4194304)
I0329 19:07:51.104991  2693 net.cpp:165] Memory required for data: 1149239488
I0329 19:07:51.104997  2693 layer_factory.hpp:77] Creating layer conv4_2
I0329 19:07:51.105012  2693 net.cpp:100] Creating Layer conv4_2
I0329 19:07:51.105020  2693 net.cpp:434] conv4_2 <- conv4_1
I0329 19:07:51.105028  2693 net.cpp:408] conv4_2 -> conv4_2
I0329 19:07:51.129256  2693 net.cpp:150] Setting up conv4_2
I0329 19:07:51.129282  2693 net.cpp:157] Top shape: 2 512 64 64 (4194304)
I0329 19:07:51.129288  2693 net.cpp:165] Memory required for data: 1166016704
I0329 19:07:51.129307  2693 layer_factory.hpp:77] Creating layer relu4_2
I0329 19:07:51.129318  2693 net.cpp:100] Creating Layer relu4_2
I0329 19:07:51.129324  2693 net.cpp:434] relu4_2 <- conv4_2
I0329 19:07:51.129335  2693 net.cpp:395] relu4_2 -> conv4_2 (in-place)
I0329 19:07:51.129568  2693 net.cpp:150] Setting up relu4_2
I0329 19:07:51.129582  2693 net.cpp:157] Top shape: 2 512 64 64 (4194304)
I0329 19:07:51.129587  2693 net.cpp:165] Memory required for data: 1182793920
I0329 19:07:51.129593  2693 layer_factory.hpp:77] Creating layer conv4_3
I0329 19:07:51.129607  2693 net.cpp:100] Creating Layer conv4_3
I0329 19:07:51.129613  2693 net.cpp:434] conv4_3 <- conv4_2
I0329 19:07:51.129623  2693 net.cpp:408] conv4_3 -> conv4_3
I0329 19:07:51.154258  2693 net.cpp:150] Setting up conv4_3
I0329 19:07:51.154286  2693 net.cpp:157] Top shape: 2 512 64 64 (4194304)
I0329 19:07:51.154292  2693 net.cpp:165] Memory required for data: 1199571136
I0329 19:07:51.154304  2693 layer_factory.hpp:77] Creating layer relu4_3
I0329 19:07:51.154314  2693 net.cpp:100] Creating Layer relu4_3
I0329 19:07:51.154322  2693 net.cpp:434] relu4_3 <- conv4_3
I0329 19:07:51.154332  2693 net.cpp:395] relu4_3 -> conv4_3 (in-place)
I0329 19:07:51.154721  2693 net.cpp:150] Setting up relu4_3
I0329 19:07:51.154739  2693 net.cpp:157] Top shape: 2 512 64 64 (4194304)
I0329 19:07:51.154744  2693 net.cpp:165] Memory required for data: 1216348352
I0329 19:07:51.154750  2693 layer_factory.hpp:77] Creating layer conv4_3_relu4_3_0_split
I0329 19:07:51.154759  2693 net.cpp:100] Creating Layer conv4_3_relu4_3_0_split
I0329 19:07:51.154767  2693 net.cpp:434] conv4_3_relu4_3_0_split <- conv4_3
I0329 19:07:51.154774  2693 net.cpp:408] conv4_3_relu4_3_0_split -> conv4_3_relu4_3_0_split_0
I0329 19:07:51.154783  2693 net.cpp:408] conv4_3_relu4_3_0_split -> conv4_3_relu4_3_0_split_1
I0329 19:07:51.154860  2693 net.cpp:150] Setting up conv4_3_relu4_3_0_split
I0329 19:07:51.154870  2693 net.cpp:157] Top shape: 2 512 64 64 (4194304)
I0329 19:07:51.154876  2693 net.cpp:157] Top shape: 2 512 64 64 (4194304)
I0329 19:07:51.154881  2693 net.cpp:165] Memory required for data: 1249902784
I0329 19:07:51.154886  2693 layer_factory.hpp:77] Creating layer pool4
I0329 19:07:51.154914  2693 net.cpp:100] Creating Layer pool4
I0329 19:07:51.154923  2693 net.cpp:434] pool4 <- conv4_3_relu4_3_0_split_0
I0329 19:07:51.154932  2693 net.cpp:408] pool4 -> pool4
I0329 19:07:51.155001  2693 net.cpp:150] Setting up pool4
I0329 19:07:51.155011  2693 net.cpp:157] Top shape: 2 512 32 32 (1048576)
I0329 19:07:51.155016  2693 net.cpp:165] Memory required for data: 1254097088
I0329 19:07:51.155021  2693 layer_factory.hpp:77] Creating layer conv5_1
I0329 19:07:51.155038  2693 net.cpp:100] Creating Layer conv5_1
I0329 19:07:51.155046  2693 net.cpp:434] conv5_1 <- pool4
I0329 19:07:51.155055  2693 net.cpp:408] conv5_1 -> conv5_1
I0329 19:07:51.179420  2693 net.cpp:150] Setting up conv5_1
I0329 19:07:51.179453  2693 net.cpp:157] Top shape: 2 512 32 32 (1048576)
I0329 19:07:51.179460  2693 net.cpp:165] Memory required for data: 1258291392
I0329 19:07:51.179471  2693 layer_factory.hpp:77] Creating layer relu5_1
I0329 19:07:51.179482  2693 net.cpp:100] Creating Layer relu5_1
I0329 19:07:51.179489  2693 net.cpp:434] relu5_1 <- conv5_1
I0329 19:07:51.179500  2693 net.cpp:395] relu5_1 -> conv5_1 (in-place)
I0329 19:07:51.179898  2693 net.cpp:150] Setting up relu5_1
I0329 19:07:51.179915  2693 net.cpp:157] Top shape: 2 512 32 32 (1048576)
I0329 19:07:51.179920  2693 net.cpp:165] Memory required for data: 1262485696
I0329 19:07:51.179926  2693 layer_factory.hpp:77] Creating layer conv5_2
I0329 19:07:51.179944  2693 net.cpp:100] Creating Layer conv5_2
I0329 19:07:51.179951  2693 net.cpp:434] conv5_2 <- conv5_1
I0329 19:07:51.179960  2693 net.cpp:408] conv5_2 -> conv5_2
I0329 19:07:51.204296  2693 net.cpp:150] Setting up conv5_2
I0329 19:07:51.204324  2693 net.cpp:157] Top shape: 2 512 32 32 (1048576)
I0329 19:07:51.204329  2693 net.cpp:165] Memory required for data: 1266680000
I0329 19:07:51.204340  2693 layer_factory.hpp:77] Creating layer relu5_2
I0329 19:07:51.204354  2693 net.cpp:100] Creating Layer relu5_2
I0329 19:07:51.204360  2693 net.cpp:434] relu5_2 <- conv5_2
I0329 19:07:51.204368  2693 net.cpp:395] relu5_2 -> conv5_2 (in-place)
I0329 19:07:51.204598  2693 net.cpp:150] Setting up relu5_2
I0329 19:07:51.204612  2693 net.cpp:157] Top shape: 2 512 32 32 (1048576)
I0329 19:07:51.204617  2693 net.cpp:165] Memory required for data: 1270874304
I0329 19:07:51.204622  2693 layer_factory.hpp:77] Creating layer conv5_3
I0329 19:07:51.204638  2693 net.cpp:100] Creating Layer conv5_3
I0329 19:07:51.204645  2693 net.cpp:434] conv5_3 <- conv5_2
I0329 19:07:51.204655  2693 net.cpp:408] conv5_3 -> conv5_3
I0329 19:07:51.228909  2693 net.cpp:150] Setting up conv5_3
I0329 19:07:51.228940  2693 net.cpp:157] Top shape: 2 512 32 32 (1048576)
I0329 19:07:51.228945  2693 net.cpp:165] Memory required for data: 1275068608
I0329 19:07:51.228955  2693 layer_factory.hpp:77] Creating layer relu5_3
I0329 19:07:51.228974  2693 net.cpp:100] Creating Layer relu5_3
I0329 19:07:51.228981  2693 net.cpp:434] relu5_3 <- conv5_3
I0329 19:07:51.228989  2693 net.cpp:395] relu5_3 -> conv5_3 (in-place)
I0329 19:07:51.229364  2693 net.cpp:150] Setting up relu5_3
I0329 19:07:51.229382  2693 net.cpp:157] Top shape: 2 512 32 32 (1048576)
I0329 19:07:51.229387  2693 net.cpp:165] Memory required for data: 1279262912
I0329 19:07:51.229393  2693 layer_factory.hpp:77] Creating layer pool5
I0329 19:07:51.229405  2693 net.cpp:100] Creating Layer pool5
I0329 19:07:51.229411  2693 net.cpp:434] pool5 <- conv5_3
I0329 19:07:51.229421  2693 net.cpp:408] pool5 -> pool5
I0329 19:07:51.229498  2693 net.cpp:150] Setting up pool5
I0329 19:07:51.229509  2693 net.cpp:157] Top shape: 2 512 32 32 (1048576)
I0329 19:07:51.229514  2693 net.cpp:165] Memory required for data: 1283457216
I0329 19:07:51.229521  2693 layer_factory.hpp:77] Creating layer fc6
I0329 19:07:51.229537  2693 net.cpp:100] Creating Layer fc6
I0329 19:07:51.229545  2693 net.cpp:434] fc6 <- pool5
I0329 19:07:51.229554  2693 net.cpp:408] fc6 -> fc6
I0329 19:07:51.276545  2693 net.cpp:150] Setting up fc6
I0329 19:07:51.276587  2693 net.cpp:157] Top shape: 2 1024 32 32 (2097152)
I0329 19:07:51.276614  2693 net.cpp:165] Memory required for data: 1291845824
I0329 19:07:51.276628  2693 layer_factory.hpp:77] Creating layer relu6
I0329 19:07:51.276643  2693 net.cpp:100] Creating Layer relu6
I0329 19:07:51.276654  2693 net.cpp:434] relu6 <- fc6
I0329 19:07:51.276662  2693 net.cpp:395] relu6 -> fc6 (in-place)
I0329 19:07:51.276967  2693 net.cpp:150] Setting up relu6
I0329 19:07:51.276980  2693 net.cpp:157] Top shape: 2 1024 32 32 (2097152)
I0329 19:07:51.276986  2693 net.cpp:165] Memory required for data: 1300234432
I0329 19:07:51.276991  2693 layer_factory.hpp:77] Creating layer fc7
I0329 19:07:51.277007  2693 net.cpp:100] Creating Layer fc7
I0329 19:07:51.277015  2693 net.cpp:434] fc7 <- fc6
I0329 19:07:51.277024  2693 net.cpp:408] fc7 -> fc7
I0329 19:07:51.288765  2693 net.cpp:150] Setting up fc7
I0329 19:07:51.288786  2693 net.cpp:157] Top shape: 2 1024 32 32 (2097152)
I0329 19:07:51.288792  2693 net.cpp:165] Memory required for data: 1308623040
I0329 19:07:51.288802  2693 layer_factory.hpp:77] Creating layer relu7
I0329 19:07:51.288813  2693 net.cpp:100] Creating Layer relu7
I0329 19:07:51.288820  2693 net.cpp:434] relu7 <- fc7
I0329 19:07:51.288827  2693 net.cpp:395] relu7 -> fc7 (in-place)
I0329 19:07:51.289064  2693 net.cpp:150] Setting up relu7
I0329 19:07:51.289078  2693 net.cpp:157] Top shape: 2 1024 32 32 (2097152)
I0329 19:07:51.289084  2693 net.cpp:165] Memory required for data: 1317011648
I0329 19:07:51.289089  2693 layer_factory.hpp:77] Creating layer fc7_relu7_0_split
I0329 19:07:51.289101  2693 net.cpp:100] Creating Layer fc7_relu7_0_split
I0329 19:07:51.289110  2693 net.cpp:434] fc7_relu7_0_split <- fc7
I0329 19:07:51.289119  2693 net.cpp:408] fc7_relu7_0_split -> fc7_relu7_0_split_0
I0329 19:07:51.289127  2693 net.cpp:408] fc7_relu7_0_split -> fc7_relu7_0_split_1
I0329 19:07:51.289145  2693 net.cpp:408] fc7_relu7_0_split -> fc7_relu7_0_split_2
I0329 19:07:51.289156  2693 net.cpp:408] fc7_relu7_0_split -> fc7_relu7_0_split_3
I0329 19:07:51.289163  2693 net.cpp:408] fc7_relu7_0_split -> fc7_relu7_0_split_4
I0329 19:07:51.289296  2693 net.cpp:150] Setting up fc7_relu7_0_split
I0329 19:07:51.289306  2693 net.cpp:157] Top shape: 2 1024 32 32 (2097152)
I0329 19:07:51.289314  2693 net.cpp:157] Top shape: 2 1024 32 32 (2097152)
I0329 19:07:51.289319  2693 net.cpp:157] Top shape: 2 1024 32 32 (2097152)
I0329 19:07:51.289325  2693 net.cpp:157] Top shape: 2 1024 32 32 (2097152)
I0329 19:07:51.289330  2693 net.cpp:157] Top shape: 2 1024 32 32 (2097152)
I0329 19:07:51.289335  2693 net.cpp:165] Memory required for data: 1358954688
I0329 19:07:51.289340  2693 layer_factory.hpp:77] Creating layer conv6_1
I0329 19:07:51.289356  2693 net.cpp:100] Creating Layer conv6_1
I0329 19:07:51.289364  2693 net.cpp:434] conv6_1 <- fc7_relu7_0_split_0
I0329 19:07:51.289373  2693 net.cpp:408] conv6_1 -> conv6_1
I0329 19:07:51.293432  2693 net.cpp:150] Setting up conv6_1
I0329 19:07:51.293452  2693 net.cpp:157] Top shape: 2 256 32 32 (524288)
I0329 19:07:51.293458  2693 net.cpp:165] Memory required for data: 1361051840
I0329 19:07:51.293467  2693 layer_factory.hpp:77] Creating layer conv6_1_relu
I0329 19:07:51.293478  2693 net.cpp:100] Creating Layer conv6_1_relu
I0329 19:07:51.293484  2693 net.cpp:434] conv6_1_relu <- conv6_1
I0329 19:07:51.293493  2693 net.cpp:395] conv6_1_relu -> conv6_1 (in-place)
I0329 19:07:51.293871  2693 net.cpp:150] Setting up conv6_1_relu
I0329 19:07:51.293889  2693 net.cpp:157] Top shape: 2 256 32 32 (524288)
I0329 19:07:51.293893  2693 net.cpp:165] Memory required for data: 1363148992
I0329 19:07:51.293900  2693 layer_factory.hpp:77] Creating layer conv6_2
I0329 19:07:51.293917  2693 net.cpp:100] Creating Layer conv6_2
I0329 19:07:51.293923  2693 net.cpp:434] conv6_2 <- conv6_1
I0329 19:07:51.293933  2693 net.cpp:408] conv6_2 -> conv6_2
I0329 19:07:51.306810  2693 net.cpp:150] Setting up conv6_2
I0329 19:07:51.306830  2693 net.cpp:157] Top shape: 2 512 16 16 (262144)
I0329 19:07:51.306838  2693 net.cpp:165] Memory required for data: 1364197568
I0329 19:07:51.306869  2693 layer_factory.hpp:77] Creating layer conv6_2_relu
I0329 19:07:51.306879  2693 net.cpp:100] Creating Layer conv6_2_relu
I0329 19:07:51.306885  2693 net.cpp:434] conv6_2_relu <- conv6_2
I0329 19:07:51.306891  2693 net.cpp:395] conv6_2_relu -> conv6_2 (in-place)
I0329 19:07:51.307132  2693 net.cpp:150] Setting up conv6_2_relu
I0329 19:07:51.307148  2693 net.cpp:157] Top shape: 2 512 16 16 (262144)
I0329 19:07:51.307153  2693 net.cpp:165] Memory required for data: 1365246144
I0329 19:07:51.307159  2693 layer_factory.hpp:77] Creating layer conv6_2_conv6_2_relu_0_split
I0329 19:07:51.307168  2693 net.cpp:100] Creating Layer conv6_2_conv6_2_relu_0_split
I0329 19:07:51.307173  2693 net.cpp:434] conv6_2_conv6_2_relu_0_split <- conv6_2
I0329 19:07:51.307180  2693 net.cpp:408] conv6_2_conv6_2_relu_0_split -> conv6_2_conv6_2_relu_0_split_0
I0329 19:07:51.307189  2693 net.cpp:408] conv6_2_conv6_2_relu_0_split -> conv6_2_conv6_2_relu_0_split_1
I0329 19:07:51.307200  2693 net.cpp:408] conv6_2_conv6_2_relu_0_split -> conv6_2_conv6_2_relu_0_split_2
I0329 19:07:51.307214  2693 net.cpp:408] conv6_2_conv6_2_relu_0_split -> conv6_2_conv6_2_relu_0_split_3
I0329 19:07:51.307222  2693 net.cpp:408] conv6_2_conv6_2_relu_0_split -> conv6_2_conv6_2_relu_0_split_4
I0329 19:07:51.307353  2693 net.cpp:150] Setting up conv6_2_conv6_2_relu_0_split
I0329 19:07:51.307363  2693 net.cpp:157] Top shape: 2 512 16 16 (262144)
I0329 19:07:51.307369  2693 net.cpp:157] Top shape: 2 512 16 16 (262144)
I0329 19:07:51.307374  2693 net.cpp:157] Top shape: 2 512 16 16 (262144)
I0329 19:07:51.307380  2693 net.cpp:157] Top shape: 2 512 16 16 (262144)
I0329 19:07:51.307385  2693 net.cpp:157] Top shape: 2 512 16 16 (262144)
I0329 19:07:51.307390  2693 net.cpp:165] Memory required for data: 1370489024
I0329 19:07:51.307395  2693 layer_factory.hpp:77] Creating layer conv7_1
I0329 19:07:51.307409  2693 net.cpp:100] Creating Layer conv7_1
I0329 19:07:51.307417  2693 net.cpp:434] conv7_1 <- conv6_2_conv6_2_relu_0_split_0
I0329 19:07:51.307427  2693 net.cpp:408] conv7_1 -> conv7_1
I0329 19:07:51.309583  2693 net.cpp:150] Setting up conv7_1
I0329 19:07:51.309602  2693 net.cpp:157] Top shape: 2 128 16 16 (65536)
I0329 19:07:51.309608  2693 net.cpp:165] Memory required for data: 1370751168
I0329 19:07:51.309617  2693 layer_factory.hpp:77] Creating layer conv7_1_relu
I0329 19:07:51.309625  2693 net.cpp:100] Creating Layer conv7_1_relu
I0329 19:07:51.309631  2693 net.cpp:434] conv7_1_relu <- conv7_1
I0329 19:07:51.309641  2693 net.cpp:395] conv7_1_relu -> conv7_1 (in-place)
I0329 19:07:51.309869  2693 net.cpp:150] Setting up conv7_1_relu
I0329 19:07:51.309885  2693 net.cpp:157] Top shape: 2 128 16 16 (65536)
I0329 19:07:51.309890  2693 net.cpp:165] Memory required for data: 1371013312
I0329 19:07:51.309895  2693 layer_factory.hpp:77] Creating layer conv7_2
I0329 19:07:51.309908  2693 net.cpp:100] Creating Layer conv7_2
I0329 19:07:51.309914  2693 net.cpp:434] conv7_2 <- conv7_1
I0329 19:07:51.309924  2693 net.cpp:408] conv7_2 -> conv7_2
I0329 19:07:51.314288  2693 net.cpp:150] Setting up conv7_2
I0329 19:07:51.314308  2693 net.cpp:157] Top shape: 2 256 8 8 (32768)
I0329 19:07:51.314314  2693 net.cpp:165] Memory required for data: 1371144384
I0329 19:07:51.314323  2693 layer_factory.hpp:77] Creating layer conv7_2_relu
I0329 19:07:51.314334  2693 net.cpp:100] Creating Layer conv7_2_relu
I0329 19:07:51.314340  2693 net.cpp:434] conv7_2_relu <- conv7_2
I0329 19:07:51.314349  2693 net.cpp:395] conv7_2_relu -> conv7_2 (in-place)
I0329 19:07:51.314728  2693 net.cpp:150] Setting up conv7_2_relu
I0329 19:07:51.314745  2693 net.cpp:157] Top shape: 2 256 8 8 (32768)
I0329 19:07:51.314751  2693 net.cpp:165] Memory required for data: 1371275456
I0329 19:07:51.314756  2693 layer_factory.hpp:77] Creating layer conv7_2_conv7_2_relu_0_split
I0329 19:07:51.314765  2693 net.cpp:100] Creating Layer conv7_2_conv7_2_relu_0_split
I0329 19:07:51.314771  2693 net.cpp:434] conv7_2_conv7_2_relu_0_split <- conv7_2
I0329 19:07:51.314781  2693 net.cpp:408] conv7_2_conv7_2_relu_0_split -> conv7_2_conv7_2_relu_0_split_0
I0329 19:07:51.314803  2693 net.cpp:408] conv7_2_conv7_2_relu_0_split -> conv7_2_conv7_2_relu_0_split_1
I0329 19:07:51.314812  2693 net.cpp:408] conv7_2_conv7_2_relu_0_split -> conv7_2_conv7_2_relu_0_split_2
I0329 19:07:51.314824  2693 net.cpp:408] conv7_2_conv7_2_relu_0_split -> conv7_2_conv7_2_relu_0_split_3
I0329 19:07:51.314832  2693 net.cpp:408] conv7_2_conv7_2_relu_0_split -> conv7_2_conv7_2_relu_0_split_4
I0329 19:07:51.314966  2693 net.cpp:150] Setting up conv7_2_conv7_2_relu_0_split
I0329 19:07:51.314977  2693 net.cpp:157] Top shape: 2 256 8 8 (32768)
I0329 19:07:51.314983  2693 net.cpp:157] Top shape: 2 256 8 8 (32768)
I0329 19:07:51.314988  2693 net.cpp:157] Top shape: 2 256 8 8 (32768)
I0329 19:07:51.314995  2693 net.cpp:157] Top shape: 2 256 8 8 (32768)
I0329 19:07:51.314999  2693 net.cpp:157] Top shape: 2 256 8 8 (32768)
I0329 19:07:51.315004  2693 net.cpp:165] Memory required for data: 1371930816
I0329 19:07:51.315009  2693 layer_factory.hpp:77] Creating layer conv8_1
I0329 19:07:51.315024  2693 net.cpp:100] Creating Layer conv8_1
I0329 19:07:51.315032  2693 net.cpp:434] conv8_1 <- conv7_2_conv7_2_relu_0_split_0
I0329 19:07:51.315042  2693 net.cpp:408] conv8_1 -> conv8_1
I0329 19:07:51.316717  2693 net.cpp:150] Setting up conv8_1
I0329 19:07:51.316736  2693 net.cpp:157] Top shape: 2 128 8 8 (16384)
I0329 19:07:51.316742  2693 net.cpp:165] Memory required for data: 1371996352
I0329 19:07:51.316751  2693 layer_factory.hpp:77] Creating layer conv8_1_relu
I0329 19:07:51.316759  2693 net.cpp:100] Creating Layer conv8_1_relu
I0329 19:07:51.316766  2693 net.cpp:434] conv8_1_relu <- conv8_1
I0329 19:07:51.316776  2693 net.cpp:395] conv8_1_relu -> conv8_1 (in-place)
I0329 19:07:51.316999  2693 net.cpp:150] Setting up conv8_1_relu
I0329 19:07:51.317013  2693 net.cpp:157] Top shape: 2 128 8 8 (16384)
I0329 19:07:51.317018  2693 net.cpp:165] Memory required for data: 1372061888
I0329 19:07:51.317023  2693 layer_factory.hpp:77] Creating layer conv8_2
I0329 19:07:51.317039  2693 net.cpp:100] Creating Layer conv8_2
I0329 19:07:51.317049  2693 net.cpp:434] conv8_2 <- conv8_1
I0329 19:07:51.317056  2693 net.cpp:408] conv8_2 -> conv8_2
I0329 19:07:51.321452  2693 net.cpp:150] Setting up conv8_2
I0329 19:07:51.321471  2693 net.cpp:157] Top shape: 2 256 6 6 (18432)
I0329 19:07:51.321477  2693 net.cpp:165] Memory required for data: 1372135616
I0329 19:07:51.321486  2693 layer_factory.hpp:77] Creating layer conv8_2_relu
I0329 19:07:51.321496  2693 net.cpp:100] Creating Layer conv8_2_relu
I0329 19:07:51.321501  2693 net.cpp:434] conv8_2_relu <- conv8_2
I0329 19:07:51.321509  2693 net.cpp:395] conv8_2_relu -> conv8_2 (in-place)
I0329 19:07:51.321737  2693 net.cpp:150] Setting up conv8_2_relu
I0329 19:07:51.321751  2693 net.cpp:157] Top shape: 2 256 6 6 (18432)
I0329 19:07:51.321756  2693 net.cpp:165] Memory required for data: 1372209344
I0329 19:07:51.321761  2693 layer_factory.hpp:77] Creating layer conv8_2_conv8_2_relu_0_split
I0329 19:07:51.321772  2693 net.cpp:100] Creating Layer conv8_2_conv8_2_relu_0_split
I0329 19:07:51.321779  2693 net.cpp:434] conv8_2_conv8_2_relu_0_split <- conv8_2
I0329 19:07:51.321785  2693 net.cpp:408] conv8_2_conv8_2_relu_0_split -> conv8_2_conv8_2_relu_0_split_0
I0329 19:07:51.321799  2693 net.cpp:408] conv8_2_conv8_2_relu_0_split -> conv8_2_conv8_2_relu_0_split_1
I0329 19:07:51.321810  2693 net.cpp:408] conv8_2_conv8_2_relu_0_split -> conv8_2_conv8_2_relu_0_split_2
I0329 19:07:51.321818  2693 net.cpp:408] conv8_2_conv8_2_relu_0_split -> conv8_2_conv8_2_relu_0_split_3
I0329 19:07:51.321830  2693 net.cpp:408] conv8_2_conv8_2_relu_0_split -> conv8_2_conv8_2_relu_0_split_4
I0329 19:07:51.321966  2693 net.cpp:150] Setting up conv8_2_conv8_2_relu_0_split
I0329 19:07:51.321977  2693 net.cpp:157] Top shape: 2 256 6 6 (18432)
I0329 19:07:51.321983  2693 net.cpp:157] Top shape: 2 256 6 6 (18432)
I0329 19:07:51.321990  2693 net.cpp:157] Top shape: 2 256 6 6 (18432)
I0329 19:07:51.321995  2693 net.cpp:157] Top shape: 2 256 6 6 (18432)
I0329 19:07:51.322000  2693 net.cpp:157] Top shape: 2 256 6 6 (18432)
I0329 19:07:51.322016  2693 net.cpp:165] Memory required for data: 1372577984
I0329 19:07:51.322022  2693 layer_factory.hpp:77] Creating layer conv9_1
I0329 19:07:51.322036  2693 net.cpp:100] Creating Layer conv9_1
I0329 19:07:51.322044  2693 net.cpp:434] conv9_1 <- conv8_2_conv8_2_relu_0_split_0
I0329 19:07:51.322054  2693 net.cpp:408] conv9_1 -> conv9_1
I0329 19:07:51.323730  2693 net.cpp:150] Setting up conv9_1
I0329 19:07:51.323750  2693 net.cpp:157] Top shape: 2 128 6 6 (9216)
I0329 19:07:51.323755  2693 net.cpp:165] Memory required for data: 1372614848
I0329 19:07:51.323763  2693 layer_factory.hpp:77] Creating layer conv9_1_relu
I0329 19:07:51.323774  2693 net.cpp:100] Creating Layer conv9_1_relu
I0329 19:07:51.323781  2693 net.cpp:434] conv9_1_relu <- conv9_1
I0329 19:07:51.323788  2693 net.cpp:395] conv9_1_relu -> conv9_1 (in-place)
I0329 19:07:51.324170  2693 net.cpp:150] Setting up conv9_1_relu
I0329 19:07:51.324189  2693 net.cpp:157] Top shape: 2 128 6 6 (9216)
I0329 19:07:51.324195  2693 net.cpp:165] Memory required for data: 1372651712
I0329 19:07:51.324201  2693 layer_factory.hpp:77] Creating layer conv9_2
I0329 19:07:51.324215  2693 net.cpp:100] Creating Layer conv9_2
I0329 19:07:51.324221  2693 net.cpp:434] conv9_2 <- conv9_1
I0329 19:07:51.324236  2693 net.cpp:408] conv9_2 -> conv9_2
I0329 19:07:51.328721  2693 net.cpp:150] Setting up conv9_2
I0329 19:07:51.328740  2693 net.cpp:157] Top shape: 2 256 4 4 (8192)
I0329 19:07:51.328747  2693 net.cpp:165] Memory required for data: 1372684480
I0329 19:07:51.328755  2693 layer_factory.hpp:77] Creating layer conv9_2_relu
I0329 19:07:51.328764  2693 net.cpp:100] Creating Layer conv9_2_relu
I0329 19:07:51.328770  2693 net.cpp:434] conv9_2_relu <- conv9_2
I0329 19:07:51.328778  2693 net.cpp:395] conv9_2_relu -> conv9_2 (in-place)
I0329 19:07:51.328992  2693 net.cpp:150] Setting up conv9_2_relu
I0329 19:07:51.329006  2693 net.cpp:157] Top shape: 2 256 4 4 (8192)
I0329 19:07:51.329011  2693 net.cpp:165] Memory required for data: 1372717248
I0329 19:07:51.329017  2693 layer_factory.hpp:77] Creating layer conv9_2_conv9_2_relu_0_split
I0329 19:07:51.329025  2693 net.cpp:100] Creating Layer conv9_2_conv9_2_relu_0_split
I0329 19:07:51.329031  2693 net.cpp:434] conv9_2_conv9_2_relu_0_split <- conv9_2
I0329 19:07:51.329040  2693 net.cpp:408] conv9_2_conv9_2_relu_0_split -> conv9_2_conv9_2_relu_0_split_0
I0329 19:07:51.329051  2693 net.cpp:408] conv9_2_conv9_2_relu_0_split -> conv9_2_conv9_2_relu_0_split_1
I0329 19:07:51.329061  2693 net.cpp:408] conv9_2_conv9_2_relu_0_split -> conv9_2_conv9_2_relu_0_split_2
I0329 19:07:51.329073  2693 net.cpp:408] conv9_2_conv9_2_relu_0_split -> conv9_2_conv9_2_relu_0_split_3
I0329 19:07:51.329200  2693 net.cpp:150] Setting up conv9_2_conv9_2_relu_0_split
I0329 19:07:51.329210  2693 net.cpp:157] Top shape: 2 256 4 4 (8192)
I0329 19:07:51.329216  2693 net.cpp:157] Top shape: 2 256 4 4 (8192)
I0329 19:07:51.329221  2693 net.cpp:157] Top shape: 2 256 4 4 (8192)
I0329 19:07:51.329226  2693 net.cpp:157] Top shape: 2 256 4 4 (8192)
I0329 19:07:51.329231  2693 net.cpp:165] Memory required for data: 1372848320
I0329 19:07:51.329236  2693 layer_factory.hpp:77] Creating layer conv4_3_norm
I0329 19:07:51.329248  2693 net.cpp:100] Creating Layer conv4_3_norm
I0329 19:07:51.329257  2693 net.cpp:434] conv4_3_norm <- conv4_3_relu4_3_0_split_1
I0329 19:07:51.329265  2693 net.cpp:408] conv4_3_norm -> conv4_3_norm
I0329 19:07:51.329524  2693 net.cpp:150] Setting up conv4_3_norm
I0329 19:07:51.329535  2693 net.cpp:157] Top shape: 2 512 64 64 (4194304)
I0329 19:07:51.329540  2693 net.cpp:165] Memory required for data: 1389625536
I0329 19:07:51.329546  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_conv4_3_norm_0_split
I0329 19:07:51.329556  2693 net.cpp:100] Creating Layer conv4_3_norm_conv4_3_norm_0_split
I0329 19:07:51.329562  2693 net.cpp:434] conv4_3_norm_conv4_3_norm_0_split <- conv4_3_norm
I0329 19:07:51.329568  2693 net.cpp:408] conv4_3_norm_conv4_3_norm_0_split -> conv4_3_norm_conv4_3_norm_0_split_0
I0329 19:07:51.329583  2693 net.cpp:408] conv4_3_norm_conv4_3_norm_0_split -> conv4_3_norm_conv4_3_norm_0_split_1
I0329 19:07:51.329604  2693 net.cpp:408] conv4_3_norm_conv4_3_norm_0_split -> conv4_3_norm_conv4_3_norm_0_split_2
I0329 19:07:51.329617  2693 net.cpp:408] conv4_3_norm_conv4_3_norm_0_split -> conv4_3_norm_conv4_3_norm_0_split_3
I0329 19:07:51.329722  2693 net.cpp:150] Setting up conv4_3_norm_conv4_3_norm_0_split
I0329 19:07:51.329733  2693 net.cpp:157] Top shape: 2 512 64 64 (4194304)
I0329 19:07:51.329740  2693 net.cpp:157] Top shape: 2 512 64 64 (4194304)
I0329 19:07:51.329744  2693 net.cpp:157] Top shape: 2 512 64 64 (4194304)
I0329 19:07:51.329751  2693 net.cpp:157] Top shape: 2 512 64 64 (4194304)
I0329 19:07:51.329754  2693 net.cpp:165] Memory required for data: 1456734400
I0329 19:07:51.329761  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_loc
I0329 19:07:51.329777  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_loc
I0329 19:07:51.329784  2693 net.cpp:434] conv4_3_norm_mbox_loc <- conv4_3_norm_conv4_3_norm_0_split_0
I0329 19:07:51.329792  2693 net.cpp:408] conv4_3_norm_mbox_loc -> conv4_3_norm_mbox_loc
I0329 19:07:51.332029  2693 net.cpp:150] Setting up conv4_3_norm_mbox_loc
I0329 19:07:51.332053  2693 net.cpp:157] Top shape: 2 16 64 64 (131072)
I0329 19:07:51.332059  2693 net.cpp:165] Memory required for data: 1457258688
I0329 19:07:51.332068  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_loc_perm
I0329 19:07:51.332079  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_loc_perm
I0329 19:07:51.332088  2693 net.cpp:434] conv4_3_norm_mbox_loc_perm <- conv4_3_norm_mbox_loc
I0329 19:07:51.332096  2693 net.cpp:408] conv4_3_norm_mbox_loc_perm -> conv4_3_norm_mbox_loc_perm
I0329 19:07:51.332285  2693 net.cpp:150] Setting up conv4_3_norm_mbox_loc_perm
I0329 19:07:51.332296  2693 net.cpp:157] Top shape: 2 64 64 16 (131072)
I0329 19:07:51.332301  2693 net.cpp:165] Memory required for data: 1457782976
I0329 19:07:51.332307  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_loc_flat
I0329 19:07:51.332319  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_loc_flat
I0329 19:07:51.332326  2693 net.cpp:434] conv4_3_norm_mbox_loc_flat <- conv4_3_norm_mbox_loc_perm
I0329 19:07:51.332334  2693 net.cpp:408] conv4_3_norm_mbox_loc_flat -> conv4_3_norm_mbox_loc_flat
I0329 19:07:51.332379  2693 net.cpp:150] Setting up conv4_3_norm_mbox_loc_flat
I0329 19:07:51.332389  2693 net.cpp:157] Top shape: 2 65536 (131072)
I0329 19:07:51.332394  2693 net.cpp:165] Memory required for data: 1458307264
I0329 19:07:51.332399  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_conf
I0329 19:07:51.332427  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_conf
I0329 19:07:51.332434  2693 net.cpp:434] conv4_3_norm_mbox_conf <- conv4_3_norm_conv4_3_norm_0_split_1
I0329 19:07:51.332445  2693 net.cpp:408] conv4_3_norm_mbox_conf -> conv4_3_norm_mbox_conf
I0329 19:07:51.334920  2693 net.cpp:150] Setting up conv4_3_norm_mbox_conf
I0329 19:07:51.334940  2693 net.cpp:157] Top shape: 2 8 64 64 (65536)
I0329 19:07:51.334945  2693 net.cpp:165] Memory required for data: 1458569408
I0329 19:07:51.334955  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_conf_perm
I0329 19:07:51.334966  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_conf_perm
I0329 19:07:51.334974  2693 net.cpp:434] conv4_3_norm_mbox_conf_perm <- conv4_3_norm_mbox_conf
I0329 19:07:51.334982  2693 net.cpp:408] conv4_3_norm_mbox_conf_perm -> conv4_3_norm_mbox_conf_perm
I0329 19:07:51.335170  2693 net.cpp:150] Setting up conv4_3_norm_mbox_conf_perm
I0329 19:07:51.335181  2693 net.cpp:157] Top shape: 2 64 64 8 (65536)
I0329 19:07:51.335186  2693 net.cpp:165] Memory required for data: 1458831552
I0329 19:07:51.335191  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_conf_flat
I0329 19:07:51.335201  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_conf_flat
I0329 19:07:51.335207  2693 net.cpp:434] conv4_3_norm_mbox_conf_flat <- conv4_3_norm_mbox_conf_perm
I0329 19:07:51.335216  2693 net.cpp:408] conv4_3_norm_mbox_conf_flat -> conv4_3_norm_mbox_conf_flat
I0329 19:07:51.335283  2693 net.cpp:150] Setting up conv4_3_norm_mbox_conf_flat
I0329 19:07:51.335295  2693 net.cpp:157] Top shape: 2 32768 (65536)
I0329 19:07:51.335300  2693 net.cpp:165] Memory required for data: 1459093696
I0329 19:07:51.335305  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_clean
I0329 19:07:51.335319  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_clean
I0329 19:07:51.335327  2693 net.cpp:434] conv4_3_norm_mbox_clean <- conv4_3_norm_conv4_3_norm_0_split_2
I0329 19:07:51.335340  2693 net.cpp:408] conv4_3_norm_mbox_clean -> conv4_3_norm_mbox_clean
I0329 19:07:51.337193  2693 net.cpp:150] Setting up conv4_3_norm_mbox_clean
I0329 19:07:51.337213  2693 net.cpp:157] Top shape: 2 4 64 64 (32768)
I0329 19:07:51.337218  2693 net.cpp:165] Memory required for data: 1459224768
I0329 19:07:51.337227  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_clean_perm
I0329 19:07:51.337239  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_clean_perm
I0329 19:07:51.337249  2693 net.cpp:434] conv4_3_norm_mbox_clean_perm <- conv4_3_norm_mbox_clean
I0329 19:07:51.337257  2693 net.cpp:408] conv4_3_norm_mbox_clean_perm -> conv4_3_norm_mbox_clean_perm
I0329 19:07:51.337445  2693 net.cpp:150] Setting up conv4_3_norm_mbox_clean_perm
I0329 19:07:51.337457  2693 net.cpp:157] Top shape: 2 64 64 4 (32768)
I0329 19:07:51.337462  2693 net.cpp:165] Memory required for data: 1459355840
I0329 19:07:51.337467  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_clean_flat
I0329 19:07:51.337476  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_clean_flat
I0329 19:07:51.337481  2693 net.cpp:434] conv4_3_norm_mbox_clean_flat <- conv4_3_norm_mbox_clean_perm
I0329 19:07:51.337491  2693 net.cpp:408] conv4_3_norm_mbox_clean_flat -> conv4_3_norm_mbox_clean_flat
I0329 19:07:51.337534  2693 net.cpp:150] Setting up conv4_3_norm_mbox_clean_flat
I0329 19:07:51.337545  2693 net.cpp:157] Top shape: 2 16384 (32768)
I0329 19:07:51.337549  2693 net.cpp:165] Memory required for data: 1459486912
I0329 19:07:51.337554  2693 layer_factory.hpp:77] Creating layer conv4_3_norm_mbox_priorbox
I0329 19:07:51.337568  2693 net.cpp:100] Creating Layer conv4_3_norm_mbox_priorbox
I0329 19:07:51.337575  2693 net.cpp:434] conv4_3_norm_mbox_priorbox <- conv4_3_norm_conv4_3_norm_0_split_3
I0329 19:07:51.337582  2693 net.cpp:434] conv4_3_norm_mbox_priorbox <- data_data_0_split_1
I0329 19:07:51.337591  2693 net.cpp:408] conv4_3_norm_mbox_priorbox -> conv4_3_norm_mbox_priorbox
I0329 19:07:51.337640  2693 net.cpp:150] Setting up conv4_3_norm_mbox_priorbox
I0329 19:07:51.337649  2693 net.cpp:157] Top shape: 1 2 65536 (131072)
I0329 19:07:51.337654  2693 net.cpp:165] Memory required for data: 1460011200
I0329 19:07:51.337659  2693 layer_factory.hpp:77] Creating layer fc7_mbox_loc
I0329 19:07:51.337677  2693 net.cpp:100] Creating Layer fc7_mbox_loc
I0329 19:07:51.337685  2693 net.cpp:434] fc7_mbox_loc <- fc7_relu7_0_split_1
I0329 19:07:51.337693  2693 net.cpp:408] fc7_mbox_loc -> fc7_mbox_loc
I0329 19:07:51.341531  2693 net.cpp:150] Setting up fc7_mbox_loc
I0329 19:07:51.341552  2693 net.cpp:157] Top shape: 2 24 32 32 (49152)
I0329 19:07:51.341557  2693 net.cpp:165] Memory required for data: 1460207808
I0329 19:07:51.341567  2693 layer_factory.hpp:77] Creating layer fc7_mbox_loc_perm
I0329 19:07:51.341578  2693 net.cpp:100] Creating Layer fc7_mbox_loc_perm
I0329 19:07:51.341588  2693 net.cpp:434] fc7_mbox_loc_perm <- fc7_mbox_loc
I0329 19:07:51.341598  2693 net.cpp:408] fc7_mbox_loc_perm -> fc7_mbox_loc_perm
I0329 19:07:51.341789  2693 net.cpp:150] Setting up fc7_mbox_loc_perm
I0329 19:07:51.341801  2693 net.cpp:157] Top shape: 2 32 32 24 (49152)
I0329 19:07:51.341806  2693 net.cpp:165] Memory required for data: 1460404416
I0329 19:07:51.341811  2693 layer_factory.hpp:77] Creating layer fc7_mbox_loc_flat
I0329 19:07:51.341820  2693 net.cpp:100] Creating Layer fc7_mbox_loc_flat
I0329 19:07:51.341825  2693 net.cpp:434] fc7_mbox_loc_flat <- fc7_mbox_loc_perm
I0329 19:07:51.341835  2693 net.cpp:408] fc7_mbox_loc_flat -> fc7_mbox_loc_flat
I0329 19:07:51.341893  2693 net.cpp:150] Setting up fc7_mbox_loc_flat
I0329 19:07:51.341904  2693 net.cpp:157] Top shape: 2 24576 (49152)
I0329 19:07:51.341909  2693 net.cpp:165] Memory required for data: 1460601024
I0329 19:07:51.341914  2693 layer_factory.hpp:77] Creating layer fc7_mbox_conf
I0329 19:07:51.341933  2693 net.cpp:100] Creating Layer fc7_mbox_conf
I0329 19:07:51.341940  2693 net.cpp:434] fc7_mbox_conf <- fc7_relu7_0_split_2
I0329 19:07:51.341949  2693 net.cpp:408] fc7_mbox_conf -> fc7_mbox_conf
I0329 19:07:51.344991  2693 net.cpp:150] Setting up fc7_mbox_conf
I0329 19:07:51.345011  2693 net.cpp:157] Top shape: 2 12 32 32 (24576)
I0329 19:07:51.345017  2693 net.cpp:165] Memory required for data: 1460699328
I0329 19:07:51.345026  2693 layer_factory.hpp:77] Creating layer fc7_mbox_conf_perm
I0329 19:07:51.345038  2693 net.cpp:100] Creating Layer fc7_mbox_conf_perm
I0329 19:07:51.345046  2693 net.cpp:434] fc7_mbox_conf_perm <- fc7_mbox_conf
I0329 19:07:51.345057  2693 net.cpp:408] fc7_mbox_conf_perm -> fc7_mbox_conf_perm
I0329 19:07:51.345252  2693 net.cpp:150] Setting up fc7_mbox_conf_perm
I0329 19:07:51.345264  2693 net.cpp:157] Top shape: 2 32 32 12 (24576)
I0329 19:07:51.345269  2693 net.cpp:165] Memory required for data: 1460797632
I0329 19:07:51.345274  2693 layer_factory.hpp:77] Creating layer fc7_mbox_conf_flat
I0329 19:07:51.345283  2693 net.cpp:100] Creating Layer fc7_mbox_conf_flat
I0329 19:07:51.345288  2693 net.cpp:434] fc7_mbox_conf_flat <- fc7_mbox_conf_perm
I0329 19:07:51.345297  2693 net.cpp:408] fc7_mbox_conf_flat -> fc7_mbox_conf_flat
I0329 19:07:51.345342  2693 net.cpp:150] Setting up fc7_mbox_conf_flat
I0329 19:07:51.345352  2693 net.cpp:157] Top shape: 2 12288 (24576)
I0329 19:07:51.345357  2693 net.cpp:165] Memory required for data: 1460895936
I0329 19:07:51.345362  2693 layer_factory.hpp:77] Creating layer fc7_mbox_clean
I0329 19:07:51.345379  2693 net.cpp:100] Creating Layer fc7_mbox_clean
I0329 19:07:51.345387  2693 net.cpp:434] fc7_mbox_clean <- fc7_relu7_0_split_3
I0329 19:07:51.345396  2693 net.cpp:408] fc7_mbox_clean -> fc7_mbox_clean
I0329 19:07:51.347460  2693 net.cpp:150] Setting up fc7_mbox_clean
I0329 19:07:51.347481  2693 net.cpp:157] Top shape: 2 6 32 32 (12288)
I0329 19:07:51.347487  2693 net.cpp:165] Memory required for data: 1460945088
I0329 19:07:51.347497  2693 layer_factory.hpp:77] Creating layer fc7_mbox_clean_perm
I0329 19:07:51.347506  2693 net.cpp:100] Creating Layer fc7_mbox_clean_perm
I0329 19:07:51.347513  2693 net.cpp:434] fc7_mbox_clean_perm <- fc7_mbox_clean
I0329 19:07:51.347520  2693 net.cpp:408] fc7_mbox_clean_perm -> fc7_mbox_clean_perm
I0329 19:07:51.347728  2693 net.cpp:150] Setting up fc7_mbox_clean_perm
I0329 19:07:51.347740  2693 net.cpp:157] Top shape: 2 32 32 6 (12288)
I0329 19:07:51.347746  2693 net.cpp:165] Memory required for data: 1460994240
I0329 19:07:51.347751  2693 layer_factory.hpp:77] Creating layer fc7_mbox_clean_flat
I0329 19:07:51.347761  2693 net.cpp:100] Creating Layer fc7_mbox_clean_flat
I0329 19:07:51.347769  2693 net.cpp:434] fc7_mbox_clean_flat <- fc7_mbox_clean_perm
I0329 19:07:51.347776  2693 net.cpp:408] fc7_mbox_clean_flat -> fc7_mbox_clean_flat
I0329 19:07:51.347822  2693 net.cpp:150] Setting up fc7_mbox_clean_flat
I0329 19:07:51.347833  2693 net.cpp:157] Top shape: 2 6144 (12288)
I0329 19:07:51.347837  2693 net.cpp:165] Memory required for data: 1461043392
I0329 19:07:51.347843  2693 layer_factory.hpp:77] Creating layer fc7_mbox_priorbox
I0329 19:07:51.347852  2693 net.cpp:100] Creating Layer fc7_mbox_priorbox
I0329 19:07:51.347861  2693 net.cpp:434] fc7_mbox_priorbox <- fc7_relu7_0_split_4
I0329 19:07:51.347867  2693 net.cpp:434] fc7_mbox_priorbox <- data_data_0_split_2
I0329 19:07:51.347875  2693 net.cpp:408] fc7_mbox_priorbox -> fc7_mbox_priorbox
I0329 19:07:51.347926  2693 net.cpp:150] Setting up fc7_mbox_priorbox
I0329 19:07:51.347937  2693 net.cpp:157] Top shape: 1 2 24576 (49152)
I0329 19:07:51.347942  2693 net.cpp:165] Memory required for data: 1461240000
I0329 19:07:51.347947  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_loc
I0329 19:07:51.347975  2693 net.cpp:100] Creating Layer conv6_2_mbox_loc
I0329 19:07:51.347985  2693 net.cpp:434] conv6_2_mbox_loc <- conv6_2_conv6_2_relu_0_split_1
I0329 19:07:51.347996  2693 net.cpp:408] conv6_2_mbox_loc -> conv6_2_mbox_loc
I0329 19:07:51.350538  2693 net.cpp:150] Setting up conv6_2_mbox_loc
I0329 19:07:51.350558  2693 net.cpp:157] Top shape: 2 24 16 16 (12288)
I0329 19:07:51.350564  2693 net.cpp:165] Memory required for data: 1461289152
I0329 19:07:51.350574  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_loc_perm
I0329 19:07:51.350586  2693 net.cpp:100] Creating Layer conv6_2_mbox_loc_perm
I0329 19:07:51.350595  2693 net.cpp:434] conv6_2_mbox_loc_perm <- conv6_2_mbox_loc
I0329 19:07:51.350603  2693 net.cpp:408] conv6_2_mbox_loc_perm -> conv6_2_mbox_loc_perm
I0329 19:07:51.350795  2693 net.cpp:150] Setting up conv6_2_mbox_loc_perm
I0329 19:07:51.350806  2693 net.cpp:157] Top shape: 2 16 16 24 (12288)
I0329 19:07:51.350812  2693 net.cpp:165] Memory required for data: 1461338304
I0329 19:07:51.350817  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_loc_flat
I0329 19:07:51.350826  2693 net.cpp:100] Creating Layer conv6_2_mbox_loc_flat
I0329 19:07:51.350831  2693 net.cpp:434] conv6_2_mbox_loc_flat <- conv6_2_mbox_loc_perm
I0329 19:07:51.350839  2693 net.cpp:408] conv6_2_mbox_loc_flat -> conv6_2_mbox_loc_flat
I0329 19:07:51.350886  2693 net.cpp:150] Setting up conv6_2_mbox_loc_flat
I0329 19:07:51.350899  2693 net.cpp:157] Top shape: 2 6144 (12288)
I0329 19:07:51.350904  2693 net.cpp:165] Memory required for data: 1461387456
I0329 19:07:51.350909  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_conf
I0329 19:07:51.350922  2693 net.cpp:100] Creating Layer conv6_2_mbox_conf
I0329 19:07:51.350931  2693 net.cpp:434] conv6_2_mbox_conf <- conv6_2_conv6_2_relu_0_split_2
I0329 19:07:51.350940  2693 net.cpp:408] conv6_2_mbox_conf -> conv6_2_mbox_conf
I0329 19:07:51.353008  2693 net.cpp:150] Setting up conv6_2_mbox_conf
I0329 19:07:51.353027  2693 net.cpp:157] Top shape: 2 12 16 16 (6144)
I0329 19:07:51.353034  2693 net.cpp:165] Memory required for data: 1461412032
I0329 19:07:51.353042  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_conf_perm
I0329 19:07:51.353055  2693 net.cpp:100] Creating Layer conv6_2_mbox_conf_perm
I0329 19:07:51.353062  2693 net.cpp:434] conv6_2_mbox_conf_perm <- conv6_2_mbox_conf
I0329 19:07:51.353071  2693 net.cpp:408] conv6_2_mbox_conf_perm -> conv6_2_mbox_conf_perm
I0329 19:07:51.353262  2693 net.cpp:150] Setting up conv6_2_mbox_conf_perm
I0329 19:07:51.353273  2693 net.cpp:157] Top shape: 2 16 16 12 (6144)
I0329 19:07:51.353278  2693 net.cpp:165] Memory required for data: 1461436608
I0329 19:07:51.353283  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_conf_flat
I0329 19:07:51.353291  2693 net.cpp:100] Creating Layer conv6_2_mbox_conf_flat
I0329 19:07:51.353297  2693 net.cpp:434] conv6_2_mbox_conf_flat <- conv6_2_mbox_conf_perm
I0329 19:07:51.353307  2693 net.cpp:408] conv6_2_mbox_conf_flat -> conv6_2_mbox_conf_flat
I0329 19:07:51.353353  2693 net.cpp:150] Setting up conv6_2_mbox_conf_flat
I0329 19:07:51.353363  2693 net.cpp:157] Top shape: 2 3072 (6144)
I0329 19:07:51.353368  2693 net.cpp:165] Memory required for data: 1461461184
I0329 19:07:51.353373  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_clean
I0329 19:07:51.353389  2693 net.cpp:100] Creating Layer conv6_2_mbox_clean
I0329 19:07:51.353396  2693 net.cpp:434] conv6_2_mbox_clean <- conv6_2_conv6_2_relu_0_split_3
I0329 19:07:51.353405  2693 net.cpp:408] conv6_2_mbox_clean -> conv6_2_mbox_clean
I0329 19:07:51.355095  2693 net.cpp:150] Setting up conv6_2_mbox_clean
I0329 19:07:51.355114  2693 net.cpp:157] Top shape: 2 6 16 16 (3072)
I0329 19:07:51.355120  2693 net.cpp:165] Memory required for data: 1461473472
I0329 19:07:51.355144  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_clean_perm
I0329 19:07:51.355155  2693 net.cpp:100] Creating Layer conv6_2_mbox_clean_perm
I0329 19:07:51.355161  2693 net.cpp:434] conv6_2_mbox_clean_perm <- conv6_2_mbox_clean
I0329 19:07:51.355185  2693 net.cpp:408] conv6_2_mbox_clean_perm -> conv6_2_mbox_clean_perm
I0329 19:07:51.355376  2693 net.cpp:150] Setting up conv6_2_mbox_clean_perm
I0329 19:07:51.355391  2693 net.cpp:157] Top shape: 2 16 16 6 (3072)
I0329 19:07:51.355396  2693 net.cpp:165] Memory required for data: 1461485760
I0329 19:07:51.355401  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_clean_flat
I0329 19:07:51.355408  2693 net.cpp:100] Creating Layer conv6_2_mbox_clean_flat
I0329 19:07:51.355414  2693 net.cpp:434] conv6_2_mbox_clean_flat <- conv6_2_mbox_clean_perm
I0329 19:07:51.355422  2693 net.cpp:408] conv6_2_mbox_clean_flat -> conv6_2_mbox_clean_flat
I0329 19:07:51.355470  2693 net.cpp:150] Setting up conv6_2_mbox_clean_flat
I0329 19:07:51.355481  2693 net.cpp:157] Top shape: 2 1536 (3072)
I0329 19:07:51.355486  2693 net.cpp:165] Memory required for data: 1461498048
I0329 19:07:51.355491  2693 layer_factory.hpp:77] Creating layer conv6_2_mbox_priorbox
I0329 19:07:51.355501  2693 net.cpp:100] Creating Layer conv6_2_mbox_priorbox
I0329 19:07:51.355507  2693 net.cpp:434] conv6_2_mbox_priorbox <- conv6_2_conv6_2_relu_0_split_4
I0329 19:07:51.355515  2693 net.cpp:434] conv6_2_mbox_priorbox <- data_data_0_split_3
I0329 19:07:51.355525  2693 net.cpp:408] conv6_2_mbox_priorbox -> conv6_2_mbox_priorbox
I0329 19:07:51.355576  2693 net.cpp:150] Setting up conv6_2_mbox_priorbox
I0329 19:07:51.355595  2693 net.cpp:157] Top shape: 1 2 6144 (12288)
I0329 19:07:51.355602  2693 net.cpp:165] Memory required for data: 1461547200
I0329 19:07:51.355607  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_loc
I0329 19:07:51.355620  2693 net.cpp:100] Creating Layer conv7_2_mbox_loc
I0329 19:07:51.355628  2693 net.cpp:434] conv7_2_mbox_loc <- conv7_2_conv7_2_relu_0_split_1
I0329 19:07:51.355643  2693 net.cpp:408] conv7_2_mbox_loc -> conv7_2_mbox_loc
I0329 19:07:51.357686  2693 net.cpp:150] Setting up conv7_2_mbox_loc
I0329 19:07:51.357704  2693 net.cpp:157] Top shape: 2 24 8 8 (3072)
I0329 19:07:51.357710  2693 net.cpp:165] Memory required for data: 1461559488
I0329 19:07:51.357720  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_loc_perm
I0329 19:07:51.357734  2693 net.cpp:100] Creating Layer conv7_2_mbox_loc_perm
I0329 19:07:51.357743  2693 net.cpp:434] conv7_2_mbox_loc_perm <- conv7_2_mbox_loc
I0329 19:07:51.357753  2693 net.cpp:408] conv7_2_mbox_loc_perm -> conv7_2_mbox_loc_perm
I0329 19:07:51.357946  2693 net.cpp:150] Setting up conv7_2_mbox_loc_perm
I0329 19:07:51.357959  2693 net.cpp:157] Top shape: 2 8 8 24 (3072)
I0329 19:07:51.357964  2693 net.cpp:165] Memory required for data: 1461571776
I0329 19:07:51.357969  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_loc_flat
I0329 19:07:51.357978  2693 net.cpp:100] Creating Layer conv7_2_mbox_loc_flat
I0329 19:07:51.357985  2693 net.cpp:434] conv7_2_mbox_loc_flat <- conv7_2_mbox_loc_perm
I0329 19:07:51.357995  2693 net.cpp:408] conv7_2_mbox_loc_flat -> conv7_2_mbox_loc_flat
I0329 19:07:51.358042  2693 net.cpp:150] Setting up conv7_2_mbox_loc_flat
I0329 19:07:51.358052  2693 net.cpp:157] Top shape: 2 1536 (3072)
I0329 19:07:51.358057  2693 net.cpp:165] Memory required for data: 1461584064
I0329 19:07:51.358062  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_conf
I0329 19:07:51.358075  2693 net.cpp:100] Creating Layer conv7_2_mbox_conf
I0329 19:07:51.358084  2693 net.cpp:434] conv7_2_mbox_conf <- conv7_2_conv7_2_relu_0_split_2
I0329 19:07:51.358096  2693 net.cpp:408] conv7_2_mbox_conf -> conv7_2_mbox_conf
I0329 19:07:51.359963  2693 net.cpp:150] Setting up conv7_2_mbox_conf
I0329 19:07:51.359990  2693 net.cpp:157] Top shape: 2 12 8 8 (1536)
I0329 19:07:51.359997  2693 net.cpp:165] Memory required for data: 1461590208
I0329 19:07:51.360005  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_conf_perm
I0329 19:07:51.360018  2693 net.cpp:100] Creating Layer conv7_2_mbox_conf_perm
I0329 19:07:51.360025  2693 net.cpp:434] conv7_2_mbox_conf_perm <- conv7_2_mbox_conf
I0329 19:07:51.360035  2693 net.cpp:408] conv7_2_mbox_conf_perm -> conv7_2_mbox_conf_perm
I0329 19:07:51.360235  2693 net.cpp:150] Setting up conv7_2_mbox_conf_perm
I0329 19:07:51.360260  2693 net.cpp:157] Top shape: 2 8 8 12 (1536)
I0329 19:07:51.360265  2693 net.cpp:165] Memory required for data: 1461596352
I0329 19:07:51.360270  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_conf_flat
I0329 19:07:51.360280  2693 net.cpp:100] Creating Layer conv7_2_mbox_conf_flat
I0329 19:07:51.360290  2693 net.cpp:434] conv7_2_mbox_conf_flat <- conv7_2_mbox_conf_perm
I0329 19:07:51.360298  2693 net.cpp:408] conv7_2_mbox_conf_flat -> conv7_2_mbox_conf_flat
I0329 19:07:51.360347  2693 net.cpp:150] Setting up conv7_2_mbox_conf_flat
I0329 19:07:51.360358  2693 net.cpp:157] Top shape: 2 768 (1536)
I0329 19:07:51.360363  2693 net.cpp:165] Memory required for data: 1461602496
I0329 19:07:51.360368  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_clean
I0329 19:07:51.360385  2693 net.cpp:100] Creating Layer conv7_2_mbox_clean
I0329 19:07:51.360393  2693 net.cpp:434] conv7_2_mbox_clean <- conv7_2_conv7_2_relu_0_split_3
I0329 19:07:51.360401  2693 net.cpp:408] conv7_2_mbox_clean -> conv7_2_mbox_clean
I0329 19:07:51.362156  2693 net.cpp:150] Setting up conv7_2_mbox_clean
I0329 19:07:51.362177  2693 net.cpp:157] Top shape: 2 6 8 8 (768)
I0329 19:07:51.362184  2693 net.cpp:165] Memory required for data: 1461605568
I0329 19:07:51.362192  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_clean_perm
I0329 19:07:51.362202  2693 net.cpp:100] Creating Layer conv7_2_mbox_clean_perm
I0329 19:07:51.362208  2693 net.cpp:434] conv7_2_mbox_clean_perm <- conv7_2_mbox_clean
I0329 19:07:51.362218  2693 net.cpp:408] conv7_2_mbox_clean_perm -> conv7_2_mbox_clean_perm
I0329 19:07:51.362416  2693 net.cpp:150] Setting up conv7_2_mbox_clean_perm
I0329 19:07:51.362428  2693 net.cpp:157] Top shape: 2 8 8 6 (768)
I0329 19:07:51.362433  2693 net.cpp:165] Memory required for data: 1461608640
I0329 19:07:51.362438  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_clean_flat
I0329 19:07:51.362447  2693 net.cpp:100] Creating Layer conv7_2_mbox_clean_flat
I0329 19:07:51.362457  2693 net.cpp:434] conv7_2_mbox_clean_flat <- conv7_2_mbox_clean_perm
I0329 19:07:51.362463  2693 net.cpp:408] conv7_2_mbox_clean_flat -> conv7_2_mbox_clean_flat
I0329 19:07:51.362511  2693 net.cpp:150] Setting up conv7_2_mbox_clean_flat
I0329 19:07:51.362521  2693 net.cpp:157] Top shape: 2 384 (768)
I0329 19:07:51.362526  2693 net.cpp:165] Memory required for data: 1461611712
I0329 19:07:51.362531  2693 layer_factory.hpp:77] Creating layer conv7_2_mbox_priorbox
I0329 19:07:51.362542  2693 net.cpp:100] Creating Layer conv7_2_mbox_priorbox
I0329 19:07:51.362548  2693 net.cpp:434] conv7_2_mbox_priorbox <- conv7_2_conv7_2_relu_0_split_4
I0329 19:07:51.362556  2693 net.cpp:434] conv7_2_mbox_priorbox <- data_data_0_split_4
I0329 19:07:51.362565  2693 net.cpp:408] conv7_2_mbox_priorbox -> conv7_2_mbox_priorbox
I0329 19:07:51.362620  2693 net.cpp:150] Setting up conv7_2_mbox_priorbox
I0329 19:07:51.362632  2693 net.cpp:157] Top shape: 1 2 1536 (3072)
I0329 19:07:51.362635  2693 net.cpp:165] Memory required for data: 1461624000
I0329 19:07:51.362640  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_loc
I0329 19:07:51.362655  2693 net.cpp:100] Creating Layer conv8_2_mbox_loc
I0329 19:07:51.362663  2693 net.cpp:434] conv8_2_mbox_loc <- conv8_2_conv8_2_relu_0_split_1
I0329 19:07:51.362673  2693 net.cpp:408] conv8_2_mbox_loc -> conv8_2_mbox_loc
I0329 19:07:51.365202  2693 net.cpp:150] Setting up conv8_2_mbox_loc
I0329 19:07:51.365222  2693 net.cpp:157] Top shape: 2 16 6 6 (1152)
I0329 19:07:51.365228  2693 net.cpp:165] Memory required for data: 1461628608
I0329 19:07:51.365237  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_loc_perm
I0329 19:07:51.365247  2693 net.cpp:100] Creating Layer conv8_2_mbox_loc_perm
I0329 19:07:51.365255  2693 net.cpp:434] conv8_2_mbox_loc_perm <- conv8_2_mbox_loc
I0329 19:07:51.365267  2693 net.cpp:408] conv8_2_mbox_loc_perm -> conv8_2_mbox_loc_perm
I0329 19:07:51.365464  2693 net.cpp:150] Setting up conv8_2_mbox_loc_perm
I0329 19:07:51.365478  2693 net.cpp:157] Top shape: 2 6 6 16 (1152)
I0329 19:07:51.365496  2693 net.cpp:165] Memory required for data: 1461633216
I0329 19:07:51.365502  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_loc_flat
I0329 19:07:51.365511  2693 net.cpp:100] Creating Layer conv8_2_mbox_loc_flat
I0329 19:07:51.365516  2693 net.cpp:434] conv8_2_mbox_loc_flat <- conv8_2_mbox_loc_perm
I0329 19:07:51.365527  2693 net.cpp:408] conv8_2_mbox_loc_flat -> conv8_2_mbox_loc_flat
I0329 19:07:51.365576  2693 net.cpp:150] Setting up conv8_2_mbox_loc_flat
I0329 19:07:51.365587  2693 net.cpp:157] Top shape: 2 576 (1152)
I0329 19:07:51.365592  2693 net.cpp:165] Memory required for data: 1461637824
I0329 19:07:51.365597  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_conf
I0329 19:07:51.365612  2693 net.cpp:100] Creating Layer conv8_2_mbox_conf
I0329 19:07:51.365620  2693 net.cpp:434] conv8_2_mbox_conf <- conv8_2_conv8_2_relu_0_split_2
I0329 19:07:51.365629  2693 net.cpp:408] conv8_2_mbox_conf -> conv8_2_mbox_conf
I0329 19:07:51.367405  2693 net.cpp:150] Setting up conv8_2_mbox_conf
I0329 19:07:51.367426  2693 net.cpp:157] Top shape: 2 8 6 6 (576)
I0329 19:07:51.367432  2693 net.cpp:165] Memory required for data: 1461640128
I0329 19:07:51.367441  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_conf_perm
I0329 19:07:51.367452  2693 net.cpp:100] Creating Layer conv8_2_mbox_conf_perm
I0329 19:07:51.367460  2693 net.cpp:434] conv8_2_mbox_conf_perm <- conv8_2_mbox_conf
I0329 19:07:51.367470  2693 net.cpp:408] conv8_2_mbox_conf_perm -> conv8_2_mbox_conf_perm
I0329 19:07:51.367691  2693 net.cpp:150] Setting up conv8_2_mbox_conf_perm
I0329 19:07:51.367704  2693 net.cpp:157] Top shape: 2 6 6 8 (576)
I0329 19:07:51.367710  2693 net.cpp:165] Memory required for data: 1461642432
I0329 19:07:51.367715  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_conf_flat
I0329 19:07:51.367725  2693 net.cpp:100] Creating Layer conv8_2_mbox_conf_flat
I0329 19:07:51.367732  2693 net.cpp:434] conv8_2_mbox_conf_flat <- conv8_2_mbox_conf_perm
I0329 19:07:51.367739  2693 net.cpp:408] conv8_2_mbox_conf_flat -> conv8_2_mbox_conf_flat
I0329 19:07:51.367789  2693 net.cpp:150] Setting up conv8_2_mbox_conf_flat
I0329 19:07:51.367800  2693 net.cpp:157] Top shape: 2 288 (576)
I0329 19:07:51.367805  2693 net.cpp:165] Memory required for data: 1461644736
I0329 19:07:51.367810  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_clean
I0329 19:07:51.367825  2693 net.cpp:100] Creating Layer conv8_2_mbox_clean
I0329 19:07:51.367833  2693 net.cpp:434] conv8_2_mbox_clean <- conv8_2_conv8_2_relu_0_split_3
I0329 19:07:51.367847  2693 net.cpp:408] conv8_2_mbox_clean -> conv8_2_mbox_clean
I0329 19:07:51.369551  2693 net.cpp:150] Setting up conv8_2_mbox_clean
I0329 19:07:51.369572  2693 net.cpp:157] Top shape: 2 4 6 6 (288)
I0329 19:07:51.369580  2693 net.cpp:165] Memory required for data: 1461645888
I0329 19:07:51.369590  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_clean_perm
I0329 19:07:51.369601  2693 net.cpp:100] Creating Layer conv8_2_mbox_clean_perm
I0329 19:07:51.369609  2693 net.cpp:434] conv8_2_mbox_clean_perm <- conv8_2_mbox_clean
I0329 19:07:51.369618  2693 net.cpp:408] conv8_2_mbox_clean_perm -> conv8_2_mbox_clean_perm
I0329 19:07:51.369827  2693 net.cpp:150] Setting up conv8_2_mbox_clean_perm
I0329 19:07:51.369838  2693 net.cpp:157] Top shape: 2 6 6 4 (288)
I0329 19:07:51.369843  2693 net.cpp:165] Memory required for data: 1461647040
I0329 19:07:51.369848  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_clean_flat
I0329 19:07:51.369858  2693 net.cpp:100] Creating Layer conv8_2_mbox_clean_flat
I0329 19:07:51.369864  2693 net.cpp:434] conv8_2_mbox_clean_flat <- conv8_2_mbox_clean_perm
I0329 19:07:51.369871  2693 net.cpp:408] conv8_2_mbox_clean_flat -> conv8_2_mbox_clean_flat
I0329 19:07:51.369921  2693 net.cpp:150] Setting up conv8_2_mbox_clean_flat
I0329 19:07:51.369931  2693 net.cpp:157] Top shape: 2 144 (288)
I0329 19:07:51.369936  2693 net.cpp:165] Memory required for data: 1461648192
I0329 19:07:51.369941  2693 layer_factory.hpp:77] Creating layer conv8_2_mbox_priorbox
I0329 19:07:51.369964  2693 net.cpp:100] Creating Layer conv8_2_mbox_priorbox
I0329 19:07:51.369973  2693 net.cpp:434] conv8_2_mbox_priorbox <- conv8_2_conv8_2_relu_0_split_4
I0329 19:07:51.369981  2693 net.cpp:434] conv8_2_mbox_priorbox <- data_data_0_split_5
I0329 19:07:51.369989  2693 net.cpp:408] conv8_2_mbox_priorbox -> conv8_2_mbox_priorbox
I0329 19:07:51.370046  2693 net.cpp:150] Setting up conv8_2_mbox_priorbox
I0329 19:07:51.370057  2693 net.cpp:157] Top shape: 1 2 576 (1152)
I0329 19:07:51.370062  2693 net.cpp:165] Memory required for data: 1461652800
I0329 19:07:51.370067  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_loc
I0329 19:07:51.370081  2693 net.cpp:100] Creating Layer conv9_2_mbox_loc
I0329 19:07:51.370090  2693 net.cpp:434] conv9_2_mbox_loc <- conv9_2_conv9_2_relu_0_split_0
I0329 19:07:51.370100  2693 net.cpp:408] conv9_2_mbox_loc -> conv9_2_mbox_loc
I0329 19:07:51.372041  2693 net.cpp:150] Setting up conv9_2_mbox_loc
I0329 19:07:51.372061  2693 net.cpp:157] Top shape: 2 16 4 4 (512)
I0329 19:07:51.372066  2693 net.cpp:165] Memory required for data: 1461654848
I0329 19:07:51.372077  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_loc_perm
I0329 19:07:51.372089  2693 net.cpp:100] Creating Layer conv9_2_mbox_loc_perm
I0329 19:07:51.372098  2693 net.cpp:434] conv9_2_mbox_loc_perm <- conv9_2_mbox_loc
I0329 19:07:51.372108  2693 net.cpp:408] conv9_2_mbox_loc_perm -> conv9_2_mbox_loc_perm
I0329 19:07:51.372311  2693 net.cpp:150] Setting up conv9_2_mbox_loc_perm
I0329 19:07:51.372323  2693 net.cpp:157] Top shape: 2 4 4 16 (512)
I0329 19:07:51.372328  2693 net.cpp:165] Memory required for data: 1461656896
I0329 19:07:51.372334  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_loc_flat
I0329 19:07:51.372341  2693 net.cpp:100] Creating Layer conv9_2_mbox_loc_flat
I0329 19:07:51.372347  2693 net.cpp:434] conv9_2_mbox_loc_flat <- conv9_2_mbox_loc_perm
I0329 19:07:51.372354  2693 net.cpp:408] conv9_2_mbox_loc_flat -> conv9_2_mbox_loc_flat
I0329 19:07:51.372401  2693 net.cpp:150] Setting up conv9_2_mbox_loc_flat
I0329 19:07:51.372411  2693 net.cpp:157] Top shape: 2 256 (512)
I0329 19:07:51.372416  2693 net.cpp:165] Memory required for data: 1461658944
I0329 19:07:51.372421  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_conf
I0329 19:07:51.372437  2693 net.cpp:100] Creating Layer conv9_2_mbox_conf
I0329 19:07:51.372447  2693 net.cpp:434] conv9_2_mbox_conf <- conv9_2_conv9_2_relu_0_split_1
I0329 19:07:51.372457  2693 net.cpp:408] conv9_2_mbox_conf -> conv9_2_mbox_conf
I0329 19:07:51.374236  2693 net.cpp:150] Setting up conv9_2_mbox_conf
I0329 19:07:51.374255  2693 net.cpp:157] Top shape: 2 8 4 4 (256)
I0329 19:07:51.374261  2693 net.cpp:165] Memory required for data: 1461659968
I0329 19:07:51.374271  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_conf_perm
I0329 19:07:51.374282  2693 net.cpp:100] Creating Layer conv9_2_mbox_conf_perm
I0329 19:07:51.374289  2693 net.cpp:434] conv9_2_mbox_conf_perm <- conv9_2_mbox_conf
I0329 19:07:51.374300  2693 net.cpp:408] conv9_2_mbox_conf_perm -> conv9_2_mbox_conf_perm
I0329 19:07:51.374564  2693 net.cpp:150] Setting up conv9_2_mbox_conf_perm
I0329 19:07:51.374588  2693 net.cpp:157] Top shape: 2 4 4 8 (256)
I0329 19:07:51.374598  2693 net.cpp:165] Memory required for data: 1461660992
I0329 19:07:51.374608  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_conf_flat
I0329 19:07:51.374629  2693 net.cpp:100] Creating Layer conv9_2_mbox_conf_flat
I0329 19:07:51.374635  2693 net.cpp:434] conv9_2_mbox_conf_flat <- conv9_2_mbox_conf_perm
I0329 19:07:51.374644  2693 net.cpp:408] conv9_2_mbox_conf_flat -> conv9_2_mbox_conf_flat
I0329 19:07:51.374698  2693 net.cpp:150] Setting up conv9_2_mbox_conf_flat
I0329 19:07:51.374709  2693 net.cpp:157] Top shape: 2 128 (256)
I0329 19:07:51.374714  2693 net.cpp:165] Memory required for data: 1461662016
I0329 19:07:51.374719  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_clean
I0329 19:07:51.374735  2693 net.cpp:100] Creating Layer conv9_2_mbox_clean
I0329 19:07:51.374742  2693 net.cpp:434] conv9_2_mbox_clean <- conv9_2_conv9_2_relu_0_split_2
I0329 19:07:51.374768  2693 net.cpp:408] conv9_2_mbox_clean -> conv9_2_mbox_clean
I0329 19:07:51.376519  2693 net.cpp:150] Setting up conv9_2_mbox_clean
I0329 19:07:51.376538  2693 net.cpp:157] Top shape: 2 4 4 4 (128)
I0329 19:07:51.376544  2693 net.cpp:165] Memory required for data: 1461662528
I0329 19:07:51.376554  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_clean_perm
I0329 19:07:51.376565  2693 net.cpp:100] Creating Layer conv9_2_mbox_clean_perm
I0329 19:07:51.376572  2693 net.cpp:434] conv9_2_mbox_clean_perm <- conv9_2_mbox_clean
I0329 19:07:51.376580  2693 net.cpp:408] conv9_2_mbox_clean_perm -> conv9_2_mbox_clean_perm
I0329 19:07:51.376786  2693 net.cpp:150] Setting up conv9_2_mbox_clean_perm
I0329 19:07:51.376798  2693 net.cpp:157] Top shape: 2 4 4 4 (128)
I0329 19:07:51.376803  2693 net.cpp:165] Memory required for data: 1461663040
I0329 19:07:51.376808  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_clean_flat
I0329 19:07:51.376818  2693 net.cpp:100] Creating Layer conv9_2_mbox_clean_flat
I0329 19:07:51.376827  2693 net.cpp:434] conv9_2_mbox_clean_flat <- conv9_2_mbox_clean_perm
I0329 19:07:51.376834  2693 net.cpp:408] conv9_2_mbox_clean_flat -> conv9_2_mbox_clean_flat
I0329 19:07:51.376883  2693 net.cpp:150] Setting up conv9_2_mbox_clean_flat
I0329 19:07:51.376893  2693 net.cpp:157] Top shape: 2 64 (128)
I0329 19:07:51.376899  2693 net.cpp:165] Memory required for data: 1461663552
I0329 19:07:51.376904  2693 layer_factory.hpp:77] Creating layer conv9_2_mbox_priorbox
I0329 19:07:51.376914  2693 net.cpp:100] Creating Layer conv9_2_mbox_priorbox
I0329 19:07:51.376920  2693 net.cpp:434] conv9_2_mbox_priorbox <- conv9_2_conv9_2_relu_0_split_3
I0329 19:07:51.376927  2693 net.cpp:434] conv9_2_mbox_priorbox <- data_data_0_split_6
I0329 19:07:51.376936  2693 net.cpp:408] conv9_2_mbox_priorbox -> conv9_2_mbox_priorbox
I0329 19:07:51.376988  2693 net.cpp:150] Setting up conv9_2_mbox_priorbox
I0329 19:07:51.376998  2693 net.cpp:157] Top shape: 1 2 256 (512)
I0329 19:07:51.377003  2693 net.cpp:165] Memory required for data: 1461665600
I0329 19:07:51.377008  2693 layer_factory.hpp:77] Creating layer mbox_loc
I0329 19:07:51.377017  2693 net.cpp:100] Creating Layer mbox_loc
I0329 19:07:51.377025  2693 net.cpp:434] mbox_loc <- conv4_3_norm_mbox_loc_flat
I0329 19:07:51.377033  2693 net.cpp:434] mbox_loc <- fc7_mbox_loc_flat
I0329 19:07:51.377039  2693 net.cpp:434] mbox_loc <- conv6_2_mbox_loc_flat
I0329 19:07:51.377046  2693 net.cpp:434] mbox_loc <- conv7_2_mbox_loc_flat
I0329 19:07:51.377053  2693 net.cpp:434] mbox_loc <- conv8_2_mbox_loc_flat
I0329 19:07:51.377058  2693 net.cpp:434] mbox_loc <- conv9_2_mbox_loc_flat
I0329 19:07:51.377069  2693 net.cpp:408] mbox_loc -> mbox_loc
I0329 19:07:51.377135  2693 net.cpp:150] Setting up mbox_loc
I0329 19:07:51.377146  2693 net.cpp:157] Top shape: 2 98624 (197248)
I0329 19:07:51.377151  2693 net.cpp:165] Memory required for data: 1462454592
I0329 19:07:51.377156  2693 layer_factory.hpp:77] Creating layer mbox_conf
I0329 19:07:51.377171  2693 net.cpp:100] Creating Layer mbox_conf
I0329 19:07:51.377178  2693 net.cpp:434] mbox_conf <- conv4_3_norm_mbox_conf_flat
I0329 19:07:51.377187  2693 net.cpp:434] mbox_conf <- fc7_mbox_conf_flat
I0329 19:07:51.377192  2693 net.cpp:434] mbox_conf <- conv6_2_mbox_conf_flat
I0329 19:07:51.377198  2693 net.cpp:434] mbox_conf <- conv7_2_mbox_conf_flat
I0329 19:07:51.377207  2693 net.cpp:434] mbox_conf <- conv8_2_mbox_conf_flat
I0329 19:07:51.377213  2693 net.cpp:434] mbox_conf <- conv9_2_mbox_conf_flat
I0329 19:07:51.377220  2693 net.cpp:408] mbox_conf -> mbox_conf
I0329 19:07:51.377269  2693 net.cpp:150] Setting up mbox_conf
I0329 19:07:51.377279  2693 net.cpp:157] Top shape: 2 49312 (98624)
I0329 19:07:51.377284  2693 net.cpp:165] Memory required for data: 1462849088
I0329 19:07:51.377288  2693 layer_factory.hpp:77] Creating layer mbox_priorbox
I0329 19:07:51.377296  2693 net.cpp:100] Creating Layer mbox_priorbox
I0329 19:07:51.377301  2693 net.cpp:434] mbox_priorbox <- conv4_3_norm_mbox_priorbox
I0329 19:07:51.377321  2693 net.cpp:434] mbox_priorbox <- fc7_mbox_priorbox
I0329 19:07:51.377331  2693 net.cpp:434] mbox_priorbox <- conv6_2_mbox_priorbox
I0329 19:07:51.377336  2693 net.cpp:434] mbox_priorbox <- conv7_2_mbox_priorbox
I0329 19:07:51.377341  2693 net.cpp:434] mbox_priorbox <- conv8_2_mbox_priorbox
I0329 19:07:51.377347  2693 net.cpp:434] mbox_priorbox <- conv9_2_mbox_priorbox
I0329 19:07:51.377357  2693 net.cpp:408] mbox_priorbox -> mbox_priorbox
I0329 19:07:51.377410  2693 net.cpp:150] Setting up mbox_priorbox
I0329 19:07:51.377420  2693 net.cpp:157] Top shape: 1 2 98624 (197248)
I0329 19:07:51.377425  2693 net.cpp:165] Memory required for data: 1463638080
I0329 19:07:51.377430  2693 layer_factory.hpp:77] Creating layer mbox_clean
I0329 19:07:51.377439  2693 net.cpp:100] Creating Layer mbox_clean
I0329 19:07:51.377444  2693 net.cpp:434] mbox_clean <- conv4_3_norm_mbox_clean_flat
I0329 19:07:51.377451  2693 net.cpp:434] mbox_clean <- fc7_mbox_clean_flat
I0329 19:07:51.377457  2693 net.cpp:434] mbox_clean <- conv6_2_mbox_clean_flat
I0329 19:07:51.377465  2693 net.cpp:434] mbox_clean <- conv7_2_mbox_clean_flat
I0329 19:07:51.377472  2693 net.cpp:434] mbox_clean <- conv8_2_mbox_clean_flat
I0329 19:07:51.377478  2693 net.cpp:434] mbox_clean <- conv9_2_mbox_clean_flat
I0329 19:07:51.377485  2693 net.cpp:408] mbox_clean -> mbox_clean
I0329 19:07:51.377533  2693 net.cpp:150] Setting up mbox_clean
I0329 19:07:51.377543  2693 net.cpp:157] Top shape: 2 24656 (49312)
I0329 19:07:51.377548  2693 net.cpp:165] Memory required for data: 1463835328
I0329 19:07:51.377553  2693 layer_factory.hpp:77] Creating layer mbox_conf_reshape
I0329 19:07:51.377565  2693 net.cpp:100] Creating Layer mbox_conf_reshape
I0329 19:07:51.377573  2693 net.cpp:434] mbox_conf_reshape <- mbox_conf
I0329 19:07:51.377580  2693 net.cpp:408] mbox_conf_reshape -> mbox_conf_reshape
I0329 19:07:51.377638  2693 net.cpp:150] Setting up mbox_conf_reshape
I0329 19:07:51.377650  2693 net.cpp:157] Top shape: 2 24656 2 (98624)
I0329 19:07:51.377653  2693 net.cpp:165] Memory required for data: 1464229824
I0329 19:07:51.377658  2693 layer_factory.hpp:77] Creating layer mbox_conf_softmax
I0329 19:07:51.377667  2693 net.cpp:100] Creating Layer mbox_conf_softmax
I0329 19:07:51.377673  2693 net.cpp:434] mbox_conf_softmax <- mbox_conf_reshape
I0329 19:07:51.377681  2693 net.cpp:408] mbox_conf_softmax -> mbox_conf_softmax
I0329 19:07:51.378041  2693 net.cpp:150] Setting up mbox_conf_softmax
I0329 19:07:51.378056  2693 net.cpp:157] Top shape: 2 24656 2 (98624)
I0329 19:07:51.378062  2693 net.cpp:165] Memory required for data: 1464624320
I0329 19:07:51.378067  2693 layer_factory.hpp:77] Creating layer mbox_conf_flatten
I0329 19:07:51.378075  2693 net.cpp:100] Creating Layer mbox_conf_flatten
I0329 19:07:51.378083  2693 net.cpp:434] mbox_conf_flatten <- mbox_conf_softmax
I0329 19:07:51.378094  2693 net.cpp:408] mbox_conf_flatten -> mbox_conf_flatten
I0329 19:07:51.378146  2693 net.cpp:150] Setting up mbox_conf_flatten
I0329 19:07:51.378157  2693 net.cpp:157] Top shape: 2 49312 (98624)
I0329 19:07:51.378162  2693 net.cpp:165] Memory required for data: 1465018816
I0329 19:07:51.378167  2693 layer_factory.hpp:77] Creating layer mbox_clean_sigmoid
I0329 19:07:51.378188  2693 net.cpp:100] Creating Layer mbox_clean_sigmoid
I0329 19:07:51.378196  2693 net.cpp:434] mbox_clean_sigmoid <- mbox_clean
I0329 19:07:51.378203  2693 net.cpp:408] mbox_clean_sigmoid -> mbox_clean_sigmoid
I0329 19:07:51.378649  2693 net.cpp:150] Setting up mbox_clean_sigmoid
I0329 19:07:51.378667  2693 net.cpp:157] Top shape: 2 24656 (49312)
I0329 19:07:51.378674  2693 net.cpp:165] Memory required for data: 1465216064
I0329 19:07:51.378679  2693 layer_factory.hpp:77] Creating layer detection_out
I0329 19:07:51.378706  2693 net.cpp:100] Creating Layer detection_out
I0329 19:07:51.378717  2693 net.cpp:434] detection_out <- mbox_loc
I0329 19:07:51.378726  2693 net.cpp:434] detection_out <- mbox_conf_flatten
I0329 19:07:51.378731  2693 net.cpp:434] detection_out <- mbox_priorbox
I0329 19:07:51.378749  2693 net.cpp:434] detection_out <- mbox_clean_sigmoid
I0329 19:07:51.378762  2693 net.cpp:408] detection_out -> detection_out
I0329 19:07:51.389685  2693 net.cpp:150] Setting up detection_out
I0329 19:07:51.389703  2693 net.cpp:157] Top shape: 1 1 1 8 (8)
I0329 19:07:51.389708  2693 net.cpp:165] Memory required for data: 1465216096
I0329 19:07:51.389714  2693 layer_factory.hpp:77] Creating layer detection_eval
I0329 19:07:51.389725  2693 net.cpp:100] Creating Layer detection_eval
I0329 19:07:51.389731  2693 net.cpp:434] detection_eval <- detection_out
I0329 19:07:51.389739  2693 net.cpp:434] detection_eval <- label
I0329 19:07:51.389749  2693 net.cpp:408] detection_eval -> detection_eval
I0329 19:07:51.396773  2693 net.cpp:150] Setting up detection_eval
I0329 19:07:51.396790  2693 net.cpp:157] Top shape: 1 1 2 5 (10)
I0329 19:07:51.396795  2693 net.cpp:165] Memory required for data: 1465216136
I0329 19:07:51.396801  2693 net.cpp:228] detection_eval does not need backward computation.
I0329 19:07:51.396808  2693 net.cpp:228] detection_out does not need backward computation.
I0329 19:07:51.396814  2693 net.cpp:228] mbox_clean_sigmoid does not need backward computation.
I0329 19:07:51.396819  2693 net.cpp:228] mbox_conf_flatten does not need backward computation.
I0329 19:07:51.396824  2693 net.cpp:228] mbox_conf_softmax does not need backward computation.
I0329 19:07:51.396831  2693 net.cpp:228] mbox_conf_reshape does not need backward computation.
I0329 19:07:51.396834  2693 net.cpp:228] mbox_clean does not need backward computation.
I0329 19:07:51.396842  2693 net.cpp:228] mbox_priorbox does not need backward computation.
I0329 19:07:51.396848  2693 net.cpp:228] mbox_conf does not need backward computation.
I0329 19:07:51.396855  2693 net.cpp:228] mbox_loc does not need backward computation.
I0329 19:07:51.396862  2693 net.cpp:228] conv9_2_mbox_priorbox does not need backward computation.
I0329 19:07:51.396868  2693 net.cpp:228] conv9_2_mbox_clean_flat does not need backward computation.
I0329 19:07:51.396874  2693 net.cpp:228] conv9_2_mbox_clean_perm does not need backward computation.
I0329 19:07:51.396879  2693 net.cpp:228] conv9_2_mbox_clean does not need backward computation.
I0329 19:07:51.396885  2693 net.cpp:228] conv9_2_mbox_conf_flat does not need backward computation.
I0329 19:07:51.396890  2693 net.cpp:228] conv9_2_mbox_conf_perm does not need backward computation.
I0329 19:07:51.396895  2693 net.cpp:228] conv9_2_mbox_conf does not need backward computation.
I0329 19:07:51.396901  2693 net.cpp:228] conv9_2_mbox_loc_flat does not need backward computation.
I0329 19:07:51.396906  2693 net.cpp:228] conv9_2_mbox_loc_perm does not need backward computation.
I0329 19:07:51.396911  2693 net.cpp:228] conv9_2_mbox_loc does not need backward computation.
I0329 19:07:51.396916  2693 net.cpp:228] conv8_2_mbox_priorbox does not need backward computation.
I0329 19:07:51.396922  2693 net.cpp:228] conv8_2_mbox_clean_flat does not need backward computation.
I0329 19:07:51.396927  2693 net.cpp:228] conv8_2_mbox_clean_perm does not need backward computation.
I0329 19:07:51.396934  2693 net.cpp:228] conv8_2_mbox_clean does not need backward computation.
I0329 19:07:51.396939  2693 net.cpp:228] conv8_2_mbox_conf_flat does not need backward computation.
I0329 19:07:51.396944  2693 net.cpp:228] conv8_2_mbox_conf_perm does not need backward computation.
I0329 19:07:51.396950  2693 net.cpp:228] conv8_2_mbox_conf does not need backward computation.
I0329 19:07:51.396955  2693 net.cpp:228] conv8_2_mbox_loc_flat does not need backward computation.
I0329 19:07:51.396960  2693 net.cpp:228] conv8_2_mbox_loc_perm does not need backward computation.
I0329 19:07:51.396965  2693 net.cpp:228] conv8_2_mbox_loc does not need backward computation.
I0329 19:07:51.396970  2693 net.cpp:228] conv7_2_mbox_priorbox does not need backward computation.
I0329 19:07:51.396975  2693 net.cpp:228] conv7_2_mbox_clean_flat does not need backward computation.
I0329 19:07:51.396981  2693 net.cpp:228] conv7_2_mbox_clean_perm does not need backward computation.
I0329 19:07:51.396998  2693 net.cpp:228] conv7_2_mbox_clean does not need backward computation.
I0329 19:07:51.397003  2693 net.cpp:228] conv7_2_mbox_conf_flat does not need backward computation.
I0329 19:07:51.397009  2693 net.cpp:228] conv7_2_mbox_conf_perm does not need backward computation.
I0329 19:07:51.397014  2693 net.cpp:228] conv7_2_mbox_conf does not need backward computation.
I0329 19:07:51.397019  2693 net.cpp:228] conv7_2_mbox_loc_flat does not need backward computation.
I0329 19:07:51.397024  2693 net.cpp:228] conv7_2_mbox_loc_perm does not need backward computation.
I0329 19:07:51.397029  2693 net.cpp:228] conv7_2_mbox_loc does not need backward computation.
I0329 19:07:51.397034  2693 net.cpp:228] conv6_2_mbox_priorbox does not need backward computation.
I0329 19:07:51.397040  2693 net.cpp:228] conv6_2_mbox_clean_flat does not need backward computation.
I0329 19:07:51.397045  2693 net.cpp:228] conv6_2_mbox_clean_perm does not need backward computation.
I0329 19:07:51.397052  2693 net.cpp:228] conv6_2_mbox_clean does not need backward computation.
I0329 19:07:51.397056  2693 net.cpp:228] conv6_2_mbox_conf_flat does not need backward computation.
I0329 19:07:51.397061  2693 net.cpp:228] conv6_2_mbox_conf_perm does not need backward computation.
I0329 19:07:51.397066  2693 net.cpp:228] conv6_2_mbox_conf does not need backward computation.
I0329 19:07:51.397071  2693 net.cpp:228] conv6_2_mbox_loc_flat does not need backward computation.
I0329 19:07:51.397078  2693 net.cpp:228] conv6_2_mbox_loc_perm does not need backward computation.
I0329 19:07:51.397083  2693 net.cpp:228] conv6_2_mbox_loc does not need backward computation.
I0329 19:07:51.397088  2693 net.cpp:228] fc7_mbox_priorbox does not need backward computation.
I0329 19:07:51.397094  2693 net.cpp:228] fc7_mbox_clean_flat does not need backward computation.
I0329 19:07:51.397099  2693 net.cpp:228] fc7_mbox_clean_perm does not need backward computation.
I0329 19:07:51.397104  2693 net.cpp:228] fc7_mbox_clean does not need backward computation.
I0329 19:07:51.397109  2693 net.cpp:228] fc7_mbox_conf_flat does not need backward computation.
I0329 19:07:51.397114  2693 net.cpp:228] fc7_mbox_conf_perm does not need backward computation.
I0329 19:07:51.397120  2693 net.cpp:228] fc7_mbox_conf does not need backward computation.
I0329 19:07:51.397125  2693 net.cpp:228] fc7_mbox_loc_flat does not need backward computation.
I0329 19:07:51.397130  2693 net.cpp:228] fc7_mbox_loc_perm does not need backward computation.
I0329 19:07:51.397135  2693 net.cpp:228] fc7_mbox_loc does not need backward computation.
I0329 19:07:51.397141  2693 net.cpp:228] conv4_3_norm_mbox_priorbox does not need backward computation.
I0329 19:07:51.397147  2693 net.cpp:228] conv4_3_norm_mbox_clean_flat does not need backward computation.
I0329 19:07:51.397152  2693 net.cpp:228] conv4_3_norm_mbox_clean_perm does not need backward computation.
I0329 19:07:51.397158  2693 net.cpp:228] conv4_3_norm_mbox_clean does not need backward computation.
I0329 19:07:51.397164  2693 net.cpp:228] conv4_3_norm_mbox_conf_flat does not need backward computation.
I0329 19:07:51.397169  2693 net.cpp:228] conv4_3_norm_mbox_conf_perm does not need backward computation.
I0329 19:07:51.397176  2693 net.cpp:228] conv4_3_norm_mbox_conf does not need backward computation.
I0329 19:07:51.397181  2693 net.cpp:228] conv4_3_norm_mbox_loc_flat does not need backward computation.
I0329 19:07:51.397186  2693 net.cpp:228] conv4_3_norm_mbox_loc_perm does not need backward computation.
I0329 19:07:51.397191  2693 net.cpp:228] conv4_3_norm_mbox_loc does not need backward computation.
I0329 19:07:51.397197  2693 net.cpp:228] conv4_3_norm_conv4_3_norm_0_split does not need backward computation.
I0329 19:07:51.397202  2693 net.cpp:228] conv4_3_norm does not need backward computation.
I0329 19:07:51.397208  2693 net.cpp:228] conv9_2_conv9_2_relu_0_split does not need backward computation.
I0329 19:07:51.397213  2693 net.cpp:228] conv9_2_relu does not need backward computation.
I0329 19:07:51.397218  2693 net.cpp:228] conv9_2 does not need backward computation.
I0329 19:07:51.397230  2693 net.cpp:228] conv9_1_relu does not need backward computation.
I0329 19:07:51.397235  2693 net.cpp:228] conv9_1 does not need backward computation.
I0329 19:07:51.397241  2693 net.cpp:228] conv8_2_conv8_2_relu_0_split does not need backward computation.
I0329 19:07:51.397246  2693 net.cpp:228] conv8_2_relu does not need backward computation.
I0329 19:07:51.397251  2693 net.cpp:228] conv8_2 does not need backward computation.
I0329 19:07:51.397256  2693 net.cpp:228] conv8_1_relu does not need backward computation.
I0329 19:07:51.397261  2693 net.cpp:228] conv8_1 does not need backward computation.
I0329 19:07:51.397267  2693 net.cpp:228] conv7_2_conv7_2_relu_0_split does not need backward computation.
I0329 19:07:51.397272  2693 net.cpp:228] conv7_2_relu does not need backward computation.
I0329 19:07:51.397277  2693 net.cpp:228] conv7_2 does not need backward computation.
I0329 19:07:51.397282  2693 net.cpp:228] conv7_1_relu does not need backward computation.
I0329 19:07:51.397287  2693 net.cpp:228] conv7_1 does not need backward computation.
I0329 19:07:51.397294  2693 net.cpp:228] conv6_2_conv6_2_relu_0_split does not need backward computation.
I0329 19:07:51.397299  2693 net.cpp:228] conv6_2_relu does not need backward computation.
I0329 19:07:51.397303  2693 net.cpp:228] conv6_2 does not need backward computation.
I0329 19:07:51.397310  2693 net.cpp:228] conv6_1_relu does not need backward computation.
I0329 19:07:51.397315  2693 net.cpp:228] conv6_1 does not need backward computation.
I0329 19:07:51.397320  2693 net.cpp:228] fc7_relu7_0_split does not need backward computation.
I0329 19:07:51.397325  2693 net.cpp:228] relu7 does not need backward computation.
I0329 19:07:51.397330  2693 net.cpp:228] fc7 does not need backward computation.
I0329 19:07:51.397336  2693 net.cpp:228] relu6 does not need backward computation.
I0329 19:07:51.397341  2693 net.cpp:228] fc6 does not need backward computation.
I0329 19:07:51.397346  2693 net.cpp:228] pool5 does not need backward computation.
I0329 19:07:51.397352  2693 net.cpp:228] relu5_3 does not need backward computation.
I0329 19:07:51.397357  2693 net.cpp:228] conv5_3 does not need backward computation.
I0329 19:07:51.397362  2693 net.cpp:228] relu5_2 does not need backward computation.
I0329 19:07:51.397367  2693 net.cpp:228] conv5_2 does not need backward computation.
I0329 19:07:51.397372  2693 net.cpp:228] relu5_1 does not need backward computation.
I0329 19:07:51.397377  2693 net.cpp:228] conv5_1 does not need backward computation.
I0329 19:07:51.397383  2693 net.cpp:228] pool4 does not need backward computation.
I0329 19:07:51.397389  2693 net.cpp:228] conv4_3_relu4_3_0_split does not need backward computation.
I0329 19:07:51.397394  2693 net.cpp:228] relu4_3 does not need backward computation.
I0329 19:07:51.397399  2693 net.cpp:228] conv4_3 does not need backward computation.
I0329 19:07:51.397404  2693 net.cpp:228] relu4_2 does not need backward computation.
I0329 19:07:51.397410  2693 net.cpp:228] conv4_2 does not need backward computation.
I0329 19:07:51.397415  2693 net.cpp:228] relu4_1 does not need backward computation.
I0329 19:07:51.397420  2693 net.cpp:228] conv4_1 does not need backward computation.
I0329 19:07:51.397425  2693 net.cpp:228] pool3 does not need backward computation.
I0329 19:07:51.397430  2693 net.cpp:228] relu3_3 does not need backward computation.
I0329 19:07:51.397436  2693 net.cpp:228] conv3_3 does not need backward computation.
I0329 19:07:51.397441  2693 net.cpp:228] relu3_2 does not need backward computation.
I0329 19:07:51.397446  2693 net.cpp:228] conv3_2 does not need backward computation.
I0329 19:07:51.397452  2693 net.cpp:228] relu3_1 does not need backward computation.
I0329 19:07:51.397457  2693 net.cpp:228] conv3_1 does not need backward computation.
I0329 19:07:51.397462  2693 net.cpp:228] pool2 does not need backward computation.
I0329 19:07:51.397467  2693 net.cpp:228] relu2_2 does not need backward computation.
I0329 19:07:51.397472  2693 net.cpp:228] conv2_2 does not need backward computation.
I0329 19:07:51.397483  2693 net.cpp:228] relu2_1 does not need backward computation.
I0329 19:07:51.397490  2693 net.cpp:228] conv2_1 does not need backward computation.
I0329 19:07:51.397495  2693 net.cpp:228] pool1 does not need backward computation.
I0329 19:07:51.397500  2693 net.cpp:228] relu1_2 does not need backward computation.
I0329 19:07:51.397505  2693 net.cpp:228] conv1_2 does not need backward computation.
I0329 19:07:51.397510  2693 net.cpp:228] relu1_1 does not need backward computation.
I0329 19:07:51.397514  2693 net.cpp:228] conv1_1 does not need backward computation.
I0329 19:07:51.397521  2693 net.cpp:228] data_data_0_split does not need backward computation.
I0329 19:07:51.397532  2693 net.cpp:228] data does not need backward computation.
I0329 19:07:51.397537  2693 net.cpp:270] This network produces output detection_eval
I0329 19:07:51.397636  2693 net.cpp:283] Network initialization done.
I0329 19:07:51.398113  2693 solver.cpp:75] Solver scaffolding done.
I0329 19:07:51.404005  2693 caffe.cpp:155] Finetuning from models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel
I0329 19:07:51.555428  2693 upgrade_proto.cpp:67] Attempting to upgrade input file specified using deprecated input fields: models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel
I0329 19:07:51.555476  2693 upgrade_proto.cpp:70] Successfully upgraded file specified using deprecated input fields.
W0329 19:07:51.555483  2693 upgrade_proto.cpp:72] Note that future Caffe releases will only support input layers and not input fields.
I0329 19:07:51.569499  2693 net.cpp:761] Ignoring source layer drop6
I0329 19:07:51.570303  2693 net.cpp:761] Ignoring source layer drop7
I0329 19:07:51.570314  2693 net.cpp:761] Ignoring source layer fc8
I0329 19:07:51.570319  2693 net.cpp:761] Ignoring source layer prob
I0329 19:07:51.697649  2693 upgrade_proto.cpp:67] Attempting to upgrade input file specified using deprecated input fields: models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel
I0329 19:07:51.697695  2693 upgrade_proto.cpp:70] Successfully upgraded file specified using deprecated input fields.
W0329 19:07:51.697700  2693 upgrade_proto.cpp:72] Note that future Caffe releases will only support input layers and not input fields.
I0329 19:07:51.711153  2693 net.cpp:761] Ignoring source layer drop6
I0329 19:07:51.711905  2693 net.cpp:761] Ignoring source layer drop7
I0329 19:07:51.711916  2693 net.cpp:761] Ignoring source layer fc8
I0329 19:07:51.711920  2693 net.cpp:761] Ignoring source layer prob
I0329 19:07:51.712771  2693 caffe.cpp:251] Starting Optimization
I0329 19:07:51.712783  2693 solver.cpp:294] Solving VGG_ssd_coco_part_clean_train
I0329 19:07:51.712787  2693 solver.cpp:295] Learning Rate Policy: step
I0329 19:07:52.394292  2693 solver.cpp:243] Iteration 0, loss = 20.0414
I0329 19:07:52.394361  2693 solver.cpp:259]     Train net output #0: mbox_loss = 20.0414 (* 1 = 20.0414 loss)
I0329 19:07:52.394383  2693 sgd_solver.cpp:138] Iteration 0, lr = 0.0005
I0329 19:10:08.442042  2693 solver.cpp:243] Iteration 100, loss = 8.97089
I0329 19:10:08.442243  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.67374 (* 1 = 6.67374 loss)
I0329 19:10:08.442260  2693 sgd_solver.cpp:138] Iteration 100, lr = 0.0005
I0329 19:12:27.156977  2693 solver.cpp:243] Iteration 200, loss = 7.27036
I0329 19:12:27.157174  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.78981 (* 1 = 6.78981 loss)
I0329 19:12:27.157191  2693 sgd_solver.cpp:138] Iteration 200, lr = 0.0005
I0329 19:14:39.969449  2693 solver.cpp:243] Iteration 300, loss = 7.04766
I0329 19:14:39.971037  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.74597 (* 1 = 5.74597 loss)
I0329 19:14:39.971056  2693 sgd_solver.cpp:138] Iteration 300, lr = 0.0005
I0329 19:16:50.351572  2693 solver.cpp:243] Iteration 400, loss = 7.05565
I0329 19:16:50.351893  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.23429 (* 1 = 6.23429 loss)
I0329 19:16:50.351910  2693 sgd_solver.cpp:138] Iteration 400, lr = 0.0005
I0329 19:19:05.726585  2693 solver.cpp:243] Iteration 500, loss = 6.73296
I0329 19:19:05.726811  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.79375 (* 1 = 6.79375 loss)
I0329 19:19:05.726830  2693 sgd_solver.cpp:138] Iteration 500, lr = 0.0005
I0329 19:21:16.475733  2693 solver.cpp:243] Iteration 600, loss = 6.50202
I0329 19:21:16.476037  2693 solver.cpp:259]     Train net output #0: mbox_loss = 7.49715 (* 1 = 7.49715 loss)
I0329 19:21:16.476097  2693 sgd_solver.cpp:138] Iteration 600, lr = 0.0005
I0329 19:23:28.160481  2693 solver.cpp:243] Iteration 700, loss = 6.52816
I0329 19:23:28.160653  2693 solver.cpp:259]     Train net output #0: mbox_loss = 7.38111 (* 1 = 7.38111 loss)
I0329 19:23:28.160670  2693 sgd_solver.cpp:138] Iteration 700, lr = 0.0005
I0329 19:25:39.782595  2693 solver.cpp:243] Iteration 800, loss = 6.48199
I0329 19:25:39.789847  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.54495 (* 1 = 6.54495 loss)
I0329 19:25:39.790004  2693 sgd_solver.cpp:138] Iteration 800, lr = 0.0005
I0329 19:27:52.363891  2693 solver.cpp:243] Iteration 900, loss = 6.37319
I0329 19:27:52.364024  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.63805 (* 1 = 5.63805 loss)
I0329 19:27:52.364042  2693 sgd_solver.cpp:138] Iteration 900, lr = 0.0005
I0329 19:30:04.943825  2693 solver.cpp:243] Iteration 1000, loss = 6.26709
I0329 19:30:04.944021  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.5495 (* 1 = 6.5495 loss)
I0329 19:30:04.944038  2693 sgd_solver.cpp:138] Iteration 1000, lr = 0.0005
I0329 19:32:15.709952  2693 solver.cpp:243] Iteration 1100, loss = 6.20186
I0329 19:32:15.710160  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.02282 (* 1 = 6.02282 loss)
I0329 19:32:15.710176  2693 sgd_solver.cpp:138] Iteration 1100, lr = 0.0005
I0329 19:34:28.157032  2693 solver.cpp:243] Iteration 1200, loss = 6.12529
I0329 19:34:28.157232  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.76501 (* 1 = 6.76501 loss)
I0329 19:34:28.157248  2693 sgd_solver.cpp:138] Iteration 1200, lr = 0.0005
I0329 19:36:40.578898  2693 solver.cpp:243] Iteration 1300, loss = 6.22851
I0329 19:36:40.579121  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.02188 (* 1 = 6.02188 loss)
I0329 19:36:40.579155  2693 sgd_solver.cpp:138] Iteration 1300, lr = 0.0005
I0329 19:38:50.385695  2693 solver.cpp:243] Iteration 1400, loss = 5.96528
I0329 19:38:50.385887  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.72784 (* 1 = 5.72784 loss)
I0329 19:38:50.385905  2693 sgd_solver.cpp:138] Iteration 1400, lr = 0.0005
I0329 19:41:00.199553  2693 solver.cpp:243] Iteration 1500, loss = 5.93454
I0329 19:41:00.199800  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.37393 (* 1 = 6.37393 loss)
I0329 19:41:00.199815  2693 sgd_solver.cpp:138] Iteration 1500, lr = 0.0005
I0329 19:43:10.878293  2693 solver.cpp:243] Iteration 1600, loss = 5.90574
I0329 19:43:10.878466  2693 solver.cpp:259]     Train net output #0: mbox_loss = 7.04962 (* 1 = 7.04962 loss)
I0329 19:43:10.878484  2693 sgd_solver.cpp:138] Iteration 1600, lr = 0.0005
I0329 19:45:20.435961  2693 solver.cpp:243] Iteration 1700, loss = 5.90083
I0329 19:45:20.436149  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.02572 (* 1 = 6.02572 loss)
I0329 19:45:20.436167  2693 sgd_solver.cpp:138] Iteration 1700, lr = 0.0005
I0329 19:47:31.272306  2693 solver.cpp:243] Iteration 1800, loss = 5.87377
I0329 19:47:31.278728  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.21842 (* 1 = 5.21842 loss)
I0329 19:47:31.278760  2693 sgd_solver.cpp:138] Iteration 1800, lr = 0.0005
I0329 19:49:41.879673  2693 solver.cpp:243] Iteration 1900, loss = 5.7365
I0329 19:49:41.879866  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.63216 (* 1 = 6.63216 loss)
I0329 19:49:41.879884  2693 sgd_solver.cpp:138] Iteration 1900, lr = 0.0005
I0329 19:51:52.489728  2693 solver.cpp:243] Iteration 2000, loss = 5.62707
I0329 19:51:52.496809  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.02921 (* 1 = 5.02921 loss)
I0329 19:51:52.496830  2693 sgd_solver.cpp:138] Iteration 2000, lr = 0.0005
I0329 19:54:04.603685  2693 solver.cpp:243] Iteration 2100, loss = 5.44668
I0329 19:54:04.603883  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.29351 (* 1 = 6.29351 loss)
I0329 19:54:04.603900  2693 sgd_solver.cpp:138] Iteration 2100, lr = 0.0005
I0329 19:56:16.132683  2693 solver.cpp:243] Iteration 2200, loss = 5.67479
I0329 19:56:16.132900  2693 solver.cpp:259]     Train net output #0: mbox_loss = 8.37384 (* 1 = 8.37384 loss)
I0329 19:56:16.132933  2693 sgd_solver.cpp:138] Iteration 2200, lr = 0.0005
I0329 19:58:26.550333  2693 solver.cpp:243] Iteration 2300, loss = 5.59711
I0329 19:58:26.550503  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.17218 (* 1 = 6.17218 loss)
I0329 19:58:26.550519  2693 sgd_solver.cpp:138] Iteration 2300, lr = 0.0005
I0329 20:00:36.603052  2693 solver.cpp:243] Iteration 2400, loss = 5.63294
I0329 20:00:36.609419  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.69258 (* 1 = 5.69258 loss)
I0329 20:00:36.609452  2693 sgd_solver.cpp:138] Iteration 2400, lr = 0.0005
I0329 20:02:47.955036  2693 solver.cpp:243] Iteration 2500, loss = 5.46919
I0329 20:02:47.955307  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.36151 (* 1 = 6.36151 loss)
I0329 20:02:47.955350  2693 sgd_solver.cpp:138] Iteration 2500, lr = 0.0005
I0329 20:05:02.806021  2693 solver.cpp:243] Iteration 2600, loss = 5.57553
I0329 20:05:02.806228  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.94676 (* 1 = 4.94676 loss)
I0329 20:05:02.806246  2693 sgd_solver.cpp:138] Iteration 2600, lr = 0.0005
I0329 20:07:14.777077  2693 solver.cpp:243] Iteration 2700, loss = 5.57871
I0329 20:07:14.777251  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.52435 (* 1 = 6.52435 loss)
I0329 20:07:14.777267  2693 sgd_solver.cpp:138] Iteration 2700, lr = 0.0005
I0329 20:09:28.312669  2693 solver.cpp:243] Iteration 2800, loss = 5.45072
I0329 20:09:28.312940  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.31061 (* 1 = 5.31061 loss)
I0329 20:09:28.312983  2693 sgd_solver.cpp:138] Iteration 2800, lr = 0.0005
I0329 20:11:43.829903  2693 solver.cpp:243] Iteration 2900, loss = 5.37189
I0329 20:11:43.840242  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.75276 (* 1 = 4.75276 loss)
I0329 20:11:43.840273  2693 sgd_solver.cpp:138] Iteration 2900, lr = 0.0005
I0329 20:13:57.408138  2693 solver.cpp:243] Iteration 3000, loss = 5.52793
I0329 20:13:57.414752  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.58816 (* 1 = 5.58816 loss)
I0329 20:13:57.414785  2693 sgd_solver.cpp:138] Iteration 3000, lr = 0.0005
I0329 20:16:09.290681  2693 solver.cpp:243] Iteration 3100, loss = 5.36696
I0329 20:16:09.290877  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.80244 (* 1 = 5.80244 loss)
I0329 20:16:09.290894  2693 sgd_solver.cpp:138] Iteration 3100, lr = 0.0005
I0329 20:18:23.735085  2693 solver.cpp:243] Iteration 3200, loss = 5.50449
I0329 20:18:23.735275  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.89016 (* 1 = 4.89016 loss)
I0329 20:18:23.735291  2693 sgd_solver.cpp:138] Iteration 3200, lr = 0.0005
I0329 20:20:35.051250  2693 solver.cpp:243] Iteration 3300, loss = 5.50676
I0329 20:20:35.051407  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.52317 (* 1 = 4.52317 loss)
I0329 20:20:35.051424  2693 sgd_solver.cpp:138] Iteration 3300, lr = 0.0005
I0329 20:22:45.757282  2693 solver.cpp:243] Iteration 3400, loss = 5.32737
I0329 20:22:45.757524  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.04563 (* 1 = 5.04563 loss)
I0329 20:22:45.757560  2693 sgd_solver.cpp:138] Iteration 3400, lr = 0.0005
I0329 20:24:58.460191  2693 solver.cpp:243] Iteration 3500, loss = 5.28521
I0329 20:24:58.466637  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.10114 (* 1 = 5.10114 loss)
I0329 20:24:58.466667  2693 sgd_solver.cpp:138] Iteration 3500, lr = 0.0005
I0329 20:27:08.356668  2693 solver.cpp:243] Iteration 3600, loss = 5.37489
I0329 20:27:08.356879  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.86623 (* 1 = 5.86623 loss)
I0329 20:27:08.356897  2693 sgd_solver.cpp:138] Iteration 3600, lr = 0.0005
I0329 20:29:23.574539  2693 solver.cpp:243] Iteration 3700, loss = 5.26338
I0329 20:29:23.574729  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.877 (* 1 = 4.877 loss)
I0329 20:29:23.574748  2693 sgd_solver.cpp:138] Iteration 3700, lr = 0.0005
I0329 20:31:35.310338  2693 solver.cpp:243] Iteration 3800, loss = 5.1824
I0329 20:31:35.310528  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.49221 (* 1 = 4.49221 loss)
I0329 20:31:35.310545  2693 sgd_solver.cpp:138] Iteration 3800, lr = 0.0005
I0329 20:33:45.588671  2693 solver.cpp:243] Iteration 3900, loss = 5.10052
I0329 20:33:45.588862  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.12167 (* 1 = 5.12167 loss)
I0329 20:33:45.588879  2693 sgd_solver.cpp:138] Iteration 3900, lr = 0.0005
I0329 20:35:57.869002  2693 solver.cpp:243] Iteration 4000, loss = 5.24764
I0329 20:35:57.869282  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.16299 (* 1 = 4.16299 loss)
I0329 20:35:57.869345  2693 sgd_solver.cpp:138] Iteration 4000, lr = 0.0005
I0329 20:38:07.826603  2693 solver.cpp:243] Iteration 4100, loss = 5.10602
I0329 20:38:07.826776  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.4686 (* 1 = 5.4686 loss)
I0329 20:38:07.826792  2693 sgd_solver.cpp:138] Iteration 4100, lr = 0.0005
I0329 20:40:20.877024  2693 solver.cpp:243] Iteration 4200, loss = 5.19314
I0329 20:40:20.877231  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.02252 (* 1 = 5.02252 loss)
I0329 20:40:20.877251  2693 sgd_solver.cpp:138] Iteration 4200, lr = 0.0005
I0329 20:42:31.561018  2693 solver.cpp:243] Iteration 4300, loss = 5.18127
I0329 20:42:31.561192  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.70931 (* 1 = 4.70931 loss)
I0329 20:42:31.561210  2693 sgd_solver.cpp:138] Iteration 4300, lr = 0.0005
I0329 20:44:44.816411  2693 solver.cpp:243] Iteration 4400, loss = 5.28732
I0329 20:44:44.816588  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.27953 (* 1 = 5.27953 loss)
I0329 20:44:44.816606  2693 sgd_solver.cpp:138] Iteration 4400, lr = 0.0005
I0329 20:46:57.892287  2693 solver.cpp:243] Iteration 4500, loss = 5.20792
I0329 20:46:57.892504  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.73694 (* 1 = 6.73694 loss)
I0329 20:46:57.892535  2693 sgd_solver.cpp:138] Iteration 4500, lr = 0.0005
I0329 20:49:10.004372  2693 solver.cpp:243] Iteration 4600, loss = 5.14443
I0329 20:49:10.004554  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.12025 (* 1 = 5.12025 loss)
I0329 20:49:10.004571  2693 sgd_solver.cpp:138] Iteration 4600, lr = 0.0005
I0329 20:51:54.520535  2693 solver.cpp:243] Iteration 4700, loss = 5.17347
I0329 20:51:54.931277  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.80698 (* 1 = 4.80698 loss)
I0329 20:51:54.931314  2693 sgd_solver.cpp:138] Iteration 4700, lr = 0.0005
I0329 20:54:06.846974  2693 solver.cpp:243] Iteration 4800, loss = 5.11603
I0329 20:54:06.847178  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.75651 (* 1 = 4.75651 loss)
I0329 20:54:06.847211  2693 sgd_solver.cpp:138] Iteration 4800, lr = 0.0005
I0329 20:56:19.694252  2693 solver.cpp:243] Iteration 4900, loss = 4.98538
I0329 20:56:19.694571  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.06431 (* 1 = 4.06431 loss)
I0329 20:56:19.694589  2693 sgd_solver.cpp:138] Iteration 4900, lr = 0.0005
I0329 20:58:31.090090  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_5000.caffemodel
I0329 20:58:32.651293  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_5000.solverstate
I0329 20:58:32.805064  2693 solver.cpp:433] Iteration 5000, Testing net (#0)
I0329 20:58:33.740360  2693 net.cpp:693] Ignoring source layer mbox_loss
I0329 20:58:35.992341  2693 blocking_queue.cpp:50] Data layer prefetch queue empty
I0329 20:59:54.296596  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.401103
I0329 20:59:54.864137  2693 solver.cpp:243] Iteration 5000, loss = 5.14252
I0329 20:59:54.864203  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.94067 (* 1 = 5.94067 loss)
I0329 20:59:54.864219  2693 sgd_solver.cpp:138] Iteration 5000, lr = 0.0005
I0329 21:02:07.768394  2693 solver.cpp:243] Iteration 5100, loss = 5.25969
I0329 21:02:07.768606  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.40776 (* 1 = 4.40776 loss)
I0329 21:02:07.768630  2693 sgd_solver.cpp:138] Iteration 5100, lr = 0.0005
I0329 21:04:18.955862  2693 solver.cpp:243] Iteration 5200, loss = 5.19579
I0329 21:04:18.956115  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.86874 (* 1 = 5.86874 loss)
I0329 21:04:18.956156  2693 sgd_solver.cpp:138] Iteration 5200, lr = 0.0005
I0329 21:06:32.484187  2693 solver.cpp:243] Iteration 5300, loss = 4.98389
I0329 21:06:32.491014  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.30558 (* 1 = 4.30558 loss)
I0329 21:06:32.491042  2693 sgd_solver.cpp:138] Iteration 5300, lr = 0.0005
I0329 21:08:45.270756  2693 solver.cpp:243] Iteration 5400, loss = 4.98886
I0329 21:08:45.611340  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.5934 (* 1 = 6.5934 loss)
I0329 21:08:45.611372  2693 sgd_solver.cpp:138] Iteration 5400, lr = 0.0005
I0329 21:10:57.571789  2693 solver.cpp:243] Iteration 5500, loss = 4.99415
I0329 21:10:57.572021  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.63991 (* 1 = 4.63991 loss)
I0329 21:10:57.572057  2693 sgd_solver.cpp:138] Iteration 5500, lr = 0.0005
I0329 21:13:09.061786  2693 solver.cpp:243] Iteration 5600, loss = 5.10158
I0329 21:13:09.062021  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.68897 (* 1 = 4.68897 loss)
I0329 21:13:09.062054  2693 sgd_solver.cpp:138] Iteration 5600, lr = 0.0005
I0329 21:15:22.368721  2693 solver.cpp:243] Iteration 5700, loss = 5.15478
I0329 21:15:22.368922  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.39884 (* 1 = 5.39884 loss)
I0329 21:15:22.368938  2693 sgd_solver.cpp:138] Iteration 5700, lr = 0.0005
I0329 21:17:35.564406  2693 solver.cpp:243] Iteration 5800, loss = 5.02323
I0329 21:17:35.564627  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.07394 (* 1 = 6.07394 loss)
I0329 21:17:35.564658  2693 sgd_solver.cpp:138] Iteration 5800, lr = 0.0005
I0329 21:19:48.130075  2693 solver.cpp:243] Iteration 5900, loss = 4.96041
I0329 21:19:48.136293  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.08305 (* 1 = 3.08305 loss)
I0329 21:19:48.136334  2693 sgd_solver.cpp:138] Iteration 5900, lr = 0.0005
I0329 21:21:59.003654  2693 solver.cpp:243] Iteration 6000, loss = 5.03663
I0329 21:21:59.003909  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.24018 (* 1 = 5.24018 loss)
I0329 21:21:59.003957  2693 sgd_solver.cpp:138] Iteration 6000, lr = 0.0005
I0329 21:24:11.602039  2693 solver.cpp:243] Iteration 6100, loss = 5.0805
I0329 21:24:11.602246  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.56292 (* 1 = 4.56292 loss)
I0329 21:24:11.602277  2693 sgd_solver.cpp:138] Iteration 6100, lr = 0.0005
I0329 21:26:23.330701  2693 solver.cpp:243] Iteration 6200, loss = 4.9489
I0329 21:26:23.330924  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.15101 (* 1 = 6.15101 loss)
I0329 21:26:23.330958  2693 sgd_solver.cpp:138] Iteration 6200, lr = 0.0005
I0329 21:28:35.053517  2693 solver.cpp:243] Iteration 6300, loss = 4.92086
I0329 21:28:35.053742  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.44244 (* 1 = 5.44244 loss)
I0329 21:28:35.053776  2693 sgd_solver.cpp:138] Iteration 6300, lr = 0.0005
I0329 21:30:45.309222  2693 solver.cpp:243] Iteration 6400, loss = 4.99851
I0329 21:30:45.309414  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.60823 (* 1 = 5.60823 loss)
I0329 21:30:45.309432  2693 sgd_solver.cpp:138] Iteration 6400, lr = 0.0005
I0329 21:32:57.657336  2693 solver.cpp:243] Iteration 6500, loss = 5.06303
I0329 21:32:57.657553  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.87083 (* 1 = 5.87083 loss)
I0329 21:32:57.657572  2693 sgd_solver.cpp:138] Iteration 6500, lr = 0.0005
I0329 21:35:09.406425  2693 solver.cpp:243] Iteration 6600, loss = 4.91571
I0329 21:35:09.406651  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.63723 (* 1 = 5.63723 loss)
I0329 21:35:09.406682  2693 sgd_solver.cpp:138] Iteration 6600, lr = 0.0005
I0329 21:37:22.045250  2693 solver.cpp:243] Iteration 6700, loss = 4.90351
I0329 21:37:22.046876  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.33106 (* 1 = 4.33106 loss)
I0329 21:37:22.046901  2693 sgd_solver.cpp:138] Iteration 6700, lr = 0.0005
I0329 21:39:36.643456  2693 solver.cpp:243] Iteration 6800, loss = 5.0696
I0329 21:39:36.899454  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.93561 (* 1 = 4.93561 loss)
I0329 21:39:36.899484  2693 sgd_solver.cpp:138] Iteration 6800, lr = 0.0005
I0329 21:41:50.832686  2693 solver.cpp:243] Iteration 6900, loss = 4.99664
I0329 21:41:51.316522  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.49342 (* 1 = 5.49342 loss)
I0329 21:41:51.316555  2693 sgd_solver.cpp:138] Iteration 6900, lr = 0.0005
I0329 21:44:04.613561  2693 solver.cpp:243] Iteration 7000, loss = 5.03614
I0329 21:44:05.194190  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.01243 (* 1 = 4.01243 loss)
I0329 21:44:05.194218  2693 sgd_solver.cpp:138] Iteration 7000, lr = 0.0005
I0329 21:46:16.819299  2693 solver.cpp:243] Iteration 7100, loss = 4.96677
I0329 21:46:16.825995  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.68699 (* 1 = 4.68699 loss)
I0329 21:46:16.826021  2693 sgd_solver.cpp:138] Iteration 7100, lr = 0.0005
I0329 21:48:29.441328  2693 solver.cpp:243] Iteration 7200, loss = 4.85129
I0329 21:48:29.441611  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.93131 (* 1 = 4.93131 loss)
I0329 21:48:29.441638  2693 sgd_solver.cpp:138] Iteration 7200, lr = 0.0005
I0329 21:50:40.356317  2693 solver.cpp:243] Iteration 7300, loss = 5.01178
I0329 21:50:40.356621  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.0662 (* 1 = 4.0662 loss)
I0329 21:50:40.356657  2693 sgd_solver.cpp:138] Iteration 7300, lr = 0.0005
I0329 21:52:49.918732  2693 solver.cpp:243] Iteration 7400, loss = 4.99606
I0329 21:52:49.918938  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.53705 (* 1 = 4.53705 loss)
I0329 21:52:49.918956  2693 sgd_solver.cpp:138] Iteration 7400, lr = 0.0005
I0329 21:55:18.031137  2693 solver.cpp:243] Iteration 7500, loss = 4.99645
I0329 21:55:18.031513  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.40442 (* 1 = 5.40442 loss)
I0329 21:55:18.031548  2693 sgd_solver.cpp:138] Iteration 7500, lr = 0.0005
I0329 21:57:15.162822  2732 blocking_queue.cpp:50] Waiting for data
I0329 21:57:31.215286  2693 solver.cpp:243] Iteration 7600, loss = 4.92552
I0329 21:57:31.215354  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.0973 (* 1 = 5.0973 loss)
I0329 21:57:31.215369  2693 sgd_solver.cpp:138] Iteration 7600, lr = 0.0005
I0329 21:59:43.144016  2693 solver.cpp:243] Iteration 7700, loss = 4.75342
I0329 21:59:43.144176  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.64934 (* 1 = 4.64934 loss)
I0329 21:59:43.144191  2693 sgd_solver.cpp:138] Iteration 7700, lr = 0.0005
I0329 22:01:55.187448  2693 solver.cpp:243] Iteration 7800, loss = 4.88086
I0329 22:01:55.187736  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.21041 (* 1 = 5.21041 loss)
I0329 22:01:55.187782  2693 sgd_solver.cpp:138] Iteration 7800, lr = 0.0005
I0329 22:04:06.443691  2693 solver.cpp:243] Iteration 7900, loss = 4.80531
I0329 22:04:06.443994  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.84381 (* 1 = 4.84381 loss)
I0329 22:04:06.444031  2693 sgd_solver.cpp:138] Iteration 7900, lr = 0.0005
I0329 22:06:17.770081  2693 solver.cpp:243] Iteration 8000, loss = 4.9004
I0329 22:06:17.771339  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.31538 (* 1 = 4.31538 loss)
I0329 22:06:17.771446  2693 sgd_solver.cpp:138] Iteration 8000, lr = 0.0005
I0329 22:08:28.571830  2693 solver.cpp:243] Iteration 8100, loss = 4.82308
I0329 22:08:28.572073  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.20479 (* 1 = 5.20479 loss)
I0329 22:08:28.572108  2693 sgd_solver.cpp:138] Iteration 8100, lr = 0.0005
I0329 22:10:39.429178  2693 solver.cpp:243] Iteration 8200, loss = 5.01813
I0329 22:10:39.429388  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.42894 (* 1 = 4.42894 loss)
I0329 22:10:39.429414  2693 sgd_solver.cpp:138] Iteration 8200, lr = 0.0005
I0329 22:12:51.199815  2693 solver.cpp:243] Iteration 8300, loss = 4.92953
I0329 22:12:51.199990  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.47456 (* 1 = 6.47456 loss)
I0329 22:12:51.200007  2693 sgd_solver.cpp:138] Iteration 8300, lr = 0.0005
I0329 22:15:03.642390  2693 solver.cpp:243] Iteration 8400, loss = 4.70224
I0329 22:15:03.642593  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.8899 (* 1 = 4.8899 loss)
I0329 22:15:03.642609  2693 sgd_solver.cpp:138] Iteration 8400, lr = 0.0005
I0329 22:17:20.640007  2693 solver.cpp:243] Iteration 8500, loss = 4.94825
I0329 22:17:20.640277  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.32904 (* 1 = 4.32904 loss)
I0329 22:17:20.640329  2693 sgd_solver.cpp:138] Iteration 8500, lr = 0.0005
I0329 22:19:38.858456  2693 solver.cpp:243] Iteration 8600, loss = 4.85635
I0329 22:19:38.864791  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.80485 (* 1 = 4.80485 loss)
I0329 22:19:38.864832  2693 sgd_solver.cpp:138] Iteration 8600, lr = 0.0005
I0329 22:21:56.474875  2693 solver.cpp:243] Iteration 8700, loss = 4.90119
I0329 22:21:56.481612  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.02192 (* 1 = 5.02192 loss)
I0329 22:21:56.481649  2693 sgd_solver.cpp:138] Iteration 8700, lr = 0.0005
I0329 22:24:14.756258  2693 solver.cpp:243] Iteration 8800, loss = 4.98189
I0329 22:24:14.756451  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.29764 (* 1 = 4.29764 loss)
I0329 22:24:14.756471  2693 sgd_solver.cpp:138] Iteration 8800, lr = 0.0005
I0329 22:26:31.164304  2693 solver.cpp:243] Iteration 8900, loss = 4.85145
I0329 22:26:31.164535  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.22104 (* 1 = 3.22104 loss)
I0329 22:26:31.164572  2693 sgd_solver.cpp:138] Iteration 8900, lr = 0.0005
I0329 22:28:48.933522  2693 solver.cpp:243] Iteration 9000, loss = 4.78104
I0329 22:28:48.939111  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.11589 (* 1 = 4.11589 loss)
I0329 22:28:48.939151  2693 sgd_solver.cpp:138] Iteration 9000, lr = 0.0005
I0329 22:31:04.833420  2693 solver.cpp:243] Iteration 9100, loss = 4.85315
I0329 22:31:04.833657  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.39216 (* 1 = 5.39216 loss)
I0329 22:31:04.833691  2693 sgd_solver.cpp:138] Iteration 9100, lr = 0.0005
I0329 22:33:20.206395  2693 solver.cpp:243] Iteration 9200, loss = 4.93252
I0329 22:33:20.211748  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.1411 (* 1 = 4.1411 loss)
I0329 22:33:20.211833  2693 sgd_solver.cpp:138] Iteration 9200, lr = 0.0005
I0329 22:35:36.864924  2693 solver.cpp:243] Iteration 9300, loss = 4.71654
I0329 22:35:36.865139  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.49006 (* 1 = 3.49006 loss)
I0329 22:35:36.865156  2693 sgd_solver.cpp:138] Iteration 9300, lr = 0.0005
I0329 22:35:39.692414  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_9303.caffemodel
I0329 22:35:40.800824  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_9303.solverstate
I0329 22:37:55.911013  2693 solver.cpp:243] Iteration 9400, loss = 4.82481
I0329 22:37:55.918658  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.39144 (* 1 = 5.39144 loss)
I0329 22:37:55.918735  2693 sgd_solver.cpp:138] Iteration 9400, lr = 0.0005
I0329 22:40:12.334695  2693 solver.cpp:243] Iteration 9500, loss = 4.67769
I0329 22:40:12.334997  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.44078 (* 1 = 5.44078 loss)
I0329 22:40:12.335052  2693 sgd_solver.cpp:138] Iteration 9500, lr = 0.0005
I0329 22:42:29.671154  2693 solver.cpp:243] Iteration 9600, loss = 4.75427
I0329 22:42:29.671473  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.18257 (* 1 = 4.18257 loss)
I0329 22:42:29.671525  2693 sgd_solver.cpp:138] Iteration 9600, lr = 0.0005
I0329 22:44:45.669692  2693 solver.cpp:243] Iteration 9700, loss = 4.62037
I0329 22:44:45.676082  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.48045 (* 1 = 4.48045 loss)
I0329 22:44:45.676120  2693 sgd_solver.cpp:138] Iteration 9700, lr = 0.0005
I0329 22:47:02.915933  2693 solver.cpp:243] Iteration 9800, loss = 4.6765
I0329 22:47:02.916224  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.92231 (* 1 = 5.92231 loss)
I0329 22:47:02.916285  2693 sgd_solver.cpp:138] Iteration 9800, lr = 0.0005
I0329 22:49:22.255352  2693 solver.cpp:243] Iteration 9900, loss = 4.8626
I0329 22:49:22.262603  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.55325 (* 1 = 4.55325 loss)
I0329 22:49:22.262711  2693 sgd_solver.cpp:138] Iteration 9900, lr = 0.0005
I0329 22:51:39.640924  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_10000.caffemodel
I0329 22:51:40.883792  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_10000.solverstate
I0329 22:51:41.052144  2693 solver.cpp:433] Iteration 10000, Testing net (#0)
I0329 22:51:41.052237  2693 net.cpp:693] Ignoring source layer mbox_loss
I0329 22:53:06.768314  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.485075
I0329 22:53:07.535792  2693 solver.cpp:243] Iteration 10000, loss = 4.93095
I0329 22:53:07.535854  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.34415 (* 1 = 5.34415 loss)
I0329 22:53:07.535869  2693 sgd_solver.cpp:138] Iteration 10000, lr = 0.0005
I0329 22:55:27.576247  2693 solver.cpp:243] Iteration 10100, loss = 4.72693
I0329 22:55:27.576496  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.80132 (* 1 = 3.80132 loss)
I0329 22:55:27.576527  2693 sgd_solver.cpp:138] Iteration 10100, lr = 0.0005
I0329 22:57:45.234272  2693 solver.cpp:243] Iteration 10200, loss = 4.80319
I0329 22:57:45.234479  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.28131 (* 1 = 4.28131 loss)
I0329 22:57:45.234498  2693 sgd_solver.cpp:138] Iteration 10200, lr = 0.0005
I0329 23:00:02.593752  2693 solver.cpp:243] Iteration 10300, loss = 4.7151
I0329 23:00:02.594038  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.41927 (* 1 = 5.41927 loss)
I0329 23:00:02.594089  2693 sgd_solver.cpp:138] Iteration 10300, lr = 0.0005
I0329 23:02:20.015136  2693 solver.cpp:243] Iteration 10400, loss = 4.80142
I0329 23:02:20.015429  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.52599 (* 1 = 5.52599 loss)
I0329 23:02:20.015476  2693 sgd_solver.cpp:138] Iteration 10400, lr = 0.0005
I0329 23:04:37.721083  2693 solver.cpp:243] Iteration 10500, loss = 4.89859
I0329 23:04:37.729730  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.65096 (* 1 = 4.65096 loss)
I0329 23:04:37.729751  2693 sgd_solver.cpp:138] Iteration 10500, lr = 0.0005
I0329 23:06:54.770097  2693 solver.cpp:243] Iteration 10600, loss = 4.80272
I0329 23:06:54.770403  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.5073 (* 1 = 5.5073 loss)
I0329 23:06:54.770465  2693 sgd_solver.cpp:138] Iteration 10600, lr = 0.0005
I0329 23:09:11.892266  2693 solver.cpp:243] Iteration 10700, loss = 4.79062
I0329 23:09:11.892560  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.81805 (* 1 = 4.81805 loss)
I0329 23:09:11.892621  2693 sgd_solver.cpp:138] Iteration 10700, lr = 0.0005
I0329 23:11:22.393043  2693 solver.cpp:243] Iteration 10800, loss = 4.67865
I0329 23:11:22.393352  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.09443 (* 1 = 4.09443 loss)
I0329 23:11:22.393373  2693 sgd_solver.cpp:138] Iteration 10800, lr = 0.0005
I0329 23:13:35.638540  2693 solver.cpp:243] Iteration 10900, loss = 4.8242
I0329 23:13:35.638749  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.98117 (* 1 = 5.98117 loss)
I0329 23:13:35.638769  2693 sgd_solver.cpp:138] Iteration 10900, lr = 0.0005
I0329 23:15:49.292356  2693 solver.cpp:243] Iteration 11000, loss = 4.70495
I0329 23:15:49.292678  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.02707 (* 1 = 5.02707 loss)
I0329 23:15:49.292748  2693 sgd_solver.cpp:138] Iteration 11000, lr = 0.0005
I0329 23:18:02.877568  2693 solver.cpp:243] Iteration 11100, loss = 4.57496
I0329 23:18:02.883939  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.86243 (* 1 = 5.86243 loss)
I0329 23:18:02.883972  2693 sgd_solver.cpp:138] Iteration 11100, lr = 0.0005
I0329 23:20:15.005007  2693 solver.cpp:243] Iteration 11200, loss = 4.69886
I0329 23:20:15.005182  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.80269 (* 1 = 4.80269 loss)
I0329 23:20:15.005198  2693 sgd_solver.cpp:138] Iteration 11200, lr = 0.0005
I0329 23:22:27.842006  2693 solver.cpp:243] Iteration 11300, loss = 4.79562
I0329 23:22:27.842244  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.94525 (* 1 = 5.94525 loss)
I0329 23:22:27.842267  2693 sgd_solver.cpp:138] Iteration 11300, lr = 0.0005
I0329 23:24:42.046342  2693 solver.cpp:243] Iteration 11400, loss = 4.823
I0329 23:24:42.046610  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.87279 (* 1 = 3.87279 loss)
I0329 23:24:42.046629  2693 sgd_solver.cpp:138] Iteration 11400, lr = 0.0005
I0329 23:26:55.768770  2693 solver.cpp:243] Iteration 11500, loss = 4.7317
I0329 23:26:55.768954  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.19163 (* 1 = 4.19163 loss)
I0329 23:26:55.768972  2693 sgd_solver.cpp:138] Iteration 11500, lr = 0.0005
I0329 23:29:07.645275  2693 solver.cpp:243] Iteration 11600, loss = 4.78679
I0329 23:29:07.645542  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.47826 (* 1 = 4.47826 loss)
I0329 23:29:07.645584  2693 sgd_solver.cpp:138] Iteration 11600, lr = 0.0005
I0329 23:31:19.584727  2693 solver.cpp:243] Iteration 11700, loss = 4.74368
I0329 23:31:19.592048  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.13807 (* 1 = 4.13807 loss)
I0329 23:31:19.592077  2693 sgd_solver.cpp:138] Iteration 11700, lr = 0.0005
I0329 23:33:31.154393  2693 solver.cpp:243] Iteration 11800, loss = 4.71801
I0329 23:33:31.154718  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.46321 (* 1 = 4.46321 loss)
I0329 23:33:31.154783  2693 sgd_solver.cpp:138] Iteration 11800, lr = 0.0005
I0329 23:35:44.820700  2693 solver.cpp:243] Iteration 11900, loss = 4.71684
I0329 23:35:44.820938  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.07002 (* 1 = 5.07002 loss)
I0329 23:35:44.820960  2693 sgd_solver.cpp:138] Iteration 11900, lr = 0.0005
I0329 23:37:56.242718  2693 solver.cpp:243] Iteration 12000, loss = 4.81818
I0329 23:37:56.243043  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.09915 (* 1 = 5.09915 loss)
I0329 23:37:56.243104  2693 sgd_solver.cpp:138] Iteration 12000, lr = 0.0005
I0329 23:40:11.580638  2693 solver.cpp:243] Iteration 12100, loss = 4.74793
I0329 23:40:11.580906  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.39728 (* 1 = 4.39728 loss)
I0329 23:40:11.580936  2693 sgd_solver.cpp:138] Iteration 12100, lr = 0.0005
I0329 23:42:24.570663  2693 solver.cpp:243] Iteration 12200, loss = 4.60578
I0329 23:42:24.573253  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.05972 (* 1 = 4.05972 loss)
I0329 23:42:24.573271  2693 sgd_solver.cpp:138] Iteration 12200, lr = 0.0005
I0329 23:44:36.716467  2693 solver.cpp:243] Iteration 12300, loss = 4.68483
I0329 23:44:36.716658  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.92509 (* 1 = 4.92509 loss)
I0329 23:44:36.716675  2693 sgd_solver.cpp:138] Iteration 12300, lr = 0.0005
I0329 23:46:50.473526  2693 solver.cpp:243] Iteration 12400, loss = 4.73954
I0329 23:46:50.485723  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.57026 (* 1 = 4.57026 loss)
I0329 23:46:50.485752  2693 sgd_solver.cpp:138] Iteration 12400, lr = 0.0005
I0329 23:49:05.121201  2693 solver.cpp:243] Iteration 12500, loss = 4.72998
I0329 23:49:05.121500  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.26276 (* 1 = 6.26276 loss)
I0329 23:49:05.121561  2693 sgd_solver.cpp:138] Iteration 12500, lr = 0.0005
I0329 23:51:21.042115  2693 solver.cpp:243] Iteration 12600, loss = 4.88205
I0329 23:51:21.042304  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.78332 (* 1 = 4.78332 loss)
I0329 23:51:21.042321  2693 sgd_solver.cpp:138] Iteration 12600, lr = 0.0005
I0329 23:53:31.812664  2693 solver.cpp:243] Iteration 12700, loss = 4.66833
I0329 23:53:31.812872  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.59365 (* 1 = 6.59365 loss)
I0329 23:53:31.812901  2693 sgd_solver.cpp:138] Iteration 12700, lr = 0.0005
I0329 23:55:41.771445  2693 solver.cpp:243] Iteration 12800, loss = 4.57465
I0329 23:55:41.771756  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.90979 (* 1 = 4.90979 loss)
I0329 23:55:41.771776  2693 sgd_solver.cpp:138] Iteration 12800, lr = 0.0005
I0329 23:57:53.674958  2693 solver.cpp:243] Iteration 12900, loss = 4.8525
I0329 23:57:53.675201  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.1154 (* 1 = 5.1154 loss)
I0329 23:57:53.675235  2693 sgd_solver.cpp:138] Iteration 12900, lr = 0.0005
I0330 00:00:05.060937  2693 solver.cpp:243] Iteration 13000, loss = 4.6448
I0330 00:00:05.061203  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.58259 (* 1 = 4.58259 loss)
I0330 00:00:05.061236  2693 sgd_solver.cpp:138] Iteration 13000, lr = 0.0005
I0330 00:02:16.720141  2693 solver.cpp:243] Iteration 13100, loss = 4.6328
I0330 00:02:16.720376  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.0938 (* 1 = 5.0938 loss)
I0330 00:02:16.720393  2693 sgd_solver.cpp:138] Iteration 13100, lr = 0.0005
I0330 00:04:30.784581  2693 solver.cpp:243] Iteration 13200, loss = 4.74738
I0330 00:04:30.784947  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.46546 (* 1 = 4.46546 loss)
I0330 00:04:30.785008  2693 sgd_solver.cpp:138] Iteration 13200, lr = 0.0005
I0330 00:06:43.822721  2693 solver.cpp:243] Iteration 13300, loss = 4.63024
I0330 00:06:43.822971  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.70248 (* 1 = 3.70248 loss)
I0330 00:06:43.823019  2693 sgd_solver.cpp:138] Iteration 13300, lr = 0.0005
I0330 00:08:53.651729  2693 solver.cpp:243] Iteration 13400, loss = 4.61489
I0330 00:08:53.651954  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.01211 (* 1 = 5.01211 loss)
I0330 00:08:53.651993  2693 sgd_solver.cpp:138] Iteration 13400, lr = 0.0005
I0330 00:11:10.017709  2693 solver.cpp:243] Iteration 13500, loss = 4.7435
I0330 00:11:10.018051  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.80356 (* 1 = 3.80356 loss)
I0330 00:11:10.018072  2693 sgd_solver.cpp:138] Iteration 13500, lr = 0.0005
I0330 00:13:27.125840  2693 solver.cpp:243] Iteration 13600, loss = 4.69438
I0330 00:13:27.126047  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.9292 (* 1 = 3.9292 loss)
I0330 00:13:27.126066  2693 sgd_solver.cpp:138] Iteration 13600, lr = 0.0005
I0330 00:15:43.834480  2693 solver.cpp:243] Iteration 13700, loss = 4.6832
I0330 00:15:43.834702  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.75485 (* 1 = 3.75485 loss)
I0330 00:15:43.834720  2693 sgd_solver.cpp:138] Iteration 13700, lr = 0.0005
I0330 00:17:59.341194  2693 solver.cpp:243] Iteration 13800, loss = 4.69949
I0330 00:17:59.341430  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.52372 (* 1 = 4.52372 loss)
I0330 00:17:59.341475  2693 sgd_solver.cpp:138] Iteration 13800, lr = 0.0005
I0330 00:20:13.825486  2693 solver.cpp:243] Iteration 13900, loss = 4.67162
I0330 00:20:13.825980  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.29082 (* 1 = 4.29082 loss)
I0330 00:20:13.826000  2693 sgd_solver.cpp:138] Iteration 13900, lr = 0.0005
I0330 00:22:30.213547  2693 solver.cpp:243] Iteration 14000, loss = 4.62469
I0330 00:22:30.213846  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.71831 (* 1 = 4.71831 loss)
I0330 00:22:30.213863  2693 sgd_solver.cpp:138] Iteration 14000, lr = 0.0005
I0330 00:24:45.055068  2693 solver.cpp:243] Iteration 14100, loss = 4.52537
I0330 00:24:45.055313  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.66709 (* 1 = 5.66709 loss)
I0330 00:24:45.055331  2693 sgd_solver.cpp:138] Iteration 14100, lr = 0.0005
I0330 00:27:00.584302  2693 solver.cpp:243] Iteration 14200, loss = 4.65247
I0330 00:27:00.584483  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.19166 (* 1 = 3.19166 loss)
I0330 00:27:00.584499  2693 sgd_solver.cpp:138] Iteration 14200, lr = 0.0005
I0330 00:29:17.721415  2693 solver.cpp:243] Iteration 14300, loss = 4.70114
I0330 00:29:17.721602  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.73319 (* 1 = 4.73319 loss)
I0330 00:29:17.721621  2693 sgd_solver.cpp:138] Iteration 14300, lr = 0.0005
I0330 00:31:35.437649  2693 solver.cpp:243] Iteration 14400, loss = 4.74614
I0330 00:31:35.437863  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.55339 (* 1 = 4.55339 loss)
I0330 00:31:35.437893  2693 sgd_solver.cpp:138] Iteration 14400, lr = 0.0005
I0330 00:33:52.244645  2693 solver.cpp:243] Iteration 14500, loss = 4.63275
I0330 00:33:52.244820  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.74476 (* 1 = 4.74476 loss)
I0330 00:33:52.244837  2693 sgd_solver.cpp:138] Iteration 14500, lr = 0.0005
I0330 00:36:09.820199  2693 solver.cpp:243] Iteration 14600, loss = 4.64986
I0330 00:36:09.820385  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.98381 (* 1 = 5.98381 loss)
I0330 00:36:09.820402  2693 sgd_solver.cpp:138] Iteration 14600, lr = 0.0005
I0330 00:38:24.657032  2693 solver.cpp:243] Iteration 14700, loss = 4.57522
I0330 00:38:24.657368  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.64316 (* 1 = 3.64316 loss)
I0330 00:38:24.657402  2693 sgd_solver.cpp:138] Iteration 14700, lr = 0.0005
I0330 00:40:41.186789  2693 solver.cpp:243] Iteration 14800, loss = 4.68935
I0330 00:40:41.187011  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.00126 (* 1 = 3.00126 loss)
I0330 00:40:41.187054  2693 sgd_solver.cpp:138] Iteration 14800, lr = 0.0005
I0330 00:42:56.626608  2693 solver.cpp:243] Iteration 14900, loss = 4.58861
I0330 00:42:56.626811  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.34845 (* 1 = 6.34845 loss)
I0330 00:42:56.626842  2693 sgd_solver.cpp:138] Iteration 14900, lr = 0.0005
I0330 00:45:11.471287  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_15000.caffemodel
I0330 00:45:12.593036  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_15000.solverstate
I0330 00:45:12.722533  2693 solver.cpp:433] Iteration 15000, Testing net (#0)
I0330 00:45:12.722618  2693 net.cpp:693] Ignoring source layer mbox_loss
I0330 00:46:34.426759  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.495681
I0330 00:46:35.140130  2693 solver.cpp:243] Iteration 15000, loss = 4.60863
I0330 00:46:35.140204  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.78452 (* 1 = 4.78452 loss)
I0330 00:46:35.140223  2693 sgd_solver.cpp:138] Iteration 15000, lr = 0.0005
I0330 00:48:50.303750  2693 solver.cpp:243] Iteration 15100, loss = 4.58632
I0330 00:48:50.303920  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.78939 (* 1 = 4.78939 loss)
I0330 00:48:50.303937  2693 sgd_solver.cpp:138] Iteration 15100, lr = 0.0005
I0330 00:51:07.361608  2693 solver.cpp:243] Iteration 15200, loss = 4.62292
I0330 00:51:07.361815  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.57644 (* 1 = 4.57644 loss)
I0330 00:51:07.361840  2693 sgd_solver.cpp:138] Iteration 15200, lr = 0.0005
I0330 00:53:23.905481  2693 solver.cpp:243] Iteration 15300, loss = 4.71449
I0330 00:53:23.905709  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.8516 (* 1 = 4.8516 loss)
I0330 00:53:23.905727  2693 sgd_solver.cpp:138] Iteration 15300, lr = 0.0005
I0330 00:55:39.226521  2693 solver.cpp:243] Iteration 15400, loss = 4.4843
I0330 00:55:39.226719  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.16684 (* 1 = 4.16684 loss)
I0330 00:55:39.226737  2693 sgd_solver.cpp:138] Iteration 15400, lr = 0.0005
I0330 00:57:57.448170  2693 solver.cpp:243] Iteration 15500, loss = 4.61347
I0330 00:57:57.448467  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.16077 (* 1 = 4.16077 loss)
I0330 00:57:57.448485  2693 sgd_solver.cpp:138] Iteration 15500, lr = 0.0005
I0330 01:00:11.664471  2693 solver.cpp:243] Iteration 15600, loss = 4.74094
I0330 01:00:11.664952  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.99107 (* 1 = 4.99107 loss)
I0330 01:00:11.664971  2693 sgd_solver.cpp:138] Iteration 15600, lr = 0.0005
I0330 01:02:28.846441  2693 solver.cpp:243] Iteration 15700, loss = 4.7141
I0330 01:02:28.846695  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.70436 (* 1 = 6.70436 loss)
I0330 01:02:28.846730  2693 sgd_solver.cpp:138] Iteration 15700, lr = 0.0005
I0330 01:04:44.194510  2693 solver.cpp:243] Iteration 15800, loss = 4.54868
I0330 01:04:44.194835  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.49755 (* 1 = 4.49755 loss)
I0330 01:04:44.194919  2693 sgd_solver.cpp:138] Iteration 15800, lr = 0.0005
I0330 01:06:58.713863  2693 solver.cpp:243] Iteration 15900, loss = 4.53983
I0330 01:06:58.714145  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.55199 (* 1 = 4.55199 loss)
I0330 01:06:58.714185  2693 sgd_solver.cpp:138] Iteration 15900, lr = 0.0005
I0330 01:09:13.987426  2693 solver.cpp:243] Iteration 16000, loss = 4.65464
I0330 01:09:13.987793  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.8447 (* 1 = 3.8447 loss)
I0330 01:09:13.987814  2693 sgd_solver.cpp:138] Iteration 16000, lr = 0.0005
I0330 01:11:28.879386  2693 solver.cpp:243] Iteration 16100, loss = 4.65317
I0330 01:11:28.879659  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.9403 (* 1 = 3.9403 loss)
I0330 01:11:28.879683  2693 sgd_solver.cpp:138] Iteration 16100, lr = 0.0005
I0330 01:13:44.880924  2693 solver.cpp:243] Iteration 16200, loss = 4.64974
I0330 01:13:44.881175  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.69981 (* 1 = 4.69981 loss)
I0330 01:13:44.881213  2693 sgd_solver.cpp:138] Iteration 16200, lr = 0.0005
I0330 01:16:01.490547  2693 solver.cpp:243] Iteration 16300, loss = 4.67896
I0330 01:16:01.490777  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.2255 (* 1 = 5.2255 loss)
I0330 01:16:01.490809  2693 sgd_solver.cpp:138] Iteration 16300, lr = 0.0005
I0330 01:18:18.319077  2693 solver.cpp:243] Iteration 16400, loss = 4.64929
I0330 01:18:18.319350  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.58392 (* 1 = 4.58392 loss)
I0330 01:18:18.319365  2693 sgd_solver.cpp:138] Iteration 16400, lr = 0.0005
I0330 01:20:33.631747  2693 solver.cpp:243] Iteration 16500, loss = 4.53036
I0330 01:20:33.631963  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.03005 (* 1 = 5.03005 loss)
I0330 01:20:33.631980  2693 sgd_solver.cpp:138] Iteration 16500, lr = 0.0005
I0330 01:22:49.985211  2693 solver.cpp:243] Iteration 16600, loss = 4.35652
I0330 01:22:49.985404  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.03496 (* 1 = 4.03496 loss)
I0330 01:22:49.985419  2693 sgd_solver.cpp:138] Iteration 16600, lr = 0.0005
I0330 01:25:06.595266  2693 solver.cpp:243] Iteration 16700, loss = 4.57449
I0330 01:25:06.595517  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.57174 (* 1 = 3.57174 loss)
I0330 01:25:06.595552  2693 sgd_solver.cpp:138] Iteration 16700, lr = 0.0005
I0330 01:27:20.666453  2693 solver.cpp:243] Iteration 16800, loss = 4.48923
I0330 01:27:20.666769  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.94818 (* 1 = 4.94818 loss)
I0330 01:27:20.666788  2693 sgd_solver.cpp:138] Iteration 16800, lr = 0.0005
I0330 01:29:34.393844  2693 solver.cpp:243] Iteration 16900, loss = 4.51693
I0330 01:29:34.394085  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.40248 (* 1 = 3.40248 loss)
I0330 01:29:34.394117  2693 sgd_solver.cpp:138] Iteration 16900, lr = 0.0005
I0330 01:31:52.928884  2693 solver.cpp:243] Iteration 17000, loss = 4.70268
I0330 01:31:52.929075  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.09314 (* 1 = 4.09314 loss)
I0330 01:31:52.929091  2693 sgd_solver.cpp:138] Iteration 17000, lr = 0.0005
I0330 01:34:09.310108  2693 solver.cpp:243] Iteration 17100, loss = 4.55107
I0330 01:34:09.310397  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.2362 (* 1 = 5.2362 loss)
I0330 01:34:09.310432  2693 sgd_solver.cpp:138] Iteration 17100, lr = 0.0005
I0330 01:36:24.751343  2693 solver.cpp:243] Iteration 17200, loss = 4.50744
I0330 01:36:24.751651  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.02487 (* 1 = 5.02487 loss)
I0330 01:36:24.751687  2693 sgd_solver.cpp:138] Iteration 17200, lr = 0.0005
I0330 01:38:40.519085  2693 solver.cpp:243] Iteration 17300, loss = 4.48705
I0330 01:38:40.519275  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.04831 (* 1 = 3.04831 loss)
I0330 01:38:40.519291  2693 sgd_solver.cpp:138] Iteration 17300, lr = 0.0005
I0330 01:40:55.924392  2693 solver.cpp:243] Iteration 17400, loss = 4.59952
I0330 01:40:55.924696  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.61519 (* 1 = 4.61519 loss)
I0330 01:40:55.924712  2693 sgd_solver.cpp:138] Iteration 17400, lr = 0.0005
I0330 01:43:12.293264  2693 solver.cpp:243] Iteration 17500, loss = 4.68487
I0330 01:43:12.293507  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.13288 (* 1 = 4.13288 loss)
I0330 01:43:12.293541  2693 sgd_solver.cpp:138] Iteration 17500, lr = 0.0005
I0330 01:45:27.644819  2693 solver.cpp:243] Iteration 17600, loss = 4.65594
I0330 01:45:27.645000  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.97603 (* 1 = 3.97603 loss)
I0330 01:45:27.645017  2693 sgd_solver.cpp:138] Iteration 17600, lr = 0.0005
I0330 01:47:43.223651  2693 solver.cpp:243] Iteration 17700, loss = 4.61037
I0330 01:47:43.230439  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.20825 (* 1 = 5.20825 loss)
I0330 01:47:43.230494  2693 sgd_solver.cpp:138] Iteration 17700, lr = 0.0005
I0330 01:49:59.569056  2693 solver.cpp:243] Iteration 17800, loss = 4.67859
I0330 01:49:59.569388  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.45803 (* 1 = 4.45803 loss)
I0330 01:49:59.569406  2693 sgd_solver.cpp:138] Iteration 17800, lr = 0.0005
I0330 01:52:15.440850  2693 solver.cpp:243] Iteration 17900, loss = 4.63094
I0330 01:52:15.441148  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.96754 (* 1 = 4.96754 loss)
I0330 01:52:15.441200  2693 sgd_solver.cpp:138] Iteration 17900, lr = 0.0005
I0330 01:54:31.566049  2693 solver.cpp:243] Iteration 18000, loss = 4.62686
I0330 01:54:31.566293  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.94946 (* 1 = 3.94946 loss)
I0330 01:54:31.566331  2693 sgd_solver.cpp:138] Iteration 18000, lr = 0.0005
I0330 01:56:48.059448  2693 solver.cpp:243] Iteration 18100, loss = 4.626
I0330 01:56:48.059777  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.76309 (* 1 = 4.76309 loss)
I0330 01:56:48.059819  2693 sgd_solver.cpp:138] Iteration 18100, lr = 0.0005
I0330 01:59:05.551978  2693 solver.cpp:243] Iteration 18200, loss = 4.65141
I0330 01:59:05.552295  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.20059 (* 1 = 5.20059 loss)
I0330 01:59:05.552345  2693 sgd_solver.cpp:138] Iteration 18200, lr = 0.0005
I0330 02:01:21.000074  2693 solver.cpp:243] Iteration 18300, loss = 4.52771
I0330 02:01:21.000351  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.49044 (* 1 = 4.49044 loss)
I0330 02:01:21.000372  2693 sgd_solver.cpp:138] Iteration 18300, lr = 0.0005
I0330 02:03:36.781689  2693 solver.cpp:243] Iteration 18400, loss = 4.53701
I0330 02:03:36.782017  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.32011 (* 1 = 5.32011 loss)
I0330 02:03:36.782057  2693 sgd_solver.cpp:138] Iteration 18400, lr = 0.0005
I0330 02:05:53.193099  2693 solver.cpp:243] Iteration 18500, loss = 4.45766
I0330 02:05:53.193306  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.52824 (* 1 = 6.52824 loss)
I0330 02:05:53.193363  2693 sgd_solver.cpp:138] Iteration 18500, lr = 0.0005
I0330 02:08:07.300704  2693 solver.cpp:243] Iteration 18600, loss = 4.65468
I0330 02:08:07.300972  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.68796 (* 1 = 5.68796 loss)
I0330 02:08:07.301002  2693 sgd_solver.cpp:138] Iteration 18600, lr = 0.0005
I0330 02:10:22.102530  2693 solver.cpp:243] Iteration 18700, loss = 4.45404
I0330 02:10:22.102929  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.73657 (* 1 = 4.73657 loss)
I0330 02:10:22.102946  2693 sgd_solver.cpp:138] Iteration 18700, lr = 0.0005
I0330 02:12:38.938976  2693 solver.cpp:243] Iteration 18800, loss = 4.58965
I0330 02:12:38.939203  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.3885 (* 1 = 4.3885 loss)
I0330 02:12:38.939232  2693 sgd_solver.cpp:138] Iteration 18800, lr = 0.0005
I0330 02:14:56.525140  2693 solver.cpp:243] Iteration 18900, loss = 4.53281
I0330 02:14:56.525377  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.34508 (* 1 = 5.34508 loss)
I0330 02:14:56.525393  2693 sgd_solver.cpp:138] Iteration 18900, lr = 0.0005
I0330 02:17:11.773288  2693 solver.cpp:243] Iteration 19000, loss = 4.36263
I0330 02:17:11.773488  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.93671 (* 1 = 3.93671 loss)
I0330 02:17:11.773507  2693 sgd_solver.cpp:138] Iteration 19000, lr = 0.0005
I0330 02:19:26.958634  2693 solver.cpp:243] Iteration 19100, loss = 4.50377
I0330 02:19:26.958850  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.47453 (* 1 = 4.47453 loss)
I0330 02:19:26.958879  2693 sgd_solver.cpp:138] Iteration 19100, lr = 0.0005
I0330 02:21:41.584997  2693 solver.cpp:243] Iteration 19200, loss = 4.64974
I0330 02:21:41.587757  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.48251 (* 1 = 3.48251 loss)
I0330 02:21:41.587790  2693 sgd_solver.cpp:138] Iteration 19200, lr = 0.0005
I0330 02:23:59.427170  2693 solver.cpp:243] Iteration 19300, loss = 4.45844
I0330 02:23:59.427368  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.6754 (* 1 = 3.6754 loss)
I0330 02:23:59.427386  2693 sgd_solver.cpp:138] Iteration 19300, lr = 0.0005
I0330 02:26:13.359305  2693 solver.cpp:243] Iteration 19400, loss = 4.44964
I0330 02:26:13.360137  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.68783 (* 1 = 4.68783 loss)
I0330 02:26:13.360157  2693 sgd_solver.cpp:138] Iteration 19400, lr = 0.0005
I0330 02:28:30.837062  2693 solver.cpp:243] Iteration 19500, loss = 4.56582
I0330 02:28:30.837378  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.59638 (* 1 = 4.59638 loss)
I0330 02:28:30.837396  2693 sgd_solver.cpp:138] Iteration 19500, lr = 0.0005
I0330 02:30:47.388339  2693 solver.cpp:243] Iteration 19600, loss = 4.54263
I0330 02:30:47.390741  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.00068 (* 1 = 5.00068 loss)
I0330 02:30:47.390761  2693 sgd_solver.cpp:138] Iteration 19600, lr = 0.0005
I0330 02:33:03.519503  2693 solver.cpp:243] Iteration 19700, loss = 4.38874
I0330 02:33:03.519896  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.66799 (* 1 = 4.66799 loss)
I0330 02:33:03.519979  2693 sgd_solver.cpp:138] Iteration 19700, lr = 0.0005
I0330 02:35:19.628679  2693 solver.cpp:243] Iteration 19800, loss = 4.4911
I0330 02:35:19.628865  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.26191 (* 1 = 4.26191 loss)
I0330 02:35:19.628881  2693 sgd_solver.cpp:138] Iteration 19800, lr = 0.0005
I0330 02:37:37.062294  2693 solver.cpp:243] Iteration 19900, loss = 4.57765
I0330 02:37:37.062624  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.62687 (* 1 = 4.62687 loss)
I0330 02:37:37.062644  2693 sgd_solver.cpp:138] Iteration 19900, lr = 0.0005
I0330 02:39:51.423709  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_20000.caffemodel
I0330 02:39:52.624141  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_20000.solverstate
I0330 02:39:52.764233  2693 solver.cpp:433] Iteration 20000, Testing net (#0)
I0330 02:39:52.764351  2693 net.cpp:693] Ignoring source layer mbox_loss
I0330 02:41:14.339416  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.501046
I0330 02:41:15.102371  2693 solver.cpp:243] Iteration 20000, loss = 4.48824
I0330 02:41:15.102449  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.11317 (* 1 = 5.11317 loss)
I0330 02:41:15.102468  2693 sgd_solver.cpp:138] Iteration 20000, lr = 0.0005
I0330 02:43:33.608420  2693 solver.cpp:243] Iteration 20100, loss = 4.62964
I0330 02:43:33.608628  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.6743 (* 1 = 3.6743 loss)
I0330 02:43:33.608645  2693 sgd_solver.cpp:138] Iteration 20100, lr = 0.0005
I0330 02:45:49.179862  2693 solver.cpp:243] Iteration 20200, loss = 4.51438
I0330 02:45:49.180059  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.25134 (* 1 = 5.25134 loss)
I0330 02:45:49.180080  2693 sgd_solver.cpp:138] Iteration 20200, lr = 0.0005
I0330 02:48:05.676407  2693 solver.cpp:243] Iteration 20300, loss = 4.58101
I0330 02:48:05.676632  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.92584 (* 1 = 5.92584 loss)
I0330 02:48:05.676661  2693 sgd_solver.cpp:138] Iteration 20300, lr = 0.0005
I0330 02:50:19.416460  2693 solver.cpp:243] Iteration 20400, loss = 4.46792
I0330 02:50:19.416726  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.26565 (* 1 = 5.26565 loss)
I0330 02:50:19.416743  2693 sgd_solver.cpp:138] Iteration 20400, lr = 0.0005
I0330 02:52:34.283073  2693 solver.cpp:243] Iteration 20500, loss = 4.54833
I0330 02:52:34.283327  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.81968 (* 1 = 5.81968 loss)
I0330 02:52:34.283356  2693 sgd_solver.cpp:138] Iteration 20500, lr = 0.0005
I0330 02:54:49.310312  2693 solver.cpp:243] Iteration 20600, loss = 4.40177
I0330 02:54:49.310493  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.55421 (* 1 = 5.55421 loss)
I0330 02:54:49.310509  2693 sgd_solver.cpp:138] Iteration 20600, lr = 0.0005
I0330 02:57:02.974444  2693 solver.cpp:243] Iteration 20700, loss = 4.53247
I0330 02:57:02.974642  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.90625 (* 1 = 3.90625 loss)
I0330 02:57:02.974658  2693 sgd_solver.cpp:138] Iteration 20700, lr = 0.0005
I0330 02:59:18.297168  2693 solver.cpp:243] Iteration 20800, loss = 4.42237
I0330 02:59:18.297360  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.76904 (* 1 = 4.76904 loss)
I0330 02:59:18.297380  2693 sgd_solver.cpp:138] Iteration 20800, lr = 0.0005
I0330 03:01:32.496549  2693 solver.cpp:243] Iteration 20900, loss = 4.48371
I0330 03:01:32.496748  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.26872 (* 1 = 4.26872 loss)
I0330 03:01:32.496772  2693 sgd_solver.cpp:138] Iteration 20900, lr = 0.0005
I0330 03:03:45.210898  2693 solver.cpp:243] Iteration 21000, loss = 4.53538
I0330 03:03:45.211074  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.35925 (* 1 = 4.35925 loss)
I0330 03:03:45.211094  2693 sgd_solver.cpp:138] Iteration 21000, lr = 0.0005
I0330 03:06:01.626416  2693 solver.cpp:243] Iteration 21100, loss = 4.53489
I0330 03:06:01.626598  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.09332 (* 1 = 5.09332 loss)
I0330 03:06:01.626616  2693 sgd_solver.cpp:138] Iteration 21100, lr = 0.0005
I0330 03:08:17.316156  2693 solver.cpp:243] Iteration 21200, loss = 4.59974
I0330 03:08:17.316383  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.68449 (* 1 = 4.68449 loss)
I0330 03:08:17.316402  2693 sgd_solver.cpp:138] Iteration 21200, lr = 0.0005
I0330 03:10:32.200176  2693 solver.cpp:243] Iteration 21300, loss = 4.60723
I0330 03:10:32.200460  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.2776 (* 1 = 3.2776 loss)
I0330 03:10:32.200553  2693 sgd_solver.cpp:138] Iteration 21300, lr = 0.0005
I0330 03:12:48.561262  2693 solver.cpp:243] Iteration 21400, loss = 4.53598
I0330 03:12:48.561594  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.10139 (* 1 = 4.10139 loss)
I0330 03:12:48.561614  2693 sgd_solver.cpp:138] Iteration 21400, lr = 0.0005
I0330 03:15:02.197165  2693 solver.cpp:243] Iteration 21500, loss = 4.49206
I0330 03:15:02.197386  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.91001 (* 1 = 2.91001 loss)
I0330 03:15:02.197407  2693 sgd_solver.cpp:138] Iteration 21500, lr = 0.0005
I0330 03:17:16.264119  2693 solver.cpp:243] Iteration 21600, loss = 4.38446
I0330 03:17:16.264292  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.32997 (* 1 = 4.32997 loss)
I0330 03:17:16.264309  2693 sgd_solver.cpp:138] Iteration 21600, lr = 0.0005
I0330 03:19:29.181180  2693 solver.cpp:243] Iteration 21700, loss = 4.62082
I0330 03:19:29.181355  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.74587 (* 1 = 6.74587 loss)
I0330 03:19:29.181375  2693 sgd_solver.cpp:138] Iteration 21700, lr = 0.0005
I0330 03:21:45.166115  2693 solver.cpp:243] Iteration 21800, loss = 4.50446
I0330 03:21:45.166391  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.98549 (* 1 = 4.98549 loss)
I0330 03:21:45.166425  2693 sgd_solver.cpp:138] Iteration 21800, lr = 0.0005
I0330 03:23:59.804127  2693 solver.cpp:243] Iteration 21900, loss = 4.59037
I0330 03:23:59.804328  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.13348 (* 1 = 5.13348 loss)
I0330 03:23:59.804345  2693 sgd_solver.cpp:138] Iteration 21900, lr = 0.0005
I0330 03:26:16.917929  2693 solver.cpp:243] Iteration 22000, loss = 4.5809
I0330 03:26:16.918184  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.45054 (* 1 = 4.45054 loss)
I0330 03:26:16.918231  2693 sgd_solver.cpp:138] Iteration 22000, lr = 0.0005
I0330 03:28:28.732017  2693 solver.cpp:243] Iteration 22100, loss = 4.2521
I0330 03:28:28.732216  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.87227 (* 1 = 3.87227 loss)
I0330 03:28:28.732234  2693 sgd_solver.cpp:138] Iteration 22100, lr = 0.0005
I0330 03:30:43.500175  2693 solver.cpp:243] Iteration 22200, loss = 4.53829
I0330 03:30:43.500373  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.78329 (* 1 = 4.78329 loss)
I0330 03:30:43.500391  2693 sgd_solver.cpp:138] Iteration 22200, lr = 0.0005
I0330 03:32:55.604410  2693 solver.cpp:243] Iteration 22300, loss = 4.39235
I0330 03:32:55.604673  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.64555 (* 1 = 4.64555 loss)
I0330 03:32:55.604696  2693 sgd_solver.cpp:138] Iteration 22300, lr = 0.0005
I0330 03:35:07.621101  2693 solver.cpp:243] Iteration 22400, loss = 4.28037
I0330 03:35:07.621294  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.84216 (* 1 = 3.84216 loss)
I0330 03:35:07.621312  2693 sgd_solver.cpp:138] Iteration 22400, lr = 0.0005
I0330 03:37:17.964856  2693 solver.cpp:243] Iteration 22500, loss = 4.46898
I0330 03:37:17.972959  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.15896 (* 1 = 4.15896 loss)
I0330 03:37:17.972988  2693 sgd_solver.cpp:138] Iteration 22500, lr = 0.0005
I0330 03:39:30.722787  2693 solver.cpp:243] Iteration 22600, loss = 4.408
I0330 03:39:30.722993  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.69518 (* 1 = 3.69518 loss)
I0330 03:39:30.723011  2693 sgd_solver.cpp:138] Iteration 22600, lr = 0.0005
I0330 03:41:44.285606  2693 solver.cpp:243] Iteration 22700, loss = 4.5546
I0330 03:41:44.285799  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.18093 (* 1 = 5.18093 loss)
I0330 03:41:44.285816  2693 sgd_solver.cpp:138] Iteration 22700, lr = 0.0005
I0330 03:43:56.233911  2693 solver.cpp:243] Iteration 22800, loss = 4.5133
I0330 03:43:56.234092  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.41401 (* 1 = 3.41401 loss)
I0330 03:43:56.234110  2693 sgd_solver.cpp:138] Iteration 22800, lr = 0.0005
I0330 03:46:07.220903  2693 solver.cpp:243] Iteration 22900, loss = 4.38169
I0330 03:46:07.221112  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.7833 (* 1 = 4.7833 loss)
I0330 03:46:07.221128  2693 sgd_solver.cpp:138] Iteration 22900, lr = 0.0005
I0330 03:48:16.252943  2693 solver.cpp:243] Iteration 23000, loss = 4.41483
I0330 03:48:16.253199  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.45664 (* 1 = 4.45664 loss)
I0330 03:48:16.253242  2693 sgd_solver.cpp:138] Iteration 23000, lr = 0.0005
I0330 03:50:28.235774  2693 solver.cpp:243] Iteration 23100, loss = 4.40592
I0330 03:50:28.235999  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.75359 (* 1 = 6.75359 loss)
I0330 03:50:28.236019  2693 sgd_solver.cpp:138] Iteration 23100, lr = 0.0005
I0330 03:52:39.245616  2693 solver.cpp:243] Iteration 23200, loss = 4.36249
I0330 03:52:39.245796  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.0592 (* 1 = 5.0592 loss)
I0330 03:52:39.245813  2693 sgd_solver.cpp:138] Iteration 23200, lr = 0.0005
I0330 03:54:51.175338  2693 solver.cpp:243] Iteration 23300, loss = 4.52889
I0330 03:54:51.175638  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.76973 (* 1 = 6.76973 loss)
I0330 03:54:51.175659  2693 sgd_solver.cpp:138] Iteration 23300, lr = 0.0005
I0330 03:57:04.524245  2693 solver.cpp:243] Iteration 23400, loss = 4.51855
I0330 03:57:04.524566  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.77578 (* 1 = 2.77578 loss)
I0330 03:57:04.524632  2693 sgd_solver.cpp:138] Iteration 23400, lr = 0.0005
I0330 03:59:17.560411  2693 solver.cpp:243] Iteration 23500, loss = 4.66635
I0330 03:59:17.560647  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.47532 (* 1 = 4.47532 loss)
I0330 03:59:17.560667  2693 sgd_solver.cpp:138] Iteration 23500, lr = 0.0005
I0330 04:01:31.325139  2693 solver.cpp:243] Iteration 23600, loss = 4.54248
I0330 04:01:31.325419  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.41734 (* 1 = 4.41734 loss)
I0330 04:01:31.325450  2693 sgd_solver.cpp:138] Iteration 23600, lr = 0.0005
I0330 04:03:43.967422  2693 solver.cpp:243] Iteration 23700, loss = 4.46614
I0330 04:03:43.967699  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.62799 (* 1 = 4.62799 loss)
I0330 04:03:43.967718  2693 sgd_solver.cpp:138] Iteration 23700, lr = 0.0005
I0330 04:05:55.134271  2693 solver.cpp:243] Iteration 23800, loss = 4.43633
I0330 04:05:55.134528  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.70736 (* 1 = 4.70736 loss)
I0330 04:05:55.134564  2693 sgd_solver.cpp:138] Iteration 23800, lr = 0.0005
I0330 04:08:07.167765  2693 solver.cpp:243] Iteration 23900, loss = 4.42997
I0330 04:08:07.168020  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.49532 (* 1 = 4.49532 loss)
I0330 04:08:07.168036  2693 sgd_solver.cpp:138] Iteration 23900, lr = 0.0005
I0330 04:10:17.428354  2693 solver.cpp:243] Iteration 24000, loss = 4.43287
I0330 04:10:17.428546  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.88306 (* 1 = 3.88306 loss)
I0330 04:10:17.428563  2693 sgd_solver.cpp:138] Iteration 24000, lr = 0.0005
I0330 04:12:29.079552  2693 solver.cpp:243] Iteration 24100, loss = 4.49572
I0330 04:12:29.079963  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.24237 (* 1 = 5.24237 loss)
I0330 04:12:29.080029  2693 sgd_solver.cpp:138] Iteration 24100, lr = 0.0005
I0330 04:14:41.186815  2693 solver.cpp:243] Iteration 24200, loss = 4.57117
I0330 04:14:41.187100  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.9688 (* 1 = 3.9688 loss)
I0330 04:14:41.187134  2693 sgd_solver.cpp:138] Iteration 24200, lr = 0.0005
I0330 04:16:51.673563  2693 solver.cpp:243] Iteration 24300, loss = 4.39198
I0330 04:16:51.673753  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.803 (* 1 = 4.803 loss)
I0330 04:16:51.673774  2693 sgd_solver.cpp:138] Iteration 24300, lr = 0.0005
I0330 04:19:05.442317  2693 solver.cpp:243] Iteration 24400, loss = 4.46085
I0330 04:19:05.450462  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.68562 (* 1 = 5.68562 loss)
I0330 04:19:05.450564  2693 sgd_solver.cpp:138] Iteration 24400, lr = 0.0005
I0330 04:21:17.695094  2693 solver.cpp:243] Iteration 24500, loss = 4.42575
I0330 04:21:17.695296  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.35766 (* 1 = 4.35766 loss)
I0330 04:21:17.695313  2693 sgd_solver.cpp:138] Iteration 24500, lr = 0.0005
I0330 04:23:29.752235  2693 solver.cpp:243] Iteration 24600, loss = 4.37292
I0330 04:23:29.752439  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.98013 (* 1 = 6.98013 loss)
I0330 04:23:29.752457  2693 sgd_solver.cpp:138] Iteration 24600, lr = 0.0005
I0330 04:25:40.923055  2693 solver.cpp:243] Iteration 24700, loss = 4.38808
I0330 04:25:40.923321  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.28705 (* 1 = 4.28705 loss)
I0330 04:25:40.923373  2693 sgd_solver.cpp:138] Iteration 24700, lr = 0.0005
I0330 04:27:52.328517  2693 solver.cpp:243] Iteration 24800, loss = 4.43032
I0330 04:27:52.328728  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.84251 (* 1 = 3.84251 loss)
I0330 04:27:52.328750  2693 sgd_solver.cpp:138] Iteration 24800, lr = 0.0005
I0330 04:30:05.376611  2693 solver.cpp:243] Iteration 24900, loss = 4.50581
I0330 04:30:05.376884  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.10636 (* 1 = 4.10636 loss)
I0330 04:30:05.376919  2693 sgd_solver.cpp:138] Iteration 24900, lr = 0.0005
I0330 04:32:16.307888  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_25000.caffemodel
I0330 04:32:17.331713  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_25000.solverstate
I0330 04:32:17.478631  2693 solver.cpp:433] Iteration 25000, Testing net (#0)
I0330 04:32:17.478723  2693 net.cpp:693] Ignoring source layer mbox_loss
I0330 04:33:38.281091  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.515933
I0330 04:33:38.823724  2693 solver.cpp:243] Iteration 25000, loss = 4.43755
I0330 04:33:38.823842  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.39918 (* 1 = 2.39918 loss)
I0330 04:33:38.823873  2693 sgd_solver.cpp:138] Iteration 25000, lr = 0.0005
I0330 04:35:49.262684  2693 solver.cpp:243] Iteration 25100, loss = 4.30696
I0330 04:35:49.262917  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.42696 (* 1 = 4.42696 loss)
I0330 04:35:49.262948  2693 sgd_solver.cpp:138] Iteration 25100, lr = 0.0005
I0330 04:37:59.562315  2693 solver.cpp:243] Iteration 25200, loss = 4.33176
I0330 04:37:59.570605  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.52516 (* 1 = 3.52516 loss)
I0330 04:37:59.570673  2693 sgd_solver.cpp:138] Iteration 25200, lr = 0.0005
I0330 04:40:12.578380  2693 solver.cpp:243] Iteration 25300, loss = 4.39354
I0330 04:40:12.586153  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.69392 (* 1 = 4.69392 loss)
I0330 04:40:12.586186  2693 sgd_solver.cpp:138] Iteration 25300, lr = 0.0005
I0330 04:42:24.582881  2693 solver.cpp:243] Iteration 25400, loss = 4.31081
I0330 04:42:24.583163  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.77223 (* 1 = 4.77223 loss)
I0330 04:42:24.583215  2693 sgd_solver.cpp:138] Iteration 25400, lr = 0.0005
I0330 04:44:37.299346  2693 solver.cpp:243] Iteration 25500, loss = 4.48606
I0330 04:44:37.299564  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.6156 (* 1 = 4.6156 loss)
I0330 04:44:37.299685  2693 sgd_solver.cpp:138] Iteration 25500, lr = 0.0005
I0330 04:46:48.225410  2693 solver.cpp:243] Iteration 25600, loss = 4.42878
I0330 04:46:48.225661  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.46031 (* 1 = 3.46031 loss)
I0330 04:46:48.225688  2693 sgd_solver.cpp:138] Iteration 25600, lr = 0.0005
I0330 04:49:00.069664  2693 solver.cpp:243] Iteration 25700, loss = 4.38135
I0330 04:49:00.069901  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.65389 (* 1 = 3.65389 loss)
I0330 04:49:00.069919  2693 sgd_solver.cpp:138] Iteration 25700, lr = 0.0005
I0330 04:51:12.334781  2693 solver.cpp:243] Iteration 25800, loss = 4.50385
I0330 04:51:12.334998  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.54928 (* 1 = 5.54928 loss)
I0330 04:51:12.335016  2693 sgd_solver.cpp:138] Iteration 25800, lr = 0.0005
I0330 04:53:25.170557  2693 solver.cpp:243] Iteration 25900, loss = 4.41616
I0330 04:53:25.170843  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.54624 (* 1 = 4.54624 loss)
I0330 04:53:25.170904  2693 sgd_solver.cpp:138] Iteration 25900, lr = 0.0005
I0330 04:55:34.852887  2693 solver.cpp:243] Iteration 26000, loss = 4.3907
I0330 04:55:34.853072  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.62096 (* 1 = 3.62096 loss)
I0330 04:55:34.853090  2693 sgd_solver.cpp:138] Iteration 26000, lr = 0.0005
I0330 04:57:47.788655  2693 solver.cpp:243] Iteration 26100, loss = 4.49318
I0330 04:57:47.788867  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.9151 (* 1 = 4.9151 loss)
I0330 04:57:47.788902  2693 sgd_solver.cpp:138] Iteration 26100, lr = 0.0005
I0330 04:59:58.309044  2693 solver.cpp:243] Iteration 26200, loss = 4.30619
I0330 04:59:58.309223  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.6272 (* 1 = 4.6272 loss)
I0330 04:59:58.309240  2693 sgd_solver.cpp:138] Iteration 26200, lr = 0.0005
I0330 05:02:12.349073  2693 solver.cpp:243] Iteration 26300, loss = 4.39138
I0330 05:02:12.349261  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.5868 (* 1 = 4.5868 loss)
I0330 05:02:12.349277  2693 sgd_solver.cpp:138] Iteration 26300, lr = 0.0005
I0330 05:04:23.614343  2693 solver.cpp:243] Iteration 26400, loss = 4.36818
I0330 05:04:23.614552  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.02029 (* 1 = 4.02029 loss)
I0330 05:04:23.614572  2693 sgd_solver.cpp:138] Iteration 26400, lr = 0.0005
I0330 05:06:35.097229  2693 solver.cpp:243] Iteration 26500, loss = 4.40082
I0330 05:06:35.097424  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.84311 (* 1 = 3.84311 loss)
I0330 05:06:35.097442  2693 sgd_solver.cpp:138] Iteration 26500, lr = 0.0005
I0330 05:08:45.309293  2693 solver.cpp:243] Iteration 26600, loss = 4.37368
I0330 05:08:45.309527  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.03556 (* 1 = 5.03556 loss)
I0330 05:08:45.309546  2693 sgd_solver.cpp:138] Iteration 26600, lr = 0.0005
I0330 05:10:57.298017  2693 solver.cpp:243] Iteration 26700, loss = 4.32928
I0330 05:10:57.298221  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.78965 (* 1 = 3.78965 loss)
I0330 05:10:57.298239  2693 sgd_solver.cpp:138] Iteration 26700, lr = 0.0005
I0330 05:13:09.895553  2693 solver.cpp:243] Iteration 26800, loss = 4.43668
I0330 05:13:09.895843  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.75608 (* 1 = 3.75608 loss)
I0330 05:13:09.895859  2693 sgd_solver.cpp:138] Iteration 26800, lr = 0.0005
I0330 05:15:21.717422  2693 solver.cpp:243] Iteration 26900, loss = 4.52591
I0330 05:15:21.717624  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.48305 (* 1 = 4.48305 loss)
I0330 05:15:21.717643  2693 sgd_solver.cpp:138] Iteration 26900, lr = 0.0005
I0330 05:17:35.074826  2693 solver.cpp:243] Iteration 27000, loss = 4.55598
I0330 05:17:35.075047  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.24762 (* 1 = 4.24762 loss)
I0330 05:17:35.075079  2693 sgd_solver.cpp:138] Iteration 27000, lr = 0.0005
I0330 05:19:47.194906  2693 solver.cpp:243] Iteration 27100, loss = 4.36722
I0330 05:19:47.195173  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.63148 (* 1 = 4.63148 loss)
I0330 05:19:47.195206  2693 sgd_solver.cpp:138] Iteration 27100, lr = 0.0005
I0330 05:21:57.108891  2693 solver.cpp:243] Iteration 27200, loss = 4.31895
I0330 05:21:57.109125  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.47014 (* 1 = 3.47014 loss)
I0330 05:21:57.109144  2693 sgd_solver.cpp:138] Iteration 27200, lr = 0.0005
I0330 05:24:09.001540  2693 solver.cpp:243] Iteration 27300, loss = 4.39815
I0330 05:24:09.010038  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.34057 (* 1 = 4.34057 loss)
I0330 05:24:09.010068  2693 sgd_solver.cpp:138] Iteration 27300, lr = 0.0005
I0330 05:26:19.990927  2693 solver.cpp:243] Iteration 27400, loss = 4.37013
I0330 05:26:19.991209  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.60807 (* 1 = 4.60807 loss)
I0330 05:26:19.991241  2693 sgd_solver.cpp:138] Iteration 27400, lr = 0.0005
I0330 05:28:30.657253  2693 solver.cpp:243] Iteration 27500, loss = 4.3643
I0330 05:28:30.657464  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.80795 (* 1 = 3.80795 loss)
I0330 05:28:30.657480  2693 sgd_solver.cpp:138] Iteration 27500, lr = 0.0005
I0330 05:30:42.510375  2693 solver.cpp:243] Iteration 27600, loss = 4.50605
I0330 05:30:42.510571  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.9354 (* 1 = 4.9354 loss)
I0330 05:30:42.510588  2693 sgd_solver.cpp:138] Iteration 27600, lr = 0.0005
I0330 05:32:53.739711  2693 solver.cpp:243] Iteration 27700, loss = 4.46333
I0330 05:32:53.739912  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.94398 (* 1 = 4.94398 loss)
I0330 05:32:53.739943  2693 sgd_solver.cpp:138] Iteration 27700, lr = 0.0005
I0330 05:35:04.951855  2693 solver.cpp:243] Iteration 27800, loss = 4.46619
I0330 05:35:04.952016  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.87951 (* 1 = 6.87951 loss)
I0330 05:35:04.952047  2693 sgd_solver.cpp:138] Iteration 27800, lr = 0.0005
I0330 05:37:16.393712  2693 solver.cpp:243] Iteration 27900, loss = 4.38641
I0330 05:37:16.393985  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.04126 (* 1 = 5.04126 loss)
I0330 05:37:16.394026  2693 sgd_solver.cpp:138] Iteration 27900, lr = 0.0005
I0330 05:39:29.506276  2693 solver.cpp:243] Iteration 28000, loss = 4.27084
I0330 05:39:29.506505  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.71804 (* 1 = 4.71804 loss)
I0330 05:39:29.506525  2693 sgd_solver.cpp:138] Iteration 28000, lr = 0.0005
I0330 05:41:40.327178  2693 solver.cpp:243] Iteration 28100, loss = 4.46831
I0330 05:41:40.327368  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.05346 (* 1 = 5.05346 loss)
I0330 05:41:40.327388  2693 sgd_solver.cpp:138] Iteration 28100, lr = 0.0005
I0330 05:43:51.669811  2693 solver.cpp:243] Iteration 28200, loss = 4.36696
I0330 05:43:51.670094  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.11054 (* 1 = 4.11054 loss)
I0330 05:43:51.670133  2693 sgd_solver.cpp:138] Iteration 28200, lr = 0.0005
I0330 05:46:05.356242  2693 solver.cpp:243] Iteration 28300, loss = 4.52199
I0330 05:46:05.356449  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.17573 (* 1 = 4.17573 loss)
I0330 05:46:05.356467  2693 sgd_solver.cpp:138] Iteration 28300, lr = 0.0005
I0330 05:48:18.463246  2693 solver.cpp:243] Iteration 28400, loss = 4.45301
I0330 05:48:18.463516  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.24367 (* 1 = 4.24367 loss)
I0330 05:48:18.463557  2693 sgd_solver.cpp:138] Iteration 28400, lr = 0.0005
I0330 05:50:30.477871  2693 solver.cpp:243] Iteration 28500, loss = 4.27506
I0330 05:50:30.478063  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.22333 (* 1 = 3.22333 loss)
I0330 05:50:30.478082  2693 sgd_solver.cpp:138] Iteration 28500, lr = 0.0005
I0330 05:52:40.300608  2693 solver.cpp:243] Iteration 28600, loss = 4.44732
I0330 05:52:40.300854  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.13343 (* 1 = 5.13343 loss)
I0330 05:52:40.300886  2693 sgd_solver.cpp:138] Iteration 28600, lr = 0.0005
I0330 05:54:52.826004  2693 solver.cpp:243] Iteration 28700, loss = 4.2713
I0330 05:54:52.826212  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.85111 (* 1 = 4.85111 loss)
I0330 05:54:52.826230  2693 sgd_solver.cpp:138] Iteration 28700, lr = 0.0005
I0330 05:57:05.976157  2693 solver.cpp:243] Iteration 28800, loss = 4.41162
I0330 05:57:05.976383  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.45278 (* 1 = 4.45278 loss)
I0330 05:57:05.976431  2693 sgd_solver.cpp:138] Iteration 28800, lr = 0.0005
I0330 05:59:17.940248  2693 solver.cpp:243] Iteration 28900, loss = 4.49533
I0330 05:59:17.940517  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.77253 (* 1 = 4.77253 loss)
I0330 05:59:17.940539  2693 sgd_solver.cpp:138] Iteration 28900, lr = 0.0005
I0330 06:01:28.270869  2693 solver.cpp:243] Iteration 29000, loss = 4.41883
I0330 06:01:28.271114  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.85714 (* 1 = 4.85714 loss)
I0330 06:01:28.271136  2693 sgd_solver.cpp:138] Iteration 29000, lr = 0.0005
I0330 06:03:39.882582  2693 solver.cpp:243] Iteration 29100, loss = 4.43888
I0330 06:03:39.882794  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.29376 (* 1 = 4.29376 loss)
I0330 06:03:39.882812  2693 sgd_solver.cpp:138] Iteration 29100, lr = 0.0005
I0330 06:05:51.212016  2693 solver.cpp:243] Iteration 29200, loss = 4.48963
I0330 06:05:51.212213  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.89204 (* 1 = 4.89204 loss)
I0330 06:05:51.212234  2693 sgd_solver.cpp:138] Iteration 29200, lr = 0.0005
I0330 06:08:03.439478  2693 solver.cpp:243] Iteration 29300, loss = 4.33317
I0330 06:08:03.439735  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.56459 (* 1 = 4.56459 loss)
I0330 06:08:03.439754  2693 sgd_solver.cpp:138] Iteration 29300, lr = 0.0005
I0330 06:10:15.318397  2693 solver.cpp:243] Iteration 29400, loss = 4.40459
I0330 06:10:15.318574  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.94476 (* 1 = 4.94476 loss)
I0330 06:10:15.318594  2693 sgd_solver.cpp:138] Iteration 29400, lr = 0.0005
I0330 06:12:28.253481  2693 solver.cpp:243] Iteration 29500, loss = 4.41386
I0330 06:12:28.253700  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.36821 (* 1 = 4.36821 loss)
I0330 06:12:28.253809  2693 sgd_solver.cpp:138] Iteration 29500, lr = 0.0005
I0330 06:14:38.807963  2693 solver.cpp:243] Iteration 29600, loss = 4.30306
I0330 06:14:38.808168  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.41079 (* 1 = 4.41079 loss)
I0330 06:14:38.808187  2693 sgd_solver.cpp:138] Iteration 29600, lr = 0.0005
I0330 06:16:51.367133  2693 solver.cpp:243] Iteration 29700, loss = 4.46109
I0330 06:16:51.367339  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.72598 (* 1 = 5.72598 loss)
I0330 06:16:51.367358  2693 sgd_solver.cpp:138] Iteration 29700, lr = 0.0005
I0330 06:19:01.929970  2693 solver.cpp:243] Iteration 29800, loss = 4.3128
I0330 06:19:01.937301  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.99915 (* 1 = 3.99915 loss)
I0330 06:19:01.937336  2693 sgd_solver.cpp:138] Iteration 29800, lr = 0.0005
I0330 06:21:12.094360  2693 solver.cpp:243] Iteration 29900, loss = 4.42843
I0330 06:21:12.094564  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.03332 (* 1 = 4.03332 loss)
I0330 06:21:12.094584  2693 sgd_solver.cpp:138] Iteration 29900, lr = 0.0005
I0330 06:23:21.300917  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_30000.caffemodel
I0330 06:23:22.247160  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_30000.solverstate
I0330 06:23:22.358901  2693 solver.cpp:433] Iteration 30000, Testing net (#0)
I0330 06:23:22.358980  2693 net.cpp:693] Ignoring source layer mbox_loss
I0330 06:24:43.206184  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.521198
I0330 06:24:43.902822  2693 solver.cpp:243] Iteration 30000, loss = 4.2688
I0330 06:24:43.902892  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.79022 (* 1 = 4.79022 loss)
I0330 06:24:43.902907  2693 sgd_solver.cpp:138] Iteration 30000, lr = 0.0005
I0330 06:26:55.680989  2693 solver.cpp:243] Iteration 30100, loss = 4.37613
I0330 06:26:55.681207  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.49852 (* 1 = 4.49852 loss)
I0330 06:26:55.681226  2693 sgd_solver.cpp:138] Iteration 30100, lr = 0.0005
I0330 06:29:06.846195  2693 solver.cpp:243] Iteration 30200, loss = 4.42614
I0330 06:29:06.846509  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.11959 (* 1 = 3.11959 loss)
I0330 06:29:06.846554  2693 sgd_solver.cpp:138] Iteration 30200, lr = 0.0005
I0330 06:31:17.363986  2693 solver.cpp:243] Iteration 30300, loss = 4.31507
I0330 06:31:17.364186  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.49555 (* 1 = 6.49555 loss)
I0330 06:31:17.364202  2693 sgd_solver.cpp:138] Iteration 30300, lr = 0.0005
I0330 06:33:27.734849  2693 solver.cpp:243] Iteration 30400, loss = 4.39042
I0330 06:33:27.735059  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.97335 (* 1 = 2.97335 loss)
I0330 06:33:27.735074  2693 sgd_solver.cpp:138] Iteration 30400, lr = 0.0005
I0330 06:35:39.333060  2693 solver.cpp:243] Iteration 30500, loss = 4.37894
I0330 06:35:39.333286  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.99186 (* 1 = 4.99186 loss)
I0330 06:35:39.333310  2693 sgd_solver.cpp:138] Iteration 30500, lr = 0.0005
I0330 06:37:49.052276  2693 solver.cpp:243] Iteration 30600, loss = 4.35081
I0330 06:37:49.052587  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.88174 (* 1 = 3.88174 loss)
I0330 06:37:49.052659  2693 sgd_solver.cpp:138] Iteration 30600, lr = 0.0005
I0330 06:39:59.474068  2693 solver.cpp:243] Iteration 30700, loss = 4.22629
I0330 06:39:59.474301  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.72201 (* 1 = 4.72201 loss)
I0330 06:39:59.474336  2693 sgd_solver.cpp:138] Iteration 30700, lr = 0.0005
I0330 06:42:11.532685  2693 solver.cpp:243] Iteration 30800, loss = 4.43473
I0330 06:42:11.540274  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.96487 (* 1 = 3.96487 loss)
I0330 06:42:11.540297  2693 sgd_solver.cpp:138] Iteration 30800, lr = 0.0005
I0330 06:44:21.983049  2693 solver.cpp:243] Iteration 30900, loss = 4.44043
I0330 06:44:21.983247  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.96654 (* 1 = 2.96654 loss)
I0330 06:44:21.983264  2693 sgd_solver.cpp:138] Iteration 30900, lr = 0.0005
I0330 06:46:33.717517  2693 solver.cpp:243] Iteration 31000, loss = 4.32356
I0330 06:46:33.719504  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.57293 (* 1 = 3.57293 loss)
I0330 06:46:33.719671  2693 sgd_solver.cpp:138] Iteration 31000, lr = 0.0005
I0330 06:48:42.933183  2693 solver.cpp:243] Iteration 31100, loss = 4.41053
I0330 06:48:42.933388  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.17898 (* 1 = 5.17898 loss)
I0330 06:48:42.933404  2693 sgd_solver.cpp:138] Iteration 31100, lr = 0.0005
I0330 06:50:54.731818  2693 solver.cpp:243] Iteration 31200, loss = 4.5029
I0330 06:50:54.732107  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.46155 (* 1 = 5.46155 loss)
I0330 06:50:54.732168  2693 sgd_solver.cpp:138] Iteration 31200, lr = 0.0005
I0330 06:53:06.118093  2693 solver.cpp:243] Iteration 31300, loss = 4.33873
I0330 06:53:06.118389  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.1258 (* 1 = 4.1258 loss)
I0330 06:53:06.118450  2693 sgd_solver.cpp:138] Iteration 31300, lr = 0.0005
I0330 06:55:19.541442  2693 solver.cpp:243] Iteration 31400, loss = 4.34475
I0330 06:55:19.541652  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.68162 (* 1 = 3.68162 loss)
I0330 06:55:19.541672  2693 sgd_solver.cpp:138] Iteration 31400, lr = 0.0005
I0330 06:57:31.048692  2693 solver.cpp:243] Iteration 31500, loss = 4.33669
I0330 06:57:31.056128  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.28878 (* 1 = 4.28878 loss)
I0330 06:57:31.056192  2693 sgd_solver.cpp:138] Iteration 31500, lr = 0.0005
I0330 06:59:41.452203  2693 solver.cpp:243] Iteration 31600, loss = 4.27904
I0330 06:59:41.452455  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.29869 (* 1 = 4.29869 loss)
I0330 06:59:41.452488  2693 sgd_solver.cpp:138] Iteration 31600, lr = 0.0005
I0330 07:01:51.775483  2693 solver.cpp:243] Iteration 31700, loss = 4.19677
I0330 07:01:51.775738  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.34293 (* 1 = 3.34293 loss)
I0330 07:01:51.775758  2693 sgd_solver.cpp:138] Iteration 31700, lr = 0.0005
I0330 07:04:02.397210  2693 solver.cpp:243] Iteration 31800, loss = 4.51316
I0330 07:04:02.397442  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.13072 (* 1 = 4.13072 loss)
I0330 07:04:02.397464  2693 sgd_solver.cpp:138] Iteration 31800, lr = 0.0005
I0330 07:06:13.882163  2693 solver.cpp:243] Iteration 31900, loss = 4.31685
I0330 07:06:13.882446  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.10119 (* 1 = 5.10119 loss)
I0330 07:06:13.882483  2693 sgd_solver.cpp:138] Iteration 31900, lr = 0.0005
I0330 07:08:23.231271  2693 solver.cpp:243] Iteration 32000, loss = 4.18455
I0330 07:08:23.237979  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.63109 (* 1 = 2.63109 loss)
I0330 07:08:23.238023  2693 sgd_solver.cpp:138] Iteration 32000, lr = 0.0005
I0330 07:10:34.008937  2693 solver.cpp:243] Iteration 32100, loss = 4.23121
I0330 07:10:34.009181  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.0331 (* 1 = 4.0331 loss)
I0330 07:10:34.009238  2693 sgd_solver.cpp:138] Iteration 32100, lr = 0.0005
I0330 07:12:45.450863  2693 solver.cpp:243] Iteration 32200, loss = 4.37067
I0330 07:12:45.451120  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.99929 (* 1 = 3.99929 loss)
I0330 07:12:45.451195  2693 sgd_solver.cpp:138] Iteration 32200, lr = 0.0005
I0330 07:14:54.970552  2693 solver.cpp:243] Iteration 32300, loss = 4.18034
I0330 07:14:54.970795  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.61887 (* 1 = 3.61887 loss)
I0330 07:14:54.970815  2693 sgd_solver.cpp:138] Iteration 32300, lr = 0.0005
I0330 07:17:07.131230  2693 solver.cpp:243] Iteration 32400, loss = 4.30577
I0330 07:17:07.138738  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.61908 (* 1 = 4.61908 loss)
I0330 07:17:07.138772  2693 sgd_solver.cpp:138] Iteration 32400, lr = 0.0005
I0330 07:19:18.007915  2693 solver.cpp:243] Iteration 32500, loss = 4.45716
I0330 07:19:18.014941  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.63603 (* 1 = 3.63603 loss)
I0330 07:19:18.014960  2693 sgd_solver.cpp:138] Iteration 32500, lr = 0.0005
I0330 07:21:30.403434  2693 solver.cpp:243] Iteration 32600, loss = 4.40784
I0330 07:21:30.410863  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.23073 (* 1 = 4.23073 loss)
I0330 07:21:30.410895  2693 sgd_solver.cpp:138] Iteration 32600, lr = 0.0005
I0330 07:23:41.817248  2693 solver.cpp:243] Iteration 32700, loss = 4.37751
I0330 07:23:41.817436  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.35925 (* 1 = 4.35925 loss)
I0330 07:23:41.817454  2693 sgd_solver.cpp:138] Iteration 32700, lr = 0.0005
I0330 07:25:53.038431  2693 solver.cpp:243] Iteration 32800, loss = 4.13498
I0330 07:25:53.038635  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.53436 (* 1 = 3.53436 loss)
I0330 07:25:53.038651  2693 sgd_solver.cpp:138] Iteration 32800, lr = 0.0005
I0330 07:28:04.320806  2693 solver.cpp:243] Iteration 32900, loss = 4.44887
I0330 07:28:04.321050  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.7705 (* 1 = 3.7705 loss)
I0330 07:28:04.321080  2693 sgd_solver.cpp:138] Iteration 32900, lr = 0.0005
I0330 07:30:14.719701  2693 solver.cpp:243] Iteration 33000, loss = 4.32471
I0330 07:30:14.719933  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.79767 (* 1 = 3.79767 loss)
I0330 07:30:14.719952  2693 sgd_solver.cpp:138] Iteration 33000, lr = 0.0005
I0330 07:32:25.261216  2693 solver.cpp:243] Iteration 33100, loss = 4.29259
I0330 07:32:25.261447  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.45959 (* 1 = 4.45959 loss)
I0330 07:32:25.261467  2693 sgd_solver.cpp:138] Iteration 33100, lr = 0.0005
I0330 07:34:35.738462  2693 solver.cpp:243] Iteration 33200, loss = 4.40223
I0330 07:34:35.738637  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.19671 (* 1 = 4.19671 loss)
I0330 07:34:35.738653  2693 sgd_solver.cpp:138] Iteration 33200, lr = 0.0005
I0330 07:36:47.934985  2693 solver.cpp:243] Iteration 33300, loss = 4.27186
I0330 07:36:47.935246  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.93301 (* 1 = 4.93301 loss)
I0330 07:36:47.935264  2693 sgd_solver.cpp:138] Iteration 33300, lr = 0.0005
I0330 07:38:57.374897  2693 solver.cpp:243] Iteration 33400, loss = 4.28094
I0330 07:38:57.375128  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.7803 (* 1 = 4.7803 loss)
I0330 07:38:57.375154  2693 sgd_solver.cpp:138] Iteration 33400, lr = 0.0005
I0330 07:41:08.790729  2693 solver.cpp:243] Iteration 33500, loss = 4.30643
I0330 07:41:08.790966  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.25433 (* 1 = 4.25433 loss)
I0330 07:41:08.790987  2693 sgd_solver.cpp:138] Iteration 33500, lr = 0.0005
I0330 07:43:19.061345  2693 solver.cpp:243] Iteration 33600, loss = 4.28739
I0330 07:43:19.068600  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.65352 (* 1 = 3.65352 loss)
I0330 07:43:19.068666  2693 sgd_solver.cpp:138] Iteration 33600, lr = 0.0005
I0330 07:45:29.061879  2693 solver.cpp:243] Iteration 33700, loss = 4.16643
I0330 07:45:29.062108  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.40991 (* 1 = 5.40991 loss)
I0330 07:45:29.062139  2693 sgd_solver.cpp:138] Iteration 33700, lr = 0.0005
I0330 07:47:37.947486  2693 solver.cpp:243] Iteration 33800, loss = 4.30871
I0330 07:47:37.947862  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.08514 (* 1 = 4.08514 loss)
I0330 07:47:37.947921  2693 sgd_solver.cpp:138] Iteration 33800, lr = 0.0005
I0330 07:49:48.052182  2693 solver.cpp:243] Iteration 33900, loss = 4.34726
I0330 07:49:48.052440  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.70327 (* 1 = 3.70327 loss)
I0330 07:49:48.052470  2693 sgd_solver.cpp:138] Iteration 33900, lr = 0.0005
I0330 07:52:00.370451  2693 solver.cpp:243] Iteration 34000, loss = 4.33435
I0330 07:52:00.370678  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.98338 (* 1 = 3.98338 loss)
I0330 07:52:00.370697  2693 sgd_solver.cpp:138] Iteration 34000, lr = 0.0005
I0330 07:54:11.666597  2693 solver.cpp:243] Iteration 34100, loss = 4.44177
I0330 07:54:11.666802  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.65391 (* 1 = 2.65391 loss)
I0330 07:54:11.666818  2693 sgd_solver.cpp:138] Iteration 34100, lr = 0.0005
I0330 07:56:20.252548  2693 solver.cpp:243] Iteration 34200, loss = 4.31534
I0330 07:56:20.267848  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.90436 (* 1 = 3.90436 loss)
I0330 07:56:20.267884  2693 sgd_solver.cpp:138] Iteration 34200, lr = 0.0005
I0330 07:58:28.984740  2693 solver.cpp:243] Iteration 34300, loss = 4.30506
I0330 07:58:28.986526  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.49107 (* 1 = 4.49107 loss)
I0330 07:58:28.986546  2693 sgd_solver.cpp:138] Iteration 34300, lr = 0.0005
I0330 08:00:39.773535  2693 solver.cpp:243] Iteration 34400, loss = 4.38644
I0330 08:00:39.773769  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.14771 (* 1 = 4.14771 loss)
I0330 08:00:39.773807  2693 sgd_solver.cpp:138] Iteration 34400, lr = 0.0005
I0330 08:02:49.691123  2693 solver.cpp:243] Iteration 34500, loss = 4.26123
I0330 08:02:49.691402  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.70543 (* 1 = 4.70543 loss)
I0330 08:02:49.691421  2693 sgd_solver.cpp:138] Iteration 34500, lr = 0.0005
I0330 08:04:59.040232  2693 solver.cpp:243] Iteration 34600, loss = 4.37324
I0330 08:04:59.042402  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.07889 (* 1 = 5.07889 loss)
I0330 08:04:59.042420  2693 sgd_solver.cpp:138] Iteration 34600, lr = 0.0005
I0330 08:07:10.429221  2693 solver.cpp:243] Iteration 34700, loss = 4.31561
I0330 08:07:10.429499  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.84921 (* 1 = 2.84921 loss)
I0330 08:07:10.429545  2693 sgd_solver.cpp:138] Iteration 34700, lr = 0.0005
I0330 08:09:21.767437  2693 solver.cpp:243] Iteration 34800, loss = 4.36931
I0330 08:09:21.767711  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.17875 (* 1 = 5.17875 loss)
I0330 08:09:21.767730  2693 sgd_solver.cpp:138] Iteration 34800, lr = 0.0005
I0330 08:11:32.426029  2693 solver.cpp:243] Iteration 34900, loss = 4.28996
I0330 08:11:32.426352  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.67488 (* 1 = 3.67488 loss)
I0330 08:11:32.426424  2693 sgd_solver.cpp:138] Iteration 34900, lr = 0.0005
I0330 08:13:42.790489  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_35000.caffemodel
I0330 08:13:43.995450  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_35000.solverstate
I0330 08:13:44.122318  2693 solver.cpp:433] Iteration 35000, Testing net (#0)
I0330 08:13:44.122401  2693 net.cpp:693] Ignoring source layer mbox_loss
I0330 08:15:04.479816  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.53637
I0330 08:15:05.293195  2693 solver.cpp:243] Iteration 35000, loss = 4.2468
I0330 08:15:05.293264  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.73614 (* 1 = 4.73614 loss)
I0330 08:15:05.293279  2693 sgd_solver.cpp:138] Iteration 35000, lr = 0.0005
I0330 08:17:17.331082  2693 solver.cpp:243] Iteration 35100, loss = 4.39432
I0330 08:17:17.331303  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.29905 (* 1 = 4.29905 loss)
I0330 08:17:17.331332  2693 sgd_solver.cpp:138] Iteration 35100, lr = 0.0005
I0330 08:19:28.256753  2693 solver.cpp:243] Iteration 35200, loss = 4.36989
I0330 08:19:28.256995  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.7684 (* 1 = 3.7684 loss)
I0330 08:19:28.257019  2693 sgd_solver.cpp:138] Iteration 35200, lr = 0.0005
I0330 08:21:37.043802  2693 solver.cpp:243] Iteration 35300, loss = 4.29854
I0330 08:21:37.044049  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.07319 (* 1 = 3.07319 loss)
I0330 08:21:37.044065  2693 sgd_solver.cpp:138] Iteration 35300, lr = 0.0005
I0330 08:23:49.284770  2693 solver.cpp:243] Iteration 35400, loss = 4.30521
I0330 08:23:49.285017  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.20741 (* 1 = 4.20741 loss)
I0330 08:23:49.285048  2693 sgd_solver.cpp:138] Iteration 35400, lr = 0.0005
I0330 08:26:00.398448  2693 solver.cpp:243] Iteration 35500, loss = 4.43392
I0330 08:26:00.398680  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.22161 (* 1 = 4.22161 loss)
I0330 08:26:00.398713  2693 sgd_solver.cpp:138] Iteration 35500, lr = 0.0005
I0330 08:28:10.072476  2693 solver.cpp:243] Iteration 35600, loss = 4.26533
I0330 08:28:10.072664  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.44625 (* 1 = 4.44625 loss)
I0330 08:28:10.072680  2693 sgd_solver.cpp:138] Iteration 35600, lr = 0.0005
I0330 08:30:21.935356  2693 solver.cpp:243] Iteration 35700, loss = 4.2551
I0330 08:30:21.935535  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.12625 (* 1 = 5.12625 loss)
I0330 08:30:21.935554  2693 sgd_solver.cpp:138] Iteration 35700, lr = 0.0005
I0330 08:32:33.170866  2693 solver.cpp:243] Iteration 35800, loss = 4.33872
I0330 08:32:33.171063  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.09584 (* 1 = 5.09584 loss)
I0330 08:32:33.171083  2693 sgd_solver.cpp:138] Iteration 35800, lr = 0.0005
I0330 08:34:43.808295  2693 solver.cpp:243] Iteration 35900, loss = 4.27494
I0330 08:34:43.808588  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.27464 (* 1 = 5.27464 loss)
I0330 08:34:43.808607  2693 sgd_solver.cpp:138] Iteration 35900, lr = 0.0005
I0330 08:36:52.294859  2693 solver.cpp:243] Iteration 36000, loss = 4.24387
I0330 08:36:52.295117  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.20875 (* 1 = 4.20875 loss)
I0330 08:36:52.295150  2693 sgd_solver.cpp:138] Iteration 36000, lr = 0.0005
I0330 08:39:02.154884  2693 solver.cpp:243] Iteration 36100, loss = 4.39334
I0330 08:39:02.155180  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.94275 (* 1 = 4.94275 loss)
I0330 08:39:02.155222  2693 sgd_solver.cpp:138] Iteration 36100, lr = 0.0005
I0330 08:41:13.777281  2693 solver.cpp:243] Iteration 36200, loss = 4.38193
I0330 08:41:13.777520  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.49025 (* 1 = 3.49025 loss)
I0330 08:41:13.777540  2693 sgd_solver.cpp:138] Iteration 36200, lr = 0.0005
I0330 08:43:24.522809  2693 solver.cpp:243] Iteration 36300, loss = 4.28342
I0330 08:43:24.522994  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.93183 (* 1 = 3.93183 loss)
I0330 08:43:24.523011  2693 sgd_solver.cpp:138] Iteration 36300, lr = 0.0005
I0330 08:45:35.683890  2693 solver.cpp:243] Iteration 36400, loss = 4.28643
I0330 08:45:35.684069  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.23958 (* 1 = 5.23958 loss)
I0330 08:45:35.684085  2693 sgd_solver.cpp:138] Iteration 36400, lr = 0.0005
I0330 08:47:46.475947  2693 solver.cpp:243] Iteration 36500, loss = 4.38403
I0330 08:47:46.476198  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.12044 (* 1 = 6.12044 loss)
I0330 08:47:46.476235  2693 sgd_solver.cpp:138] Iteration 36500, lr = 0.0005
I0330 08:49:55.257340  2693 solver.cpp:243] Iteration 36600, loss = 4.29661
I0330 08:49:55.257545  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.1754 (* 1 = 4.1754 loss)
I0330 08:49:55.257565  2693 sgd_solver.cpp:138] Iteration 36600, lr = 0.0005
I0330 08:52:05.163482  2693 solver.cpp:243] Iteration 36700, loss = 4.29789
I0330 08:52:05.163751  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.6196 (* 1 = 4.6196 loss)
I0330 08:52:05.163769  2693 sgd_solver.cpp:138] Iteration 36700, lr = 0.0005
I0330 08:54:16.106267  2693 solver.cpp:243] Iteration 36800, loss = 4.30278
I0330 08:54:16.106451  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.13348 (* 1 = 3.13348 loss)
I0330 08:54:16.106468  2693 sgd_solver.cpp:138] Iteration 36800, lr = 0.0005
I0330 08:56:26.901592  2693 solver.cpp:243] Iteration 36900, loss = 4.37235
I0330 08:56:26.901782  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.56177 (* 1 = 4.56177 loss)
I0330 08:56:26.901800  2693 sgd_solver.cpp:138] Iteration 36900, lr = 0.0005
I0330 08:58:38.332542  2693 solver.cpp:243] Iteration 37000, loss = 4.19321
I0330 08:58:38.332788  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.59086 (* 1 = 3.59086 loss)
I0330 08:58:38.332816  2693 sgd_solver.cpp:138] Iteration 37000, lr = 0.0005
I0330 09:00:49.730026  2693 solver.cpp:243] Iteration 37100, loss = 4.27134
I0330 09:00:49.730216  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.08855 (* 1 = 3.08855 loss)
I0330 09:00:49.730234  2693 sgd_solver.cpp:138] Iteration 37100, lr = 0.0005
I0330 09:03:01.635046  2693 solver.cpp:243] Iteration 37200, loss = 4.22526
I0330 09:03:01.635304  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.71458 (* 1 = 4.71458 loss)
I0330 09:03:01.635336  2693 sgd_solver.cpp:138] Iteration 37200, lr = 0.0005
I0330 09:05:11.668320  2693 solver.cpp:243] Iteration 37300, loss = 4.34251
I0330 09:05:11.668514  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.87385 (* 1 = 3.87385 loss)
I0330 09:05:11.668534  2693 sgd_solver.cpp:138] Iteration 37300, lr = 0.0005
I0330 09:07:21.891827  2693 solver.cpp:243] Iteration 37400, loss = 4.17826
I0330 09:07:21.899976  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.38755 (* 1 = 5.38755 loss)
I0330 09:07:21.899994  2693 sgd_solver.cpp:138] Iteration 37400, lr = 0.0005
I0330 09:09:31.148041  2693 solver.cpp:243] Iteration 37500, loss = 4.19072
I0330 09:09:31.148365  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.72184 (* 1 = 2.72184 loss)
I0330 09:09:31.148423  2693 sgd_solver.cpp:138] Iteration 37500, lr = 0.0005
I0330 09:11:42.551479  2693 solver.cpp:243] Iteration 37600, loss = 4.43506
I0330 09:11:42.551759  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.40138 (* 1 = 4.40138 loss)
I0330 09:11:42.551777  2693 sgd_solver.cpp:138] Iteration 37600, lr = 0.0005
I0330 09:13:52.064838  2693 solver.cpp:243] Iteration 37700, loss = 4.11441
I0330 09:13:52.065035  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.01187 (* 1 = 4.01187 loss)
I0330 09:13:52.065052  2693 sgd_solver.cpp:138] Iteration 37700, lr = 0.0005
I0330 09:16:02.417856  2693 solver.cpp:243] Iteration 37800, loss = 4.18302
I0330 09:16:02.418120  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.6044 (* 1 = 3.6044 loss)
I0330 09:16:02.418138  2693 sgd_solver.cpp:138] Iteration 37800, lr = 0.0005
I0330 09:18:11.351907  2693 solver.cpp:243] Iteration 37900, loss = 4.27072
I0330 09:18:11.360478  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.79759 (* 1 = 3.79759 loss)
I0330 09:18:11.360611  2693 sgd_solver.cpp:138] Iteration 37900, lr = 0.0005
I0330 09:20:21.119020  2693 solver.cpp:243] Iteration 38000, loss = 4.24345
I0330 09:20:21.119263  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.48216 (* 1 = 4.48216 loss)
I0330 09:20:21.119297  2693 sgd_solver.cpp:138] Iteration 38000, lr = 0.0005
I0330 09:22:33.375296  2693 solver.cpp:243] Iteration 38100, loss = 4.2953
I0330 09:22:33.375516  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.42942 (* 1 = 4.42942 loss)
I0330 09:22:33.375543  2693 sgd_solver.cpp:138] Iteration 38100, lr = 0.0005
I0330 09:24:43.280184  2693 solver.cpp:243] Iteration 38200, loss = 4.32368
I0330 09:24:43.280417  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.06872 (* 1 = 5.06872 loss)
I0330 09:24:43.280436  2693 sgd_solver.cpp:138] Iteration 38200, lr = 0.0005
I0330 09:26:55.600059  2693 solver.cpp:243] Iteration 38300, loss = 4.34152
I0330 09:26:55.600241  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.27671 (* 1 = 4.27671 loss)
I0330 09:26:55.600258  2693 sgd_solver.cpp:138] Iteration 38300, lr = 0.0005
I0330 09:29:07.148141  2693 solver.cpp:243] Iteration 38400, loss = 4.20115
I0330 09:29:07.148327  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.7184 (* 1 = 4.7184 loss)
I0330 09:29:07.148344  2693 sgd_solver.cpp:138] Iteration 38400, lr = 0.0005
I0330 09:31:15.317502  2693 solver.cpp:243] Iteration 38500, loss = 4.28168
I0330 09:31:15.317714  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.38697 (* 1 = 3.38697 loss)
I0330 09:31:15.317733  2693 sgd_solver.cpp:138] Iteration 38500, lr = 0.0005
I0330 09:33:25.129761  2693 solver.cpp:243] Iteration 38600, loss = 4.39805
I0330 09:33:25.129987  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.76447 (* 1 = 4.76447 loss)
I0330 09:33:25.130007  2693 sgd_solver.cpp:138] Iteration 38600, lr = 0.0005
I0330 09:35:35.825945  2693 solver.cpp:243] Iteration 38700, loss = 4.2598
I0330 09:35:35.826143  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.8888 (* 1 = 4.8888 loss)
I0330 09:35:35.826159  2693 sgd_solver.cpp:138] Iteration 38700, lr = 0.0005
I0330 09:37:46.018254  2693 solver.cpp:243] Iteration 38800, loss = 4.18399
I0330 09:37:46.018479  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.49819 (* 1 = 4.49819 loss)
I0330 09:37:46.018502  2693 sgd_solver.cpp:138] Iteration 38800, lr = 0.0005
I0330 09:39:56.096495  2693 solver.cpp:243] Iteration 38900, loss = 4.40282
I0330 09:39:56.096740  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.67062 (* 1 = 4.67062 loss)
I0330 09:39:56.096782  2693 sgd_solver.cpp:138] Iteration 38900, lr = 0.0005
I0330 09:42:06.444705  2693 solver.cpp:243] Iteration 39000, loss = 4.2406
I0330 09:42:06.444957  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.28896 (* 1 = 5.28896 loss)
I0330 09:42:06.444979  2693 sgd_solver.cpp:138] Iteration 39000, lr = 0.0005
I0330 09:44:17.311117  2693 solver.cpp:243] Iteration 39100, loss = 4.2253
I0330 09:44:17.311347  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.21996 (* 1 = 4.21996 loss)
I0330 09:44:17.311367  2693 sgd_solver.cpp:138] Iteration 39100, lr = 0.0005
I0330 09:46:27.309496  2693 solver.cpp:243] Iteration 39200, loss = 4.23575
I0330 09:46:27.309710  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.12327 (* 1 = 4.12327 loss)
I0330 09:46:27.309729  2693 sgd_solver.cpp:138] Iteration 39200, lr = 0.0005
I0330 09:48:37.502982  2693 solver.cpp:243] Iteration 39300, loss = 4.1231
I0330 09:48:37.503252  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.45589 (* 1 = 3.45589 loss)
I0330 09:48:37.503268  2693 sgd_solver.cpp:138] Iteration 39300, lr = 0.0005
I0330 09:50:45.263128  2693 solver.cpp:243] Iteration 39400, loss = 4.19156
I0330 09:50:45.263336  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.12495 (* 1 = 4.12495 loss)
I0330 09:50:45.263355  2693 sgd_solver.cpp:138] Iteration 39400, lr = 0.0005
I0330 09:52:53.919695  2693 solver.cpp:243] Iteration 39500, loss = 4.32772
I0330 09:52:53.919888  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.43132 (* 1 = 5.43132 loss)
I0330 09:52:53.919908  2693 sgd_solver.cpp:138] Iteration 39500, lr = 0.0005
I0330 09:55:06.593456  2693 solver.cpp:243] Iteration 39600, loss = 4.42507
I0330 09:55:06.593670  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.61694 (* 1 = 4.61694 loss)
I0330 09:55:06.593688  2693 sgd_solver.cpp:138] Iteration 39600, lr = 0.0005
I0330 09:57:17.141607  2693 solver.cpp:243] Iteration 39700, loss = 4.30782
I0330 09:57:17.141903  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.7593 (* 1 = 3.7593 loss)
I0330 09:57:17.141955  2693 sgd_solver.cpp:138] Iteration 39700, lr = 0.0005
I0330 09:59:27.718463  2693 solver.cpp:243] Iteration 39800, loss = 4.16469
I0330 09:59:27.718689  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.09847 (* 1 = 5.09847 loss)
I0330 09:59:27.718719  2693 sgd_solver.cpp:138] Iteration 39800, lr = 0.0005
I0330 10:01:35.696246  2693 solver.cpp:243] Iteration 39900, loss = 4.09
I0330 10:01:35.696446  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.53964 (* 1 = 3.53964 loss)
I0330 10:01:35.696463  2693 sgd_solver.cpp:138] Iteration 39900, lr = 0.0005
I0330 10:03:44.656399  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_40000.caffemodel
I0330 10:03:45.831532  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_40000.solverstate
I0330 10:03:45.988922  2693 solver.cpp:433] Iteration 40000, Testing net (#0)
I0330 10:03:45.989006  2693 net.cpp:693] Ignoring source layer mbox_loss
I0330 10:05:06.182164  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.553266
I0330 10:05:06.956290  2693 solver.cpp:243] Iteration 40000, loss = 4.18036
I0330 10:05:06.956367  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.06921 (* 1 = 5.06921 loss)
I0330 10:05:06.956387  2693 sgd_solver.cpp:138] Iteration 40000, lr = 0.0005
I0330 10:07:18.474478  2693 solver.cpp:243] Iteration 40100, loss = 4.28794
I0330 10:07:18.474684  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.04004 (* 1 = 4.04004 loss)
I0330 10:07:18.474701  2693 sgd_solver.cpp:138] Iteration 40100, lr = 0.0005
I0330 10:09:28.172657  2693 solver.cpp:243] Iteration 40200, loss = 4.26152
I0330 10:09:28.172868  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.47068 (* 1 = 4.47068 loss)
I0330 10:09:28.172885  2693 sgd_solver.cpp:138] Iteration 40200, lr = 0.0005
I0330 10:11:37.639197  2693 solver.cpp:243] Iteration 40300, loss = 4.36766
I0330 10:11:37.639417  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.14346 (* 1 = 5.14346 loss)
I0330 10:11:37.639436  2693 sgd_solver.cpp:138] Iteration 40300, lr = 0.0005
I0330 10:13:50.308908  2693 solver.cpp:243] Iteration 40400, loss = 4.32445
I0330 10:13:50.309092  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.67908 (* 1 = 3.67908 loss)
I0330 10:13:50.309109  2693 sgd_solver.cpp:138] Iteration 40400, lr = 0.0005
I0330 10:16:00.462169  2693 solver.cpp:243] Iteration 40500, loss = 4.38587
I0330 10:16:00.462429  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.74009 (* 1 = 5.74009 loss)
I0330 10:16:00.462455  2693 sgd_solver.cpp:138] Iteration 40500, lr = 0.0005
I0330 10:18:11.936377  2693 solver.cpp:243] Iteration 40600, loss = 4.29829
I0330 10:18:11.936595  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.76134 (* 1 = 4.76134 loss)
I0330 10:18:11.936611  2693 sgd_solver.cpp:138] Iteration 40600, lr = 0.0005
I0330 10:20:21.681671  2693 solver.cpp:243] Iteration 40700, loss = 4.34883
I0330 10:20:21.681923  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.36119 (* 1 = 5.36119 loss)
I0330 10:20:21.681942  2693 sgd_solver.cpp:138] Iteration 40700, lr = 0.0005
I0330 10:22:33.374320  2693 solver.cpp:243] Iteration 40800, loss = 4.33816
I0330 10:22:33.374620  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.01722 (* 1 = 3.01722 loss)
I0330 10:22:33.374681  2693 sgd_solver.cpp:138] Iteration 40800, lr = 0.0005
I0330 10:24:43.377890  2693 solver.cpp:243] Iteration 40900, loss = 4.26646
I0330 10:24:43.378113  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.58786 (* 1 = 4.58786 loss)
I0330 10:24:43.378130  2693 sgd_solver.cpp:138] Iteration 40900, lr = 0.0005
I0330 10:26:53.160009  2693 solver.cpp:243] Iteration 41000, loss = 4.31925
I0330 10:26:53.160229  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.15254 (* 1 = 4.15254 loss)
I0330 10:26:53.160245  2693 sgd_solver.cpp:138] Iteration 41000, lr = 0.0005
I0330 10:29:04.082077  2693 solver.cpp:243] Iteration 41100, loss = 4.21641
I0330 10:29:04.082345  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.19799 (* 1 = 5.19799 loss)
I0330 10:29:04.082389  2693 sgd_solver.cpp:138] Iteration 41100, lr = 0.0005
I0330 10:31:12.504091  2693 solver.cpp:243] Iteration 41200, loss = 4.33094
I0330 10:31:12.512491  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.39601 (* 1 = 4.39601 loss)
I0330 10:31:12.512557  2693 sgd_solver.cpp:138] Iteration 41200, lr = 0.0005
I0330 10:33:21.185535  2693 solver.cpp:243] Iteration 41300, loss = 4.31487
I0330 10:33:21.185775  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.85446 (* 1 = 3.85446 loss)
I0330 10:33:21.185807  2693 sgd_solver.cpp:138] Iteration 41300, lr = 0.0005
I0330 10:35:32.810845  2693 solver.cpp:243] Iteration 41400, loss = 4.26994
I0330 10:35:32.811101  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.5671 (* 1 = 4.5671 loss)
I0330 10:35:32.811133  2693 sgd_solver.cpp:138] Iteration 41400, lr = 0.0005
I0330 10:37:44.211128  2693 solver.cpp:243] Iteration 41500, loss = 4.18064
I0330 10:37:44.213268  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.12055 (* 1 = 4.12055 loss)
I0330 10:37:44.213315  2693 sgd_solver.cpp:138] Iteration 41500, lr = 0.0005
I0330 10:39:52.313482  2693 solver.cpp:243] Iteration 41600, loss = 4.13307
I0330 10:39:52.313758  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.43711 (* 1 = 4.43711 loss)
I0330 10:39:52.313797  2693 sgd_solver.cpp:138] Iteration 41600, lr = 0.0005
I0330 10:42:03.217730  2693 solver.cpp:243] Iteration 41700, loss = 4.23551
I0330 10:42:03.218040  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.027 (* 1 = 3.027 loss)
I0330 10:42:03.218071  2693 sgd_solver.cpp:138] Iteration 41700, lr = 0.0005
I0330 10:44:12.415369  2693 solver.cpp:243] Iteration 41800, loss = 4.27998
I0330 10:44:12.415573  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.10382 (* 1 = 4.10382 loss)
I0330 10:44:12.415668  2693 sgd_solver.cpp:138] Iteration 41800, lr = 0.0005
I0330 10:46:22.173732  2693 solver.cpp:243] Iteration 41900, loss = 4.20425
I0330 10:46:22.173976  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.57767 (* 1 = 4.57767 loss)
I0330 10:46:22.174007  2693 sgd_solver.cpp:138] Iteration 41900, lr = 0.0005
I0330 10:48:34.691882  2693 solver.cpp:243] Iteration 42000, loss = 4.21461
I0330 10:48:34.692114  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.75792 (* 1 = 3.75792 loss)
I0330 10:48:34.692147  2693 sgd_solver.cpp:138] Iteration 42000, lr = 0.0005
I0330 10:50:47.371381  2693 solver.cpp:243] Iteration 42100, loss = 4.28701
I0330 10:50:47.371677  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.09777 (* 1 = 4.09777 loss)
I0330 10:50:47.371711  2693 sgd_solver.cpp:138] Iteration 42100, lr = 0.0005
I0330 10:53:01.025264  2693 solver.cpp:243] Iteration 42200, loss = 4.32486
I0330 10:53:01.026268  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.00721 (* 1 = 3.00721 loss)
I0330 10:53:01.026288  2693 sgd_solver.cpp:138] Iteration 42200, lr = 0.0005
I0330 10:55:14.597704  2693 solver.cpp:243] Iteration 42300, loss = 4.29717
I0330 10:55:14.597967  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.48777 (* 1 = 5.48777 loss)
I0330 10:55:14.598008  2693 sgd_solver.cpp:138] Iteration 42300, lr = 0.0005
I0330 10:57:28.001224  2693 solver.cpp:243] Iteration 42400, loss = 4.19125
I0330 10:57:28.001631  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.75464 (* 1 = 4.75464 loss)
I0330 10:57:28.001667  2693 sgd_solver.cpp:138] Iteration 42400, lr = 0.0005
I0330 10:59:42.080461  2693 solver.cpp:243] Iteration 42500, loss = 4.35234
I0330 10:59:42.081441  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.23566 (* 1 = 3.23566 loss)
I0330 10:59:42.081462  2693 sgd_solver.cpp:138] Iteration 42500, lr = 0.0005
I0330 11:01:54.937289  2693 solver.cpp:243] Iteration 42600, loss = 4.22303
I0330 11:01:54.937472  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.89568 (* 1 = 3.89568 loss)
I0330 11:01:54.937492  2693 sgd_solver.cpp:138] Iteration 42600, lr = 0.0005
I0330 11:04:10.465528  2693 solver.cpp:243] Iteration 42700, loss = 4.19251
I0330 11:04:10.465740  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.89843 (* 1 = 4.89843 loss)
I0330 11:04:10.465766  2693 sgd_solver.cpp:138] Iteration 42700, lr = 0.0005
I0330 11:06:23.903214  2693 solver.cpp:243] Iteration 42800, loss = 4.25047
I0330 11:06:23.903523  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.53331 (* 1 = 2.53331 loss)
I0330 11:06:23.903549  2693 sgd_solver.cpp:138] Iteration 42800, lr = 0.0005
I0330 11:08:38.549345  2693 solver.cpp:243] Iteration 42900, loss = 4.25637
I0330 11:08:38.549574  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.95321 (* 1 = 4.95321 loss)
I0330 11:08:38.549592  2693 sgd_solver.cpp:138] Iteration 42900, lr = 0.0005
I0330 11:10:51.286689  2693 solver.cpp:243] Iteration 43000, loss = 4.21205
I0330 11:10:51.286960  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.25054 (* 1 = 5.25054 loss)
I0330 11:10:51.287004  2693 sgd_solver.cpp:138] Iteration 43000, lr = 0.0005
I0330 11:13:07.553223  2693 solver.cpp:243] Iteration 43100, loss = 4.2742
I0330 11:13:07.553421  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.61971 (* 1 = 4.61971 loss)
I0330 11:13:07.553436  2693 sgd_solver.cpp:138] Iteration 43100, lr = 0.0005
I0330 11:15:21.388244  2693 solver.cpp:243] Iteration 43200, loss = 4.1904
I0330 11:15:21.389196  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.01977 (* 1 = 4.01977 loss)
I0330 11:15:21.389214  2693 sgd_solver.cpp:138] Iteration 43200, lr = 0.0005
I0330 11:17:34.026254  2693 solver.cpp:243] Iteration 43300, loss = 4.25612
I0330 11:17:34.026433  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.78491 (* 1 = 3.78491 loss)
I0330 11:17:34.026450  2693 sgd_solver.cpp:138] Iteration 43300, lr = 0.0005
I0330 11:19:48.352414  2693 solver.cpp:243] Iteration 43400, loss = 4.07469
I0330 11:19:48.352660  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.15965 (* 1 = 3.15965 loss)
I0330 11:19:48.352715  2693 sgd_solver.cpp:138] Iteration 43400, lr = 0.0005
I0330 11:22:01.260462  2693 solver.cpp:243] Iteration 43500, loss = 4.20363
I0330 11:22:01.260785  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.26307 (* 1 = 4.26307 loss)
I0330 11:22:01.260852  2693 sgd_solver.cpp:138] Iteration 43500, lr = 0.0005
I0330 11:24:13.596480  2693 solver.cpp:243] Iteration 43600, loss = 4.03742
I0330 11:24:13.597546  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.62865 (* 1 = 4.62865 loss)
I0330 11:24:13.597564  2693 sgd_solver.cpp:138] Iteration 43600, lr = 0.0005
I0330 11:26:27.226351  2693 solver.cpp:243] Iteration 43700, loss = 4.14023
I0330 11:26:27.226570  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.50476 (* 1 = 3.50476 loss)
I0330 11:26:27.226610  2693 sgd_solver.cpp:138] Iteration 43700, lr = 0.0005
I0330 11:28:41.975248  2693 solver.cpp:243] Iteration 43800, loss = 4.29988
I0330 11:28:41.977587  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.33683 (* 1 = 4.33683 loss)
I0330 11:28:41.977653  2693 sgd_solver.cpp:138] Iteration 43800, lr = 0.0005
I0330 11:30:56.151520  2693 solver.cpp:243] Iteration 43900, loss = 4.30991
I0330 11:30:56.152575  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.97064 (* 1 = 3.97064 loss)
I0330 11:30:56.152595  2693 sgd_solver.cpp:138] Iteration 43900, lr = 0.0005
I0330 11:33:11.247748  2693 solver.cpp:243] Iteration 44000, loss = 4.25271
I0330 11:33:11.247952  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.99947 (* 1 = 3.99947 loss)
I0330 11:33:11.247972  2693 sgd_solver.cpp:138] Iteration 44000, lr = 0.0005
I0330 11:35:25.622789  2693 solver.cpp:243] Iteration 44100, loss = 4.17048
I0330 11:35:25.623008  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.51725 (* 1 = 4.51725 loss)
I0330 11:35:25.623049  2693 sgd_solver.cpp:138] Iteration 44100, lr = 0.0005
I0330 11:37:40.151193  2693 solver.cpp:243] Iteration 44200, loss = 4.22452
I0330 11:37:40.151427  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.98928 (* 1 = 4.98928 loss)
I0330 11:37:40.151461  2693 sgd_solver.cpp:138] Iteration 44200, lr = 0.0005
I0330 11:39:52.982816  2693 solver.cpp:243] Iteration 44300, loss = 4.06966
I0330 11:39:52.983024  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.21806 (* 1 = 3.21806 loss)
I0330 11:39:52.983044  2693 sgd_solver.cpp:138] Iteration 44300, lr = 0.0005
I0330 11:42:08.894901  2693 solver.cpp:243] Iteration 44400, loss = 4.15354
I0330 11:42:08.895131  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.7799 (* 1 = 3.7799 loss)
I0330 11:42:08.895149  2693 sgd_solver.cpp:138] Iteration 44400, lr = 0.0005
I0330 11:44:22.866418  2693 solver.cpp:243] Iteration 44500, loss = 4.1931
I0330 11:44:22.866644  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.0602 (* 1 = 5.0602 loss)
I0330 11:44:22.866678  2693 sgd_solver.cpp:138] Iteration 44500, lr = 0.0005
I0330 11:46:36.949595  2693 solver.cpp:243] Iteration 44600, loss = 4.3308
I0330 11:46:36.949771  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.25723 (* 1 = 4.25723 loss)
I0330 11:46:36.949789  2693 sgd_solver.cpp:138] Iteration 44600, lr = 0.0005
I0330 11:48:49.104374  2693 solver.cpp:243] Iteration 44700, loss = 4.03732
I0330 11:48:49.104589  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.94832 (* 1 = 4.94832 loss)
I0330 11:48:49.104619  2693 sgd_solver.cpp:138] Iteration 44700, lr = 0.0005
I0330 11:51:04.011826  2693 solver.cpp:243] Iteration 44800, loss = 4.28333
I0330 11:51:04.012367  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.19792 (* 1 = 4.19792 loss)
I0330 11:51:04.012385  2693 sgd_solver.cpp:138] Iteration 44800, lr = 0.0005
I0330 11:53:17.368355  2693 solver.cpp:243] Iteration 44900, loss = 4.15115
I0330 11:53:17.368618  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.44824 (* 1 = 3.44824 loss)
I0330 11:53:17.368655  2693 sgd_solver.cpp:138] Iteration 44900, lr = 0.0005
I0330 11:55:30.136868  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_45000.caffemodel
I0330 11:55:31.112810  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_45000.solverstate
I0330 11:55:31.248687  2693 solver.cpp:433] Iteration 45000, Testing net (#0)
I0330 11:55:31.248769  2693 net.cpp:693] Ignoring source layer mbox_loss
I0330 11:56:52.023960  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.557218
I0330 11:56:52.659711  2693 solver.cpp:243] Iteration 45000, loss = 4.07074
I0330 11:56:52.659788  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.46247 (* 1 = 4.46247 loss)
I0330 11:56:52.659804  2693 sgd_solver.cpp:138] Iteration 45000, lr = 0.0005
I0330 11:59:04.314883  2693 solver.cpp:243] Iteration 45100, loss = 4.19322
I0330 11:59:04.315201  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.81713 (* 1 = 4.81713 loss)
I0330 11:59:04.315248  2693 sgd_solver.cpp:138] Iteration 45100, lr = 0.0005
I0330 12:01:18.196190  2693 solver.cpp:243] Iteration 45200, loss = 4.27664
I0330 12:01:18.196373  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.80894 (* 1 = 3.80894 loss)
I0330 12:01:18.196391  2693 sgd_solver.cpp:138] Iteration 45200, lr = 0.0005
I0330 12:03:32.951297  2693 solver.cpp:243] Iteration 45300, loss = 4.33885
I0330 12:03:32.951511  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.47964 (* 1 = 3.47964 loss)
I0330 12:03:32.951544  2693 sgd_solver.cpp:138] Iteration 45300, lr = 0.0005
I0330 12:05:46.705157  2693 solver.cpp:243] Iteration 45400, loss = 4.28046
I0330 12:05:46.705346  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.67596 (* 1 = 3.67596 loss)
I0330 12:05:46.705363  2693 sgd_solver.cpp:138] Iteration 45400, lr = 0.0005
I0330 12:07:58.162706  2693 solver.cpp:243] Iteration 45500, loss = 4.2877
I0330 12:07:58.162926  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.55817 (* 1 = 5.55817 loss)
I0330 12:07:58.162958  2693 sgd_solver.cpp:138] Iteration 45500, lr = 0.0005
I0330 12:10:10.230979  2693 solver.cpp:243] Iteration 45600, loss = 4.18103
I0330 12:10:10.231227  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.38793 (* 1 = 5.38793 loss)
I0330 12:10:10.231262  2693 sgd_solver.cpp:138] Iteration 45600, lr = 0.0005
I0330 12:12:24.195413  2693 solver.cpp:243] Iteration 45700, loss = 4.29974
I0330 12:12:24.195714  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.88502 (* 1 = 4.88502 loss)
I0330 12:12:24.195745  2693 sgd_solver.cpp:138] Iteration 45700, lr = 0.0005
I0330 12:14:37.769263  2693 solver.cpp:243] Iteration 45800, loss = 4.14031
I0330 12:14:37.769491  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.03642 (* 1 = 3.03642 loss)
I0330 12:14:37.769527  2693 sgd_solver.cpp:138] Iteration 45800, lr = 0.0005
I0330 12:16:50.568835  2693 solver.cpp:243] Iteration 45900, loss = 4.38155
I0330 12:16:50.569007  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.80501 (* 1 = 4.80501 loss)
I0330 12:16:50.569025  2693 sgd_solver.cpp:138] Iteration 45900, lr = 0.0005
I0330 12:19:04.283537  2693 solver.cpp:243] Iteration 46000, loss = 4.19952
I0330 12:19:04.284009  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.56137 (* 1 = 4.56137 loss)
I0330 12:19:04.284029  2693 sgd_solver.cpp:138] Iteration 46000, lr = 0.0005
I0330 12:21:17.121561  2693 solver.cpp:243] Iteration 46100, loss = 4.21567
I0330 12:21:17.121816  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.0335 (* 1 = 4.0335 loss)
I0330 12:21:17.121853  2693 sgd_solver.cpp:138] Iteration 46100, lr = 0.0005
I0330 12:23:31.955379  2693 solver.cpp:243] Iteration 46200, loss = 4.24649
I0330 12:23:31.955658  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.51109 (* 1 = 6.51109 loss)
I0330 12:23:31.955687  2693 sgd_solver.cpp:138] Iteration 46200, lr = 0.0005
I0330 12:25:44.431407  2693 solver.cpp:243] Iteration 46300, loss = 4.29132
I0330 12:25:44.431674  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.51019 (* 1 = 4.51019 loss)
I0330 12:25:44.431691  2693 sgd_solver.cpp:138] Iteration 46300, lr = 0.0005
I0330 12:27:59.355896  2693 solver.cpp:243] Iteration 46400, loss = 4.30758
I0330 12:27:59.356091  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.74042 (* 1 = 4.74042 loss)
I0330 12:27:59.356108  2693 sgd_solver.cpp:138] Iteration 46400, lr = 0.0005
I0330 12:30:11.015779  2693 solver.cpp:243] Iteration 46500, loss = 4.14467
I0330 12:30:11.015944  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.45268 (* 1 = 3.45268 loss)
I0330 12:30:11.015960  2693 sgd_solver.cpp:138] Iteration 46500, lr = 0.0005
I0330 12:32:23.720199  2693 solver.cpp:243] Iteration 46600, loss = 4.178
I0330 12:32:23.720398  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.85019 (* 1 = 5.85019 loss)
I0330 12:32:23.720413  2693 sgd_solver.cpp:138] Iteration 46600, lr = 0.0005
I0330 12:34:36.106324  2693 solver.cpp:243] Iteration 46700, loss = 4.12749
I0330 12:34:36.106585  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.3525 (* 1 = 4.3525 loss)
I0330 12:34:36.106616  2693 sgd_solver.cpp:138] Iteration 46700, lr = 0.0005
I0330 12:36:48.375257  2693 solver.cpp:243] Iteration 46800, loss = 4.2781
I0330 12:36:48.375458  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.84322 (* 1 = 3.84322 loss)
I0330 12:36:48.375474  2693 sgd_solver.cpp:138] Iteration 46800, lr = 0.0005
I0330 12:38:59.406519  2693 solver.cpp:243] Iteration 46900, loss = 4.12963
I0330 12:38:59.406733  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.7507 (* 1 = 2.7507 loss)
I0330 12:38:59.406754  2693 sgd_solver.cpp:138] Iteration 46900, lr = 0.0005
I0330 12:41:14.134457  2693 solver.cpp:243] Iteration 47000, loss = 4.23667
I0330 12:41:14.134698  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.89548 (* 1 = 4.89548 loss)
I0330 12:41:14.134732  2693 sgd_solver.cpp:138] Iteration 47000, lr = 0.0005
I0330 12:43:27.059721  2693 solver.cpp:243] Iteration 47100, loss = 4.18808
I0330 12:43:27.060194  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.60885 (* 1 = 3.60885 loss)
I0330 12:43:27.060211  2693 sgd_solver.cpp:138] Iteration 47100, lr = 0.0005
I0330 12:45:40.904594  2693 solver.cpp:243] Iteration 47200, loss = 4.17923
I0330 12:45:40.904783  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.34033 (* 1 = 4.34033 loss)
I0330 12:45:40.904800  2693 sgd_solver.cpp:138] Iteration 47200, lr = 0.0005
I0330 12:47:52.276654  2693 solver.cpp:243] Iteration 47300, loss = 4.11793
I0330 12:47:52.276846  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.81746 (* 1 = 4.81746 loss)
I0330 12:47:52.276865  2693 sgd_solver.cpp:138] Iteration 47300, lr = 0.0005
I0330 12:50:04.255833  2693 solver.cpp:243] Iteration 47400, loss = 4.12187
I0330 12:50:04.263442  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.0932 (* 1 = 4.0932 loss)
I0330 12:50:04.263460  2693 sgd_solver.cpp:138] Iteration 47400, lr = 0.0005
I0330 12:52:18.781535  2693 solver.cpp:243] Iteration 47500, loss = 4.29905
I0330 12:52:18.781747  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.11577 (* 1 = 5.11577 loss)
I0330 12:52:18.781766  2693 sgd_solver.cpp:138] Iteration 47500, lr = 0.0005
I0330 12:54:30.466588  2693 solver.cpp:243] Iteration 47600, loss = 4.12546
I0330 12:54:30.466851  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.60524 (* 1 = 3.60524 loss)
I0330 12:54:30.466895  2693 sgd_solver.cpp:138] Iteration 47600, lr = 0.0005
I0330 12:56:44.507511  2693 solver.cpp:243] Iteration 47700, loss = 4.22312
I0330 12:56:44.507756  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.19165 (* 1 = 5.19165 loss)
I0330 12:56:44.507776  2693 sgd_solver.cpp:138] Iteration 47700, lr = 0.0005
I0330 12:58:59.060308  2693 solver.cpp:243] Iteration 47800, loss = 4.27875
I0330 12:58:59.060755  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.37313 (* 1 = 4.37313 loss)
I0330 12:58:59.060773  2693 sgd_solver.cpp:138] Iteration 47800, lr = 0.0005
I0330 13:01:12.247145  2693 solver.cpp:243] Iteration 47900, loss = 4.18908
I0330 13:01:12.247370  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.57665 (* 1 = 5.57665 loss)
I0330 13:01:12.247404  2693 sgd_solver.cpp:138] Iteration 47900, lr = 0.0005
I0330 13:03:26.759343  2693 solver.cpp:243] Iteration 48000, loss = 4.17633
I0330 13:03:26.759546  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.93309 (* 1 = 4.93309 loss)
I0330 13:03:26.759564  2693 sgd_solver.cpp:138] Iteration 48000, lr = 0.0005
I0330 13:05:40.053951  2693 solver.cpp:243] Iteration 48100, loss = 4.13104
I0330 13:05:40.054172  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.37487 (* 1 = 4.37487 loss)
I0330 13:05:40.054188  2693 sgd_solver.cpp:138] Iteration 48100, lr = 0.0005
I0330 13:07:53.137282  2693 solver.cpp:243] Iteration 48200, loss = 4.2669
I0330 13:07:53.137697  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.53578 (* 1 = 3.53578 loss)
I0330 13:07:53.137717  2693 sgd_solver.cpp:138] Iteration 48200, lr = 0.0005
I0330 13:10:07.236076  2693 solver.cpp:243] Iteration 48300, loss = 4.17525
I0330 13:10:07.236333  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.98569 (* 1 = 3.98569 loss)
I0330 13:10:07.236353  2693 sgd_solver.cpp:138] Iteration 48300, lr = 0.0005
I0330 13:12:21.187705  2693 solver.cpp:243] Iteration 48400, loss = 4.16235
I0330 13:12:21.187942  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.96277 (* 1 = 3.96277 loss)
I0330 13:12:21.187997  2693 sgd_solver.cpp:138] Iteration 48400, lr = 0.0005
I0330 13:14:35.283565  2693 solver.cpp:243] Iteration 48500, loss = 4.1099
I0330 13:14:35.283818  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.20879 (* 1 = 3.20879 loss)
I0330 13:14:35.283834  2693 sgd_solver.cpp:138] Iteration 48500, lr = 0.0005
I0330 13:16:48.663154  2693 solver.cpp:243] Iteration 48600, loss = 4.12075
I0330 13:16:48.671128  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.16653 (* 1 = 4.16653 loss)
I0330 13:16:48.671217  2693 sgd_solver.cpp:138] Iteration 48600, lr = 0.0005
I0330 13:19:01.957041  2693 solver.cpp:243] Iteration 48700, loss = 4.06289
I0330 13:19:01.957257  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.90522 (* 1 = 3.90522 loss)
I0330 13:19:01.957285  2693 sgd_solver.cpp:138] Iteration 48700, lr = 0.0005
I0330 13:21:15.007439  2693 solver.cpp:243] Iteration 48800, loss = 4.14239
I0330 13:21:15.007719  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.48561 (* 1 = 3.48561 loss)
I0330 13:21:15.007737  2693 sgd_solver.cpp:138] Iteration 48800, lr = 0.0005
I0330 13:23:29.090224  2693 solver.cpp:243] Iteration 48900, loss = 4.28855
I0330 13:23:29.094419  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.63918 (* 1 = 3.63918 loss)
I0330 13:23:29.094457  2693 sgd_solver.cpp:138] Iteration 48900, lr = 0.0005
I0330 13:25:39.310441  2693 solver.cpp:243] Iteration 49000, loss = 4.05646
I0330 13:25:39.316965  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.18518 (* 1 = 5.18518 loss)
I0330 13:25:39.317083  2693 sgd_solver.cpp:138] Iteration 49000, lr = 0.0005
I0330 13:27:49.740445  2693 solver.cpp:243] Iteration 49100, loss = 4.11734
I0330 13:27:49.740720  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.26794 (* 1 = 4.26794 loss)
I0330 13:27:49.740757  2693 sgd_solver.cpp:138] Iteration 49100, lr = 0.0005
I0330 13:29:59.272029  2693 solver.cpp:243] Iteration 49200, loss = 4.06858
I0330 13:29:59.278393  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.3368 (* 1 = 4.3368 loss)
I0330 13:29:59.278409  2693 sgd_solver.cpp:138] Iteration 49200, lr = 0.0005
I0330 13:32:09.437650  2693 solver.cpp:243] Iteration 49300, loss = 4.012
I0330 13:32:09.437834  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.45189 (* 1 = 3.45189 loss)
I0330 13:32:09.437850  2693 sgd_solver.cpp:138] Iteration 49300, lr = 0.0005
I0330 13:34:22.884668  2693 solver.cpp:243] Iteration 49400, loss = 4.26702
I0330 13:34:22.884850  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.48051 (* 1 = 5.48051 loss)
I0330 13:34:22.884867  2693 sgd_solver.cpp:138] Iteration 49400, lr = 0.0005
I0330 13:36:33.734993  2693 solver.cpp:243] Iteration 49500, loss = 4.17984
I0330 13:36:33.735249  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.99264 (* 1 = 4.99264 loss)
I0330 13:36:33.735276  2693 sgd_solver.cpp:138] Iteration 49500, lr = 0.0005
I0330 13:38:47.109737  2693 solver.cpp:243] Iteration 49600, loss = 4.25432
I0330 13:38:47.111004  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.09318 (* 1 = 3.09318 loss)
I0330 13:38:47.111176  2693 sgd_solver.cpp:138] Iteration 49600, lr = 0.0005
I0330 13:40:59.680584  2693 solver.cpp:243] Iteration 49700, loss = 4.15666
I0330 13:40:59.680850  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.87827 (* 1 = 4.87827 loss)
I0330 13:40:59.680873  2693 sgd_solver.cpp:138] Iteration 49700, lr = 0.0005
I0330 13:43:09.070982  2693 solver.cpp:243] Iteration 49800, loss = 4.07236
I0330 13:43:09.071305  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.5257 (* 1 = 4.5257 loss)
I0330 13:43:09.071341  2693 sgd_solver.cpp:138] Iteration 49800, lr = 0.0005
I0330 13:45:19.231119  2693 solver.cpp:243] Iteration 49900, loss = 4.21384
I0330 13:45:19.231385  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.91589 (* 1 = 3.91589 loss)
I0330 13:45:19.231417  2693 sgd_solver.cpp:138] Iteration 49900, lr = 0.0005
I0330 13:47:27.962736  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_50000.caffemodel
I0330 13:47:29.012722  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_50000.solverstate
I0330 13:47:29.165585  2693 solver.cpp:433] Iteration 50000, Testing net (#0)
I0330 13:47:29.165691  2693 net.cpp:693] Ignoring source layer mbox_loss
I0330 13:48:49.341518  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.58511
I0330 13:48:50.161304  2693 solver.cpp:243] Iteration 50000, loss = 4.15163
I0330 13:48:50.161375  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.44259 (* 1 = 5.44259 loss)
I0330 13:48:50.161391  2693 sgd_solver.cpp:138] Iteration 50000, lr = 0.0005
I0330 13:51:00.260565  2693 solver.cpp:243] Iteration 50100, loss = 4.1756
I0330 13:51:00.260771  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.32953 (* 1 = 5.32953 loss)
I0330 13:51:00.260787  2693 sgd_solver.cpp:138] Iteration 50100, lr = 0.0005
I0330 13:53:11.243343  2693 solver.cpp:243] Iteration 50200, loss = 4.22617
I0330 13:53:11.243532  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.12987 (* 1 = 6.12987 loss)
I0330 13:53:11.243548  2693 sgd_solver.cpp:138] Iteration 50200, lr = 0.0005
I0330 13:55:21.453768  2693 solver.cpp:243] Iteration 50300, loss = 4.08059
I0330 13:55:21.454032  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.94968 (* 1 = 2.94968 loss)
I0330 13:55:21.454058  2693 sgd_solver.cpp:138] Iteration 50300, lr = 0.0005
I0330 13:57:34.832417  2693 solver.cpp:243] Iteration 50400, loss = 4.19136
I0330 13:57:34.832623  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.46144 (* 1 = 5.46144 loss)
I0330 13:57:34.832643  2693 sgd_solver.cpp:138] Iteration 50400, lr = 0.0005
I0330 13:59:45.068701  2693 solver.cpp:243] Iteration 50500, loss = 4.08625
I0330 13:59:45.068945  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.1498 (* 1 = 3.1498 loss)
I0330 13:59:45.069000  2693 sgd_solver.cpp:138] Iteration 50500, lr = 0.0005
I0330 14:01:55.047145  2693 solver.cpp:243] Iteration 50600, loss = 4.22647
I0330 14:01:55.047349  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.48128 (* 1 = 5.48128 loss)
I0330 14:01:55.047366  2693 sgd_solver.cpp:138] Iteration 50600, lr = 0.0005
I0330 14:04:04.037858  2693 solver.cpp:243] Iteration 50700, loss = 4.22952
I0330 14:04:04.038055  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.01648 (* 1 = 4.01648 loss)
I0330 14:04:04.038072  2693 sgd_solver.cpp:138] Iteration 50700, lr = 0.0005
I0330 14:06:14.580221  2693 solver.cpp:243] Iteration 50800, loss = 4.06592
I0330 14:06:14.580471  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.62786 (* 1 = 4.62786 loss)
I0330 14:06:14.580591  2693 sgd_solver.cpp:138] Iteration 50800, lr = 0.0005
I0330 14:08:27.642765  2693 solver.cpp:243] Iteration 50900, loss = 4.22156
I0330 14:08:27.643013  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.30191 (* 1 = 4.30191 loss)
I0330 14:08:27.643040  2693 sgd_solver.cpp:138] Iteration 50900, lr = 0.0005
I0330 14:10:40.719380  2693 solver.cpp:243] Iteration 51000, loss = 4.16591
I0330 14:10:40.719708  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.67491 (* 1 = 4.67491 loss)
I0330 14:10:40.719734  2693 sgd_solver.cpp:138] Iteration 51000, lr = 0.0005
I0330 14:12:51.963423  2693 solver.cpp:243] Iteration 51100, loss = 4.10261
I0330 14:12:51.963709  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.08265 (* 1 = 5.08265 loss)
I0330 14:12:51.963740  2693 sgd_solver.cpp:138] Iteration 51100, lr = 0.0005
I0330 14:15:02.124853  2693 solver.cpp:243] Iteration 51200, loss = 4.30783
I0330 14:15:02.125090  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.1091 (* 1 = 5.1091 loss)
I0330 14:15:02.125108  2693 sgd_solver.cpp:138] Iteration 51200, lr = 0.0005
I0330 14:17:13.761973  2693 solver.cpp:243] Iteration 51300, loss = 4.1304
I0330 14:17:13.762177  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.50126 (* 1 = 4.50126 loss)
I0330 14:17:13.762197  2693 sgd_solver.cpp:138] Iteration 51300, lr = 0.0005
I0330 14:19:25.404397  2693 solver.cpp:243] Iteration 51400, loss = 4.17256
I0330 14:19:25.404606  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.1985 (* 1 = 6.1985 loss)
I0330 14:19:25.404626  2693 sgd_solver.cpp:138] Iteration 51400, lr = 0.0005
I0330 14:21:35.559661  2693 solver.cpp:243] Iteration 51500, loss = 4.09584
I0330 14:21:35.559845  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.94695 (* 1 = 5.94695 loss)
I0330 14:21:35.559864  2693 sgd_solver.cpp:138] Iteration 51500, lr = 0.0005
I0330 14:23:45.369467  2693 solver.cpp:243] Iteration 51600, loss = 4.26615
I0330 14:23:45.369730  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.2587 (* 1 = 5.2587 loss)
I0330 14:23:45.369774  2693 sgd_solver.cpp:138] Iteration 51600, lr = 0.0005
I0330 14:25:56.670127  2693 solver.cpp:243] Iteration 51700, loss = 4.20479
I0330 14:25:56.670367  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.42729 (* 1 = 4.42729 loss)
I0330 14:25:56.670398  2693 sgd_solver.cpp:138] Iteration 51700, lr = 0.0005
I0330 14:28:07.019795  2693 solver.cpp:243] Iteration 51800, loss = 4.12493
I0330 14:28:07.020051  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.21484 (* 1 = 4.21484 loss)
I0330 14:28:07.020082  2693 sgd_solver.cpp:138] Iteration 51800, lr = 0.0005
I0330 14:30:19.030597  2693 solver.cpp:243] Iteration 51900, loss = 4.16767
I0330 14:30:19.030838  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.41182 (* 1 = 4.41182 loss)
I0330 14:30:19.030864  2693 sgd_solver.cpp:138] Iteration 51900, lr = 0.0005
I0330 14:32:30.168782  2693 solver.cpp:243] Iteration 52000, loss = 4.23203
I0330 14:32:30.169040  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.79156 (* 1 = 2.79156 loss)
I0330 14:32:30.169076  2693 sgd_solver.cpp:138] Iteration 52000, lr = 0.0005
I0330 14:34:42.846557  2693 solver.cpp:243] Iteration 52100, loss = 4.29594
I0330 14:34:42.852787  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.70415 (* 1 = 2.70415 loss)
I0330 14:34:42.852803  2693 sgd_solver.cpp:138] Iteration 52100, lr = 0.0005
I0330 14:36:52.809871  2693 solver.cpp:243] Iteration 52200, loss = 4.18682
I0330 14:36:52.810078  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.04568 (* 1 = 5.04568 loss)
I0330 14:36:52.810096  2693 sgd_solver.cpp:138] Iteration 52200, lr = 0.0005
I0330 14:39:03.167459  2693 solver.cpp:243] Iteration 52300, loss = 4.27192
I0330 14:39:03.167795  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.01616 (* 1 = 3.01616 loss)
I0330 14:39:03.167832  2693 sgd_solver.cpp:138] Iteration 52300, lr = 0.0005
I0330 14:41:13.186007  2693 solver.cpp:243] Iteration 52400, loss = 4.17134
I0330 14:41:13.186259  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.96125 (* 1 = 5.96125 loss)
I0330 14:41:13.186300  2693 sgd_solver.cpp:138] Iteration 52400, lr = 0.0005
I0330 14:43:22.662061  2693 solver.cpp:243] Iteration 52500, loss = 4.27183
I0330 14:43:22.662259  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.45647 (* 1 = 3.45647 loss)
I0330 14:43:22.662276  2693 sgd_solver.cpp:138] Iteration 52500, lr = 0.0005
I0330 14:45:33.080143  2693 solver.cpp:243] Iteration 52600, loss = 4.12574
I0330 14:45:33.080327  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.40299 (* 1 = 3.40299 loss)
I0330 14:45:33.080346  2693 sgd_solver.cpp:138] Iteration 52600, lr = 0.0005
I0330 14:47:45.685747  2693 solver.cpp:243] Iteration 52700, loss = 4.27196
I0330 14:47:45.686035  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.79588 (* 1 = 3.79588 loss)
I0330 14:47:45.686058  2693 sgd_solver.cpp:138] Iteration 52700, lr = 0.0005
I0330 14:49:57.124966  2693 solver.cpp:243] Iteration 52800, loss = 4.07017
I0330 14:49:57.125074  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.04215 (* 1 = 4.04215 loss)
I0330 14:49:57.125092  2693 sgd_solver.cpp:138] Iteration 52800, lr = 0.0005
I0330 14:52:07.158691  2693 solver.cpp:243] Iteration 52900, loss = 4.102
I0330 14:52:07.158888  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.83209 (* 1 = 2.83209 loss)
I0330 14:52:07.158906  2693 sgd_solver.cpp:138] Iteration 52900, lr = 0.0005
I0330 14:54:18.927748  2693 solver.cpp:243] Iteration 53000, loss = 4.15887
I0330 14:54:18.927954  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.71253 (* 1 = 5.71253 loss)
I0330 14:54:18.927971  2693 sgd_solver.cpp:138] Iteration 53000, lr = 0.0005
I0330 14:56:29.247807  2693 solver.cpp:243] Iteration 53100, loss = 4.13912
I0330 14:56:29.248119  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.74414 (* 1 = 4.74414 loss)
I0330 14:56:29.248139  2693 sgd_solver.cpp:138] Iteration 53100, lr = 0.0005
I0330 14:58:41.040515  2693 solver.cpp:243] Iteration 53200, loss = 4.06316
I0330 14:58:41.040712  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.73344 (* 1 = 4.73344 loss)
I0330 14:58:41.040730  2693 sgd_solver.cpp:138] Iteration 53200, lr = 0.0005
I0330 15:00:50.231937  2693 solver.cpp:243] Iteration 53300, loss = 4.18471
I0330 15:00:50.232143  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.58431 (* 1 = 5.58431 loss)
I0330 15:00:50.232161  2693 sgd_solver.cpp:138] Iteration 53300, lr = 0.0005
I0330 15:03:00.188675  2693 solver.cpp:243] Iteration 53400, loss = 4.13218
I0330 15:03:00.188923  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.99272 (* 1 = 4.99272 loss)
I0330 15:03:00.188972  2693 sgd_solver.cpp:138] Iteration 53400, lr = 0.0005
I0330 15:05:13.210458  2693 solver.cpp:243] Iteration 53500, loss = 4.22974
I0330 15:05:13.210697  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.53494 (* 1 = 3.53494 loss)
I0330 15:05:13.210721  2693 sgd_solver.cpp:138] Iteration 53500, lr = 0.0005
I0330 15:07:23.382134  2693 solver.cpp:243] Iteration 53600, loss = 4.08953
I0330 15:07:23.382328  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.91923 (* 1 = 5.91923 loss)
I0330 15:07:23.382344  2693 sgd_solver.cpp:138] Iteration 53600, lr = 0.0005
I0330 15:09:33.725836  2693 solver.cpp:243] Iteration 53700, loss = 4.03236
I0330 15:09:33.726039  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.953 (* 1 = 3.953 loss)
I0330 15:09:33.726058  2693 sgd_solver.cpp:138] Iteration 53700, lr = 0.0005
I0330 15:11:46.873246  2693 solver.cpp:243] Iteration 53800, loss = 4.11093
I0330 15:11:46.873484  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.78284 (* 1 = 2.78284 loss)
I0330 15:11:46.873517  2693 sgd_solver.cpp:138] Iteration 53800, lr = 0.0005
I0330 15:13:57.909910  2693 solver.cpp:243] Iteration 53900, loss = 4.06784
I0330 15:13:57.910156  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.12095 (* 1 = 3.12095 loss)
I0330 15:13:57.910190  2693 sgd_solver.cpp:138] Iteration 53900, lr = 0.0005
I0330 15:16:10.621953  2693 solver.cpp:243] Iteration 54000, loss = 4.17659
I0330 15:16:10.622311  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.15884 (* 1 = 3.15884 loss)
I0330 15:16:10.622342  2693 sgd_solver.cpp:138] Iteration 54000, lr = 0.0005
I0330 15:18:21.728389  2693 solver.cpp:243] Iteration 54100, loss = 4.14934
I0330 15:18:21.728577  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.26158 (* 1 = 4.26158 loss)
I0330 15:18:21.728595  2693 sgd_solver.cpp:138] Iteration 54100, lr = 0.0005
I0330 15:20:32.385579  2693 solver.cpp:243] Iteration 54200, loss = 4.07245
I0330 15:20:32.385829  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.56283 (* 1 = 3.56283 loss)
I0330 15:20:32.385857  2693 sgd_solver.cpp:138] Iteration 54200, lr = 0.0005
I0330 15:22:43.293758  2693 solver.cpp:243] Iteration 54300, loss = 4.21369
I0330 15:22:43.294003  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.10499 (* 1 = 5.10499 loss)
I0330 15:22:43.294024  2693 sgd_solver.cpp:138] Iteration 54300, lr = 0.0005
I0330 15:24:54.128789  2693 solver.cpp:243] Iteration 54400, loss = 4.18769
I0330 15:24:54.128976  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.44768 (* 1 = 3.44768 loss)
I0330 15:24:54.128991  2693 sgd_solver.cpp:138] Iteration 54400, lr = 0.0005
I0330 15:27:04.443852  2693 solver.cpp:243] Iteration 54500, loss = 4.0371
I0330 15:27:04.444135  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.06891 (* 1 = 4.06891 loss)
I0330 15:27:04.444159  2693 sgd_solver.cpp:138] Iteration 54500, lr = 0.0005
I0330 15:29:14.488791  2693 solver.cpp:243] Iteration 54600, loss = 4.1005
I0330 15:29:14.489063  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.83434 (* 1 = 3.83434 loss)
I0330 15:29:14.489100  2693 sgd_solver.cpp:138] Iteration 54600, lr = 0.0005
I0330 15:31:24.909065  2693 solver.cpp:243] Iteration 54700, loss = 4.10196
I0330 15:31:24.909432  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.55629 (* 1 = 4.55629 loss)
I0330 15:31:24.909479  2693 sgd_solver.cpp:138] Iteration 54700, lr = 0.0005
I0330 15:33:35.613937  2693 solver.cpp:243] Iteration 54800, loss = 4.11176
I0330 15:33:35.614153  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.06554 (* 1 = 5.06554 loss)
I0330 15:33:35.614171  2693 sgd_solver.cpp:138] Iteration 54800, lr = 0.0005
I0330 15:35:45.528491  2693 solver.cpp:243] Iteration 54900, loss = 3.94541
I0330 15:35:45.528741  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.87982 (* 1 = 4.87982 loss)
I0330 15:35:45.528777  2693 sgd_solver.cpp:138] Iteration 54900, lr = 0.0005
I0330 15:37:56.921099  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_55000.caffemodel
I0330 15:37:57.877220  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_55000.solverstate
I0330 15:37:57.991516  2693 solver.cpp:433] Iteration 55000, Testing net (#0)
I0330 15:37:57.991614  2693 net.cpp:693] Ignoring source layer mbox_loss
I0330 15:39:18.442076  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.570982
I0330 15:39:19.119005  2693 solver.cpp:243] Iteration 55000, loss = 4.23067
I0330 15:39:19.119107  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.96479 (* 1 = 3.96479 loss)
I0330 15:39:19.119143  2693 sgd_solver.cpp:138] Iteration 55000, lr = 0.0005
I0330 15:41:30.283013  2693 solver.cpp:243] Iteration 55100, loss = 4.13713
I0330 15:41:30.283221  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.49751 (* 1 = 4.49751 loss)
I0330 15:41:30.283239  2693 sgd_solver.cpp:138] Iteration 55100, lr = 0.0005
I0330 15:43:41.407652  2693 solver.cpp:243] Iteration 55200, loss = 4.27716
I0330 15:43:41.407888  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.25652 (* 1 = 3.25652 loss)
I0330 15:43:41.407907  2693 sgd_solver.cpp:138] Iteration 55200, lr = 0.0005
I0330 15:45:53.391072  2693 solver.cpp:243] Iteration 55300, loss = 4.24855
I0330 15:45:53.391297  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.84485 (* 1 = 2.84485 loss)
I0330 15:45:53.391327  2693 sgd_solver.cpp:138] Iteration 55300, lr = 0.0005
I0330 15:48:02.507851  2693 solver.cpp:243] Iteration 55400, loss = 4.12334
I0330 15:48:02.514546  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.70996 (* 1 = 3.70996 loss)
I0330 15:48:02.514603  2693 sgd_solver.cpp:138] Iteration 55400, lr = 0.0005
I0330 15:50:14.237371  2693 solver.cpp:243] Iteration 55500, loss = 4.27006
I0330 15:50:14.237596  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.95325 (* 1 = 3.95325 loss)
I0330 15:50:14.237615  2693 sgd_solver.cpp:138] Iteration 55500, lr = 0.0005
I0330 15:52:25.241051  2693 solver.cpp:243] Iteration 55600, loss = 4.09962
I0330 15:52:25.241364  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.72323 (* 1 = 4.72323 loss)
I0330 15:52:25.241387  2693 sgd_solver.cpp:138] Iteration 55600, lr = 0.0005
I0330 15:54:36.991102  2693 solver.cpp:243] Iteration 55700, loss = 4.19028
I0330 15:54:36.991345  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.25511 (* 1 = 5.25511 loss)
I0330 15:54:36.991371  2693 sgd_solver.cpp:138] Iteration 55700, lr = 0.0005
I0330 15:56:48.164427  2693 solver.cpp:243] Iteration 55800, loss = 4.2323
I0330 15:56:48.164659  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.93589 (* 1 = 4.93589 loss)
I0330 15:56:48.164685  2693 sgd_solver.cpp:138] Iteration 55800, lr = 0.0005
I0330 15:59:00.706233  2693 solver.cpp:243] Iteration 55900, loss = 4.11979
I0330 15:59:00.706485  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.90633 (* 1 = 3.90633 loss)
I0330 15:59:00.706506  2693 sgd_solver.cpp:138] Iteration 55900, lr = 0.0005
I0330 16:01:10.260677  2693 solver.cpp:243] Iteration 56000, loss = 4.05308
I0330 16:01:10.260856  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.97061 (* 1 = 3.97061 loss)
I0330 16:01:10.260872  2693 sgd_solver.cpp:138] Iteration 56000, lr = 0.0005
I0330 16:03:21.888813  2693 solver.cpp:243] Iteration 56100, loss = 4.15086
I0330 16:03:21.889077  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.21135 (* 1 = 4.21135 loss)
I0330 16:03:21.889113  2693 sgd_solver.cpp:138] Iteration 56100, lr = 0.0005
I0330 16:05:33.101910  2693 solver.cpp:243] Iteration 56200, loss = 4.04056
I0330 16:05:33.114364  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.10498 (* 1 = 3.10498 loss)
I0330 16:05:33.114382  2693 sgd_solver.cpp:138] Iteration 56200, lr = 0.0005
I0330 16:07:43.667172  2693 solver.cpp:243] Iteration 56300, loss = 4.02401
I0330 16:07:43.672015  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.89508 (* 1 = 2.89508 loss)
I0330 16:07:43.672046  2693 sgd_solver.cpp:138] Iteration 56300, lr = 0.0005
I0330 16:09:52.646255  2693 solver.cpp:243] Iteration 56400, loss = 4.07911
I0330 16:09:52.647316  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.31454 (* 1 = 3.31454 loss)
I0330 16:09:52.647342  2693 sgd_solver.cpp:138] Iteration 56400, lr = 0.0005
I0330 16:12:05.570267  2693 solver.cpp:243] Iteration 56500, loss = 4.29057
I0330 16:12:05.570469  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.48231 (* 1 = 5.48231 loss)
I0330 16:12:05.570487  2693 sgd_solver.cpp:138] Iteration 56500, lr = 0.0005
I0330 16:14:17.390693  2693 solver.cpp:243] Iteration 56600, loss = 4.15082
I0330 16:14:17.390888  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.70397 (* 1 = 5.70397 loss)
I0330 16:14:17.390905  2693 sgd_solver.cpp:138] Iteration 56600, lr = 0.0005
I0330 16:16:29.921372  2693 solver.cpp:243] Iteration 56700, loss = 4.19398
I0330 16:16:29.921594  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.72739 (* 1 = 3.72739 loss)
I0330 16:16:29.921613  2693 sgd_solver.cpp:138] Iteration 56700, lr = 0.0005
I0330 16:18:39.645407  2693 solver.cpp:243] Iteration 56800, loss = 4.03229
I0330 16:18:39.645665  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.67816 (* 1 = 4.67816 loss)
I0330 16:18:39.645700  2693 sgd_solver.cpp:138] Iteration 56800, lr = 0.0005
I0330 16:20:49.837474  2693 solver.cpp:243] Iteration 56900, loss = 4.0746
I0330 16:20:49.837793  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.1776 (* 1 = 3.1776 loss)
I0330 16:20:49.837828  2693 sgd_solver.cpp:138] Iteration 56900, lr = 0.0005
I0330 16:23:01.412151  2693 solver.cpp:243] Iteration 57000, loss = 4.09107
I0330 16:23:01.412433  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.19665 (* 1 = 3.19665 loss)
I0330 16:23:01.412466  2693 sgd_solver.cpp:138] Iteration 57000, lr = 0.0005
I0330 16:25:12.019994  2693 solver.cpp:243] Iteration 57100, loss = 4.07327
I0330 16:25:12.020244  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.26163 (* 1 = 3.26163 loss)
I0330 16:25:12.020264  2693 sgd_solver.cpp:138] Iteration 57100, lr = 0.0005
I0330 16:27:23.946449  2693 solver.cpp:243] Iteration 57200, loss = 4.14205
I0330 16:27:23.946722  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.01332 (* 1 = 5.01332 loss)
I0330 16:27:23.946745  2693 sgd_solver.cpp:138] Iteration 57200, lr = 0.0005
I0330 16:29:36.967277  2693 solver.cpp:243] Iteration 57300, loss = 4.14904
I0330 16:29:36.967452  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.10767 (* 1 = 4.10767 loss)
I0330 16:29:36.967469  2693 sgd_solver.cpp:138] Iteration 57300, lr = 0.0005
I0330 16:31:48.409950  2693 solver.cpp:243] Iteration 57400, loss = 4.04915
I0330 16:31:48.410204  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.9179 (* 1 = 3.9179 loss)
I0330 16:31:48.410229  2693 sgd_solver.cpp:138] Iteration 57400, lr = 0.0005
I0330 16:34:00.848219  2693 solver.cpp:243] Iteration 57500, loss = 4.08095
I0330 16:34:00.851809  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.88622 (* 1 = 4.88622 loss)
I0330 16:34:00.851831  2693 sgd_solver.cpp:138] Iteration 57500, lr = 0.0005
I0330 16:36:13.757230  2693 solver.cpp:243] Iteration 57600, loss = 4.25946
I0330 16:36:13.757463  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.42115 (* 1 = 4.42115 loss)
I0330 16:36:13.757498  2693 sgd_solver.cpp:138] Iteration 57600, lr = 0.0005
I0330 16:38:27.014338  2693 solver.cpp:243] Iteration 57700, loss = 4.26975
I0330 16:38:27.014607  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.72327 (* 1 = 3.72327 loss)
I0330 16:38:27.014631  2693 sgd_solver.cpp:138] Iteration 57700, lr = 0.0005
I0330 16:40:37.907008  2693 solver.cpp:243] Iteration 57800, loss = 4.26753
I0330 16:40:37.914082  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.55214 (* 1 = 3.55214 loss)
I0330 16:40:37.914180  2693 sgd_solver.cpp:138] Iteration 57800, lr = 0.0005
I0330 16:42:48.421954  2693 solver.cpp:243] Iteration 57900, loss = 4.26509
I0330 16:42:48.422199  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.95249 (* 1 = 3.95249 loss)
I0330 16:42:48.422227  2693 sgd_solver.cpp:138] Iteration 57900, lr = 0.0005
I0330 16:44:58.976356  2693 solver.cpp:243] Iteration 58000, loss = 4.00959
I0330 16:44:58.976541  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.86602 (* 1 = 3.86602 loss)
I0330 16:44:58.976557  2693 sgd_solver.cpp:138] Iteration 58000, lr = 0.0005
I0330 16:47:10.607257  2693 solver.cpp:243] Iteration 58100, loss = 4.32835
I0330 16:47:10.607517  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.80016 (* 1 = 4.80016 loss)
I0330 16:47:10.607543  2693 sgd_solver.cpp:138] Iteration 58100, lr = 0.0005
I0330 16:49:20.098212  2693 solver.cpp:243] Iteration 58200, loss = 4.16016
I0330 16:49:20.117992  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.10518 (* 1 = 5.10518 loss)
I0330 16:49:20.118028  2693 sgd_solver.cpp:138] Iteration 58200, lr = 0.0005
I0330 16:51:31.910641  2693 solver.cpp:243] Iteration 58300, loss = 4.12179
I0330 16:51:31.910876  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.56774 (* 1 = 4.56774 loss)
I0330 16:51:31.910899  2693 sgd_solver.cpp:138] Iteration 58300, lr = 0.0005
I0330 16:53:43.188859  2693 solver.cpp:243] Iteration 58400, loss = 3.99193
I0330 16:53:43.189113  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.00102 (* 1 = 5.00102 loss)
I0330 16:53:43.189136  2693 sgd_solver.cpp:138] Iteration 58400, lr = 0.0005
I0330 16:55:53.183254  2693 solver.cpp:243] Iteration 58500, loss = 4.05909
I0330 16:55:53.183533  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.28905 (* 1 = 3.28905 loss)
I0330 16:55:53.183565  2693 sgd_solver.cpp:138] Iteration 58500, lr = 0.0005
I0330 16:58:03.397745  2693 solver.cpp:243] Iteration 58600, loss = 4.18037
I0330 16:58:03.404160  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.61697 (* 1 = 2.61697 loss)
I0330 16:58:03.404186  2693 sgd_solver.cpp:138] Iteration 58600, lr = 0.0005
I0330 17:00:13.679231  2693 solver.cpp:243] Iteration 58700, loss = 4.10003
I0330 17:00:13.679509  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.26948 (* 1 = 4.26948 loss)
I0330 17:00:13.679534  2693 sgd_solver.cpp:138] Iteration 58700, lr = 0.0005
I0330 17:02:24.339313  2693 solver.cpp:243] Iteration 58800, loss = 4.07709
I0330 17:02:24.339567  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.37477 (* 1 = 2.37477 loss)
I0330 17:02:24.339624  2693 sgd_solver.cpp:138] Iteration 58800, lr = 0.0005
I0330 17:04:35.181352  2693 solver.cpp:243] Iteration 58900, loss = 4.06397
I0330 17:04:35.181614  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.74364 (* 1 = 4.74364 loss)
I0330 17:04:35.181659  2693 sgd_solver.cpp:138] Iteration 58900, lr = 0.0005
I0330 17:06:47.092334  2693 solver.cpp:243] Iteration 59000, loss = 4.1196
I0330 17:06:47.092566  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.77425 (* 1 = 3.77425 loss)
I0330 17:06:47.092592  2693 sgd_solver.cpp:138] Iteration 59000, lr = 0.0005
I0330 17:08:59.274384  2693 solver.cpp:243] Iteration 59100, loss = 4.13829
I0330 17:08:59.277968  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.76266 (* 1 = 4.76266 loss)
I0330 17:08:59.278056  2693 sgd_solver.cpp:138] Iteration 59100, lr = 0.0005
I0330 17:11:10.554206  2693 solver.cpp:243] Iteration 59200, loss = 4.14179
I0330 17:11:10.554452  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.11605 (* 1 = 2.11605 loss)
I0330 17:11:10.554481  2693 sgd_solver.cpp:138] Iteration 59200, lr = 0.0005
I0330 17:13:22.887574  2693 solver.cpp:243] Iteration 59300, loss = 4.17389
I0330 17:13:22.887883  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.36751 (* 1 = 3.36751 loss)
I0330 17:13:22.887908  2693 sgd_solver.cpp:138] Iteration 59300, lr = 0.0005
I0330 17:15:34.948317  2693 solver.cpp:243] Iteration 59400, loss = 4.01439
I0330 17:15:34.948509  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.89667 (* 1 = 3.89667 loss)
I0330 17:15:34.948528  2693 sgd_solver.cpp:138] Iteration 59400, lr = 0.0005
I0330 17:17:47.378654  2693 solver.cpp:243] Iteration 59500, loss = 4.25286
I0330 17:17:47.378907  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.20886 (* 1 = 4.20886 loss)
I0330 17:17:47.378934  2693 sgd_solver.cpp:138] Iteration 59500, lr = 0.0005
I0330 17:20:01.279247  2693 solver.cpp:243] Iteration 59600, loss = 4.10296
I0330 17:20:01.279495  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.71521 (* 1 = 3.71521 loss)
I0330 17:20:01.279517  2693 sgd_solver.cpp:138] Iteration 59600, lr = 0.0005
I0330 17:22:12.823905  2693 solver.cpp:243] Iteration 59700, loss = 4.18381
I0330 17:22:12.855667  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.9636 (* 1 = 3.9636 loss)
I0330 17:22:12.855710  2693 sgd_solver.cpp:138] Iteration 59700, lr = 0.0005
I0330 17:24:24.679522  2693 solver.cpp:243] Iteration 59800, loss = 4.05335
I0330 17:24:24.679805  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.37028 (* 1 = 4.37028 loss)
I0330 17:24:24.679823  2693 sgd_solver.cpp:138] Iteration 59800, lr = 0.0005
I0330 17:26:35.972082  2693 solver.cpp:243] Iteration 59900, loss = 4.24407
I0330 17:26:35.972332  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.00803 (* 1 = 4.00803 loss)
I0330 17:26:35.972378  2693 sgd_solver.cpp:138] Iteration 59900, lr = 0.0005
I0330 17:28:45.762526  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_60000.caffemodel
I0330 17:28:47.023636  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_60000.solverstate
I0330 17:28:47.149494  2693 solver.cpp:433] Iteration 60000, Testing net (#0)
I0330 17:28:47.149576  2693 net.cpp:693] Ignoring source layer mbox_loss
I0330 17:30:07.361133  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.544912
I0330 17:30:08.007112  2693 solver.cpp:243] Iteration 60000, loss = 4.09544
I0330 17:30:08.007186  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.82717 (* 1 = 4.82717 loss)
I0330 17:30:08.007202  2693 sgd_solver.cpp:138] Iteration 60000, lr = 0.0005
I0330 17:32:17.511265  2693 solver.cpp:243] Iteration 60100, loss = 4.12497
I0330 17:32:17.511548  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.63567 (* 1 = 3.63567 loss)
I0330 17:32:17.511569  2693 sgd_solver.cpp:138] Iteration 60100, lr = 0.0005
I0330 17:34:30.291012  2693 solver.cpp:243] Iteration 60200, loss = 4.22707
I0330 17:34:30.306825  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.0193 (* 1 = 4.0193 loss)
I0330 17:34:30.306860  2693 sgd_solver.cpp:138] Iteration 60200, lr = 0.0005
I0330 17:36:40.732184  2693 solver.cpp:243] Iteration 60300, loss = 4.0508
I0330 17:36:40.732450  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.05783 (* 1 = 6.05783 loss)
I0330 17:36:40.732484  2693 sgd_solver.cpp:138] Iteration 60300, lr = 0.0005
I0330 17:38:51.588032  2693 solver.cpp:243] Iteration 60400, loss = 3.97308
I0330 17:38:51.588282  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.12035 (* 1 = 4.12035 loss)
I0330 17:38:51.588315  2693 sgd_solver.cpp:138] Iteration 60400, lr = 0.0005
I0330 17:41:02.203817  2693 solver.cpp:243] Iteration 60500, loss = 4.15409
I0330 17:41:02.204067  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.48907 (* 1 = 4.48907 loss)
I0330 17:41:02.204088  2693 sgd_solver.cpp:138] Iteration 60500, lr = 0.0005
I0330 17:43:12.854980  2693 solver.cpp:243] Iteration 60600, loss = 4.14022
I0330 17:43:12.855227  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.79223 (* 1 = 4.79223 loss)
I0330 17:43:12.855252  2693 sgd_solver.cpp:138] Iteration 60600, lr = 0.0005
I0330 17:45:26.330689  2693 solver.cpp:243] Iteration 60700, loss = 4.18386
I0330 17:45:26.330929  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.59718 (* 1 = 3.59718 loss)
I0330 17:45:26.330958  2693 sgd_solver.cpp:138] Iteration 60700, lr = 0.0005
I0330 17:47:36.525259  2693 solver.cpp:243] Iteration 60800, loss = 4.20393
I0330 17:47:36.525496  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.25924 (* 1 = 4.25924 loss)
I0330 17:47:36.525516  2693 sgd_solver.cpp:138] Iteration 60800, lr = 0.0005
I0330 17:49:49.176343  2693 solver.cpp:243] Iteration 60900, loss = 4.19105
I0330 17:49:49.176589  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.0235 (* 1 = 4.0235 loss)
I0330 17:49:49.176626  2693 sgd_solver.cpp:138] Iteration 60900, lr = 0.0005
I0330 17:51:59.912883  2693 solver.cpp:243] Iteration 61000, loss = 4.03406
I0330 17:51:59.913127  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.95659 (* 1 = 3.95659 loss)
I0330 17:51:59.913161  2693 sgd_solver.cpp:138] Iteration 61000, lr = 0.0005
I0330 17:54:09.558712  2693 solver.cpp:243] Iteration 61100, loss = 4.06801
I0330 17:54:09.558961  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.9386 (* 1 = 3.9386 loss)
I0330 17:54:09.558985  2693 sgd_solver.cpp:138] Iteration 61100, lr = 0.0005
I0330 17:56:20.965262  2693 solver.cpp:243] Iteration 61200, loss = 4.10267
I0330 17:56:20.965505  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.78115 (* 1 = 4.78115 loss)
I0330 17:56:20.965530  2693 sgd_solver.cpp:138] Iteration 61200, lr = 0.0005
I0330 17:58:31.578660  2693 solver.cpp:243] Iteration 61300, loss = 4.16129
I0330 17:58:31.578872  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.37653 (* 1 = 5.37653 loss)
I0330 17:58:31.578897  2693 sgd_solver.cpp:138] Iteration 61300, lr = 0.0005
I0330 18:00:42.733491  2693 solver.cpp:243] Iteration 61400, loss = 4.18056
I0330 18:00:42.733705  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.68175 (* 1 = 4.68175 loss)
I0330 18:00:42.733747  2693 sgd_solver.cpp:138] Iteration 61400, lr = 0.0005
I0330 18:02:55.309242  2693 solver.cpp:243] Iteration 61500, loss = 4.15283
I0330 18:02:55.309448  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.53017 (* 1 = 4.53017 loss)
I0330 18:02:55.309468  2693 sgd_solver.cpp:138] Iteration 61500, lr = 0.0005
I0330 18:05:06.327405  2693 solver.cpp:243] Iteration 61600, loss = 4.05019
I0330 18:05:06.344209  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.32761 (* 1 = 4.32761 loss)
I0330 18:05:06.344378  2693 sgd_solver.cpp:138] Iteration 61600, lr = 0.0005
I0330 18:07:17.538697  2693 solver.cpp:243] Iteration 61700, loss = 4.11375
I0330 18:07:17.538897  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.97937 (* 1 = 2.97937 loss)
I0330 18:07:17.538914  2693 sgd_solver.cpp:138] Iteration 61700, lr = 0.0005
I0330 18:09:26.235791  2693 solver.cpp:243] Iteration 61800, loss = 3.99597
I0330 18:09:26.236017  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.43493 (* 1 = 4.43493 loss)
I0330 18:09:26.236034  2693 sgd_solver.cpp:138] Iteration 61800, lr = 0.0005
I0330 18:11:37.487658  2693 solver.cpp:243] Iteration 61900, loss = 3.9981
I0330 18:11:37.487897  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.78406 (* 1 = 4.78406 loss)
I0330 18:11:37.487929  2693 sgd_solver.cpp:138] Iteration 61900, lr = 0.0005
I0330 18:13:47.380769  2693 solver.cpp:243] Iteration 62000, loss = 4.10819
I0330 18:13:47.381000  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.31614 (* 1 = 3.31614 loss)
I0330 18:13:47.381021  2693 sgd_solver.cpp:138] Iteration 62000, lr = 0.0005
I0330 18:15:57.872040  2693 solver.cpp:243] Iteration 62100, loss = 4.18768
I0330 18:15:57.872242  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.15749 (* 1 = 3.15749 loss)
I0330 18:15:57.872259  2693 sgd_solver.cpp:138] Iteration 62100, lr = 0.0005
I0330 18:18:10.421538  2693 solver.cpp:243] Iteration 62200, loss = 4.24339
I0330 18:18:10.421775  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.69664 (* 1 = 4.69664 loss)
I0330 18:18:10.421797  2693 sgd_solver.cpp:138] Iteration 62200, lr = 0.0005
I0330 18:20:21.214781  2693 solver.cpp:243] Iteration 62300, loss = 4.07538
I0330 18:20:21.215065  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.95583 (* 1 = 3.95583 loss)
I0330 18:20:21.215104  2693 sgd_solver.cpp:138] Iteration 62300, lr = 0.0005
I0330 18:22:32.534709  2693 solver.cpp:243] Iteration 62400, loss = 4.0282
I0330 18:22:32.534906  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.97797 (* 1 = 3.97797 loss)
I0330 18:22:32.534924  2693 sgd_solver.cpp:138] Iteration 62400, lr = 0.0005
I0330 18:24:41.768913  2693 solver.cpp:243] Iteration 62500, loss = 4.04694
I0330 18:24:41.769155  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.32746 (* 1 = 3.32746 loss)
I0330 18:24:41.769181  2693 sgd_solver.cpp:138] Iteration 62500, lr = 0.0005
I0330 18:26:51.913851  2693 solver.cpp:243] Iteration 62600, loss = 4.04693
I0330 18:26:51.914106  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.85291 (* 1 = 3.85291 loss)
I0330 18:26:51.914139  2693 sgd_solver.cpp:138] Iteration 62600, lr = 0.0005
I0330 18:29:03.183234  2693 solver.cpp:243] Iteration 62700, loss = 4.08523
I0330 18:29:03.183426  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.55506 (* 1 = 4.55506 loss)
I0330 18:29:03.183444  2693 sgd_solver.cpp:138] Iteration 62700, lr = 0.0005
I0330 18:31:14.215519  2693 solver.cpp:243] Iteration 62800, loss = 4.0627
I0330 18:31:14.215852  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.1272 (* 1 = 4.1272 loss)
I0330 18:31:14.215880  2693 sgd_solver.cpp:138] Iteration 62800, lr = 0.0005
I0330 18:33:24.117319  2693 solver.cpp:243] Iteration 62900, loss = 4.18546
I0330 18:33:24.124351  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.3123 (* 1 = 4.3123 loss)
I0330 18:33:24.124370  2693 sgd_solver.cpp:138] Iteration 62900, lr = 0.0005
I0330 18:35:36.245110  2693 solver.cpp:243] Iteration 63000, loss = 4.11783
I0330 18:35:36.245367  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.08476 (* 1 = 4.08476 loss)
I0330 18:35:36.245406  2693 sgd_solver.cpp:138] Iteration 63000, lr = 0.0005
I0330 18:37:47.044839  2693 solver.cpp:243] Iteration 63100, loss = 4.11849
I0330 18:37:47.045073  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.21284 (* 1 = 4.21284 loss)
I0330 18:37:47.045092  2693 sgd_solver.cpp:138] Iteration 63100, lr = 0.0005
I0330 18:39:57.737200  2693 solver.cpp:243] Iteration 63200, loss = 4.18051
I0330 18:39:57.737452  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.41862 (* 1 = 3.41862 loss)
I0330 18:39:57.737470  2693 sgd_solver.cpp:138] Iteration 63200, lr = 0.0005
I0330 18:42:09.635416  2693 solver.cpp:243] Iteration 63300, loss = 4.24189
I0330 18:42:09.656230  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.9519 (* 1 = 3.9519 loss)
I0330 18:42:09.656266  2693 sgd_solver.cpp:138] Iteration 63300, lr = 0.0005
I0330 18:44:21.165747  2693 solver.cpp:243] Iteration 63400, loss = 4.28975
I0330 18:44:21.165956  2693 solver.cpp:259]     Train net output #0: mbox_loss = 7.10182 (* 1 = 7.10182 loss)
I0330 18:44:21.165973  2693 sgd_solver.cpp:138] Iteration 63400, lr = 0.0005
I0330 18:46:31.622992  2693 solver.cpp:243] Iteration 63500, loss = 4.12627
I0330 18:46:31.623251  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.54413 (* 1 = 2.54413 loss)
I0330 18:46:31.623284  2693 sgd_solver.cpp:138] Iteration 63500, lr = 0.0005
I0330 18:48:42.566138  2693 solver.cpp:243] Iteration 63600, loss = 4.19139
I0330 18:48:42.566352  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.1012 (* 1 = 5.1012 loss)
I0330 18:48:42.566383  2693 sgd_solver.cpp:138] Iteration 63600, lr = 0.0005
I0330 18:50:53.536931  2693 solver.cpp:243] Iteration 63700, loss = 4.06436
I0330 18:50:53.537220  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.08743 (* 1 = 4.08743 loss)
I0330 18:50:53.537256  2693 sgd_solver.cpp:138] Iteration 63700, lr = 0.0005
I0330 18:53:02.466269  2693 solver.cpp:243] Iteration 63800, loss = 4.16407
I0330 18:53:02.466531  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.71278 (* 1 = 3.71278 loss)
I0330 18:53:02.466562  2693 sgd_solver.cpp:138] Iteration 63800, lr = 0.0005
I0330 18:55:12.428163  2693 solver.cpp:243] Iteration 63900, loss = 4.00527
I0330 18:55:12.428357  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.7093 (* 1 = 4.7093 loss)
I0330 18:55:12.428375  2693 sgd_solver.cpp:138] Iteration 63900, lr = 0.0005
I0330 18:57:24.257596  2693 solver.cpp:243] Iteration 64000, loss = 4.19524
I0330 18:57:24.257813  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.45111 (* 1 = 2.45111 loss)
I0330 18:57:24.257830  2693 sgd_solver.cpp:138] Iteration 64000, lr = 0.0005
I0330 18:59:35.616780  2693 solver.cpp:243] Iteration 64100, loss = 4.11427
I0330 18:59:35.617035  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.4259 (* 1 = 4.4259 loss)
I0330 18:59:35.617077  2693 sgd_solver.cpp:138] Iteration 64100, lr = 0.0005
I0330 19:01:44.410181  2693 solver.cpp:243] Iteration 64200, loss = 4.05682
I0330 19:01:44.417410  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.85586 (* 1 = 3.85586 loss)
I0330 19:01:44.417449  2693 sgd_solver.cpp:138] Iteration 64200, lr = 0.0005
I0330 19:03:55.495779  2693 solver.cpp:243] Iteration 64300, loss = 4.07375
I0330 19:03:55.495918  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.01173 (* 1 = 4.01173 loss)
I0330 19:03:55.495935  2693 sgd_solver.cpp:138] Iteration 64300, lr = 0.0005
I0330 19:06:07.248287  2693 solver.cpp:243] Iteration 64400, loss = 4.04172
I0330 19:06:07.248473  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.5959 (* 1 = 3.5959 loss)
I0330 19:06:07.248492  2693 sgd_solver.cpp:138] Iteration 64400, lr = 0.0005
I0330 19:08:17.505239  2693 solver.cpp:243] Iteration 64500, loss = 4.04304
I0330 19:08:17.513072  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.71673 (* 1 = 2.71673 loss)
I0330 19:08:17.513113  2693 sgd_solver.cpp:138] Iteration 64500, lr = 0.0005
I0330 19:10:27.928205  2693 solver.cpp:243] Iteration 64600, loss = 3.96766
I0330 19:10:27.928432  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.477 (* 1 = 4.477 loss)
I0330 19:10:27.928473  2693 sgd_solver.cpp:138] Iteration 64600, lr = 0.0005
I0330 19:12:38.327795  2693 solver.cpp:243] Iteration 64700, loss = 4.17437
I0330 19:12:38.328104  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.8802 (* 1 = 3.8802 loss)
I0330 19:12:38.328148  2693 sgd_solver.cpp:138] Iteration 64700, lr = 0.0005
I0330 19:14:50.951782  2693 solver.cpp:243] Iteration 64800, loss = 4.15944
I0330 19:14:50.951963  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.36058 (* 1 = 4.36058 loss)
I0330 19:14:50.951979  2693 sgd_solver.cpp:138] Iteration 64800, lr = 0.0005
I0330 19:17:00.830824  2693 solver.cpp:243] Iteration 64900, loss = 3.98305
I0330 19:17:00.831076  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.89677 (* 1 = 3.89677 loss)
I0330 19:17:00.831102  2693 sgd_solver.cpp:138] Iteration 64900, lr = 0.0005
I0330 19:19:10.370829  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_65000.caffemodel
I0330 19:19:11.598500  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_65000.solverstate
I0330 19:19:11.717084  2693 solver.cpp:433] Iteration 65000, Testing net (#0)
I0330 19:19:11.717175  2693 net.cpp:693] Ignoring source layer mbox_loss
I0330 19:20:31.929523  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.58637
I0330 19:20:32.665765  2693 solver.cpp:243] Iteration 65000, loss = 4.02289
I0330 19:20:32.665838  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.017 (* 1 = 4.017 loss)
I0330 19:20:32.665853  2693 sgd_solver.cpp:138] Iteration 65000, lr = 0.0005
I0330 19:22:43.253640  2693 solver.cpp:243] Iteration 65100, loss = 4.0824
I0330 19:22:43.253873  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.89135 (* 1 = 3.89135 loss)
I0330 19:22:43.253891  2693 sgd_solver.cpp:138] Iteration 65100, lr = 0.0005
I0330 19:24:54.193877  2693 solver.cpp:243] Iteration 65200, loss = 4.17267
I0330 19:24:54.194123  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.2695 (* 1 = 4.2695 loss)
I0330 19:24:54.194160  2693 sgd_solver.cpp:138] Iteration 65200, lr = 0.0005
I0330 19:27:06.778574  2693 solver.cpp:243] Iteration 65300, loss = 4.09152
I0330 19:27:06.778806  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.79795 (* 1 = 2.79795 loss)
I0330 19:27:06.778841  2693 sgd_solver.cpp:138] Iteration 65300, lr = 0.0005
I0330 19:29:17.387194  2693 solver.cpp:243] Iteration 65400, loss = 4.10901
I0330 19:29:17.387439  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.03754 (* 1 = 5.03754 loss)
I0330 19:29:17.387460  2693 sgd_solver.cpp:138] Iteration 65400, lr = 0.0005
I0330 19:31:29.817334  2693 solver.cpp:243] Iteration 65500, loss = 4.05065
I0330 19:31:29.817594  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.83849 (* 1 = 2.83849 loss)
I0330 19:31:29.817632  2693 sgd_solver.cpp:138] Iteration 65500, lr = 0.0005
I0330 19:33:41.629096  2693 solver.cpp:243] Iteration 65600, loss = 4.15589
I0330 19:33:41.629287  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.87389 (* 1 = 3.87389 loss)
I0330 19:33:41.629305  2693 sgd_solver.cpp:138] Iteration 65600, lr = 0.0005
I0330 19:35:51.058645  2693 solver.cpp:243] Iteration 65700, loss = 4.1649
I0330 19:35:51.059751  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.66569 (* 1 = 3.66569 loss)
I0330 19:35:51.059797  2693 sgd_solver.cpp:138] Iteration 65700, lr = 0.0005
I0330 19:38:02.510534  2693 solver.cpp:243] Iteration 65800, loss = 4.05032
I0330 19:38:02.510795  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.93113 (* 1 = 3.93113 loss)
I0330 19:38:02.510833  2693 sgd_solver.cpp:138] Iteration 65800, lr = 0.0005
I0330 19:40:13.984441  2693 solver.cpp:243] Iteration 65900, loss = 3.95503
I0330 19:40:13.984702  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.28127 (* 1 = 4.28127 loss)
I0330 19:40:13.984726  2693 sgd_solver.cpp:138] Iteration 65900, lr = 0.0005
I0330 19:42:25.672327  2693 solver.cpp:243] Iteration 66000, loss = 3.95429
I0330 19:42:25.672514  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.10486 (* 1 = 5.10486 loss)
I0330 19:42:25.672530  2693 sgd_solver.cpp:138] Iteration 66000, lr = 0.0005
I0330 19:44:37.397975  2693 solver.cpp:243] Iteration 66100, loss = 4.04572
I0330 19:44:37.398174  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.01005 (* 1 = 5.01005 loss)
I0330 19:44:37.398192  2693 sgd_solver.cpp:138] Iteration 66100, lr = 0.0005
I0330 19:46:46.179975  2693 solver.cpp:243] Iteration 66200, loss = 3.81021
I0330 19:46:46.180224  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.00665 (* 1 = 3.00665 loss)
I0330 19:46:46.180259  2693 sgd_solver.cpp:138] Iteration 66200, lr = 0.0005
I0330 19:48:58.554127  2693 solver.cpp:243] Iteration 66300, loss = 4.12353
I0330 19:48:58.554404  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.11062 (* 1 = 4.11062 loss)
I0330 19:48:58.554437  2693 sgd_solver.cpp:138] Iteration 66300, lr = 0.0005
I0330 19:51:09.054172  2693 solver.cpp:243] Iteration 66400, loss = 4.03791
I0330 19:51:09.077399  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.63354 (* 1 = 3.63354 loss)
I0330 19:51:09.077436  2693 sgd_solver.cpp:138] Iteration 66400, lr = 0.0005
I0330 19:53:20.126296  2693 solver.cpp:243] Iteration 66500, loss = 4.29358
I0330 19:53:20.126520  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.7171 (* 1 = 4.7171 loss)
I0330 19:53:20.126544  2693 sgd_solver.cpp:138] Iteration 66500, lr = 0.0005
I0330 19:55:31.078550  2693 solver.cpp:243] Iteration 66600, loss = 4.10483
I0330 19:55:31.078796  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.50132 (* 1 = 4.50132 loss)
I0330 19:55:31.078833  2693 sgd_solver.cpp:138] Iteration 66600, lr = 0.0005
I0330 19:57:41.314548  2693 solver.cpp:243] Iteration 66700, loss = 4.04699
I0330 19:57:41.314761  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.36101 (* 1 = 3.36101 loss)
I0330 19:57:41.314782  2693 sgd_solver.cpp:138] Iteration 66700, lr = 0.0005
I0330 19:59:52.897377  2693 solver.cpp:243] Iteration 66800, loss = 4.14622
I0330 19:59:52.900061  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.70588 (* 1 = 3.70588 loss)
I0330 19:59:52.900086  2693 sgd_solver.cpp:138] Iteration 66800, lr = 0.0005
I0330 20:02:02.361124  2693 solver.cpp:243] Iteration 66900, loss = 3.93985
I0330 20:02:02.363749  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.44172 (* 1 = 4.44172 loss)
I0330 20:02:02.363783  2693 sgd_solver.cpp:138] Iteration 66900, lr = 0.0005
I0330 20:04:14.461393  2693 solver.cpp:243] Iteration 67000, loss = 4.13298
I0330 20:04:14.468514  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.82299 (* 1 = 4.82299 loss)
I0330 20:04:14.468575  2693 sgd_solver.cpp:138] Iteration 67000, lr = 0.0005
I0330 20:06:26.117496  2693 solver.cpp:243] Iteration 67100, loss = 4.08595
I0330 20:06:26.117776  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.97606 (* 1 = 3.97606 loss)
I0330 20:06:26.117800  2693 sgd_solver.cpp:138] Iteration 67100, lr = 0.0005
I0330 20:08:37.828699  2693 solver.cpp:243] Iteration 67200, loss = 4.23084
I0330 20:08:37.828927  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.22692 (* 1 = 3.22692 loss)
I0330 20:08:37.828949  2693 sgd_solver.cpp:138] Iteration 67200, lr = 0.0005
I0330 20:10:47.867112  2693 solver.cpp:243] Iteration 67300, loss = 3.98521
I0330 20:10:47.867352  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.67005 (* 1 = 5.67005 loss)
I0330 20:10:47.867373  2693 sgd_solver.cpp:138] Iteration 67300, lr = 0.0005
I0330 20:12:58.885938  2693 solver.cpp:243] Iteration 67400, loss = 3.96075
I0330 20:12:58.886184  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.18399 (* 1 = 4.18399 loss)
I0330 20:12:58.886204  2693 sgd_solver.cpp:138] Iteration 67400, lr = 0.0005
I0330 20:15:10.305871  2693 solver.cpp:243] Iteration 67500, loss = 4.00487
I0330 20:15:10.306114  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.91806 (* 1 = 4.91806 loss)
I0330 20:15:10.306160  2693 sgd_solver.cpp:138] Iteration 67500, lr = 0.0005
I0330 20:17:18.864768  2693 solver.cpp:243] Iteration 67600, loss = 3.96074
I0330 20:17:18.864982  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.99174 (* 1 = 2.99174 loss)
I0330 20:17:18.864997  2693 sgd_solver.cpp:138] Iteration 67600, lr = 0.0005
I0330 20:19:28.124429  2693 solver.cpp:243] Iteration 67700, loss = 4.04221
I0330 20:19:28.124663  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.80782 (* 1 = 4.80782 loss)
I0330 20:19:28.124699  2693 sgd_solver.cpp:138] Iteration 67700, lr = 0.0005
I0330 20:21:39.126070  2693 solver.cpp:243] Iteration 67800, loss = 4.07677
I0330 20:21:39.126276  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.16867 (* 1 = 4.16867 loss)
I0330 20:21:39.126292  2693 sgd_solver.cpp:138] Iteration 67800, lr = 0.0005
I0330 20:23:50.525804  2693 solver.cpp:243] Iteration 67900, loss = 4.1155
I0330 20:23:50.525987  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.07594 (* 1 = 4.07594 loss)
I0330 20:23:50.526005  2693 sgd_solver.cpp:138] Iteration 67900, lr = 0.0005
I0330 20:26:01.641693  2693 solver.cpp:243] Iteration 68000, loss = 4.1816
I0330 20:26:01.641955  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.54994 (* 1 = 3.54994 loss)
I0330 20:26:01.642006  2693 sgd_solver.cpp:138] Iteration 68000, lr = 0.0005
I0330 20:28:10.997066  2693 solver.cpp:243] Iteration 68100, loss = 4.01285
I0330 20:28:10.997373  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.68746 (* 1 = 3.68746 loss)
I0330 20:28:10.997407  2693 sgd_solver.cpp:138] Iteration 68100, lr = 0.0005
I0330 20:30:21.469629  2693 solver.cpp:243] Iteration 68200, loss = 4.13895
I0330 20:30:21.469869  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.78258 (* 1 = 3.78258 loss)
I0330 20:30:21.469894  2693 sgd_solver.cpp:138] Iteration 68200, lr = 0.0005
I0330 20:32:32.038861  2693 solver.cpp:243] Iteration 68300, loss = 4.04432
I0330 20:32:32.039105  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.61615 (* 1 = 4.61615 loss)
I0330 20:32:32.039125  2693 sgd_solver.cpp:138] Iteration 68300, lr = 0.0005
I0330 20:34:43.547946  2693 solver.cpp:243] Iteration 68400, loss = 3.96887
I0330 20:34:43.548133  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.38426 (* 1 = 3.38426 loss)
I0330 20:34:43.548151  2693 sgd_solver.cpp:138] Iteration 68400, lr = 0.0005
I0330 20:36:52.261633  2693 solver.cpp:243] Iteration 68500, loss = 4.01603
I0330 20:36:52.277940  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.35925 (* 1 = 4.35925 loss)
I0330 20:36:52.277978  2693 sgd_solver.cpp:138] Iteration 68500, lr = 0.0005
I0330 20:39:03.259562  2693 solver.cpp:243] Iteration 68600, loss = 4.05179
I0330 20:39:03.259902  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.83621 (* 1 = 3.83621 loss)
I0330 20:39:03.259934  2693 sgd_solver.cpp:138] Iteration 68600, lr = 0.0005
I0330 20:41:14.438941  2693 solver.cpp:243] Iteration 68700, loss = 4.0667
I0330 20:41:14.439149  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.40976 (* 1 = 4.40976 loss)
I0330 20:41:14.439170  2693 sgd_solver.cpp:138] Iteration 68700, lr = 0.0005
I0330 20:43:24.523341  2693 solver.cpp:243] Iteration 68800, loss = 3.99375
I0330 20:43:24.523555  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.79735 (* 1 = 6.79735 loss)
I0330 20:43:24.523623  2693 sgd_solver.cpp:138] Iteration 68800, lr = 0.0005
I0330 20:45:36.524884  2693 solver.cpp:243] Iteration 68900, loss = 4.06807
I0330 20:45:36.525107  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.94849 (* 1 = 3.94849 loss)
I0330 20:45:36.525142  2693 sgd_solver.cpp:138] Iteration 68900, lr = 0.0005
I0330 20:47:49.404363  2693 solver.cpp:243] Iteration 69000, loss = 4.10859
I0330 20:47:49.409245  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.61677 (* 1 = 3.61677 loss)
I0330 20:47:49.409276  2693 sgd_solver.cpp:138] Iteration 69000, lr = 0.0005
I0330 20:50:00.028836  2693 solver.cpp:243] Iteration 69100, loss = 4.10196
I0330 20:50:00.029027  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.22965 (* 1 = 3.22965 loss)
I0330 20:50:00.029045  2693 sgd_solver.cpp:138] Iteration 69100, lr = 0.0005
I0330 20:52:10.137781  2693 solver.cpp:243] Iteration 69200, loss = 4.07231
I0330 20:52:10.138098  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.34578 (* 1 = 3.34578 loss)
I0330 20:52:10.138128  2693 sgd_solver.cpp:138] Iteration 69200, lr = 0.0005
I0330 20:54:22.471936  2693 solver.cpp:243] Iteration 69300, loss = 3.8847
I0330 20:54:22.472173  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.01465 (* 1 = 4.01465 loss)
I0330 20:54:22.472196  2693 sgd_solver.cpp:138] Iteration 69300, lr = 0.0005
I0330 20:56:33.656689  2693 solver.cpp:243] Iteration 69400, loss = 4.13418
I0330 20:56:33.656941  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.63114 (* 1 = 4.63114 loss)
I0330 20:56:33.656975  2693 sgd_solver.cpp:138] Iteration 69400, lr = 0.0005
I0330 20:58:43.439438  2693 solver.cpp:243] Iteration 69500, loss = 3.92657
I0330 20:58:43.439721  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.34972 (* 1 = 5.34972 loss)
I0330 20:58:43.439740  2693 sgd_solver.cpp:138] Iteration 69500, lr = 0.0005
I0330 21:00:54.664175  2693 solver.cpp:243] Iteration 69600, loss = 3.99956
I0330 21:00:54.664345  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.20961 (* 1 = 3.20961 loss)
I0330 21:00:54.664363  2693 sgd_solver.cpp:138] Iteration 69600, lr = 0.0005
I0330 21:03:05.765008  2693 solver.cpp:243] Iteration 69700, loss = 4.08039
I0330 21:03:05.765211  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.54493 (* 1 = 3.54493 loss)
I0330 21:03:05.765229  2693 sgd_solver.cpp:138] Iteration 69700, lr = 0.0005
I0330 21:05:15.058364  2693 solver.cpp:243] Iteration 69800, loss = 4.01493
I0330 21:05:15.058580  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.71074 (* 1 = 4.71074 loss)
I0330 21:05:15.058598  2693 sgd_solver.cpp:138] Iteration 69800, lr = 0.0005
I0330 21:07:22.760406  2693 solver.cpp:243] Iteration 69900, loss = 4.05327
I0330 21:07:22.760632  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.90246 (* 1 = 3.90246 loss)
I0330 21:07:22.760649  2693 sgd_solver.cpp:138] Iteration 69900, lr = 0.0005
I0330 21:09:30.404729  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_70000.caffemodel
I0330 21:09:31.631142  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_70000.solverstate
I0330 21:09:31.780671  2693 solver.cpp:433] Iteration 70000, Testing net (#0)
I0330 21:09:31.780764  2693 net.cpp:693] Ignoring source layer mbox_loss
I0330 21:10:51.709509  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.576204
I0330 21:10:52.615357  2693 solver.cpp:243] Iteration 70000, loss = 4.0499
I0330 21:10:52.615428  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.74413 (* 1 = 4.74413 loss)
I0330 21:10:52.615443  2693 sgd_solver.cpp:138] Iteration 70000, lr = 0.0005
I0330 21:13:03.714304  2693 solver.cpp:243] Iteration 70100, loss = 3.94537
I0330 21:13:03.714515  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.06428 (* 1 = 6.06428 loss)
I0330 21:13:03.714532  2693 sgd_solver.cpp:138] Iteration 70100, lr = 0.0005
I0330 21:15:14.824600  2693 solver.cpp:243] Iteration 70200, loss = 4.0551
I0330 21:15:14.824895  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.14881 (* 1 = 3.14881 loss)
I0330 21:15:14.824934  2693 sgd_solver.cpp:138] Iteration 70200, lr = 0.0005
I0330 21:17:28.254163  2693 solver.cpp:243] Iteration 70300, loss = 4.01701
I0330 21:17:28.254427  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.93435 (* 1 = 3.93435 loss)
I0330 21:17:28.254452  2693 sgd_solver.cpp:138] Iteration 70300, lr = 0.0005
I0330 21:19:39.999675  2693 solver.cpp:243] Iteration 70400, loss = 4.19977
I0330 21:19:39.999891  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.17878 (* 1 = 4.17878 loss)
I0330 21:19:39.999909  2693 sgd_solver.cpp:138] Iteration 70400, lr = 0.0005
I0330 21:21:51.842345  2693 solver.cpp:243] Iteration 70500, loss = 4.01179
I0330 21:21:51.847195  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.92676 (* 1 = 3.92676 loss)
I0330 21:21:51.847221  2693 sgd_solver.cpp:138] Iteration 70500, lr = 0.0005
I0330 21:24:04.216529  2693 solver.cpp:243] Iteration 70600, loss = 4.09752
I0330 21:24:04.216778  2693 solver.cpp:259]     Train net output #0: mbox_loss = 1.97386 (* 1 = 1.97386 loss)
I0330 21:24:04.216804  2693 sgd_solver.cpp:138] Iteration 70600, lr = 0.0005
I0330 21:26:17.544729  2693 solver.cpp:243] Iteration 70700, loss = 4.07599
I0330 21:26:17.544970  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.18354 (* 1 = 4.18354 loss)
I0330 21:26:17.544996  2693 sgd_solver.cpp:138] Iteration 70700, lr = 0.0005
I0330 21:28:56.184350  2693 solver.cpp:243] Iteration 70800, loss = 4.1621
I0330 21:28:56.184550  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.2327 (* 1 = 4.2327 loss)
I0330 21:28:56.184566  2693 sgd_solver.cpp:138] Iteration 70800, lr = 0.0005
I0330 21:31:50.445457  2693 solver.cpp:243] Iteration 70900, loss = 4.03382
I0330 21:31:50.445735  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.57401 (* 1 = 4.57401 loss)
I0330 21:31:50.445775  2693 sgd_solver.cpp:138] Iteration 70900, lr = 0.0005
I0330 21:34:21.514053  2693 solver.cpp:243] Iteration 71000, loss = 4.18132
I0330 21:34:21.520814  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.18327 (* 1 = 4.18327 loss)
I0330 21:34:21.520833  2693 sgd_solver.cpp:138] Iteration 71000, lr = 0.0005
I0330 21:37:01.525205  2693 solver.cpp:243] Iteration 71100, loss = 4.04462
I0330 21:37:01.525481  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.51686 (* 1 = 3.51686 loss)
I0330 21:37:01.525508  2693 sgd_solver.cpp:138] Iteration 71100, lr = 0.0005
I0330 21:39:52.417616  2693 solver.cpp:243] Iteration 71200, loss = 3.97497
I0330 21:39:52.417871  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.49966 (* 1 = 4.49966 loss)
I0330 21:39:52.417906  2693 sgd_solver.cpp:138] Iteration 71200, lr = 0.0005
I0330 21:42:42.955574  2693 solver.cpp:243] Iteration 71300, loss = 4.04875
I0330 21:42:42.955848  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.09993 (* 1 = 4.09993 loss)
I0330 21:42:42.955866  2693 sgd_solver.cpp:138] Iteration 71300, lr = 0.0005
I0330 21:45:34.759994  2693 solver.cpp:243] Iteration 71400, loss = 4.01662
I0330 21:45:34.760236  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.99768 (* 1 = 3.99768 loss)
I0330 21:45:34.760277  2693 sgd_solver.cpp:138] Iteration 71400, lr = 0.0005
I0330 21:48:27.859403  2693 solver.cpp:243] Iteration 71500, loss = 4.11102
I0330 21:48:27.859689  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.85067 (* 1 = 3.85067 loss)
I0330 21:48:27.859714  2693 sgd_solver.cpp:138] Iteration 71500, lr = 0.0005
I0330 21:51:20.558264  2693 solver.cpp:243] Iteration 71600, loss = 4.05889
I0330 21:51:20.558521  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.85939 (* 1 = 2.85939 loss)
I0330 21:51:20.558571  2693 sgd_solver.cpp:138] Iteration 71600, lr = 0.0005
I0330 21:54:12.956830  2693 solver.cpp:243] Iteration 71700, loss = 3.96469
I0330 21:54:12.957069  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.51858 (* 1 = 5.51858 loss)
I0330 21:54:12.957098  2693 sgd_solver.cpp:138] Iteration 71700, lr = 0.0005
I0330 21:57:05.704043  2693 solver.cpp:243] Iteration 71800, loss = 4.02478
I0330 21:57:05.704221  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.81729 (* 1 = 2.81729 loss)
I0330 21:57:05.704239  2693 sgd_solver.cpp:138] Iteration 71800, lr = 0.0005
I0330 21:59:57.028863  2693 solver.cpp:243] Iteration 71900, loss = 4.08831
I0330 21:59:57.029132  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.98575 (* 1 = 4.98575 loss)
I0330 21:59:57.029165  2693 sgd_solver.cpp:138] Iteration 71900, lr = 0.0005
I0330 22:02:51.170822  2693 solver.cpp:243] Iteration 72000, loss = 4.11125
I0330 22:02:51.171111  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.10913 (* 1 = 5.10913 loss)
I0330 22:02:51.171139  2693 sgd_solver.cpp:138] Iteration 72000, lr = 0.0005
I0330 22:05:35.226213  2693 solver.cpp:243] Iteration 72100, loss = 4.01574
I0330 22:05:35.226538  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.14828 (* 1 = 4.14828 loss)
I0330 22:05:35.226574  2693 sgd_solver.cpp:138] Iteration 72100, lr = 0.0005
I0330 22:07:50.085064  2693 solver.cpp:243] Iteration 72200, loss = 4.05869
I0330 22:07:50.085345  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.38132 (* 1 = 4.38132 loss)
I0330 22:07:50.085374  2693 sgd_solver.cpp:138] Iteration 72200, lr = 0.0005
I0330 22:10:00.671051  2693 solver.cpp:243] Iteration 72300, loss = 4.01486
I0330 22:10:00.671303  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.34909 (* 1 = 5.34909 loss)
I0330 22:10:00.671339  2693 sgd_solver.cpp:138] Iteration 72300, lr = 0.0005
I0330 22:12:26.688601  2693 solver.cpp:243] Iteration 72400, loss = 4.1037
I0330 22:12:26.697227  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.73847 (* 1 = 3.73847 loss)
I0330 22:12:26.697259  2693 sgd_solver.cpp:138] Iteration 72400, lr = 0.0005
I0330 22:15:18.431450  2693 solver.cpp:243] Iteration 72500, loss = 4.15471
I0330 22:15:18.443840  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.86201 (* 1 = 3.86201 loss)
I0330 22:15:18.443877  2693 sgd_solver.cpp:138] Iteration 72500, lr = 0.0005
I0330 22:18:10.092005  2693 solver.cpp:243] Iteration 72600, loss = 4.11444
I0330 22:18:10.092233  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.62107 (* 1 = 3.62107 loss)
I0330 22:18:10.092253  2693 sgd_solver.cpp:138] Iteration 72600, lr = 0.0005
I0330 22:21:02.366114  2693 solver.cpp:243] Iteration 72700, loss = 4.09875
I0330 22:21:02.366312  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.43402 (* 1 = 5.43402 loss)
I0330 22:21:02.366330  2693 sgd_solver.cpp:138] Iteration 72700, lr = 0.0005
I0330 22:23:22.893303  2693 solver.cpp:243] Iteration 72800, loss = 4.08988
I0330 22:23:22.893636  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.36037 (* 1 = 4.36037 loss)
I0330 22:23:22.893705  2693 sgd_solver.cpp:138] Iteration 72800, lr = 0.0005
I0330 22:26:03.456722  2693 solver.cpp:243] Iteration 72900, loss = 3.92536
I0330 22:26:03.456991  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.06685 (* 1 = 4.06685 loss)
I0330 22:26:03.457020  2693 sgd_solver.cpp:138] Iteration 72900, lr = 0.0005
I0330 22:28:56.606142  2693 solver.cpp:243] Iteration 73000, loss = 4.0514
I0330 22:28:56.606391  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.47567 (* 1 = 2.47567 loss)
I0330 22:28:56.606410  2693 sgd_solver.cpp:138] Iteration 73000, lr = 0.0005
I0330 22:31:46.703799  2693 solver.cpp:243] Iteration 73100, loss = 4.10789
I0330 22:31:46.704006  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.86252 (* 1 = 3.86252 loss)
I0330 22:31:46.704023  2693 sgd_solver.cpp:138] Iteration 73100, lr = 0.0005
I0330 22:34:39.382295  2693 solver.cpp:243] Iteration 73200, loss = 4.00958
I0330 22:34:39.382575  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.96765 (* 1 = 4.96765 loss)
I0330 22:34:39.382611  2693 sgd_solver.cpp:138] Iteration 73200, lr = 0.0005
I0330 22:37:29.713572  2693 solver.cpp:243] Iteration 73300, loss = 4.09961
I0330 22:37:29.713838  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.75498 (* 1 = 3.75498 loss)
I0330 22:37:29.713874  2693 sgd_solver.cpp:138] Iteration 73300, lr = 0.0005
I0330 22:40:21.252190  2693 solver.cpp:243] Iteration 73400, loss = 4.07239
I0330 22:40:21.252420  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.71164 (* 1 = 4.71164 loss)
I0330 22:40:21.252440  2693 sgd_solver.cpp:138] Iteration 73400, lr = 0.0005
I0330 22:43:10.923933  2693 solver.cpp:243] Iteration 73500, loss = 4.14811
I0330 22:43:10.924216  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.2585 (* 1 = 4.2585 loss)
I0330 22:43:10.924280  2693 sgd_solver.cpp:138] Iteration 73500, lr = 0.0005
I0330 22:45:37.913599  2693 solver.cpp:243] Iteration 73600, loss = 4.04838
I0330 22:45:37.927863  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.76315 (* 1 = 2.76315 loss)
I0330 22:45:37.927893  2693 sgd_solver.cpp:138] Iteration 73600, lr = 0.0005
I0330 22:47:52.968679  2693 solver.cpp:243] Iteration 73700, loss = 3.97563
I0330 22:47:52.968924  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.58376 (* 1 = 3.58376 loss)
I0330 22:47:52.968962  2693 sgd_solver.cpp:138] Iteration 73700, lr = 0.0005
I0330 22:50:01.506197  2693 solver.cpp:243] Iteration 73800, loss = 4.08006
I0330 22:50:01.506404  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.55989 (* 1 = 4.55989 loss)
I0330 22:50:01.506424  2693 sgd_solver.cpp:138] Iteration 73800, lr = 0.0005
I0330 22:52:10.846961  2693 solver.cpp:243] Iteration 73900, loss = 4.04296
I0330 22:52:10.847210  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.25956 (* 1 = 5.25956 loss)
I0330 22:52:10.847239  2693 sgd_solver.cpp:138] Iteration 73900, lr = 0.0005
I0330 22:54:21.019533  2693 solver.cpp:243] Iteration 74000, loss = 4.12192
I0330 22:54:21.019875  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.80291 (* 1 = 3.80291 loss)
I0330 22:54:21.019901  2693 sgd_solver.cpp:138] Iteration 74000, lr = 0.0005
I0330 22:56:30.437897  2693 solver.cpp:243] Iteration 74100, loss = 4.07665
I0330 22:56:30.438099  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.74915 (* 1 = 2.74915 loss)
I0330 22:56:30.438117  2693 sgd_solver.cpp:138] Iteration 74100, lr = 0.0005
I0330 22:58:40.283143  2693 solver.cpp:243] Iteration 74200, loss = 3.96246
I0330 22:58:40.283360  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.53531 (* 1 = 6.53531 loss)
I0330 22:58:40.283377  2693 sgd_solver.cpp:138] Iteration 74200, lr = 0.0005
I0330 23:00:49.914285  2693 solver.cpp:243] Iteration 74300, loss = 4.13735
I0330 23:00:49.914515  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.76192 (* 1 = 4.76192 loss)
I0330 23:00:49.914546  2693 sgd_solver.cpp:138] Iteration 74300, lr = 0.0005
I0330 23:02:58.748875  2693 solver.cpp:243] Iteration 74400, loss = 4.06955
I0330 23:02:58.749119  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.91135 (* 1 = 2.91135 loss)
I0330 23:02:58.749150  2693 sgd_solver.cpp:138] Iteration 74400, lr = 0.0005
I0330 23:05:09.770498  2693 solver.cpp:243] Iteration 74500, loss = 4.11637
I0330 23:05:09.770758  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.13795 (* 1 = 4.13795 loss)
I0330 23:05:09.770799  2693 sgd_solver.cpp:138] Iteration 74500, lr = 0.0005
I0330 23:07:18.948091  2693 solver.cpp:243] Iteration 74600, loss = 4.0309
I0330 23:07:18.948288  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.17418 (* 1 = 3.17418 loss)
I0330 23:07:18.948307  2693 sgd_solver.cpp:138] Iteration 74600, lr = 0.0005
I0330 23:09:28.601583  2693 solver.cpp:243] Iteration 74700, loss = 4.0158
I0330 23:09:28.610056  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.0567 (* 1 = 3.0567 loss)
I0330 23:09:28.610091  2693 sgd_solver.cpp:138] Iteration 74700, lr = 0.0005
I0330 23:11:37.844213  2693 solver.cpp:243] Iteration 74800, loss = 3.99619
I0330 23:11:37.844436  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.27183 (* 1 = 3.27183 loss)
I0330 23:11:37.844455  2693 sgd_solver.cpp:138] Iteration 74800, lr = 0.0005
I0330 23:13:47.866801  2693 solver.cpp:243] Iteration 74900, loss = 3.99895
I0330 23:13:47.867023  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.19499 (* 1 = 4.19499 loss)
I0330 23:13:47.867043  2693 sgd_solver.cpp:138] Iteration 74900, lr = 0.0005
I0330 23:15:56.703155  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_75000.caffemodel
I0330 23:15:57.639924  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_75000.solverstate
I0330 23:15:57.753638  2693 solver.cpp:433] Iteration 75000, Testing net (#0)
I0330 23:15:57.753721  2693 net.cpp:693] Ignoring source layer mbox_loss
I0330 23:17:17.566596  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.550799
I0330 23:17:18.242370  2693 solver.cpp:243] Iteration 75000, loss = 4.0613
I0330 23:17:18.242449  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.61037 (* 1 = 4.61037 loss)
I0330 23:17:18.242466  2693 sgd_solver.cpp:138] Iteration 75000, lr = 0.0005
I0330 23:19:28.158179  2693 solver.cpp:243] Iteration 75100, loss = 4.07876
I0330 23:19:28.158419  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.5172 (* 1 = 2.5172 loss)
I0330 23:19:28.158440  2693 sgd_solver.cpp:138] Iteration 75100, lr = 0.0005
I0330 23:21:38.415819  2693 solver.cpp:243] Iteration 75200, loss = 4.02134
I0330 23:21:38.422464  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.53602 (* 1 = 3.53602 loss)
I0330 23:21:38.422482  2693 sgd_solver.cpp:138] Iteration 75200, lr = 0.0005
I0330 23:23:49.631731  2693 solver.cpp:243] Iteration 75300, loss = 3.94008
I0330 23:23:49.631919  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.2096 (* 1 = 5.2096 loss)
I0330 23:23:49.631938  2693 sgd_solver.cpp:138] Iteration 75300, lr = 0.0005
I0330 23:25:59.914499  2693 solver.cpp:243] Iteration 75400, loss = 4.04165
I0330 23:25:59.914721  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.6896 (* 1 = 4.6896 loss)
I0330 23:25:59.914736  2693 sgd_solver.cpp:138] Iteration 75400, lr = 0.0005
I0330 23:28:08.921397  2693 solver.cpp:243] Iteration 75500, loss = 4.09176
I0330 23:28:08.931444  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.34413 (* 1 = 3.34413 loss)
I0330 23:28:08.931468  2693 sgd_solver.cpp:138] Iteration 75500, lr = 0.0005
I0330 23:30:17.718696  2693 solver.cpp:243] Iteration 75600, loss = 4.01201
I0330 23:30:17.718920  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.81984 (* 1 = 2.81984 loss)
I0330 23:30:17.718940  2693 sgd_solver.cpp:138] Iteration 75600, lr = 0.0005
I0330 23:32:27.785361  2693 solver.cpp:243] Iteration 75700, loss = 3.99545
I0330 23:32:27.785596  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.77113 (* 1 = 2.77113 loss)
I0330 23:32:27.785639  2693 sgd_solver.cpp:138] Iteration 75700, lr = 0.0005
I0330 23:34:37.265017  2693 solver.cpp:243] Iteration 75800, loss = 3.97106
I0330 23:34:37.265271  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.47081 (* 1 = 4.47081 loss)
I0330 23:34:37.265319  2693 sgd_solver.cpp:138] Iteration 75800, lr = 0.0005
I0330 23:36:47.206271  2693 solver.cpp:243] Iteration 75900, loss = 3.86658
I0330 23:36:47.206475  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.94423 (* 1 = 3.94423 loss)
I0330 23:36:47.206493  2693 sgd_solver.cpp:138] Iteration 75900, lr = 0.0005
I0330 23:38:55.564435  2693 solver.cpp:243] Iteration 76000, loss = 4.06864
I0330 23:38:55.564625  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.5884 (* 1 = 3.5884 loss)
I0330 23:38:55.564641  2693 sgd_solver.cpp:138] Iteration 76000, lr = 0.0005
I0330 23:41:06.514927  2693 solver.cpp:243] Iteration 76100, loss = 4.09628
I0330 23:41:06.515147  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.20034 (* 1 = 3.20034 loss)
I0330 23:41:06.515168  2693 sgd_solver.cpp:138] Iteration 76100, lr = 0.0005
I0330 23:43:15.534278  2693 solver.cpp:243] Iteration 76200, loss = 4.01757
I0330 23:43:15.534468  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.00132 (* 1 = 3.00132 loss)
I0330 23:43:15.534485  2693 sgd_solver.cpp:138] Iteration 76200, lr = 0.0005
I0330 23:45:24.794855  2693 solver.cpp:243] Iteration 76300, loss = 4.02042
I0330 23:45:24.795047  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.03882 (* 1 = 6.03882 loss)
I0330 23:45:24.795063  2693 sgd_solver.cpp:138] Iteration 76300, lr = 0.0005
I0330 23:47:35.392011  2693 solver.cpp:243] Iteration 76400, loss = 4.15383
I0330 23:47:35.392441  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.24878 (* 1 = 5.24878 loss)
I0330 23:47:35.392458  2693 sgd_solver.cpp:138] Iteration 76400, lr = 0.0005
I0330 23:49:46.329718  2693 solver.cpp:243] Iteration 76500, loss = 4.1087
I0330 23:49:46.329900  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.98767 (* 1 = 3.98767 loss)
I0330 23:49:46.329916  2693 sgd_solver.cpp:138] Iteration 76500, lr = 0.0005
I0330 23:51:58.051209  2693 solver.cpp:243] Iteration 76600, loss = 4.01667
I0330 23:51:58.051506  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.85869 (* 1 = 3.85869 loss)
I0330 23:51:58.051538  2693 sgd_solver.cpp:138] Iteration 76600, lr = 0.0005
I0330 23:54:07.493261  2693 solver.cpp:243] Iteration 76700, loss = 4.04347
I0330 23:54:07.493522  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.39925 (* 1 = 3.39925 loss)
I0330 23:54:07.493546  2693 sgd_solver.cpp:138] Iteration 76700, lr = 0.0005
I0330 23:56:19.077872  2693 solver.cpp:243] Iteration 76800, loss = 3.96377
I0330 23:56:19.078063  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.49869 (* 1 = 3.49869 loss)
I0330 23:56:19.078081  2693 sgd_solver.cpp:138] Iteration 76800, lr = 0.0005
I0330 23:58:29.494374  2693 solver.cpp:243] Iteration 76900, loss = 3.96952
I0330 23:58:29.494626  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.2687 (* 1 = 3.2687 loss)
I0330 23:58:29.494655  2693 sgd_solver.cpp:138] Iteration 76900, lr = 0.0005
I0331 00:00:37.797726  2693 solver.cpp:243] Iteration 77000, loss = 4.05746
I0331 00:00:37.797924  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.24765 (* 1 = 4.24765 loss)
I0331 00:00:37.797943  2693 sgd_solver.cpp:138] Iteration 77000, lr = 0.0005
I0331 00:02:47.694193  2693 solver.cpp:243] Iteration 77100, loss = 4.03539
I0331 00:02:47.694480  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.157 (* 1 = 3.157 loss)
I0331 00:02:47.694538  2693 sgd_solver.cpp:138] Iteration 77100, lr = 0.0005
I0331 00:04:57.089793  2693 solver.cpp:243] Iteration 77200, loss = 3.98482
I0331 00:04:57.089993  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.59291 (* 1 = 3.59291 loss)
I0331 00:04:57.090011  2693 sgd_solver.cpp:138] Iteration 77200, lr = 0.0005
I0331 00:07:07.029000  2693 solver.cpp:243] Iteration 77300, loss = 3.90885
I0331 00:07:07.029212  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.32154 (* 1 = 3.32154 loss)
I0331 00:07:07.029229  2693 sgd_solver.cpp:138] Iteration 77300, lr = 0.0005
I0331 00:09:17.313863  2693 solver.cpp:243] Iteration 77400, loss = 4.09282
I0331 00:09:17.314116  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.12699 (* 1 = 3.12699 loss)
I0331 00:09:17.314165  2693 sgd_solver.cpp:138] Iteration 77400, lr = 0.0005
I0331 00:11:27.044701  2693 solver.cpp:243] Iteration 77500, loss = 3.90493
I0331 00:11:27.044905  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.03676 (* 1 = 4.03676 loss)
I0331 00:11:27.044924  2693 sgd_solver.cpp:138] Iteration 77500, lr = 0.0005
I0331 00:13:38.107448  2693 solver.cpp:243] Iteration 77600, loss = 4.14765
I0331 00:13:38.107758  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.28769 (* 1 = 4.28769 loss)
I0331 00:13:38.107779  2693 sgd_solver.cpp:138] Iteration 77600, lr = 0.0005
I0331 00:15:47.536841  2693 solver.cpp:243] Iteration 77700, loss = 4.2274
I0331 00:15:47.537034  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.18999 (* 1 = 3.18999 loss)
I0331 00:15:47.537053  2693 sgd_solver.cpp:138] Iteration 77700, lr = 0.0005
I0331 00:18:00.076635  2693 solver.cpp:243] Iteration 77800, loss = 4.10453
I0331 00:18:00.076861  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.09969 (* 1 = 4.09969 loss)
I0331 00:18:00.076881  2693 sgd_solver.cpp:138] Iteration 77800, lr = 0.0005
I0331 00:20:11.506844  2693 solver.cpp:243] Iteration 77900, loss = 4.0077
I0331 00:20:11.507088  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.66709 (* 1 = 3.66709 loss)
I0331 00:20:11.507122  2693 sgd_solver.cpp:138] Iteration 77900, lr = 0.0005
I0331 00:22:21.039386  2693 solver.cpp:243] Iteration 78000, loss = 4.03577
I0331 00:22:21.039676  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.66808 (* 1 = 4.66808 loss)
I0331 00:22:21.039710  2693 sgd_solver.cpp:138] Iteration 78000, lr = 0.0005
I0331 00:24:32.265143  2693 solver.cpp:243] Iteration 78100, loss = 4.16603
I0331 00:24:32.265413  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.76265 (* 1 = 2.76265 loss)
I0331 00:24:32.265445  2693 sgd_solver.cpp:138] Iteration 78100, lr = 0.0005
I0331 00:26:42.840222  2693 solver.cpp:243] Iteration 78200, loss = 3.91013
I0331 00:26:42.840342  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.94365 (* 1 = 3.94365 loss)
I0331 00:26:42.840358  2693 sgd_solver.cpp:138] Iteration 78200, lr = 0.0005
I0331 00:28:52.909170  2693 solver.cpp:243] Iteration 78300, loss = 4.04138
I0331 00:28:52.909432  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.77352 (* 1 = 3.77352 loss)
I0331 00:28:52.909459  2693 sgd_solver.cpp:138] Iteration 78300, lr = 0.0005
I0331 00:31:02.967551  2693 solver.cpp:243] Iteration 78400, loss = 4.05463
I0331 00:31:02.967866  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.96144 (* 1 = 3.96144 loss)
I0331 00:31:02.967898  2693 sgd_solver.cpp:138] Iteration 78400, lr = 0.0005
I0331 00:33:13.940430  2693 solver.cpp:243] Iteration 78500, loss = 4.08106
I0331 00:33:13.940618  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.11189 (* 1 = 5.11189 loss)
I0331 00:33:13.940631  2693 sgd_solver.cpp:138] Iteration 78500, lr = 0.0005
I0331 00:35:24.449272  2693 solver.cpp:243] Iteration 78600, loss = 4.02987
I0331 00:35:24.449514  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.55481 (* 1 = 5.55481 loss)
I0331 00:35:24.449548  2693 sgd_solver.cpp:138] Iteration 78600, lr = 0.0005
I0331 00:37:36.462136  2693 solver.cpp:243] Iteration 78700, loss = 4.12748
I0331 00:37:36.462352  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.65322 (* 1 = 4.65322 loss)
I0331 00:37:36.462383  2693 sgd_solver.cpp:138] Iteration 78700, lr = 0.0005
I0331 00:39:48.787904  2693 solver.cpp:243] Iteration 78800, loss = 3.96606
I0331 00:39:48.788131  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.44885 (* 1 = 4.44885 loss)
I0331 00:39:48.788172  2693 sgd_solver.cpp:138] Iteration 78800, lr = 0.0005
I0331 00:41:56.659557  2693 solver.cpp:243] Iteration 78900, loss = 4.00772
I0331 00:41:56.659854  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.24678 (* 1 = 4.24678 loss)
I0331 00:41:56.659878  2693 sgd_solver.cpp:138] Iteration 78900, lr = 0.0005
I0331 00:44:07.452172  2693 solver.cpp:243] Iteration 79000, loss = 3.92629
I0331 00:44:07.452394  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.47818 (* 1 = 3.47818 loss)
I0331 00:44:07.452425  2693 sgd_solver.cpp:138] Iteration 79000, lr = 0.0005
I0331 00:46:17.546618  2693 solver.cpp:243] Iteration 79100, loss = 4.224
I0331 00:46:17.546888  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.50467 (* 1 = 4.50467 loss)
I0331 00:46:17.546917  2693 sgd_solver.cpp:138] Iteration 79100, lr = 0.0005
I0331 00:48:28.090620  2693 solver.cpp:243] Iteration 79200, loss = 4.13399
I0331 00:48:28.090811  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.91508 (* 1 = 3.91508 loss)
I0331 00:48:28.090829  2693 sgd_solver.cpp:138] Iteration 79200, lr = 0.0005
I0331 00:50:38.254580  2693 solver.cpp:243] Iteration 79300, loss = 3.95485
I0331 00:50:38.254777  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.13169 (* 1 = 4.13169 loss)
I0331 00:50:38.254796  2693 sgd_solver.cpp:138] Iteration 79300, lr = 0.0005
I0331 00:52:47.966472  2693 solver.cpp:243] Iteration 79400, loss = 4.02581
I0331 00:52:47.966677  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.51513 (* 1 = 3.51513 loss)
I0331 00:52:47.966696  2693 sgd_solver.cpp:138] Iteration 79400, lr = 0.0005
I0331 00:54:57.753974  2693 solver.cpp:243] Iteration 79500, loss = 3.962
I0331 00:54:57.754238  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.73735 (* 1 = 3.73735 loss)
I0331 00:54:57.754276  2693 sgd_solver.cpp:138] Iteration 79500, lr = 0.0005
I0331 00:57:08.522809  2693 solver.cpp:243] Iteration 79600, loss = 3.98512
I0331 00:57:08.523015  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.50027 (* 1 = 3.50027 loss)
I0331 00:57:08.523041  2693 sgd_solver.cpp:138] Iteration 79600, lr = 0.0005
I0331 00:59:18.843765  2693 solver.cpp:243] Iteration 79700, loss = 4.02276
I0331 00:59:18.844007  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.07603 (* 1 = 6.07603 loss)
I0331 00:59:18.844028  2693 sgd_solver.cpp:138] Iteration 79700, lr = 0.0005
I0331 01:01:28.076125  2693 solver.cpp:243] Iteration 79800, loss = 4.07343
I0331 01:01:28.076361  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.90383 (* 1 = 2.90383 loss)
I0331 01:01:28.076401  2693 sgd_solver.cpp:138] Iteration 79800, lr = 0.0005
I0331 01:03:39.974565  2693 solver.cpp:243] Iteration 79900, loss = 4.02835
I0331 01:03:39.974809  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.75094 (* 1 = 3.75094 loss)
I0331 01:03:39.974853  2693 sgd_solver.cpp:138] Iteration 79900, lr = 0.0005
I0331 01:05:47.870154  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_80000.caffemodel
I0331 01:05:48.887336  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_80000.solverstate
I0331 01:05:49.024997  2693 solver.cpp:433] Iteration 80000, Testing net (#0)
I0331 01:05:49.025087  2693 net.cpp:693] Ignoring source layer mbox_loss
I0331 01:07:08.863829  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.575671
I0331 01:07:09.664870  2693 solver.cpp:243] Iteration 80000, loss = 4.12474
I0331 01:07:09.664942  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.65107 (* 1 = 4.65107 loss)
I0331 01:07:09.664957  2693 sgd_solver.cpp:138] Iteration 80000, lr = 0.0005
I0331 01:09:18.982374  2693 solver.cpp:243] Iteration 80100, loss = 3.98762
I0331 01:09:18.982558  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.18122 (* 1 = 5.18122 loss)
I0331 01:09:18.982576  2693 sgd_solver.cpp:138] Iteration 80100, lr = 0.0005
I0331 01:11:29.651414  2693 solver.cpp:243] Iteration 80200, loss = 4.0846
I0331 01:11:29.651718  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.77372 (* 1 = 3.77372 loss)
I0331 01:11:29.651738  2693 sgd_solver.cpp:138] Iteration 80200, lr = 0.0005
I0331 01:13:39.948508  2693 solver.cpp:243] Iteration 80300, loss = 3.98094
I0331 01:13:39.948752  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.38982 (* 1 = 6.38982 loss)
I0331 01:13:39.948787  2693 sgd_solver.cpp:138] Iteration 80300, lr = 0.0005
I0331 01:15:49.852485  2693 solver.cpp:243] Iteration 80400, loss = 4.05196
I0331 01:15:49.852730  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.48459 (* 1 = 4.48459 loss)
I0331 01:15:49.852751  2693 sgd_solver.cpp:138] Iteration 80400, lr = 0.0005
I0331 01:17:59.502784  2693 solver.cpp:243] Iteration 80500, loss = 3.98054
I0331 01:17:59.505285  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.89784 (* 1 = 3.89784 loss)
I0331 01:17:59.505347  2693 sgd_solver.cpp:138] Iteration 80500, lr = 0.0005
I0331 01:20:09.137934  2693 solver.cpp:243] Iteration 80600, loss = 3.84578
I0331 01:20:09.138178  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.71343 (* 1 = 3.71343 loss)
I0331 01:20:09.138206  2693 sgd_solver.cpp:138] Iteration 80600, lr = 0.0005
I0331 01:22:20.988986  2693 solver.cpp:243] Iteration 80700, loss = 4.21744
I0331 01:22:20.989255  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.01329 (* 1 = 5.01329 loss)
I0331 01:22:20.989300  2693 sgd_solver.cpp:138] Iteration 80700, lr = 0.0005
I0331 01:24:29.097921  2693 solver.cpp:243] Iteration 80800, loss = 3.98125
I0331 01:24:29.098122  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.19653 (* 1 = 4.19653 loss)
I0331 01:24:29.098141  2693 sgd_solver.cpp:138] Iteration 80800, lr = 0.0005
I0331 01:26:39.082774  2693 solver.cpp:243] Iteration 80900, loss = 4.04832
I0331 01:26:39.083056  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.82013 (* 1 = 3.82013 loss)
I0331 01:26:39.083076  2693 sgd_solver.cpp:138] Iteration 80900, lr = 0.0005
I0331 01:28:50.059521  2693 solver.cpp:243] Iteration 81000, loss = 4.0697
I0331 01:28:50.059782  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.30875 (* 1 = 4.30875 loss)
I0331 01:28:50.059800  2693 sgd_solver.cpp:138] Iteration 81000, lr = 0.0005
I0331 01:30:59.402984  2693 solver.cpp:243] Iteration 81100, loss = 3.87039
I0331 01:30:59.403180  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.14019 (* 1 = 3.14019 loss)
I0331 01:30:59.403199  2693 sgd_solver.cpp:138] Iteration 81100, lr = 0.0005
I0331 01:33:10.266001  2693 solver.cpp:243] Iteration 81200, loss = 3.99434
I0331 01:33:10.266232  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.92349 (* 1 = 3.92349 loss)
I0331 01:33:10.266252  2693 sgd_solver.cpp:138] Iteration 81200, lr = 0.0005
I0331 01:35:18.972059  2693 solver.cpp:243] Iteration 81300, loss = 4.08372
I0331 01:35:18.972265  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.78302 (* 1 = 5.78302 loss)
I0331 01:35:18.972283  2693 sgd_solver.cpp:138] Iteration 81300, lr = 0.0005
I0331 01:37:28.799332  2693 solver.cpp:243] Iteration 81400, loss = 3.92959
I0331 01:37:28.799664  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.11784 (* 1 = 4.11784 loss)
I0331 01:37:28.799702  2693 sgd_solver.cpp:138] Iteration 81400, lr = 0.0005
I0331 01:39:37.488596  2693 solver.cpp:243] Iteration 81500, loss = 3.8756
I0331 01:39:37.488888  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.5648 (* 1 = 3.5648 loss)
I0331 01:39:37.488919  2693 sgd_solver.cpp:138] Iteration 81500, lr = 0.0005
I0331 01:41:47.975178  2693 solver.cpp:243] Iteration 81600, loss = 4.24916
I0331 01:41:47.975360  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.70951 (* 1 = 3.70951 loss)
I0331 01:41:47.975379  2693 sgd_solver.cpp:138] Iteration 81600, lr = 0.0005
I0331 01:44:00.163295  2693 solver.cpp:243] Iteration 81700, loss = 4.06525
I0331 01:44:00.163516  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.71984 (* 1 = 3.71984 loss)
I0331 01:44:00.163534  2693 sgd_solver.cpp:138] Iteration 81700, lr = 0.0005
I0331 01:46:10.414665  2693 solver.cpp:243] Iteration 81800, loss = 3.97982
I0331 01:46:10.414933  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.17853 (* 1 = 3.17853 loss)
I0331 01:46:10.414950  2693 sgd_solver.cpp:138] Iteration 81800, lr = 0.0005
I0331 01:48:19.929278  2693 solver.cpp:243] Iteration 81900, loss = 4.06559
I0331 01:48:19.937891  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.17884 (* 1 = 4.17884 loss)
I0331 01:48:19.937965  2693 sgd_solver.cpp:138] Iteration 81900, lr = 0.0005
I0331 01:50:30.515902  2693 solver.cpp:243] Iteration 82000, loss = 3.93436
I0331 01:50:30.516160  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.89636 (* 1 = 3.89636 loss)
I0331 01:50:30.516191  2693 sgd_solver.cpp:138] Iteration 82000, lr = 0.0005
I0331 01:52:40.779570  2693 solver.cpp:243] Iteration 82100, loss = 4.00661
I0331 01:52:40.779904  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.34805 (* 1 = 3.34805 loss)
I0331 01:52:40.779956  2693 sgd_solver.cpp:138] Iteration 82100, lr = 0.0005
I0331 01:54:53.895045  2693 solver.cpp:243] Iteration 82200, loss = 4.03869
I0331 01:54:53.895257  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.18046 (* 1 = 5.18046 loss)
I0331 01:54:53.895277  2693 sgd_solver.cpp:138] Iteration 82200, lr = 0.0005
I0331 01:57:03.961439  2693 solver.cpp:243] Iteration 82300, loss = 4.04102
I0331 01:57:03.961706  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.29238 (* 1 = 3.29238 loss)
I0331 01:57:03.961733  2693 sgd_solver.cpp:138] Iteration 82300, lr = 0.0005
I0331 01:59:15.592123  2693 solver.cpp:243] Iteration 82400, loss = 4.10807
I0331 01:59:15.592401  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.5343 (* 1 = 3.5343 loss)
I0331 01:59:15.592442  2693 sgd_solver.cpp:138] Iteration 82400, lr = 0.0005
I0331 02:01:23.962605  2693 solver.cpp:243] Iteration 82500, loss = 3.91141
I0331 02:01:23.962793  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.47835 (* 1 = 4.47835 loss)
I0331 02:01:23.962811  2693 sgd_solver.cpp:138] Iteration 82500, lr = 0.0005
I0331 02:03:34.100383  2693 solver.cpp:243] Iteration 82600, loss = 4.05433
I0331 02:03:34.102210  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.21041 (* 1 = 4.21041 loss)
I0331 02:03:34.102246  2693 sgd_solver.cpp:138] Iteration 82600, lr = 0.0005
I0331 02:05:44.944298  2693 solver.cpp:243] Iteration 82700, loss = 4.04561
I0331 02:05:44.944571  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.83079 (* 1 = 4.83079 loss)
I0331 02:05:44.944610  2693 sgd_solver.cpp:138] Iteration 82700, lr = 0.0005
I0331 02:07:55.964009  2693 solver.cpp:243] Iteration 82800, loss = 4.04975
I0331 02:07:55.964217  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.62634 (* 1 = 4.62634 loss)
I0331 02:07:55.964237  2693 sgd_solver.cpp:138] Iteration 82800, lr = 0.0005
I0331 02:10:05.657117  2693 solver.cpp:243] Iteration 82900, loss = 3.78931
I0331 02:10:05.657335  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.76884 (* 1 = 4.76884 loss)
I0331 02:10:05.657356  2693 sgd_solver.cpp:138] Iteration 82900, lr = 0.0005
I0331 02:12:16.340466  2693 solver.cpp:243] Iteration 83000, loss = 3.86817
I0331 02:12:16.340718  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.07039 (* 1 = 5.07039 loss)
I0331 02:12:16.340749  2693 sgd_solver.cpp:138] Iteration 83000, lr = 0.0005
I0331 02:14:25.773948  2693 solver.cpp:243] Iteration 83100, loss = 3.88086
I0331 02:14:25.774185  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.87234 (* 1 = 2.87234 loss)
I0331 02:14:25.774207  2693 sgd_solver.cpp:138] Iteration 83100, lr = 0.0005
I0331 02:16:36.002485  2693 solver.cpp:243] Iteration 83200, loss = 3.91735
I0331 02:16:36.002660  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.65968 (* 1 = 4.65968 loss)
I0331 02:16:36.002676  2693 sgd_solver.cpp:138] Iteration 83200, lr = 0.0005
I0331 02:18:47.433101  2693 solver.cpp:243] Iteration 83300, loss = 4.07915
I0331 02:18:47.433367  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.82199 (* 1 = 3.82199 loss)
I0331 02:18:47.433413  2693 sgd_solver.cpp:138] Iteration 83300, lr = 0.0005
I0331 02:20:58.482128  2693 solver.cpp:243] Iteration 83400, loss = 4.05405
I0331 02:20:58.482412  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.21373 (* 1 = 4.21373 loss)
I0331 02:20:58.482446  2693 sgd_solver.cpp:138] Iteration 83400, lr = 0.0005
I0331 02:23:09.386389  2693 solver.cpp:243] Iteration 83500, loss = 4.14569
I0331 02:23:09.386615  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.22959 (* 1 = 4.22959 loss)
I0331 02:23:09.386636  2693 sgd_solver.cpp:138] Iteration 83500, lr = 0.0005
I0331 02:25:19.152122  2693 solver.cpp:243] Iteration 83600, loss = 3.97928
I0331 02:25:19.152384  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.86749 (* 1 = 3.86749 loss)
I0331 02:25:19.152410  2693 sgd_solver.cpp:138] Iteration 83600, lr = 0.0005
I0331 02:27:29.471324  2693 solver.cpp:243] Iteration 83700, loss = 3.89324
I0331 02:27:29.471725  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.95698 (* 1 = 3.95698 loss)
I0331 02:27:29.471792  2693 sgd_solver.cpp:138] Iteration 83700, lr = 0.0005
I0331 02:29:38.059032  2693 solver.cpp:243] Iteration 83800, loss = 4.08837
I0331 02:29:38.059320  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.72506 (* 1 = 3.72506 loss)
I0331 02:29:38.059355  2693 sgd_solver.cpp:138] Iteration 83800, lr = 0.0005
I0331 02:31:47.688843  2693 solver.cpp:243] Iteration 83900, loss = 4.12954
I0331 02:31:47.689108  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.56009 (* 1 = 4.56009 loss)
I0331 02:31:47.689154  2693 sgd_solver.cpp:138] Iteration 83900, lr = 0.0005
I0331 02:33:57.260967  2693 solver.cpp:243] Iteration 84000, loss = 4.04483
I0331 02:33:57.261283  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.01727 (* 1 = 4.01727 loss)
I0331 02:33:57.261337  2693 sgd_solver.cpp:138] Iteration 84000, lr = 0.0005
I0331 02:36:07.600896  2693 solver.cpp:243] Iteration 84100, loss = 4.20848
I0331 02:36:07.601132  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.11571 (* 1 = 4.11571 loss)
I0331 02:36:07.601151  2693 sgd_solver.cpp:138] Iteration 84100, lr = 0.0005
I0331 02:38:17.136492  2693 solver.cpp:243] Iteration 84200, loss = 3.91299
I0331 02:38:17.136699  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.34295 (* 1 = 3.34295 loss)
I0331 02:38:17.136718  2693 sgd_solver.cpp:138] Iteration 84200, lr = 0.0005
I0331 02:40:28.216861  2693 solver.cpp:243] Iteration 84300, loss = 3.92001
I0331 02:40:28.217108  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.00077 (* 1 = 4.00077 loss)
I0331 02:40:28.217139  2693 sgd_solver.cpp:138] Iteration 84300, lr = 0.0005
I0331 02:42:37.955067  2693 solver.cpp:243] Iteration 84400, loss = 3.98424
I0331 02:42:37.955265  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.13589 (* 1 = 5.13589 loss)
I0331 02:42:37.955281  2693 sgd_solver.cpp:138] Iteration 84400, lr = 0.0005
I0331 02:44:47.605237  2693 solver.cpp:243] Iteration 84500, loss = 3.93065
I0331 02:44:47.605509  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.30157 (* 1 = 4.30157 loss)
I0331 02:44:47.605556  2693 sgd_solver.cpp:138] Iteration 84500, lr = 0.0005
I0331 02:46:56.890549  2693 solver.cpp:243] Iteration 84600, loss = 3.95816
I0331 02:46:56.898200  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.18284 (* 1 = 4.18284 loss)
I0331 02:46:56.898234  2693 sgd_solver.cpp:138] Iteration 84600, lr = 0.0005
I0331 02:49:07.760687  2693 solver.cpp:243] Iteration 84700, loss = 4.00019
I0331 02:49:07.760885  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.06544 (* 1 = 4.06544 loss)
I0331 02:49:07.760903  2693 sgd_solver.cpp:138] Iteration 84700, lr = 0.0005
I0331 02:51:18.970216  2693 solver.cpp:243] Iteration 84800, loss = 4.07953
I0331 02:51:18.970412  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.81329 (* 1 = 3.81329 loss)
I0331 02:51:18.970429  2693 sgd_solver.cpp:138] Iteration 84800, lr = 0.0005
I0331 02:53:30.096420  2693 solver.cpp:243] Iteration 84900, loss = 3.98171
I0331 02:53:30.096601  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.9707 (* 1 = 3.9707 loss)
I0331 02:53:30.096619  2693 sgd_solver.cpp:138] Iteration 84900, lr = 0.0005
I0331 02:55:37.249778  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_85000.caffemodel
I0331 02:55:38.185655  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_85000.solverstate
I0331 02:55:38.297138  2693 solver.cpp:433] Iteration 85000, Testing net (#0)
I0331 02:55:38.297220  2693 net.cpp:693] Ignoring source layer mbox_loss
I0331 02:56:58.368641  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.57987
I0331 02:56:59.150867  2693 solver.cpp:243] Iteration 85000, loss = 3.9902
I0331 02:56:59.150938  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.62608 (* 1 = 4.62608 loss)
I0331 02:56:59.150954  2693 sgd_solver.cpp:138] Iteration 85000, lr = 0.0005
I0331 02:59:08.275032  2693 solver.cpp:243] Iteration 85100, loss = 3.96402
I0331 02:59:08.275302  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.41657 (* 1 = 3.41657 loss)
I0331 02:59:08.275339  2693 sgd_solver.cpp:138] Iteration 85100, lr = 0.0005
I0331 03:01:18.164577  2693 solver.cpp:243] Iteration 85200, loss = 4.00365
I0331 03:01:18.164757  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.67395 (* 1 = 3.67395 loss)
I0331 03:01:18.164774  2693 sgd_solver.cpp:138] Iteration 85200, lr = 0.0005
I0331 03:03:28.014961  2693 solver.cpp:243] Iteration 85300, loss = 3.88494
I0331 03:03:28.019700  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.8672 (* 1 = 3.8672 loss)
I0331 03:03:28.019722  2693 sgd_solver.cpp:138] Iteration 85300, lr = 0.0005
I0331 03:05:36.601737  2693 solver.cpp:243] Iteration 85400, loss = 3.99039
I0331 03:05:36.602010  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.43706 (* 1 = 3.43706 loss)
I0331 03:05:36.602053  2693 sgd_solver.cpp:138] Iteration 85400, lr = 0.0005
I0331 03:07:47.314836  2693 solver.cpp:243] Iteration 85500, loss = 4.01451
I0331 03:07:47.315124  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.62425 (* 1 = 3.62425 loss)
I0331 03:07:47.315165  2693 sgd_solver.cpp:138] Iteration 85500, lr = 0.0005
I0331 03:09:58.476384  2693 solver.cpp:243] Iteration 85600, loss = 4.14238
I0331 03:09:58.476634  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.23502 (* 1 = 3.23502 loss)
I0331 03:09:58.476680  2693 sgd_solver.cpp:138] Iteration 85600, lr = 0.0005
I0331 03:12:07.802680  2693 solver.cpp:243] Iteration 85700, loss = 4.07258
I0331 03:12:07.802922  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.56889 (* 1 = 2.56889 loss)
I0331 03:12:07.802964  2693 sgd_solver.cpp:138] Iteration 85700, lr = 0.0005
I0331 03:14:18.907469  2693 solver.cpp:243] Iteration 85800, loss = 4.0581
I0331 03:14:18.907830  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.99179 (* 1 = 3.99179 loss)
I0331 03:14:18.907891  2693 sgd_solver.cpp:138] Iteration 85800, lr = 0.0005
I0331 03:16:28.350539  2693 solver.cpp:243] Iteration 85900, loss = 4.2019
I0331 03:16:28.350796  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.82776 (* 1 = 4.82776 loss)
I0331 03:16:28.350841  2693 sgd_solver.cpp:138] Iteration 85900, lr = 0.0005
I0331 03:18:38.952857  2693 solver.cpp:243] Iteration 86000, loss = 4.07945
I0331 03:18:38.953058  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.56334 (* 1 = 4.56334 loss)
I0331 03:18:38.953075  2693 sgd_solver.cpp:138] Iteration 86000, lr = 0.0005
I0331 03:20:48.918328  2693 solver.cpp:243] Iteration 86100, loss = 4.07299
I0331 03:20:48.918622  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.84664 (* 1 = 3.84664 loss)
I0331 03:20:48.918675  2693 sgd_solver.cpp:138] Iteration 86100, lr = 0.0005
I0331 03:22:57.707146  2693 solver.cpp:243] Iteration 86200, loss = 3.95378
I0331 03:22:57.707388  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.78907 (* 1 = 3.78907 loss)
I0331 03:22:57.707412  2693 sgd_solver.cpp:138] Iteration 86200, lr = 0.0005
I0331 03:25:09.053165  2693 solver.cpp:243] Iteration 86300, loss = 4.16076
I0331 03:25:09.055526  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.90441 (* 1 = 3.90441 loss)
I0331 03:25:09.055544  2693 sgd_solver.cpp:138] Iteration 86300, lr = 0.0005
I0331 03:27:17.485708  2693 solver.cpp:243] Iteration 86400, loss = 3.96039
I0331 03:27:17.491963  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.60337 (* 1 = 2.60337 loss)
I0331 03:27:17.492025  2693 sgd_solver.cpp:138] Iteration 86400, lr = 0.0005
I0331 03:29:26.853700  2693 solver.cpp:243] Iteration 86500, loss = 4.04556
I0331 03:29:26.853996  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.02207 (* 1 = 4.02207 loss)
I0331 03:29:26.854048  2693 sgd_solver.cpp:138] Iteration 86500, lr = 0.0005
I0331 03:31:37.820027  2693 solver.cpp:243] Iteration 86600, loss = 3.97182
I0331 03:31:37.820242  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.1915 (* 1 = 3.1915 loss)
I0331 03:31:37.820263  2693 sgd_solver.cpp:138] Iteration 86600, lr = 0.0005
I0331 03:33:48.573709  2693 solver.cpp:243] Iteration 86700, loss = 3.92004
I0331 03:33:48.573890  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.90296 (* 1 = 2.90296 loss)
I0331 03:33:48.573909  2693 sgd_solver.cpp:138] Iteration 86700, lr = 0.0005
I0331 03:35:56.668126  2693 solver.cpp:243] Iteration 86800, loss = 3.88441
I0331 03:35:56.668346  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.77291 (* 1 = 3.77291 loss)
I0331 03:35:56.668375  2693 sgd_solver.cpp:138] Iteration 86800, lr = 0.0005
I0331 03:38:07.240442  2693 solver.cpp:243] Iteration 86900, loss = 4.07223
I0331 03:38:07.240713  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.28003 (* 1 = 4.28003 loss)
I0331 03:38:07.240741  2693 sgd_solver.cpp:138] Iteration 86900, lr = 0.0005
I0331 03:40:18.061197  2693 solver.cpp:243] Iteration 87000, loss = 4.05822
I0331 03:40:18.061470  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.30015 (* 1 = 5.30015 loss)
I0331 03:40:18.061488  2693 sgd_solver.cpp:138] Iteration 87000, lr = 0.0005
I0331 03:42:26.882864  2693 solver.cpp:243] Iteration 87100, loss = 4.03354
I0331 03:42:26.883080  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.18319 (* 1 = 4.18319 loss)
I0331 03:42:26.883105  2693 sgd_solver.cpp:138] Iteration 87100, lr = 0.0005
I0331 03:44:35.689524  2693 solver.cpp:243] Iteration 87200, loss = 4.04086
I0331 03:44:35.689741  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.30153 (* 1 = 4.30153 loss)
I0331 03:44:35.689760  2693 sgd_solver.cpp:138] Iteration 87200, lr = 0.0005
I0331 03:46:45.755766  2693 solver.cpp:243] Iteration 87300, loss = 4.09874
I0331 03:46:45.756024  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.25219 (* 1 = 4.25219 loss)
I0331 03:46:45.756053  2693 sgd_solver.cpp:138] Iteration 87300, lr = 0.0005
I0331 03:48:58.249761  2693 solver.cpp:243] Iteration 87400, loss = 3.96131
I0331 03:48:58.249934  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.24448 (* 1 = 4.24448 loss)
I0331 03:48:58.249948  2693 sgd_solver.cpp:138] Iteration 87400, lr = 0.0005
I0331 03:51:08.246984  2693 solver.cpp:243] Iteration 87500, loss = 3.91469
I0331 03:51:08.254240  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.73625 (* 1 = 3.73625 loss)
I0331 03:51:08.254305  2693 sgd_solver.cpp:138] Iteration 87500, lr = 0.0005
I0331 03:53:17.676085  2693 solver.cpp:243] Iteration 87600, loss = 3.83235
I0331 03:53:17.676339  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.1103 (* 1 = 4.1103 loss)
I0331 03:53:17.676373  2693 sgd_solver.cpp:138] Iteration 87600, lr = 0.0005
I0331 03:55:28.926208  2693 solver.cpp:243] Iteration 87700, loss = 4.08647
I0331 03:55:28.926489  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.34301 (* 1 = 4.34301 loss)
I0331 03:55:28.926523  2693 sgd_solver.cpp:138] Iteration 87700, lr = 0.0005
I0331 03:57:40.358208  2693 solver.cpp:243] Iteration 87800, loss = 4.01601
I0331 03:57:40.358449  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.37082 (* 1 = 3.37082 loss)
I0331 03:57:40.358484  2693 sgd_solver.cpp:138] Iteration 87800, lr = 0.0005
I0331 03:59:52.124229  2693 solver.cpp:243] Iteration 87900, loss = 4.02282
I0331 03:59:52.125536  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.62592 (* 1 = 2.62592 loss)
I0331 03:59:52.125556  2693 sgd_solver.cpp:138] Iteration 87900, lr = 0.0005
I0331 04:02:03.112607  2693 solver.cpp:243] Iteration 88000, loss = 4.03584
I0331 04:02:03.112824  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.85888 (* 1 = 3.85888 loss)
I0331 04:02:03.112843  2693 sgd_solver.cpp:138] Iteration 88000, lr = 0.0005
I0331 04:04:12.547118  2693 solver.cpp:243] Iteration 88100, loss = 3.97646
I0331 04:04:12.547313  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.24345 (* 1 = 5.24345 loss)
I0331 04:04:12.547333  2693 sgd_solver.cpp:138] Iteration 88100, lr = 0.0005
I0331 04:06:23.861330  2693 solver.cpp:243] Iteration 88200, loss = 3.91441
I0331 04:06:23.861568  2693 solver.cpp:259]     Train net output #0: mbox_loss = 1.89898 (* 1 = 1.89898 loss)
I0331 04:06:23.861593  2693 sgd_solver.cpp:138] Iteration 88200, lr = 0.0005
I0331 04:08:34.612280  2693 solver.cpp:243] Iteration 88300, loss = 3.89868
I0331 04:08:34.612493  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.59196 (* 1 = 3.59196 loss)
I0331 04:08:34.612509  2693 sgd_solver.cpp:138] Iteration 88300, lr = 0.0005
I0331 04:10:45.421165  2693 solver.cpp:243] Iteration 88400, loss = 4.09897
I0331 04:10:45.421360  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.58492 (* 1 = 3.58492 loss)
I0331 04:10:45.421378  2693 sgd_solver.cpp:138] Iteration 88400, lr = 0.0005
I0331 04:12:55.181829  2693 solver.cpp:243] Iteration 88500, loss = 3.99062
I0331 04:12:55.182054  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.77402 (* 1 = 3.77402 loss)
I0331 04:12:55.182072  2693 sgd_solver.cpp:138] Iteration 88500, lr = 0.0005
I0331 04:15:04.453443  2693 solver.cpp:243] Iteration 88600, loss = 3.89682
I0331 04:15:04.453773  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.10968 (* 1 = 3.10968 loss)
I0331 04:15:04.453816  2693 sgd_solver.cpp:138] Iteration 88600, lr = 0.0005
I0331 04:17:14.012028  2693 solver.cpp:243] Iteration 88700, loss = 3.89511
I0331 04:17:14.012269  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.15611 (* 1 = 3.15611 loss)
I0331 04:17:14.012303  2693 sgd_solver.cpp:138] Iteration 88700, lr = 0.0005
I0331 04:19:23.213173  2693 solver.cpp:243] Iteration 88800, loss = 3.95525
I0331 04:19:23.213467  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.28261 (* 1 = 4.28261 loss)
I0331 04:19:23.213529  2693 sgd_solver.cpp:138] Iteration 88800, lr = 0.0005
I0331 04:21:35.319226  2693 solver.cpp:243] Iteration 88900, loss = 4.0032
I0331 04:21:35.319442  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.52 (* 1 = 3.52 loss)
I0331 04:21:35.319461  2693 sgd_solver.cpp:138] Iteration 88900, lr = 0.0005
I0331 04:23:45.915791  2693 solver.cpp:243] Iteration 89000, loss = 4.08578
I0331 04:23:45.923461  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.52743 (* 1 = 3.52743 loss)
I0331 04:23:45.923481  2693 sgd_solver.cpp:138] Iteration 89000, lr = 0.0005
I0331 04:25:58.908156  2693 solver.cpp:243] Iteration 89100, loss = 4.09215
I0331 04:25:58.908344  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.34283 (* 1 = 4.34283 loss)
I0331 04:25:58.908361  2693 sgd_solver.cpp:138] Iteration 89100, lr = 0.0005
I0331 04:28:08.741999  2693 solver.cpp:243] Iteration 89200, loss = 3.84707
I0331 04:28:08.742254  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.49191 (* 1 = 4.49191 loss)
I0331 04:28:08.742300  2693 sgd_solver.cpp:138] Iteration 89200, lr = 0.0005
I0331 04:30:17.869590  2693 solver.cpp:243] Iteration 89300, loss = 4.05224
I0331 04:30:17.871762  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.75434 (* 1 = 3.75434 loss)
I0331 04:30:17.871809  2693 sgd_solver.cpp:138] Iteration 89300, lr = 0.0005
I0331 04:32:27.965474  2693 solver.cpp:243] Iteration 89400, loss = 4.0255
I0331 04:32:27.965697  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.0252 (* 1 = 5.0252 loss)
I0331 04:32:27.965716  2693 sgd_solver.cpp:138] Iteration 89400, lr = 0.0005
I0331 04:34:38.873039  2693 solver.cpp:243] Iteration 89500, loss = 4.07976
I0331 04:34:38.873242  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.57321 (* 1 = 3.57321 loss)
I0331 04:34:38.873260  2693 sgd_solver.cpp:138] Iteration 89500, lr = 0.0005
I0331 04:36:48.861578  2693 solver.cpp:243] Iteration 89600, loss = 4.10178
I0331 04:36:48.861827  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.22501 (* 1 = 3.22501 loss)
I0331 04:36:48.861876  2693 sgd_solver.cpp:138] Iteration 89600, lr = 0.0005
I0331 04:38:59.889523  2693 solver.cpp:243] Iteration 89700, loss = 4.16697
I0331 04:38:59.889717  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.76737 (* 1 = 3.76737 loss)
I0331 04:38:59.889735  2693 sgd_solver.cpp:138] Iteration 89700, lr = 0.0005
I0331 04:41:10.166930  2693 solver.cpp:243] Iteration 89800, loss = 4.07614
I0331 04:41:10.167213  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.2769 (* 1 = 4.2769 loss)
I0331 04:41:10.167268  2693 sgd_solver.cpp:138] Iteration 89800, lr = 0.0005
I0331 04:43:18.649837  2693 solver.cpp:243] Iteration 89900, loss = 3.84439
I0331 04:43:18.653393  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.49411 (* 1 = 3.49411 loss)
I0331 04:43:18.653426  2693 sgd_solver.cpp:138] Iteration 89900, lr = 0.0005
I0331 04:45:28.560453  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_90000.caffemodel
I0331 04:45:29.515727  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_90000.solverstate
I0331 04:45:29.628182  2693 solver.cpp:433] Iteration 90000, Testing net (#0)
I0331 04:45:29.628260  2693 net.cpp:693] Ignoring source layer mbox_loss
I0331 04:46:49.647425  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.558829
I0331 04:46:50.249402  2693 solver.cpp:243] Iteration 90000, loss = 3.99797
I0331 04:46:50.249469  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.44445 (* 1 = 4.44445 loss)
I0331 04:46:50.249483  2693 sgd_solver.cpp:138] Iteration 90000, lr = 0.0005
I0331 04:48:59.535100  2693 solver.cpp:243] Iteration 90100, loss = 3.84016
I0331 04:48:59.535378  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.38586 (* 1 = 3.38586 loss)
I0331 04:48:59.535398  2693 sgd_solver.cpp:138] Iteration 90100, lr = 0.0005
I0331 04:51:08.284926  2693 solver.cpp:243] Iteration 90200, loss = 4.01463
I0331 04:51:08.285174  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.97004 (* 1 = 3.97004 loss)
I0331 04:51:08.285195  2693 sgd_solver.cpp:138] Iteration 90200, lr = 0.0005
I0331 04:53:16.952662  2693 solver.cpp:243] Iteration 90300, loss = 3.96426
I0331 04:53:16.952862  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.7921 (* 1 = 3.7921 loss)
I0331 04:53:16.952893  2693 sgd_solver.cpp:138] Iteration 90300, lr = 0.0005
I0331 04:55:28.031062  2693 solver.cpp:243] Iteration 90400, loss = 4.22157
I0331 04:55:28.031301  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.77551 (* 1 = 3.77551 loss)
I0331 04:55:28.031334  2693 sgd_solver.cpp:138] Iteration 90400, lr = 0.0005
I0331 04:57:37.775696  2693 solver.cpp:243] Iteration 90500, loss = 4.09036
I0331 04:57:37.775952  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.42998 (* 1 = 4.42998 loss)
I0331 04:57:37.775985  2693 sgd_solver.cpp:138] Iteration 90500, lr = 0.0005
I0331 04:59:48.899710  2693 solver.cpp:243] Iteration 90600, loss = 3.94662
I0331 04:59:48.899893  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.52388 (* 1 = 3.52388 loss)
I0331 04:59:48.899910  2693 sgd_solver.cpp:138] Iteration 90600, lr = 0.0005
I0331 05:01:56.547487  2693 solver.cpp:243] Iteration 90700, loss = 4.01216
I0331 05:01:56.547916  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.3933 (* 1 = 3.3933 loss)
I0331 05:01:56.547979  2693 sgd_solver.cpp:138] Iteration 90700, lr = 0.0005
I0331 05:04:06.371659  2693 solver.cpp:243] Iteration 90800, loss = 3.88418
I0331 05:04:06.376818  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.22356 (* 1 = 4.22356 loss)
I0331 05:04:06.376839  2693 sgd_solver.cpp:138] Iteration 90800, lr = 0.0005
I0331 05:06:15.873392  2693 solver.cpp:243] Iteration 90900, loss = 3.97088
I0331 05:06:15.873603  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.72316 (* 1 = 2.72316 loss)
I0331 05:06:15.873625  2693 sgd_solver.cpp:138] Iteration 90900, lr = 0.0005
I0331 05:08:27.165808  2693 solver.cpp:243] Iteration 91000, loss = 4.01552
I0331 05:08:27.166028  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.62345 (* 1 = 4.62345 loss)
I0331 05:08:27.166049  2693 sgd_solver.cpp:138] Iteration 91000, lr = 0.0005
I0331 05:10:35.682695  2693 solver.cpp:243] Iteration 91100, loss = 3.93741
I0331 05:10:35.682901  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.95673 (* 1 = 4.95673 loss)
I0331 05:10:35.682919  2693 sgd_solver.cpp:138] Iteration 91100, lr = 0.0005
I0331 05:12:46.915666  2693 solver.cpp:243] Iteration 91200, loss = 4.02277
I0331 05:12:46.915882  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.47055 (* 1 = 3.47055 loss)
I0331 05:12:46.915902  2693 sgd_solver.cpp:138] Iteration 91200, lr = 0.0005
I0331 05:14:57.452868  2693 solver.cpp:243] Iteration 91300, loss = 4.04168
I0331 05:14:57.453078  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.99591 (* 1 = 4.99591 loss)
I0331 05:14:57.453097  2693 sgd_solver.cpp:138] Iteration 91300, lr = 0.0005
I0331 05:17:07.027034  2693 solver.cpp:243] Iteration 91400, loss = 3.87827
I0331 05:17:07.027220  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.22655 (* 1 = 3.22655 loss)
I0331 05:17:07.027236  2693 sgd_solver.cpp:138] Iteration 91400, lr = 0.0005
I0331 05:19:18.011523  2693 solver.cpp:243] Iteration 91500, loss = 4.0114
I0331 05:19:18.017129  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.92911 (* 1 = 4.92911 loss)
I0331 05:19:18.017164  2693 sgd_solver.cpp:138] Iteration 91500, lr = 0.0005
I0331 05:21:27.423010  2693 solver.cpp:243] Iteration 91600, loss = 3.86338
I0331 05:21:27.423236  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.73488 (* 1 = 3.73488 loss)
I0331 05:21:27.423280  2693 sgd_solver.cpp:138] Iteration 91600, lr = 0.0005
I0331 05:23:37.348539  2693 solver.cpp:243] Iteration 91700, loss = 3.97912
I0331 05:23:37.348857  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.76952 (* 1 = 3.76952 loss)
I0331 05:23:37.348917  2693 sgd_solver.cpp:138] Iteration 91700, lr = 0.0005
I0331 05:25:47.872941  2693 solver.cpp:243] Iteration 91800, loss = 3.94649
I0331 05:25:47.873117  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.14688 (* 1 = 4.14688 loss)
I0331 05:25:47.873134  2693 sgd_solver.cpp:138] Iteration 91800, lr = 0.0005
I0331 05:27:55.383229  2693 solver.cpp:243] Iteration 91900, loss = 3.95613
I0331 05:27:55.383493  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.37641 (* 1 = 3.37641 loss)
I0331 05:27:55.383514  2693 sgd_solver.cpp:138] Iteration 91900, lr = 0.0005
I0331 05:30:05.070513  2693 solver.cpp:243] Iteration 92000, loss = 4.05885
I0331 05:30:05.070691  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.71304 (* 1 = 3.71304 loss)
I0331 05:30:05.070710  2693 sgd_solver.cpp:138] Iteration 92000, lr = 0.0005
I0331 05:32:14.256774  2693 solver.cpp:243] Iteration 92100, loss = 4.02113
I0331 05:32:14.256955  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.34053 (* 1 = 4.34053 loss)
I0331 05:32:14.256974  2693 sgd_solver.cpp:138] Iteration 92100, lr = 0.0005
I0331 05:34:23.790637  2693 solver.cpp:243] Iteration 92200, loss = 4.04155
I0331 05:34:23.790828  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.24349 (* 1 = 3.24349 loss)
I0331 05:34:23.790845  2693 sgd_solver.cpp:138] Iteration 92200, lr = 0.0005
I0331 05:36:34.949996  2693 solver.cpp:243] Iteration 92300, loss = 3.93913
I0331 05:36:34.950189  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.3018 (* 1 = 3.3018 loss)
I0331 05:36:34.950206  2693 sgd_solver.cpp:138] Iteration 92300, lr = 0.0005
I0331 05:38:46.257650  2693 solver.cpp:243] Iteration 92400, loss = 3.95035
I0331 05:38:46.257865  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.88394 (* 1 = 3.88394 loss)
I0331 05:38:46.257882  2693 sgd_solver.cpp:138] Iteration 92400, lr = 0.0005
I0331 05:40:55.128911  2693 solver.cpp:243] Iteration 92500, loss = 3.96056
I0331 05:40:55.129098  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.88988 (* 1 = 3.88988 loss)
I0331 05:40:55.129118  2693 sgd_solver.cpp:138] Iteration 92500, lr = 0.0005
I0331 05:43:03.957919  2693 solver.cpp:243] Iteration 92600, loss = 3.88472
I0331 05:43:03.958145  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.05866 (* 1 = 4.05866 loss)
I0331 05:43:03.958165  2693 sgd_solver.cpp:138] Iteration 92600, lr = 0.0005
I0331 05:45:14.588379  2693 solver.cpp:243] Iteration 92700, loss = 4.01195
I0331 05:45:14.588670  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.59565 (* 1 = 3.59565 loss)
I0331 05:45:14.588690  2693 sgd_solver.cpp:138] Iteration 92700, lr = 0.0005
I0331 05:47:23.253391  2693 solver.cpp:243] Iteration 92800, loss = 3.82568
I0331 05:47:23.253607  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.43678 (* 1 = 3.43678 loss)
I0331 05:47:23.253626  2693 sgd_solver.cpp:138] Iteration 92800, lr = 0.0005
I0331 05:49:34.741045  2693 solver.cpp:243] Iteration 92900, loss = 4.11777
I0331 05:49:34.741219  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.59671 (* 1 = 3.59671 loss)
I0331 05:49:34.741236  2693 sgd_solver.cpp:138] Iteration 92900, lr = 0.0005
I0331 05:51:44.219055  2693 solver.cpp:243] Iteration 93000, loss = 3.92096
I0331 05:51:44.219245  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.045 (* 1 = 3.045 loss)
I0331 05:51:44.219264  2693 sgd_solver.cpp:138] Iteration 93000, lr = 0.0005
I0331 05:53:55.748055  2693 solver.cpp:243] Iteration 93100, loss = 3.98
I0331 05:53:55.748311  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.81741 (* 1 = 4.81741 loss)
I0331 05:53:55.748329  2693 sgd_solver.cpp:138] Iteration 93100, lr = 0.0005
I0331 05:56:05.198293  2693 solver.cpp:243] Iteration 93200, loss = 4.04415
I0331 05:56:05.198503  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.08711 (* 1 = 5.08711 loss)
I0331 05:56:05.198523  2693 sgd_solver.cpp:138] Iteration 93200, lr = 0.0005
I0331 05:58:15.353363  2693 solver.cpp:243] Iteration 93300, loss = 4.00249
I0331 05:58:15.353569  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.03823 (* 1 = 4.03823 loss)
I0331 05:58:15.353588  2693 sgd_solver.cpp:138] Iteration 93300, lr = 0.0005
I0331 06:00:25.814604  2693 solver.cpp:243] Iteration 93400, loss = 3.99345
I0331 06:00:25.814779  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.87456 (* 1 = 3.87456 loss)
I0331 06:00:25.814797  2693 sgd_solver.cpp:138] Iteration 93400, lr = 0.0005
I0331 06:02:37.703827  2693 solver.cpp:243] Iteration 93500, loss = 4.006
I0331 06:02:37.711282  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.27419 (* 1 = 5.27419 loss)
I0331 06:02:37.711300  2693 sgd_solver.cpp:138] Iteration 93500, lr = 0.0005
I0331 06:04:47.713636  2693 solver.cpp:243] Iteration 93600, loss = 4.02633
I0331 06:04:47.713851  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.43671 (* 1 = 4.43671 loss)
I0331 06:04:47.713871  2693 sgd_solver.cpp:138] Iteration 93600, lr = 0.0005
I0331 06:07:00.152758  2693 solver.cpp:243] Iteration 93700, loss = 4.04055
I0331 06:07:00.152947  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.99382 (* 1 = 3.99382 loss)
I0331 06:07:00.152966  2693 sgd_solver.cpp:138] Iteration 93700, lr = 0.0005
I0331 06:09:09.244932  2693 solver.cpp:243] Iteration 93800, loss = 3.90609
I0331 06:09:09.245193  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.46881 (* 1 = 3.46881 loss)
I0331 06:09:09.245229  2693 sgd_solver.cpp:138] Iteration 93800, lr = 0.0005
I0331 06:11:19.612010  2693 solver.cpp:243] Iteration 93900, loss = 3.99065
I0331 06:11:19.612356  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.58624 (* 1 = 4.58624 loss)
I0331 06:11:19.612411  2693 sgd_solver.cpp:138] Iteration 93900, lr = 0.0005
I0331 06:13:29.458863  2693 solver.cpp:243] Iteration 94000, loss = 3.91622
I0331 06:13:29.471679  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.01793 (* 1 = 4.01793 loss)
I0331 06:13:29.471701  2693 sgd_solver.cpp:138] Iteration 94000, lr = 0.0005
I0331 06:15:39.388428  2693 solver.cpp:243] Iteration 94100, loss = 3.89391
I0331 06:15:39.388608  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.00466 (* 1 = 3.00466 loss)
I0331 06:15:39.388624  2693 sgd_solver.cpp:138] Iteration 94100, lr = 0.0005
I0331 06:17:49.417789  2693 solver.cpp:243] Iteration 94200, loss = 3.89893
I0331 06:17:49.418046  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.27916 (* 1 = 3.27916 loss)
I0331 06:17:49.418082  2693 sgd_solver.cpp:138] Iteration 94200, lr = 0.0005
I0331 06:19:59.741758  2693 solver.cpp:243] Iteration 94300, loss = 4.04479
I0331 06:19:59.742008  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.41739 (* 1 = 3.41739 loss)
I0331 06:19:59.742040  2693 sgd_solver.cpp:138] Iteration 94300, lr = 0.0005
I0331 06:22:08.294419  2693 solver.cpp:243] Iteration 94400, loss = 3.81008
I0331 06:22:08.294603  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.82259 (* 1 = 5.82259 loss)
I0331 06:22:08.294620  2693 sgd_solver.cpp:138] Iteration 94400, lr = 0.0005
I0331 06:24:18.918462  2693 solver.cpp:243] Iteration 94500, loss = 3.96765
I0331 06:24:18.925873  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.23448 (* 1 = 4.23448 loss)
I0331 06:24:18.925890  2693 sgd_solver.cpp:138] Iteration 94500, lr = 0.0005
I0331 06:26:31.469674  2693 solver.cpp:243] Iteration 94600, loss = 4.03102
I0331 06:26:31.470002  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.63734 (* 1 = 3.63734 loss)
I0331 06:26:31.470029  2693 sgd_solver.cpp:138] Iteration 94600, lr = 0.0005
I0331 06:28:42.166527  2693 solver.cpp:243] Iteration 94700, loss = 4.00058
I0331 06:28:42.166733  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.22684 (* 1 = 4.22684 loss)
I0331 06:28:42.166751  2693 sgd_solver.cpp:138] Iteration 94700, lr = 0.0005
I0331 06:30:53.638161  2693 solver.cpp:243] Iteration 94800, loss = 3.98746
I0331 06:30:53.638360  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.2959 (* 1 = 5.2959 loss)
I0331 06:30:53.638381  2693 sgd_solver.cpp:138] Iteration 94800, lr = 0.0005
I0331 06:33:04.242332  2693 solver.cpp:243] Iteration 94900, loss = 3.96921
I0331 06:33:04.242578  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.88662 (* 1 = 4.88662 loss)
I0331 06:33:04.242599  2693 sgd_solver.cpp:138] Iteration 94900, lr = 0.0005
I0331 06:35:12.536695  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_95000.caffemodel
I0331 06:35:13.847865  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_95000.solverstate
I0331 06:35:13.964355  2693 solver.cpp:433] Iteration 95000, Testing net (#0)
I0331 06:35:13.964439  2693 net.cpp:693] Ignoring source layer mbox_loss
I0331 06:36:34.086813  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.580314
I0331 06:36:34.992560  2693 solver.cpp:243] Iteration 95000, loss = 4.02219
I0331 06:36:34.992630  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.71105 (* 1 = 3.71105 loss)
I0331 06:36:34.992646  2693 sgd_solver.cpp:138] Iteration 95000, lr = 0.0005
I0331 06:38:43.632774  2693 solver.cpp:243] Iteration 95100, loss = 4.02586
I0331 06:38:43.632999  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.53966 (* 1 = 3.53966 loss)
I0331 06:38:43.633021  2693 sgd_solver.cpp:138] Iteration 95100, lr = 0.0005
I0331 06:40:54.788185  2693 solver.cpp:243] Iteration 95200, loss = 3.98176
I0331 06:40:54.788475  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.97916 (* 1 = 2.97916 loss)
I0331 06:40:54.788508  2693 sgd_solver.cpp:138] Iteration 95200, lr = 0.0005
I0331 06:43:04.520236  2693 solver.cpp:243] Iteration 95300, loss = 3.95375
I0331 06:43:04.520473  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.3318 (* 1 = 3.3318 loss)
I0331 06:43:04.520517  2693 sgd_solver.cpp:138] Iteration 95300, lr = 0.0005
I0331 06:45:17.069010  2693 solver.cpp:243] Iteration 95400, loss = 4.02684
I0331 06:45:17.069263  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.3918 (* 1 = 5.3918 loss)
I0331 06:45:17.069301  2693 sgd_solver.cpp:138] Iteration 95400, lr = 0.0005
I0331 06:47:24.801658  2693 solver.cpp:243] Iteration 95500, loss = 3.88911
I0331 06:47:24.801851  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.5213 (* 1 = 2.5213 loss)
I0331 06:47:24.801868  2693 sgd_solver.cpp:138] Iteration 95500, lr = 0.0005
I0331 06:49:35.316272  2693 solver.cpp:243] Iteration 95600, loss = 3.95607
I0331 06:49:35.316498  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.43694 (* 1 = 4.43694 loss)
I0331 06:49:35.316519  2693 sgd_solver.cpp:138] Iteration 95600, lr = 0.0005
I0331 06:51:45.483062  2693 solver.cpp:243] Iteration 95700, loss = 3.90082
I0331 06:51:45.483284  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.45744 (* 1 = 2.45744 loss)
I0331 06:51:45.483302  2693 sgd_solver.cpp:138] Iteration 95700, lr = 0.0005
I0331 06:53:55.121959  2693 solver.cpp:243] Iteration 95800, loss = 3.93487
I0331 06:53:55.122220  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.56379 (* 1 = 3.56379 loss)
I0331 06:53:55.122236  2693 sgd_solver.cpp:138] Iteration 95800, lr = 0.0005
I0331 06:56:02.957159  2693 solver.cpp:243] Iteration 95900, loss = 3.93541
I0331 06:56:02.957351  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.84466 (* 1 = 3.84466 loss)
I0331 06:56:02.957370  2693 sgd_solver.cpp:138] Iteration 95900, lr = 0.0005
I0331 06:58:11.582952  2693 solver.cpp:243] Iteration 96000, loss = 3.97489
I0331 06:58:11.583228  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.21973 (* 1 = 3.21973 loss)
I0331 06:58:11.583250  2693 sgd_solver.cpp:138] Iteration 96000, lr = 0.0005
I0331 07:00:22.397003  2693 solver.cpp:243] Iteration 96100, loss = 3.97474
I0331 07:00:22.397195  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.19129 (* 1 = 4.19129 loss)
I0331 07:00:22.397214  2693 sgd_solver.cpp:138] Iteration 96100, lr = 0.0005
I0331 07:02:31.510090  2693 solver.cpp:243] Iteration 96200, loss = 3.98414
I0331 07:02:31.510268  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.12905 (* 1 = 3.12905 loss)
I0331 07:02:31.510284  2693 sgd_solver.cpp:138] Iteration 96200, lr = 0.0005
I0331 07:04:39.770576  2693 solver.cpp:243] Iteration 96300, loss = 3.86438
I0331 07:04:39.777818  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.71 (* 1 = 3.71 loss)
I0331 07:04:39.777881  2693 sgd_solver.cpp:138] Iteration 96300, lr = 0.0005
I0331 07:06:48.150013  2693 solver.cpp:243] Iteration 96400, loss = 3.92845
I0331 07:06:48.150248  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.62205 (* 1 = 3.62205 loss)
I0331 07:06:48.150279  2693 sgd_solver.cpp:138] Iteration 96400, lr = 0.0005
I0331 07:08:58.952975  2693 solver.cpp:243] Iteration 96500, loss = 3.95582
I0331 07:08:58.953219  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.10434 (* 1 = 5.10434 loss)
I0331 07:08:58.953245  2693 sgd_solver.cpp:138] Iteration 96500, lr = 0.0005
I0331 07:11:07.926713  2693 solver.cpp:243] Iteration 96600, loss = 3.90765
I0331 07:11:07.933331  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.44752 (* 1 = 4.44752 loss)
I0331 07:11:07.933375  2693 sgd_solver.cpp:138] Iteration 96600, lr = 0.0005
I0331 07:13:16.462198  2693 solver.cpp:243] Iteration 96700, loss = 3.91841
I0331 07:13:16.462471  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.05257 (* 1 = 4.05257 loss)
I0331 07:13:16.462515  2693 sgd_solver.cpp:138] Iteration 96700, lr = 0.0005
I0331 07:15:28.336616  2693 solver.cpp:243] Iteration 96800, loss = 4.05637
I0331 07:15:28.336861  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.40279 (* 1 = 3.40279 loss)
I0331 07:15:28.336892  2693 sgd_solver.cpp:138] Iteration 96800, lr = 0.0005
I0331 07:17:38.836206  2693 solver.cpp:243] Iteration 96900, loss = 4.06859
I0331 07:17:38.837919  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.17853 (* 1 = 4.17853 loss)
I0331 07:17:38.837952  2693 sgd_solver.cpp:138] Iteration 96900, lr = 0.0005
I0331 07:19:47.177006  2693 solver.cpp:243] Iteration 97000, loss = 4.00312
I0331 07:19:47.177222  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.02315 (* 1 = 4.02315 loss)
I0331 07:19:47.177253  2693 sgd_solver.cpp:138] Iteration 97000, lr = 0.0005
I0331 07:21:57.548001  2693 solver.cpp:243] Iteration 97100, loss = 4.03439
I0331 07:21:57.566041  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.71966 (* 1 = 5.71966 loss)
I0331 07:21:57.566061  2693 sgd_solver.cpp:138] Iteration 97100, lr = 0.0005
I0331 07:24:09.441932  2693 solver.cpp:243] Iteration 97200, loss = 4.10431
I0331 07:24:09.442178  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.94236 (* 1 = 4.94236 loss)
I0331 07:24:09.442212  2693 sgd_solver.cpp:138] Iteration 97200, lr = 0.0005
I0331 07:26:19.939273  2693 solver.cpp:243] Iteration 97300, loss = 4.12546
I0331 07:26:19.939524  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.74486 (* 1 = 4.74486 loss)
I0331 07:26:19.939558  2693 sgd_solver.cpp:138] Iteration 97300, lr = 0.0005
I0331 07:28:29.587790  2693 solver.cpp:243] Iteration 97400, loss = 4.00568
I0331 07:28:29.588016  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.51541 (* 1 = 3.51541 loss)
I0331 07:28:29.588048  2693 sgd_solver.cpp:138] Iteration 97400, lr = 0.0005
I0331 07:30:38.761349  2693 solver.cpp:243] Iteration 97500, loss = 4.0187
I0331 07:30:38.765014  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.22275 (* 1 = 4.22275 loss)
I0331 07:30:38.765044  2693 sgd_solver.cpp:138] Iteration 97500, lr = 0.0005
I0331 07:32:48.977843  2693 solver.cpp:243] Iteration 97600, loss = 3.94427
I0331 07:32:48.978081  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.00725 (* 1 = 4.00725 loss)
I0331 07:32:48.978121  2693 sgd_solver.cpp:138] Iteration 97600, lr = 0.0005
I0331 07:34:58.234119  2693 solver.cpp:243] Iteration 97700, loss = 3.9326
I0331 07:34:58.234346  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.80246 (* 1 = 2.80246 loss)
I0331 07:34:58.234385  2693 sgd_solver.cpp:138] Iteration 97700, lr = 0.0005
I0331 07:37:09.801405  2693 solver.cpp:243] Iteration 97800, loss = 4.05298
I0331 07:37:09.801590  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.77721 (* 1 = 3.77721 loss)
I0331 07:37:09.801607  2693 sgd_solver.cpp:138] Iteration 97800, lr = 0.0005
I0331 07:39:18.809448  2693 solver.cpp:243] Iteration 97900, loss = 4.05719
I0331 07:39:18.809630  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.67843 (* 1 = 3.67843 loss)
I0331 07:39:18.809648  2693 sgd_solver.cpp:138] Iteration 97900, lr = 0.0005
I0331 07:41:28.612537  2693 solver.cpp:243] Iteration 98000, loss = 3.87184
I0331 07:41:28.612853  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.54615 (* 1 = 3.54615 loss)
I0331 07:41:28.612918  2693 sgd_solver.cpp:138] Iteration 98000, lr = 0.0005
I0331 07:43:37.303378  2693 solver.cpp:243] Iteration 98100, loss = 3.88633
I0331 07:43:37.303654  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.79538 (* 1 = 5.79538 loss)
I0331 07:43:37.303674  2693 sgd_solver.cpp:138] Iteration 98100, lr = 0.0005
I0331 07:45:46.384901  2693 solver.cpp:243] Iteration 98200, loss = 4.04378
I0331 07:45:46.385092  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.02407 (* 1 = 4.02407 loss)
I0331 07:45:46.385110  2693 sgd_solver.cpp:138] Iteration 98200, lr = 0.0005
I0331 07:47:56.762305  2693 solver.cpp:243] Iteration 98300, loss = 3.97347
I0331 07:47:56.762496  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.60665 (* 1 = 2.60665 loss)
I0331 07:47:56.762516  2693 sgd_solver.cpp:138] Iteration 98300, lr = 0.0005
I0331 07:50:06.598960  2693 solver.cpp:243] Iteration 98400, loss = 3.98128
I0331 07:50:06.599164  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.16944 (* 1 = 4.16944 loss)
I0331 07:50:06.599182  2693 sgd_solver.cpp:138] Iteration 98400, lr = 0.0005
I0331 07:52:16.229828  2693 solver.cpp:243] Iteration 98500, loss = 3.93671
I0331 07:52:16.230020  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.63804 (* 1 = 3.63804 loss)
I0331 07:52:16.230037  2693 sgd_solver.cpp:138] Iteration 98500, lr = 0.0005
I0331 07:54:27.363521  2693 solver.cpp:243] Iteration 98600, loss = 4.08112
I0331 07:54:27.363826  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.13018 (* 1 = 3.13018 loss)
I0331 07:54:27.363849  2693 sgd_solver.cpp:138] Iteration 98600, lr = 0.0005
I0331 07:56:38.255434  2693 solver.cpp:243] Iteration 98700, loss = 3.9499
I0331 07:56:38.259687  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.65794 (* 1 = 3.65794 loss)
I0331 07:56:38.259707  2693 sgd_solver.cpp:138] Iteration 98700, lr = 0.0005
I0331 07:58:47.004855  2693 solver.cpp:243] Iteration 98800, loss = 3.89409
I0331 07:58:47.005043  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.84133 (* 1 = 3.84133 loss)
I0331 07:58:47.005060  2693 sgd_solver.cpp:138] Iteration 98800, lr = 0.0005
I0331 08:00:57.180451  2693 solver.cpp:243] Iteration 98900, loss = 3.93179
I0331 08:00:57.180701  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.45225 (* 1 = 3.45225 loss)
I0331 08:00:57.180737  2693 sgd_solver.cpp:138] Iteration 98900, lr = 0.0005
I0331 08:03:08.230190  2693 solver.cpp:243] Iteration 99000, loss = 3.98135
I0331 08:03:08.230438  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.83297 (* 1 = 3.83297 loss)
I0331 08:03:08.230481  2693 sgd_solver.cpp:138] Iteration 99000, lr = 0.0005
I0331 08:05:20.624486  2693 solver.cpp:243] Iteration 99100, loss = 3.98457
I0331 08:05:20.624779  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.42243 (* 1 = 3.42243 loss)
I0331 08:05:20.624802  2693 sgd_solver.cpp:138] Iteration 99100, lr = 0.0005
I0331 08:07:31.314043  2693 solver.cpp:243] Iteration 99200, loss = 3.99247
I0331 08:07:31.314245  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.53956 (* 1 = 5.53956 loss)
I0331 08:07:31.314262  2693 sgd_solver.cpp:138] Iteration 99200, lr = 0.0005
I0331 08:09:43.409770  2693 solver.cpp:243] Iteration 99300, loss = 3.9789
I0331 08:09:43.409998  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.32352 (* 1 = 4.32352 loss)
I0331 08:09:43.410022  2693 sgd_solver.cpp:138] Iteration 99300, lr = 0.0005
I0331 08:11:53.310570  2693 solver.cpp:243] Iteration 99400, loss = 3.83566
I0331 08:11:53.310817  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.91427 (* 1 = 2.91427 loss)
I0331 08:11:53.310842  2693 sgd_solver.cpp:138] Iteration 99400, lr = 0.0005
I0331 08:14:04.041805  2693 solver.cpp:243] Iteration 99500, loss = 3.91756
I0331 08:14:04.042003  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.54536 (* 1 = 3.54536 loss)
I0331 08:14:04.042021  2693 sgd_solver.cpp:138] Iteration 99500, lr = 0.0005
I0331 08:16:12.928522  2693 solver.cpp:243] Iteration 99600, loss = 3.81394
I0331 08:16:12.928768  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.31433 (* 1 = 4.31433 loss)
I0331 08:16:12.928789  2693 sgd_solver.cpp:138] Iteration 99600, lr = 0.0005
I0331 08:18:22.698691  2693 solver.cpp:243] Iteration 99700, loss = 3.95023
I0331 08:18:22.698866  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.9565 (* 1 = 3.9565 loss)
I0331 08:18:22.698884  2693 sgd_solver.cpp:138] Iteration 99700, lr = 0.0005
I0331 08:20:31.109609  2693 solver.cpp:243] Iteration 99800, loss = 3.78169
I0331 08:20:31.109848  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.98314 (* 1 = 4.98314 loss)
I0331 08:20:31.109882  2693 sgd_solver.cpp:138] Iteration 99800, lr = 0.0005
I0331 08:22:41.093140  2693 solver.cpp:243] Iteration 99900, loss = 3.86688
I0331 08:22:41.093327  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.71154 (* 1 = 3.71154 loss)
I0331 08:22:41.093343  2693 sgd_solver.cpp:138] Iteration 99900, lr = 0.0005
I0331 08:24:49.781380  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_100000.caffemodel
I0331 08:24:50.971489  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_100000.solverstate
I0331 08:24:51.088037  2693 solver.cpp:433] Iteration 100000, Testing net (#0)
I0331 08:24:51.088114  2693 net.cpp:693] Ignoring source layer mbox_loss
I0331 08:26:10.410356  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.620337
I0331 08:26:11.042004  2693 solver.cpp:243] Iteration 100000, loss = 3.98061
I0331 08:26:11.042074  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.5322 (* 1 = 3.5322 loss)
I0331 08:26:11.042090  2693 sgd_solver.cpp:138] Iteration 100000, lr = 5e-05
I0331 08:28:20.252781  2693 solver.cpp:243] Iteration 100100, loss = 3.84457
I0331 08:28:20.253013  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.77074 (* 1 = 4.77074 loss)
I0331 08:28:20.253044  2693 sgd_solver.cpp:138] Iteration 100100, lr = 5e-05
I0331 08:30:30.287827  2693 solver.cpp:243] Iteration 100200, loss = 3.9154
I0331 08:30:30.288094  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.06168 (* 1 = 3.06168 loss)
I0331 08:30:30.288120  2693 sgd_solver.cpp:138] Iteration 100200, lr = 5e-05
I0331 08:32:40.036943  2693 solver.cpp:243] Iteration 100300, loss = 3.93781
I0331 08:32:40.043887  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.0146 (* 1 = 4.0146 loss)
I0331 08:32:40.043917  2693 sgd_solver.cpp:138] Iteration 100300, lr = 5e-05
I0331 08:34:52.217258  2693 solver.cpp:243] Iteration 100400, loss = 3.84251
I0331 08:34:52.217555  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.57196 (* 1 = 3.57196 loss)
I0331 08:34:52.217587  2693 sgd_solver.cpp:138] Iteration 100400, lr = 5e-05
I0331 08:37:02.462927  2693 solver.cpp:243] Iteration 100500, loss = 3.72227
I0331 08:37:02.463179  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.06662 (* 1 = 3.06662 loss)
I0331 08:37:02.463209  2693 sgd_solver.cpp:138] Iteration 100500, lr = 5e-05
I0331 08:39:12.101750  2693 solver.cpp:243] Iteration 100600, loss = 3.83317
I0331 08:39:12.102002  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.7667 (* 1 = 3.7667 loss)
I0331 08:39:12.102035  2693 sgd_solver.cpp:138] Iteration 100600, lr = 5e-05
I0331 08:41:22.518479  2693 solver.cpp:243] Iteration 100700, loss = 3.83604
I0331 08:41:22.518723  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.54793 (* 1 = 3.54793 loss)
I0331 08:41:22.518754  2693 sgd_solver.cpp:138] Iteration 100700, lr = 5e-05
I0331 08:43:33.596388  2693 solver.cpp:243] Iteration 100800, loss = 3.753
I0331 08:43:33.596576  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.93991 (* 1 = 2.93991 loss)
I0331 08:43:33.596596  2693 sgd_solver.cpp:138] Iteration 100800, lr = 5e-05
I0331 08:45:43.114190  2693 solver.cpp:243] Iteration 100900, loss = 3.76664
I0331 08:45:43.114424  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.14517 (* 1 = 3.14517 loss)
I0331 08:45:43.114456  2693 sgd_solver.cpp:138] Iteration 100900, lr = 5e-05
I0331 08:47:53.343392  2693 solver.cpp:243] Iteration 101000, loss = 3.73732
I0331 08:47:53.343664  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.071 (* 1 = 4.071 loss)
I0331 08:47:53.343682  2693 sgd_solver.cpp:138] Iteration 101000, lr = 5e-05
I0331 08:50:03.420585  2693 solver.cpp:243] Iteration 101100, loss = 3.85628
I0331 08:50:03.420791  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.12783 (* 1 = 2.12783 loss)
I0331 08:50:03.420822  2693 sgd_solver.cpp:138] Iteration 101100, lr = 5e-05
I0331 08:52:13.398087  2693 solver.cpp:243] Iteration 101200, loss = 3.81301
I0331 08:52:13.398356  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.56269 (* 1 = 4.56269 loss)
I0331 08:52:13.398404  2693 sgd_solver.cpp:138] Iteration 101200, lr = 5e-05
I0331 08:54:23.422731  2693 solver.cpp:243] Iteration 101300, loss = 3.76897
I0331 08:54:23.422946  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.39912 (* 1 = 3.39912 loss)
I0331 08:54:23.422968  2693 sgd_solver.cpp:138] Iteration 101300, lr = 5e-05
I0331 08:56:33.842910  2693 solver.cpp:243] Iteration 101400, loss = 3.69865
I0331 08:56:33.843173  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.78473 (* 1 = 3.78473 loss)
I0331 08:56:33.843205  2693 sgd_solver.cpp:138] Iteration 101400, lr = 5e-05
I0331 08:58:41.648624  2693 solver.cpp:243] Iteration 101500, loss = 3.62329
I0331 08:58:41.656211  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.16084 (* 1 = 5.16084 loss)
I0331 08:58:41.656239  2693 sgd_solver.cpp:138] Iteration 101500, lr = 5e-05
I0331 09:00:49.507031  2693 solver.cpp:243] Iteration 101600, loss = 3.57034
I0331 09:00:49.507225  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.52667 (* 1 = 2.52667 loss)
I0331 09:00:49.507243  2693 sgd_solver.cpp:138] Iteration 101600, lr = 5e-05
I0331 09:03:01.210459  2693 solver.cpp:243] Iteration 101700, loss = 3.8924
I0331 09:03:01.210721  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.45723 (* 1 = 2.45723 loss)
I0331 09:03:01.210746  2693 sgd_solver.cpp:138] Iteration 101700, lr = 5e-05
I0331 09:05:10.634876  2693 solver.cpp:243] Iteration 101800, loss = 3.70009
I0331 09:05:10.635110  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.98476 (* 1 = 2.98476 loss)
I0331 09:05:10.635131  2693 sgd_solver.cpp:138] Iteration 101800, lr = 5e-05
I0331 09:07:20.052574  2693 solver.cpp:243] Iteration 101900, loss = 3.60503
I0331 09:07:20.052783  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.31618 (* 1 = 3.31618 loss)
I0331 09:07:20.052803  2693 sgd_solver.cpp:138] Iteration 101900, lr = 5e-05
I0331 09:09:28.988266  2693 solver.cpp:243] Iteration 102000, loss = 3.63075
I0331 09:09:28.988513  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.33797 (* 1 = 4.33797 loss)
I0331 09:09:28.988531  2693 sgd_solver.cpp:138] Iteration 102000, lr = 5e-05
I0331 09:11:38.881922  2693 solver.cpp:243] Iteration 102100, loss = 3.72535
I0331 09:11:38.882159  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.24107 (* 1 = 2.24107 loss)
I0331 09:11:38.882179  2693 sgd_solver.cpp:138] Iteration 102100, lr = 5e-05
I0331 09:13:49.428633  2693 solver.cpp:243] Iteration 102200, loss = 3.72394
I0331 09:13:49.428844  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.10666 (* 1 = 3.10666 loss)
I0331 09:13:49.428864  2693 sgd_solver.cpp:138] Iteration 102200, lr = 5e-05
I0331 09:15:58.442301  2693 solver.cpp:243] Iteration 102300, loss = 3.8087
I0331 09:15:58.442864  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.92099 (* 1 = 3.92099 loss)
I0331 09:15:58.442885  2693 sgd_solver.cpp:138] Iteration 102300, lr = 5e-05
I0331 09:18:07.030581  2693 solver.cpp:243] Iteration 102400, loss = 3.64141
I0331 09:18:07.030783  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.447 (* 1 = 2.447 loss)
I0331 09:18:07.030802  2693 sgd_solver.cpp:138] Iteration 102400, lr = 5e-05
I0331 09:20:17.201194  2693 solver.cpp:243] Iteration 102500, loss = 3.76588
I0331 09:20:17.201411  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.61648 (* 1 = 4.61648 loss)
I0331 09:20:17.201442  2693 sgd_solver.cpp:138] Iteration 102500, lr = 5e-05
I0331 09:22:25.826881  2693 solver.cpp:243] Iteration 102600, loss = 3.7555
I0331 09:22:25.827105  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.6352 (* 1 = 3.6352 loss)
I0331 09:22:25.827122  2693 sgd_solver.cpp:138] Iteration 102600, lr = 5e-05
I0331 09:24:35.147743  2693 solver.cpp:243] Iteration 102700, loss = 3.73478
I0331 09:24:35.147956  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.74214 (* 1 = 3.74214 loss)
I0331 09:24:35.147975  2693 sgd_solver.cpp:138] Iteration 102700, lr = 5e-05
I0331 09:26:45.349241  2693 solver.cpp:243] Iteration 102800, loss = 3.83911
I0331 09:26:45.349490  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.79745 (* 1 = 3.79745 loss)
I0331 09:26:45.349524  2693 sgd_solver.cpp:138] Iteration 102800, lr = 5e-05
I0331 09:28:56.276408  2693 solver.cpp:243] Iteration 102900, loss = 3.7572
I0331 09:28:56.276667  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.94669 (* 1 = 3.94669 loss)
I0331 09:28:56.276690  2693 sgd_solver.cpp:138] Iteration 102900, lr = 5e-05
I0331 09:31:04.345525  2693 solver.cpp:243] Iteration 103000, loss = 3.65421
I0331 09:31:04.345784  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.73542 (* 1 = 2.73542 loss)
I0331 09:31:04.345815  2693 sgd_solver.cpp:138] Iteration 103000, lr = 5e-05
I0331 09:33:13.626688  2693 solver.cpp:243] Iteration 103100, loss = 3.71276
I0331 09:33:13.626910  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.25555 (* 1 = 3.25555 loss)
I0331 09:33:13.626965  2693 sgd_solver.cpp:138] Iteration 103100, lr = 5e-05
I0331 09:35:24.066524  2693 solver.cpp:243] Iteration 103200, loss = 3.73643
I0331 09:35:24.066783  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.88946 (* 1 = 2.88946 loss)
I0331 09:35:24.066815  2693 sgd_solver.cpp:138] Iteration 103200, lr = 5e-05
I0331 09:37:33.477870  2693 solver.cpp:243] Iteration 103300, loss = 3.74478
I0331 09:37:33.478147  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.33506 (* 1 = 4.33506 loss)
I0331 09:37:33.478180  2693 sgd_solver.cpp:138] Iteration 103300, lr = 5e-05
I0331 09:39:43.425648  2693 solver.cpp:243] Iteration 103400, loss = 3.70163
I0331 09:39:43.425973  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.64843 (* 1 = 2.64843 loss)
I0331 09:39:43.426034  2693 sgd_solver.cpp:138] Iteration 103400, lr = 5e-05
I0331 09:41:54.656267  2693 solver.cpp:243] Iteration 103500, loss = 3.81902
I0331 09:41:54.656533  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.92906 (* 1 = 2.92906 loss)
I0331 09:41:54.656551  2693 sgd_solver.cpp:138] Iteration 103500, lr = 5e-05
I0331 09:44:04.242905  2693 solver.cpp:243] Iteration 103600, loss = 3.72403
I0331 09:44:04.243185  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.69863 (* 1 = 4.69863 loss)
I0331 09:44:04.243222  2693 sgd_solver.cpp:138] Iteration 103600, lr = 5e-05
I0331 09:46:12.240161  2693 solver.cpp:243] Iteration 103700, loss = 3.58941
I0331 09:46:12.240409  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.07356 (* 1 = 3.07356 loss)
I0331 09:46:12.240434  2693 sgd_solver.cpp:138] Iteration 103700, lr = 5e-05
I0331 09:48:21.489219  2693 solver.cpp:243] Iteration 103800, loss = 3.77299
I0331 09:48:21.489408  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.55114 (* 1 = 3.55114 loss)
I0331 09:48:21.489423  2693 sgd_solver.cpp:138] Iteration 103800, lr = 5e-05
I0331 09:50:30.877800  2693 solver.cpp:243] Iteration 103900, loss = 3.69647
I0331 09:50:30.878041  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.83498 (* 1 = 3.83498 loss)
I0331 09:50:30.878080  2693 sgd_solver.cpp:138] Iteration 103900, lr = 5e-05
I0331 09:52:41.338922  2693 solver.cpp:243] Iteration 104000, loss = 3.74973
I0331 09:52:41.339155  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.33527 (* 1 = 3.33527 loss)
I0331 09:52:41.339174  2693 sgd_solver.cpp:138] Iteration 104000, lr = 5e-05
I0331 09:54:50.268113  2693 solver.cpp:243] Iteration 104100, loss = 3.61241
I0331 09:54:50.268368  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.68669 (* 1 = 2.68669 loss)
I0331 09:54:50.268404  2693 sgd_solver.cpp:138] Iteration 104100, lr = 5e-05
I0331 09:57:00.955724  2693 solver.cpp:243] Iteration 104200, loss = 3.76837
I0331 09:57:00.955936  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.89994 (* 1 = 3.89994 loss)
I0331 09:57:00.955955  2693 sgd_solver.cpp:138] Iteration 104200, lr = 5e-05
I0331 09:59:10.212096  2693 solver.cpp:243] Iteration 104300, loss = 3.68493
I0331 09:59:10.212303  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.35048 (* 1 = 4.35048 loss)
I0331 09:59:10.212321  2693 sgd_solver.cpp:138] Iteration 104300, lr = 5e-05
I0331 10:01:19.858983  2693 solver.cpp:243] Iteration 104400, loss = 3.81268
I0331 10:01:19.866619  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.7426 (* 1 = 3.7426 loss)
I0331 10:01:19.866652  2693 sgd_solver.cpp:138] Iteration 104400, lr = 5e-05
I0331 10:03:29.823757  2693 solver.cpp:243] Iteration 104500, loss = 3.62413
I0331 10:03:29.823988  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.63909 (* 1 = 3.63909 loss)
I0331 10:03:29.824021  2693 sgd_solver.cpp:138] Iteration 104500, lr = 5e-05
I0331 10:05:40.917415  2693 solver.cpp:243] Iteration 104600, loss = 3.70647
I0331 10:05:40.917645  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.30847 (* 1 = 3.30847 loss)
I0331 10:05:40.917673  2693 sgd_solver.cpp:138] Iteration 104600, lr = 5e-05
I0331 10:07:51.097874  2693 solver.cpp:243] Iteration 104700, loss = 3.76073
I0331 10:07:51.098095  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.21202 (* 1 = 3.21202 loss)
I0331 10:07:51.098114  2693 sgd_solver.cpp:138] Iteration 104700, lr = 5e-05
I0331 10:10:04.008500  2693 solver.cpp:243] Iteration 104800, loss = 3.83963
I0331 10:10:04.017714  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.05538 (* 1 = 4.05538 loss)
I0331 10:10:04.017786  2693 sgd_solver.cpp:138] Iteration 104800, lr = 5e-05
I0331 10:12:13.979763  2693 solver.cpp:243] Iteration 104900, loss = 3.66042
I0331 10:12:13.979982  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.05307 (* 1 = 4.05307 loss)
I0331 10:12:13.980013  2693 sgd_solver.cpp:138] Iteration 104900, lr = 5e-05
I0331 10:14:24.397579  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_105000.caffemodel
I0331 10:14:25.575140  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_105000.solverstate
I0331 10:14:25.691682  2693 solver.cpp:433] Iteration 105000, Testing net (#0)
I0331 10:14:25.691772  2693 net.cpp:693] Ignoring source layer mbox_loss
I0331 10:15:45.771481  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.624016
I0331 10:15:46.374311  2693 solver.cpp:243] Iteration 105000, loss = 3.62912
I0331 10:15:46.374389  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.47483 (* 1 = 2.47483 loss)
I0331 10:15:46.374405  2693 sgd_solver.cpp:138] Iteration 105000, lr = 5e-05
I0331 10:17:55.651165  2693 solver.cpp:243] Iteration 105100, loss = 3.64995
I0331 10:17:55.651423  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.26503 (* 1 = 4.26503 loss)
I0331 10:17:55.651453  2693 sgd_solver.cpp:138] Iteration 105100, lr = 5e-05
I0331 10:20:05.693102  2693 solver.cpp:243] Iteration 105200, loss = 3.74723
I0331 10:20:05.693331  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.52734 (* 1 = 3.52734 loss)
I0331 10:20:05.693348  2693 sgd_solver.cpp:138] Iteration 105200, lr = 5e-05
I0331 10:22:14.525833  2693 solver.cpp:243] Iteration 105300, loss = 3.62341
I0331 10:22:14.526134  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.912 (* 1 = 2.912 loss)
I0331 10:22:14.526166  2693 sgd_solver.cpp:138] Iteration 105300, lr = 5e-05
I0331 10:24:24.404258  2693 solver.cpp:243] Iteration 105400, loss = 3.7419
I0331 10:24:24.404489  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.78775 (* 1 = 2.78775 loss)
I0331 10:24:24.404517  2693 sgd_solver.cpp:138] Iteration 105400, lr = 5e-05
I0331 10:26:34.122506  2693 solver.cpp:243] Iteration 105500, loss = 3.52214
I0331 10:26:34.122714  2693 solver.cpp:259]     Train net output #0: mbox_loss = 1.96965 (* 1 = 1.96965 loss)
I0331 10:26:34.122733  2693 sgd_solver.cpp:138] Iteration 105500, lr = 5e-05
I0331 10:28:42.800045  2693 solver.cpp:243] Iteration 105600, loss = 3.64213
I0331 10:28:42.800281  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.30726 (* 1 = 4.30726 loss)
I0331 10:28:42.800320  2693 sgd_solver.cpp:138] Iteration 105600, lr = 5e-05
I0331 10:30:52.723415  2693 solver.cpp:243] Iteration 105700, loss = 3.61366
I0331 10:30:52.723762  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.48343 (* 1 = 3.48343 loss)
I0331 10:30:52.723811  2693 sgd_solver.cpp:138] Iteration 105700, lr = 5e-05
I0331 10:33:03.658375  2693 solver.cpp:243] Iteration 105800, loss = 3.689
I0331 10:33:03.658603  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.97493 (* 1 = 5.97493 loss)
I0331 10:33:03.658622  2693 sgd_solver.cpp:138] Iteration 105800, lr = 5e-05
I0331 10:35:13.604801  2693 solver.cpp:243] Iteration 105900, loss = 3.79764
I0331 10:35:13.607478  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.82156 (* 1 = 2.82156 loss)
I0331 10:35:13.607507  2693 sgd_solver.cpp:138] Iteration 105900, lr = 5e-05
I0331 10:37:23.969243  2693 solver.cpp:243] Iteration 106000, loss = 3.7758
I0331 10:37:23.969446  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.68548 (* 1 = 4.68548 loss)
I0331 10:37:23.969465  2693 sgd_solver.cpp:138] Iteration 106000, lr = 5e-05
I0331 10:39:34.720712  2693 solver.cpp:243] Iteration 106100, loss = 3.80763
I0331 10:39:34.728548  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.64041 (* 1 = 3.64041 loss)
I0331 10:39:34.728576  2693 sgd_solver.cpp:138] Iteration 106100, lr = 5e-05
I0331 10:41:45.319269  2693 solver.cpp:243] Iteration 106200, loss = 3.72647
I0331 10:41:45.319510  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.65133 (* 1 = 3.65133 loss)
I0331 10:41:45.319542  2693 sgd_solver.cpp:138] Iteration 106200, lr = 5e-05
I0331 10:43:55.393921  2693 solver.cpp:243] Iteration 106300, loss = 3.64708
I0331 10:43:55.394151  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.31451 (* 1 = 4.31451 loss)
I0331 10:43:55.394183  2693 sgd_solver.cpp:138] Iteration 106300, lr = 5e-05
I0331 10:46:04.870018  2693 solver.cpp:243] Iteration 106400, loss = 3.61742
I0331 10:46:04.870226  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.86358 (* 1 = 3.86358 loss)
I0331 10:46:04.870244  2693 sgd_solver.cpp:138] Iteration 106400, lr = 5e-05
I0331 10:48:16.870549  2693 solver.cpp:243] Iteration 106500, loss = 3.68837
I0331 10:48:16.870781  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.81111 (* 1 = 3.81111 loss)
I0331 10:48:16.870798  2693 sgd_solver.cpp:138] Iteration 106500, lr = 5e-05
I0331 10:50:27.770232  2693 solver.cpp:243] Iteration 106600, loss = 3.7194
I0331 10:50:27.770468  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.14398 (* 1 = 5.14398 loss)
I0331 10:50:27.770499  2693 sgd_solver.cpp:138] Iteration 106600, lr = 5e-05
I0331 10:52:38.521122  2693 solver.cpp:243] Iteration 106700, loss = 3.75309
I0331 10:52:38.521317  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.04801 (* 1 = 3.04801 loss)
I0331 10:52:38.521332  2693 sgd_solver.cpp:138] Iteration 106700, lr = 5e-05
I0331 10:54:46.859096  2693 solver.cpp:243] Iteration 106800, loss = 3.53904
I0331 10:54:46.859298  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.6896 (* 1 = 3.6896 loss)
I0331 10:54:46.859314  2693 sgd_solver.cpp:138] Iteration 106800, lr = 5e-05
I0331 10:56:57.628178  2693 solver.cpp:243] Iteration 106900, loss = 3.62244
I0331 10:56:57.628443  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.80511 (* 1 = 3.80511 loss)
I0331 10:56:57.628481  2693 sgd_solver.cpp:138] Iteration 106900, lr = 5e-05
I0331 10:59:07.214340  2693 solver.cpp:243] Iteration 107000, loss = 3.71533
I0331 10:59:07.214608  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.34584 (* 1 = 4.34584 loss)
I0331 10:59:07.214725  2693 sgd_solver.cpp:138] Iteration 107000, lr = 5e-05
I0331 11:01:16.881310  2693 solver.cpp:243] Iteration 107100, loss = 3.55328
I0331 11:01:16.881569  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.93678 (* 1 = 2.93678 loss)
I0331 11:01:16.881629  2693 sgd_solver.cpp:138] Iteration 107100, lr = 5e-05
I0331 11:03:25.662518  2693 solver.cpp:243] Iteration 107200, loss = 3.59272
I0331 11:03:25.662714  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.88521 (* 1 = 2.88521 loss)
I0331 11:03:25.662731  2693 sgd_solver.cpp:138] Iteration 107200, lr = 5e-05
I0331 11:05:36.084929  2693 solver.cpp:243] Iteration 107300, loss = 3.56603
I0331 11:05:36.085137  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.05498 (* 1 = 4.05498 loss)
I0331 11:05:36.085157  2693 sgd_solver.cpp:138] Iteration 107300, lr = 5e-05
I0331 11:07:46.866937  2693 solver.cpp:243] Iteration 107400, loss = 3.69684
I0331 11:07:46.867173  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.84777 (* 1 = 2.84777 loss)
I0331 11:07:46.867192  2693 sgd_solver.cpp:138] Iteration 107400, lr = 5e-05
I0331 11:09:57.788470  2693 solver.cpp:243] Iteration 107500, loss = 3.74705
I0331 11:09:57.788666  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.01513 (* 1 = 5.01513 loss)
I0331 11:09:57.788683  2693 sgd_solver.cpp:138] Iteration 107500, lr = 5e-05
I0331 11:12:06.819888  2693 solver.cpp:243] Iteration 107600, loss = 3.46886
I0331 11:12:06.820065  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.71767 (* 1 = 2.71767 loss)
I0331 11:12:06.820082  2693 sgd_solver.cpp:138] Iteration 107600, lr = 5e-05
I0331 11:14:16.004739  2693 solver.cpp:243] Iteration 107700, loss = 3.70946
I0331 11:14:16.005023  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.19481 (* 1 = 3.19481 loss)
I0331 11:14:16.005071  2693 sgd_solver.cpp:138] Iteration 107700, lr = 5e-05
I0331 11:16:27.180660  2693 solver.cpp:243] Iteration 107800, loss = 3.74275
I0331 11:16:27.180852  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.75288 (* 1 = 2.75288 loss)
I0331 11:16:27.180868  2693 sgd_solver.cpp:138] Iteration 107800, lr = 5e-05
I0331 11:18:37.362224  2693 solver.cpp:243] Iteration 107900, loss = 3.60944
I0331 11:18:37.362493  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.32525 (* 1 = 2.32525 loss)
I0331 11:18:37.362560  2693 sgd_solver.cpp:138] Iteration 107900, lr = 5e-05
I0331 11:20:47.874976  2693 solver.cpp:243] Iteration 108000, loss = 3.73889
I0331 11:20:47.882872  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.48671 (* 1 = 3.48671 loss)
I0331 11:20:47.882891  2693 sgd_solver.cpp:138] Iteration 108000, lr = 5e-05
I0331 11:22:57.516376  2693 solver.cpp:243] Iteration 108100, loss = 3.70944
I0331 11:22:57.516577  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.28489 (* 1 = 3.28489 loss)
I0331 11:22:57.516597  2693 sgd_solver.cpp:138] Iteration 108100, lr = 5e-05
I0331 11:25:07.566596  2693 solver.cpp:243] Iteration 108200, loss = 3.79014
I0331 11:25:07.566789  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.25324 (* 1 = 3.25324 loss)
I0331 11:25:07.566807  2693 sgd_solver.cpp:138] Iteration 108200, lr = 5e-05
I0331 11:27:16.575940  2693 solver.cpp:243] Iteration 108300, loss = 3.74789
I0331 11:27:16.576161  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.52224 (* 1 = 2.52224 loss)
I0331 11:27:16.576179  2693 sgd_solver.cpp:138] Iteration 108300, lr = 5e-05
I0331 11:29:26.736004  2693 solver.cpp:243] Iteration 108400, loss = 3.7096
I0331 11:29:26.736306  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.63163 (* 1 = 2.63163 loss)
I0331 11:29:26.736363  2693 sgd_solver.cpp:138] Iteration 108400, lr = 5e-05
I0331 11:31:37.431845  2693 solver.cpp:243] Iteration 108500, loss = 3.75181
I0331 11:31:37.432090  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.75421 (* 1 = 4.75421 loss)
I0331 11:31:37.432113  2693 sgd_solver.cpp:138] Iteration 108500, lr = 5e-05
I0331 11:33:47.916787  2693 solver.cpp:243] Iteration 108600, loss = 3.77366
I0331 11:33:47.916956  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.46768 (* 1 = 2.46768 loss)
I0331 11:33:47.916973  2693 sgd_solver.cpp:138] Iteration 108600, lr = 5e-05
I0331 11:35:57.412909  2693 solver.cpp:243] Iteration 108700, loss = 3.69837
I0331 11:35:57.413166  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.97605 (* 1 = 2.97605 loss)
I0331 11:35:57.413198  2693 sgd_solver.cpp:138] Iteration 108700, lr = 5e-05
I0331 11:38:06.770057  2693 solver.cpp:243] Iteration 108800, loss = 3.57263
I0331 11:38:06.770273  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.38237 (* 1 = 4.38237 loss)
I0331 11:38:06.770298  2693 sgd_solver.cpp:138] Iteration 108800, lr = 5e-05
I0331 11:40:18.757442  2693 solver.cpp:243] Iteration 108900, loss = 3.80052
I0331 11:40:18.757717  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.02779 (* 1 = 3.02779 loss)
I0331 11:40:18.757748  2693 sgd_solver.cpp:138] Iteration 108900, lr = 5e-05
I0331 11:42:26.112812  2693 solver.cpp:243] Iteration 109000, loss = 3.67592
I0331 11:42:26.113042  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.45296 (* 1 = 2.45296 loss)
I0331 11:42:26.113073  2693 sgd_solver.cpp:138] Iteration 109000, lr = 5e-05
I0331 11:44:38.724043  2693 solver.cpp:243] Iteration 109100, loss = 3.65384
I0331 11:44:38.724241  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.02014 (* 1 = 3.02014 loss)
I0331 11:44:38.724256  2693 sgd_solver.cpp:138] Iteration 109100, lr = 5e-05
I0331 11:46:49.700896  2693 solver.cpp:243] Iteration 109200, loss = 3.67057
I0331 11:46:49.701124  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.14753 (* 1 = 4.14753 loss)
I0331 11:46:49.701156  2693 sgd_solver.cpp:138] Iteration 109200, lr = 5e-05
I0331 11:48:59.717253  2693 solver.cpp:243] Iteration 109300, loss = 3.62388
I0331 11:48:59.717463  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.78569 (* 1 = 2.78569 loss)
I0331 11:48:59.717480  2693 sgd_solver.cpp:138] Iteration 109300, lr = 5e-05
I0331 11:51:07.743654  2693 solver.cpp:243] Iteration 109400, loss = 3.71849
I0331 11:51:07.743896  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.63786 (* 1 = 4.63786 loss)
I0331 11:51:07.743927  2693 sgd_solver.cpp:138] Iteration 109400, lr = 5e-05
I0331 11:53:16.984735  2693 solver.cpp:243] Iteration 109500, loss = 3.71724
I0331 11:53:16.985046  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.05971 (* 1 = 4.05971 loss)
I0331 11:53:16.985080  2693 sgd_solver.cpp:138] Iteration 109500, lr = 5e-05
I0331 11:55:28.673455  2693 solver.cpp:243] Iteration 109600, loss = 3.7252
I0331 11:55:28.673676  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.24989 (* 1 = 4.24989 loss)
I0331 11:55:28.673707  2693 sgd_solver.cpp:138] Iteration 109600, lr = 5e-05
I0331 11:57:39.707950  2693 solver.cpp:243] Iteration 109700, loss = 3.64265
I0331 11:57:39.708174  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.04984 (* 1 = 4.04984 loss)
I0331 11:57:39.708196  2693 sgd_solver.cpp:138] Iteration 109700, lr = 5e-05
I0331 11:59:51.447340  2693 solver.cpp:243] Iteration 109800, loss = 3.66579
I0331 11:59:51.447628  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.4729 (* 1 = 3.4729 loss)
I0331 11:59:51.447656  2693 sgd_solver.cpp:138] Iteration 109800, lr = 5e-05
I0331 12:02:03.093861  2693 solver.cpp:243] Iteration 109900, loss = 3.68646
I0331 12:02:03.094108  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.11849 (* 1 = 3.11849 loss)
I0331 12:02:03.094138  2693 sgd_solver.cpp:138] Iteration 109900, lr = 5e-05
I0331 12:04:12.768414  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_110000.caffemodel
I0331 12:04:14.016532  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_110000.solverstate
I0331 12:04:14.135720  2693 solver.cpp:433] Iteration 110000, Testing net (#0)
I0331 12:04:14.135802  2693 net.cpp:693] Ignoring source layer mbox_loss
I0331 12:05:43.759268  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.615867
I0331 12:05:44.596793  2693 solver.cpp:243] Iteration 110000, loss = 3.80101
I0331 12:05:44.596858  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.87149 (* 1 = 4.87149 loss)
I0331 12:05:44.596873  2693 sgd_solver.cpp:138] Iteration 110000, lr = 5e-05
I0331 12:08:43.636299  2693 solver.cpp:243] Iteration 110100, loss = 3.6024
I0331 12:08:43.636639  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.06697 (* 1 = 4.06697 loss)
I0331 12:08:43.636699  2693 sgd_solver.cpp:138] Iteration 110100, lr = 5e-05
I0331 12:11:41.688555  2693 solver.cpp:243] Iteration 110200, loss = 3.65558
I0331 12:11:41.688880  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.74106 (* 1 = 2.74106 loss)
I0331 12:11:41.688948  2693 sgd_solver.cpp:138] Iteration 110200, lr = 5e-05
I0331 12:14:11.963275  2693 solver.cpp:243] Iteration 110300, loss = 3.75483
I0331 12:14:11.963475  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.27361 (* 1 = 3.27361 loss)
I0331 12:14:11.963492  2693 sgd_solver.cpp:138] Iteration 110300, lr = 5e-05
I0331 12:16:34.484012  2693 solver.cpp:243] Iteration 110400, loss = 3.81365
I0331 12:16:34.484256  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.3289 (* 1 = 3.3289 loss)
I0331 12:16:34.484274  2693 sgd_solver.cpp:138] Iteration 110400, lr = 5e-05
I0331 12:19:31.597338  2693 solver.cpp:243] Iteration 110500, loss = 3.69541
I0331 12:19:31.597579  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.28442 (* 1 = 3.28442 loss)
I0331 12:19:31.597611  2693 sgd_solver.cpp:138] Iteration 110500, lr = 5e-05
I0331 12:22:31.461676  2693 solver.cpp:243] Iteration 110600, loss = 3.7231
I0331 12:22:31.461912  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.74839 (* 1 = 3.74839 loss)
I0331 12:22:31.461946  2693 sgd_solver.cpp:138] Iteration 110600, lr = 5e-05
I0331 12:25:01.010018  2693 solver.cpp:243] Iteration 110700, loss = 3.70672
I0331 12:25:01.010254  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.07848 (* 1 = 4.07848 loss)
I0331 12:25:01.010287  2693 sgd_solver.cpp:138] Iteration 110700, lr = 5e-05
I0331 12:27:10.993320  2693 solver.cpp:243] Iteration 110800, loss = 3.67818
I0331 12:27:10.993566  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.64982 (* 1 = 3.64982 loss)
I0331 12:27:10.993585  2693 sgd_solver.cpp:138] Iteration 110800, lr = 5e-05
I0331 12:29:20.194396  2693 solver.cpp:243] Iteration 110900, loss = 3.70088
I0331 12:29:20.194815  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.57071 (* 1 = 3.57071 loss)
I0331 12:29:20.194890  2693 sgd_solver.cpp:138] Iteration 110900, lr = 5e-05
I0331 12:31:32.411562  2693 solver.cpp:243] Iteration 111000, loss = 3.63148
I0331 12:31:32.411931  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.30475 (* 1 = 4.30475 loss)
I0331 12:31:32.411990  2693 sgd_solver.cpp:138] Iteration 111000, lr = 5e-05
I0331 12:33:42.184756  2693 solver.cpp:243] Iteration 111100, loss = 3.5602
I0331 12:33:42.184983  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.21448 (* 1 = 3.21448 loss)
I0331 12:33:42.185001  2693 sgd_solver.cpp:138] Iteration 111100, lr = 5e-05
I0331 12:35:53.476263  2693 solver.cpp:243] Iteration 111200, loss = 3.56548
I0331 12:35:53.476513  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.65745 (* 1 = 3.65745 loss)
I0331 12:35:53.476533  2693 sgd_solver.cpp:138] Iteration 111200, lr = 5e-05
I0331 12:38:03.446795  2693 solver.cpp:243] Iteration 111300, loss = 3.55877
I0331 12:38:03.447052  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.54036 (* 1 = 3.54036 loss)
I0331 12:38:03.447084  2693 sgd_solver.cpp:138] Iteration 111300, lr = 5e-05
I0331 12:40:12.834946  2693 solver.cpp:243] Iteration 111400, loss = 3.64184
I0331 12:40:12.835181  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.87725 (* 1 = 2.87725 loss)
I0331 12:40:12.835203  2693 sgd_solver.cpp:138] Iteration 111400, lr = 5e-05
I0331 12:42:24.362485  2693 solver.cpp:243] Iteration 111500, loss = 3.70765
I0331 12:42:24.362756  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.75244 (* 1 = 3.75244 loss)
I0331 12:42:24.362792  2693 sgd_solver.cpp:138] Iteration 111500, lr = 5e-05
I0331 12:44:34.129782  2693 solver.cpp:243] Iteration 111600, loss = 3.70374
I0331 12:44:34.130072  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.95323 (* 1 = 3.95323 loss)
I0331 12:44:34.130120  2693 sgd_solver.cpp:138] Iteration 111600, lr = 5e-05
I0331 12:46:46.820452  2693 solver.cpp:243] Iteration 111700, loss = 3.74862
I0331 12:46:46.820672  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.7881 (* 1 = 3.7881 loss)
I0331 12:46:46.820703  2693 sgd_solver.cpp:138] Iteration 111700, lr = 5e-05
I0331 12:48:58.213970  2693 solver.cpp:243] Iteration 111800, loss = 3.71583
I0331 12:48:58.230998  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.97689 (* 1 = 3.97689 loss)
I0331 12:48:58.231032  2693 sgd_solver.cpp:138] Iteration 111800, lr = 5e-05
I0331 12:51:08.124757  2693 solver.cpp:243] Iteration 111900, loss = 3.72361
I0331 12:51:08.124982  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.77643 (* 1 = 3.77643 loss)
I0331 12:51:08.125015  2693 sgd_solver.cpp:138] Iteration 111900, lr = 5e-05
I0331 12:53:18.154888  2693 solver.cpp:243] Iteration 112000, loss = 3.71252
I0331 12:53:18.155149  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.60159 (* 1 = 6.60159 loss)
I0331 12:53:18.155189  2693 sgd_solver.cpp:138] Iteration 112000, lr = 5e-05
I0331 12:55:28.305239  2693 solver.cpp:243] Iteration 112100, loss = 3.65477
I0331 12:55:28.305449  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.56715 (* 1 = 3.56715 loss)
I0331 12:55:28.305466  2693 sgd_solver.cpp:138] Iteration 112100, lr = 5e-05
I0331 12:57:39.338784  2693 solver.cpp:243] Iteration 112200, loss = 3.57424
I0331 12:57:39.339015  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.8954 (* 1 = 3.8954 loss)
I0331 12:57:39.339033  2693 sgd_solver.cpp:138] Iteration 112200, lr = 5e-05
I0331 12:59:49.969930  2693 solver.cpp:243] Iteration 112300, loss = 3.78997
I0331 12:59:49.970131  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.77732 (* 1 = 3.77732 loss)
I0331 12:59:49.970149  2693 sgd_solver.cpp:138] Iteration 112300, lr = 5e-05
I0331 13:02:00.999920  2693 solver.cpp:243] Iteration 112400, loss = 3.56251
I0331 13:02:01.000246  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.2733 (* 1 = 3.2733 loss)
I0331 13:02:01.000314  2693 sgd_solver.cpp:138] Iteration 112400, lr = 5e-05
I0331 13:04:10.514189  2693 solver.cpp:243] Iteration 112500, loss = 3.64598
I0331 13:04:10.514427  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.13671 (* 1 = 3.13671 loss)
I0331 13:04:10.514457  2693 sgd_solver.cpp:138] Iteration 112500, lr = 5e-05
I0331 13:06:19.712731  2693 solver.cpp:243] Iteration 112600, loss = 3.68264
I0331 13:06:19.718987  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.2165 (* 1 = 3.2165 loss)
I0331 13:06:19.719045  2693 sgd_solver.cpp:138] Iteration 112600, lr = 5e-05
I0331 13:08:30.082321  2693 solver.cpp:243] Iteration 112700, loss = 3.56804
I0331 13:08:30.082588  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.43002 (* 1 = 3.43002 loss)
I0331 13:08:30.082618  2693 sgd_solver.cpp:138] Iteration 112700, lr = 5e-05
I0331 13:10:38.649917  2693 solver.cpp:243] Iteration 112800, loss = 3.55523
I0331 13:10:38.650122  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.23219 (* 1 = 3.23219 loss)
I0331 13:10:38.650141  2693 sgd_solver.cpp:138] Iteration 112800, lr = 5e-05
I0331 13:12:47.120403  2693 solver.cpp:243] Iteration 112900, loss = 3.61293
I0331 13:12:47.120642  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.71271 (* 1 = 2.71271 loss)
I0331 13:12:47.120666  2693 sgd_solver.cpp:138] Iteration 112900, lr = 5e-05
I0331 13:14:59.565210  2693 solver.cpp:243] Iteration 113000, loss = 3.81875
I0331 13:14:59.565462  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.11164 (* 1 = 2.11164 loss)
I0331 13:14:59.565490  2693 sgd_solver.cpp:138] Iteration 113000, lr = 5e-05
I0331 13:17:10.778705  2693 solver.cpp:243] Iteration 113100, loss = 3.76436
I0331 13:17:10.778939  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.23336 (* 1 = 4.23336 loss)
I0331 13:17:10.778960  2693 sgd_solver.cpp:138] Iteration 113100, lr = 5e-05
I0331 13:19:21.329773  2693 solver.cpp:243] Iteration 113200, loss = 3.64521
I0331 13:19:21.330119  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.29152 (* 1 = 4.29152 loss)
I0331 13:19:21.330363  2693 sgd_solver.cpp:138] Iteration 113200, lr = 5e-05
I0331 13:21:30.264176  2693 solver.cpp:243] Iteration 113300, loss = 3.57008
I0331 13:21:30.264487  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.9544 (* 1 = 2.9544 loss)
I0331 13:21:30.264539  2693 sgd_solver.cpp:138] Iteration 113300, lr = 5e-05
I0331 13:23:41.222724  2693 solver.cpp:243] Iteration 113400, loss = 3.57576
I0331 13:23:41.222916  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.41647 (* 1 = 4.41647 loss)
I0331 13:23:41.222932  2693 sgd_solver.cpp:138] Iteration 113400, lr = 5e-05
I0331 13:25:54.352025  2693 solver.cpp:243] Iteration 113500, loss = 3.66396
I0331 13:25:54.352305  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.22482 (* 1 = 3.22482 loss)
I0331 13:25:54.352368  2693 sgd_solver.cpp:138] Iteration 113500, lr = 5e-05
I0331 13:28:04.062829  2693 solver.cpp:243] Iteration 113600, loss = 3.66672
I0331 13:28:04.063067  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.56551 (* 1 = 4.56551 loss)
I0331 13:28:04.063093  2693 sgd_solver.cpp:138] Iteration 113600, lr = 5e-05
I0331 13:30:12.167862  2693 solver.cpp:243] Iteration 113700, loss = 3.59936
I0331 13:30:12.168077  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.56556 (* 1 = 3.56556 loss)
I0331 13:30:12.168097  2693 sgd_solver.cpp:138] Iteration 113700, lr = 5e-05
I0331 13:32:22.460155  2693 solver.cpp:243] Iteration 113800, loss = 3.63068
I0331 13:32:22.460403  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.39055 (* 1 = 3.39055 loss)
I0331 13:32:22.460435  2693 sgd_solver.cpp:138] Iteration 113800, lr = 5e-05
I0331 13:34:32.762810  2693 solver.cpp:243] Iteration 113900, loss = 3.70217
I0331 13:34:32.763020  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.50424 (* 1 = 3.50424 loss)
I0331 13:34:32.763037  2693 sgd_solver.cpp:138] Iteration 113900, lr = 5e-05
I0331 13:36:43.200284  2693 solver.cpp:243] Iteration 114000, loss = 3.76277
I0331 13:36:43.200598  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.87051 (* 1 = 4.87051 loss)
I0331 13:36:43.200659  2693 sgd_solver.cpp:138] Iteration 114000, lr = 5e-05
I0331 13:38:54.053936  2693 solver.cpp:243] Iteration 114100, loss = 3.64528
I0331 13:38:54.054175  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.2636 (* 1 = 3.2636 loss)
I0331 13:38:54.054198  2693 sgd_solver.cpp:138] Iteration 114100, lr = 5e-05
I0331 13:41:05.033043  2693 solver.cpp:243] Iteration 114200, loss = 3.75629
I0331 13:41:05.033293  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.30326 (* 1 = 4.30326 loss)
I0331 13:41:05.033318  2693 sgd_solver.cpp:138] Iteration 114200, lr = 5e-05
I0331 13:43:15.980535  2693 solver.cpp:243] Iteration 114300, loss = 3.62446
I0331 13:43:15.980716  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.61793 (* 1 = 3.61793 loss)
I0331 13:43:15.980736  2693 sgd_solver.cpp:138] Iteration 114300, lr = 5e-05
I0331 13:45:26.066859  2693 solver.cpp:243] Iteration 114400, loss = 3.59631
I0331 13:45:26.067092  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.05975 (* 1 = 3.05975 loss)
I0331 13:45:26.067111  2693 sgd_solver.cpp:138] Iteration 114400, lr = 5e-05
I0331 13:47:36.991647  2693 solver.cpp:243] Iteration 114500, loss = 3.72192
I0331 13:47:36.991899  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.8601 (* 1 = 4.8601 loss)
I0331 13:47:36.991930  2693 sgd_solver.cpp:138] Iteration 114500, lr = 5e-05
I0331 13:49:45.543915  2693 solver.cpp:243] Iteration 114600, loss = 3.64018
I0331 13:49:45.544209  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.53353 (* 1 = 2.53353 loss)
I0331 13:49:45.544244  2693 sgd_solver.cpp:138] Iteration 114600, lr = 5e-05
I0331 13:51:54.716804  2693 solver.cpp:243] Iteration 114700, loss = 3.59194
I0331 13:51:54.717010  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.11237 (* 1 = 4.11237 loss)
I0331 13:51:54.717033  2693 sgd_solver.cpp:138] Iteration 114700, lr = 5e-05
I0331 13:54:04.777719  2693 solver.cpp:243] Iteration 114800, loss = 3.62499
I0331 13:54:04.777891  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.00861 (* 1 = 4.00861 loss)
I0331 13:54:04.777909  2693 sgd_solver.cpp:138] Iteration 114800, lr = 5e-05
I0331 13:56:15.928243  2693 solver.cpp:243] Iteration 114900, loss = 3.71077
I0331 13:56:15.928479  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.32838 (* 1 = 3.32838 loss)
I0331 13:56:15.928498  2693 sgd_solver.cpp:138] Iteration 114900, lr = 5e-05
I0331 13:58:23.951236  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_115000.caffemodel
I0331 13:58:24.882872  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_115000.solverstate
I0331 13:58:24.999243  2693 solver.cpp:433] Iteration 115000, Testing net (#0)
I0331 13:58:24.999325  2693 net.cpp:693] Ignoring source layer mbox_loss
I0331 13:59:45.314411  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.62278
I0331 13:59:46.124029  2693 solver.cpp:243] Iteration 115000, loss = 3.56382
I0331 13:59:46.124142  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.87398 (* 1 = 3.87398 loss)
I0331 13:59:46.124172  2693 sgd_solver.cpp:138] Iteration 115000, lr = 5e-05
I0331 14:01:55.786785  2693 solver.cpp:243] Iteration 115100, loss = 3.5616
I0331 14:01:55.787051  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.82777 (* 1 = 2.82777 loss)
I0331 14:01:55.787086  2693 sgd_solver.cpp:138] Iteration 115100, lr = 5e-05
I0331 14:04:06.380437  2693 solver.cpp:243] Iteration 115200, loss = 3.50533
I0331 14:04:06.380591  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.70062 (* 1 = 2.70062 loss)
I0331 14:04:06.380607  2693 sgd_solver.cpp:138] Iteration 115200, lr = 5e-05
I0331 14:06:17.558141  2693 solver.cpp:243] Iteration 115300, loss = 3.51608
I0331 14:06:17.558507  2693 solver.cpp:259]     Train net output #0: mbox_loss = 1.83802 (* 1 = 1.83802 loss)
I0331 14:06:17.558540  2693 sgd_solver.cpp:138] Iteration 115300, lr = 5e-05
I0331 14:08:28.361938  2693 solver.cpp:243] Iteration 115400, loss = 3.60426
I0331 14:08:28.362139  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.27413 (* 1 = 5.27413 loss)
I0331 14:08:28.362155  2693 sgd_solver.cpp:138] Iteration 115400, lr = 5e-05
I0331 14:10:37.490082  2693 solver.cpp:243] Iteration 115500, loss = 3.72165
I0331 14:10:37.490337  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.33917 (* 1 = 4.33917 loss)
I0331 14:10:37.490368  2693 sgd_solver.cpp:138] Iteration 115500, lr = 5e-05
I0331 14:12:48.876289  2693 solver.cpp:243] Iteration 115600, loss = 3.77947
I0331 14:12:48.876538  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.77298 (* 1 = 4.77298 loss)
I0331 14:12:48.876579  2693 sgd_solver.cpp:138] Iteration 115600, lr = 5e-05
I0331 14:14:58.813370  2693 solver.cpp:243] Iteration 115700, loss = 3.72544
I0331 14:14:58.813583  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.66207 (* 1 = 3.66207 loss)
I0331 14:14:58.813602  2693 sgd_solver.cpp:138] Iteration 115700, lr = 5e-05
I0331 14:17:08.970237  2693 solver.cpp:243] Iteration 115800, loss = 3.5914
I0331 14:17:08.970435  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.39578 (* 1 = 4.39578 loss)
I0331 14:17:08.970451  2693 sgd_solver.cpp:138] Iteration 115800, lr = 5e-05
I0331 14:19:20.139310  2693 solver.cpp:243] Iteration 115900, loss = 3.7129
I0331 14:19:20.139549  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.34254 (* 1 = 4.34254 loss)
I0331 14:19:20.139644  2693 sgd_solver.cpp:138] Iteration 115900, lr = 5e-05
I0331 14:21:30.736668  2693 solver.cpp:243] Iteration 116000, loss = 3.64753
I0331 14:21:30.744107  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.31723 (* 1 = 3.31723 loss)
I0331 14:21:30.744139  2693 sgd_solver.cpp:138] Iteration 116000, lr = 5e-05
I0331 14:23:43.567575  2693 solver.cpp:243] Iteration 116100, loss = 3.7971
I0331 14:23:43.567878  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.7728 (* 1 = 2.7728 loss)
I0331 14:23:43.567909  2693 sgd_solver.cpp:138] Iteration 116100, lr = 5e-05
I0331 14:25:53.479831  2693 solver.cpp:243] Iteration 116200, loss = 3.71474
I0331 14:25:53.480082  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.61746 (* 1 = 3.61746 loss)
I0331 14:25:53.480113  2693 sgd_solver.cpp:138] Iteration 116200, lr = 5e-05
I0331 14:28:03.827536  2693 solver.cpp:243] Iteration 116300, loss = 3.74226
I0331 14:28:03.827802  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.86302 (* 1 = 2.86302 loss)
I0331 14:28:03.827836  2693 sgd_solver.cpp:138] Iteration 116300, lr = 5e-05
I0331 14:30:13.724462  2693 solver.cpp:243] Iteration 116400, loss = 3.64986
I0331 14:30:13.724722  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.4202 (* 1 = 3.4202 loss)
I0331 14:30:13.724756  2693 sgd_solver.cpp:138] Iteration 116400, lr = 5e-05
I0331 14:32:24.423516  2693 solver.cpp:243] Iteration 116500, loss = 3.60725
I0331 14:32:24.423866  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.05721 (* 1 = 5.05721 loss)
I0331 14:32:24.423904  2693 sgd_solver.cpp:138] Iteration 116500, lr = 5e-05
I0331 14:34:35.402384  2693 solver.cpp:243] Iteration 116600, loss = 3.68185
I0331 14:34:35.402642  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.23899 (* 1 = 3.23899 loss)
I0331 14:34:35.402683  2693 sgd_solver.cpp:138] Iteration 116600, lr = 5e-05
I0331 14:36:46.438125  2693 solver.cpp:243] Iteration 116700, loss = 3.54607
I0331 14:36:46.438362  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.39131 (* 1 = 3.39131 loss)
I0331 14:36:46.438405  2693 sgd_solver.cpp:138] Iteration 116700, lr = 5e-05
I0331 14:38:57.354137  2693 solver.cpp:243] Iteration 116800, loss = 3.5717
I0331 14:38:57.354465  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.31858 (* 1 = 3.31858 loss)
I0331 14:38:57.354504  2693 sgd_solver.cpp:138] Iteration 116800, lr = 5e-05
I0331 14:41:07.393641  2693 solver.cpp:243] Iteration 116900, loss = 3.54159
I0331 14:41:07.393887  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.12311 (* 1 = 4.12311 loss)
I0331 14:41:07.393934  2693 sgd_solver.cpp:138] Iteration 116900, lr = 5e-05
I0331 14:43:17.679807  2693 solver.cpp:243] Iteration 117000, loss = 3.5255
I0331 14:43:17.680011  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.27108 (* 1 = 5.27108 loss)
I0331 14:43:17.680037  2693 sgd_solver.cpp:138] Iteration 117000, lr = 5e-05
I0331 14:45:29.448261  2693 solver.cpp:243] Iteration 117100, loss = 3.7484
I0331 14:45:29.448462  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.81396 (* 1 = 3.81396 loss)
I0331 14:45:29.448479  2693 sgd_solver.cpp:138] Iteration 117100, lr = 5e-05
I0331 14:47:42.501576  2693 solver.cpp:243] Iteration 117200, loss = 3.79861
I0331 14:47:42.501775  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.11417 (* 1 = 4.11417 loss)
I0331 14:47:42.501791  2693 sgd_solver.cpp:138] Iteration 117200, lr = 5e-05
I0331 14:49:55.458282  2693 solver.cpp:243] Iteration 117300, loss = 3.75102
I0331 14:49:55.466372  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.30327 (* 1 = 4.30327 loss)
I0331 14:49:55.466398  2693 sgd_solver.cpp:138] Iteration 117300, lr = 5e-05
I0331 14:52:07.231647  2693 solver.cpp:243] Iteration 117400, loss = 3.62657
I0331 14:52:07.231848  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.31401 (* 1 = 3.31401 loss)
I0331 14:52:07.231874  2693 sgd_solver.cpp:138] Iteration 117400, lr = 5e-05
I0331 14:54:18.625722  2693 solver.cpp:243] Iteration 117500, loss = 3.68276
I0331 14:54:18.625921  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.41804 (* 1 = 2.41804 loss)
I0331 14:54:18.625938  2693 sgd_solver.cpp:138] Iteration 117500, lr = 5e-05
I0331 14:56:29.710518  2693 solver.cpp:243] Iteration 117600, loss = 3.68523
I0331 14:56:29.717401  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.76295 (* 1 = 3.76295 loss)
I0331 14:56:29.717444  2693 sgd_solver.cpp:138] Iteration 117600, lr = 5e-05
I0331 14:58:39.416960  2693 solver.cpp:243] Iteration 117700, loss = 3.63049
I0331 14:58:39.417177  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.1092 (* 1 = 3.1092 loss)
I0331 14:58:39.417196  2693 sgd_solver.cpp:138] Iteration 117700, lr = 5e-05
I0331 15:00:52.096070  2693 solver.cpp:243] Iteration 117800, loss = 3.66233
I0331 15:00:52.096262  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.70489 (* 1 = 2.70489 loss)
I0331 15:00:52.096282  2693 sgd_solver.cpp:138] Iteration 117800, lr = 5e-05
I0331 15:03:03.042989  2693 solver.cpp:243] Iteration 117900, loss = 3.75355
I0331 15:03:03.043187  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.69072 (* 1 = 3.69072 loss)
I0331 15:03:03.043207  2693 sgd_solver.cpp:138] Iteration 117900, lr = 5e-05
I0331 15:05:15.306195  2693 solver.cpp:243] Iteration 118000, loss = 3.71037
I0331 15:05:15.306470  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.52055 (* 1 = 3.52055 loss)
I0331 15:05:15.306501  2693 sgd_solver.cpp:138] Iteration 118000, lr = 5e-05
I0331 15:07:26.246757  2693 solver.cpp:243] Iteration 118100, loss = 3.5549
I0331 15:07:26.246919  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.14144 (* 1 = 5.14144 loss)
I0331 15:07:26.246939  2693 sgd_solver.cpp:138] Iteration 118100, lr = 5e-05
I0331 15:09:39.104399  2693 solver.cpp:243] Iteration 118200, loss = 3.75146
I0331 15:09:39.104631  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.11033 (* 1 = 4.11033 loss)
I0331 15:09:39.104655  2693 sgd_solver.cpp:138] Iteration 118200, lr = 5e-05
I0331 15:11:49.844491  2693 solver.cpp:243] Iteration 118300, loss = 3.56609
I0331 15:11:49.844684  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.13918 (* 1 = 3.13918 loss)
I0331 15:11:49.844703  2693 sgd_solver.cpp:138] Iteration 118300, lr = 5e-05
I0331 15:14:01.346020  2693 solver.cpp:243] Iteration 118400, loss = 3.68246
I0331 15:14:01.346349  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.71487 (* 1 = 2.71487 loss)
I0331 15:14:01.346385  2693 sgd_solver.cpp:138] Iteration 118400, lr = 5e-05
I0331 15:16:10.235087  2693 solver.cpp:243] Iteration 118500, loss = 3.60939
I0331 15:16:10.235297  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.63125 (* 1 = 3.63125 loss)
I0331 15:16:10.235313  2693 sgd_solver.cpp:138] Iteration 118500, lr = 5e-05
I0331 15:18:21.631742  2693 solver.cpp:243] Iteration 118600, loss = 3.6812
I0331 15:18:21.631944  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.13371 (* 1 = 3.13371 loss)
I0331 15:18:21.631960  2693 sgd_solver.cpp:138] Iteration 118600, lr = 5e-05
I0331 15:20:35.347055  2693 solver.cpp:243] Iteration 118700, loss = 3.68893
I0331 15:20:35.347327  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.69318 (* 1 = 3.69318 loss)
I0331 15:20:35.347375  2693 sgd_solver.cpp:138] Iteration 118700, lr = 5e-05
I0331 15:22:47.079375  2693 solver.cpp:243] Iteration 118800, loss = 3.63051
I0331 15:22:47.079689  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.02711 (* 1 = 4.02711 loss)
I0331 15:22:47.079720  2693 sgd_solver.cpp:138] Iteration 118800, lr = 5e-05
I0331 15:24:58.397781  2693 solver.cpp:243] Iteration 118900, loss = 3.58826
I0331 15:24:58.398051  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.06276 (* 1 = 4.06276 loss)
I0331 15:24:58.398082  2693 sgd_solver.cpp:138] Iteration 118900, lr = 5e-05
I0331 15:27:09.292119  2693 solver.cpp:243] Iteration 119000, loss = 3.69743
I0331 15:27:09.292351  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.50737 (* 1 = 4.50737 loss)
I0331 15:27:09.292372  2693 sgd_solver.cpp:138] Iteration 119000, lr = 5e-05
I0331 15:29:22.247437  2693 solver.cpp:243] Iteration 119100, loss = 3.6874
I0331 15:29:22.247732  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.49218 (* 1 = 3.49218 loss)
I0331 15:29:22.247751  2693 sgd_solver.cpp:138] Iteration 119100, lr = 5e-05
I0331 15:31:33.703155  2693 solver.cpp:243] Iteration 119200, loss = 3.4924
I0331 15:31:33.703411  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.38561 (* 1 = 2.38561 loss)
I0331 15:31:33.703441  2693 sgd_solver.cpp:138] Iteration 119200, lr = 5e-05
I0331 15:33:45.429141  2693 solver.cpp:243] Iteration 119300, loss = 3.75457
I0331 15:33:45.429417  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.42051 (* 1 = 3.42051 loss)
I0331 15:33:45.429450  2693 sgd_solver.cpp:138] Iteration 119300, lr = 5e-05
I0331 15:35:57.159983  2693 solver.cpp:243] Iteration 119400, loss = 3.74961
I0331 15:35:57.160285  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.6695 (* 1 = 4.6695 loss)
I0331 15:35:57.160322  2693 sgd_solver.cpp:138] Iteration 119400, lr = 5e-05
I0331 15:38:09.914633  2693 solver.cpp:243] Iteration 119500, loss = 3.70015
I0331 15:38:09.914885  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.1073 (* 1 = 3.1073 loss)
I0331 15:38:09.914937  2693 sgd_solver.cpp:138] Iteration 119500, lr = 5e-05
I0331 15:40:21.078033  2693 solver.cpp:243] Iteration 119600, loss = 3.61556
I0331 15:40:21.081789  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.23961 (* 1 = 3.23961 loss)
I0331 15:40:21.081823  2693 sgd_solver.cpp:138] Iteration 119600, lr = 5e-05
I0331 15:42:34.379232  2693 solver.cpp:243] Iteration 119700, loss = 3.64551
I0331 15:42:34.379462  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.47381 (* 1 = 3.47381 loss)
I0331 15:42:34.379480  2693 sgd_solver.cpp:138] Iteration 119700, lr = 5e-05
I0331 15:44:45.323423  2693 solver.cpp:243] Iteration 119800, loss = 3.71581
I0331 15:44:45.323693  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.60383 (* 1 = 2.60383 loss)
I0331 15:44:45.323710  2693 sgd_solver.cpp:138] Iteration 119800, lr = 5e-05
I0331 15:46:56.408347  2693 solver.cpp:243] Iteration 119900, loss = 3.61867
I0331 15:46:56.408649  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.47315 (* 1 = 2.47315 loss)
I0331 15:46:56.408681  2693 sgd_solver.cpp:138] Iteration 119900, lr = 5e-05
I0331 15:49:06.267457  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_120000.caffemodel
I0331 15:49:07.186417  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_120000.solverstate
I0331 15:49:07.300609  2693 solver.cpp:433] Iteration 120000, Testing net (#0)
I0331 15:49:07.300689  2693 net.cpp:693] Ignoring source layer mbox_loss
I0331 15:50:28.215142  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.620014
I0331 15:50:28.751363  2693 solver.cpp:243] Iteration 120000, loss = 3.72285
I0331 15:50:28.751436  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.57194 (* 1 = 3.57194 loss)
I0331 15:50:28.751451  2693 sgd_solver.cpp:138] Iteration 120000, lr = 5e-05
I0331 15:52:39.208855  2693 solver.cpp:243] Iteration 120100, loss = 3.56372
I0331 15:52:39.209060  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.59451 (* 1 = 2.59451 loss)
I0331 15:52:39.209077  2693 sgd_solver.cpp:138] Iteration 120100, lr = 5e-05
I0331 15:54:52.242940  2693 solver.cpp:243] Iteration 120200, loss = 3.75515
I0331 15:54:52.243216  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.5469 (* 1 = 3.5469 loss)
I0331 15:54:52.243253  2693 sgd_solver.cpp:138] Iteration 120200, lr = 5e-05
I0331 15:57:02.112726  2693 solver.cpp:243] Iteration 120300, loss = 3.65832
I0331 15:57:02.112965  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.25979 (* 1 = 3.25979 loss)
I0331 15:57:02.112996  2693 sgd_solver.cpp:138] Iteration 120300, lr = 5e-05
I0331 15:59:13.417644  2693 solver.cpp:243] Iteration 120400, loss = 3.66326
I0331 15:59:13.417878  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.75451 (* 1 = 3.75451 loss)
I0331 15:59:13.417920  2693 sgd_solver.cpp:138] Iteration 120400, lr = 5e-05
I0331 16:01:26.852282  2693 solver.cpp:243] Iteration 120500, loss = 3.81989
I0331 16:01:26.873108  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.96183 (* 1 = 3.96183 loss)
I0331 16:01:26.873142  2693 sgd_solver.cpp:138] Iteration 120500, lr = 5e-05
I0331 16:03:37.241763  2693 solver.cpp:243] Iteration 120600, loss = 3.49982
I0331 16:03:37.242002  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.0581 (* 1 = 5.0581 loss)
I0331 16:03:37.242036  2693 sgd_solver.cpp:138] Iteration 120600, lr = 5e-05
I0331 16:05:47.755237  2693 solver.cpp:243] Iteration 120700, loss = 3.62699
I0331 16:05:47.755452  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.41399 (* 1 = 4.41399 loss)
I0331 16:05:47.755476  2693 sgd_solver.cpp:138] Iteration 120700, lr = 5e-05
I0331 16:07:58.442989  2693 solver.cpp:243] Iteration 120800, loss = 3.6473
I0331 16:07:58.443240  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.18221 (* 1 = 3.18221 loss)
I0331 16:07:58.443265  2693 sgd_solver.cpp:138] Iteration 120800, lr = 5e-05
I0331 16:10:11.289415  2693 solver.cpp:243] Iteration 120900, loss = 3.59258
I0331 16:10:11.289624  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.16754 (* 1 = 3.16754 loss)
I0331 16:10:11.289641  2693 sgd_solver.cpp:138] Iteration 120900, lr = 5e-05
I0331 16:12:22.350760  2693 solver.cpp:243] Iteration 121000, loss = 3.417
I0331 16:12:22.350955  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.01992 (* 1 = 3.01992 loss)
I0331 16:12:22.350972  2693 sgd_solver.cpp:138] Iteration 121000, lr = 5e-05
I0331 16:14:33.642721  2693 solver.cpp:243] Iteration 121100, loss = 3.66063
I0331 16:14:33.642933  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.71289 (* 1 = 3.71289 loss)
I0331 16:14:33.642952  2693 sgd_solver.cpp:138] Iteration 121100, lr = 5e-05
I0331 16:16:45.143666  2693 solver.cpp:243] Iteration 121200, loss = 3.62055
I0331 16:16:45.143863  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.48749 (* 1 = 3.48749 loss)
I0331 16:16:45.143882  2693 sgd_solver.cpp:138] Iteration 121200, lr = 5e-05
I0331 16:18:56.182292  2693 solver.cpp:243] Iteration 121300, loss = 3.74561
I0331 16:18:56.182608  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.1256 (* 1 = 3.1256 loss)
I0331 16:18:56.182626  2693 sgd_solver.cpp:138] Iteration 121300, lr = 5e-05
I0331 16:21:08.332578  2693 solver.cpp:243] Iteration 121400, loss = 3.64467
I0331 16:21:08.332880  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.13832 (* 1 = 4.13832 loss)
I0331 16:21:08.332919  2693 sgd_solver.cpp:138] Iteration 121400, lr = 5e-05
I0331 16:23:21.448923  2693 solver.cpp:243] Iteration 121500, loss = 3.74029
I0331 16:23:21.449213  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.30928 (* 1 = 4.30928 loss)
I0331 16:23:21.449255  2693 sgd_solver.cpp:138] Iteration 121500, lr = 5e-05
I0331 16:25:33.995923  2693 solver.cpp:243] Iteration 121600, loss = 3.50481
I0331 16:25:33.996106  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.18604 (* 1 = 4.18604 loss)
I0331 16:25:33.996122  2693 sgd_solver.cpp:138] Iteration 121600, lr = 5e-05
I0331 16:27:47.102664  2693 solver.cpp:243] Iteration 121700, loss = 3.66892
I0331 16:27:47.102926  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.96416 (* 1 = 2.96416 loss)
I0331 16:27:47.102952  2693 sgd_solver.cpp:138] Iteration 121700, lr = 5e-05
I0331 16:29:59.603965  2693 solver.cpp:243] Iteration 121800, loss = 3.81781
I0331 16:29:59.604210  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.19541 (* 1 = 5.19541 loss)
I0331 16:29:59.604234  2693 sgd_solver.cpp:138] Iteration 121800, lr = 5e-05
I0331 16:32:12.306529  2693 solver.cpp:243] Iteration 121900, loss = 3.65876
I0331 16:32:12.306813  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.44638 (* 1 = 4.44638 loss)
I0331 16:32:12.306850  2693 sgd_solver.cpp:138] Iteration 121900, lr = 5e-05
I0331 16:34:22.531491  2693 solver.cpp:243] Iteration 122000, loss = 3.63429
I0331 16:34:22.531783  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.15296 (* 1 = 4.15296 loss)
I0331 16:34:22.531803  2693 sgd_solver.cpp:138] Iteration 122000, lr = 5e-05
I0331 16:36:33.072885  2693 solver.cpp:243] Iteration 122100, loss = 3.57863
I0331 16:36:33.073107  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.10753 (* 1 = 3.10753 loss)
I0331 16:36:33.073125  2693 sgd_solver.cpp:138] Iteration 122100, lr = 5e-05
I0331 16:38:43.837013  2693 solver.cpp:243] Iteration 122200, loss = 3.53588
I0331 16:38:43.837230  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.05998 (* 1 = 2.05998 loss)
I0331 16:38:43.837249  2693 sgd_solver.cpp:138] Iteration 122200, lr = 5e-05
I0331 16:40:56.390962  2693 solver.cpp:243] Iteration 122300, loss = 3.7634
I0331 16:40:56.391237  2693 solver.cpp:259]     Train net output #0: mbox_loss = 1.92749 (* 1 = 1.92749 loss)
I0331 16:40:56.391268  2693 sgd_solver.cpp:138] Iteration 122300, lr = 5e-05
I0331 16:43:07.194247  2693 solver.cpp:243] Iteration 122400, loss = 3.60292
I0331 16:43:07.194491  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.39064 (* 1 = 3.39064 loss)
I0331 16:43:07.194515  2693 sgd_solver.cpp:138] Iteration 122400, lr = 5e-05
I0331 16:45:18.996011  2693 solver.cpp:243] Iteration 122500, loss = 3.46767
I0331 16:45:18.996276  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.40104 (* 1 = 3.40104 loss)
I0331 16:45:18.996302  2693 sgd_solver.cpp:138] Iteration 122500, lr = 5e-05
I0331 16:47:30.002761  2693 solver.cpp:243] Iteration 122600, loss = 3.57206
I0331 16:47:30.004335  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.66153 (* 1 = 3.66153 loss)
I0331 16:47:30.004350  2693 sgd_solver.cpp:138] Iteration 122600, lr = 5e-05
I0331 16:49:41.089532  2693 solver.cpp:243] Iteration 122700, loss = 3.50875
I0331 16:49:41.089721  2693 solver.cpp:259]     Train net output #0: mbox_loss = 1.93398 (* 1 = 1.93398 loss)
I0331 16:49:41.089738  2693 sgd_solver.cpp:138] Iteration 122700, lr = 5e-05
I0331 16:51:54.627125  2693 solver.cpp:243] Iteration 122800, loss = 3.7366
I0331 16:51:54.627379  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.30555 (* 1 = 4.30555 loss)
I0331 16:51:54.627398  2693 sgd_solver.cpp:138] Iteration 122800, lr = 5e-05
I0331 16:54:06.221506  2693 solver.cpp:243] Iteration 122900, loss = 3.65759
I0331 16:54:06.221774  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.82126 (* 1 = 3.82126 loss)
I0331 16:54:06.221804  2693 sgd_solver.cpp:138] Iteration 122900, lr = 5e-05
I0331 16:56:19.383235  2693 solver.cpp:243] Iteration 123000, loss = 3.78942
I0331 16:56:19.383471  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.60632 (* 1 = 3.60632 loss)
I0331 16:56:19.383499  2693 sgd_solver.cpp:138] Iteration 123000, lr = 5e-05
I0331 16:58:30.259999  2693 solver.cpp:243] Iteration 123100, loss = 3.60672
I0331 16:58:30.260191  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.04057 (* 1 = 3.04057 loss)
I0331 16:58:30.260206  2693 sgd_solver.cpp:138] Iteration 123100, lr = 5e-05
I0331 17:00:41.255422  2693 solver.cpp:243] Iteration 123200, loss = 3.52745
I0331 17:00:41.262609  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.9981 (* 1 = 2.9981 loss)
I0331 17:00:41.262626  2693 sgd_solver.cpp:138] Iteration 123200, lr = 5e-05
I0331 17:02:52.595082  2693 solver.cpp:243] Iteration 123300, loss = 3.71563
I0331 17:02:52.609519  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.62023 (* 1 = 4.62023 loss)
I0331 17:02:52.609542  2693 sgd_solver.cpp:138] Iteration 123300, lr = 5e-05
I0331 17:05:03.446979  2693 solver.cpp:243] Iteration 123400, loss = 3.76042
I0331 17:05:03.447183  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.7609 (* 1 = 3.7609 loss)
I0331 17:05:03.447202  2693 sgd_solver.cpp:138] Iteration 123400, lr = 5e-05
I0331 17:07:15.469319  2693 solver.cpp:243] Iteration 123500, loss = 3.62505
I0331 17:07:15.469543  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.18747 (* 1 = 3.18747 loss)
I0331 17:07:15.469584  2693 sgd_solver.cpp:138] Iteration 123500, lr = 5e-05
I0331 17:09:27.376333  2693 solver.cpp:243] Iteration 123600, loss = 3.66774
I0331 17:09:27.376554  2693 solver.cpp:259]     Train net output #0: mbox_loss = 6.06191 (* 1 = 6.06191 loss)
I0331 17:09:27.376583  2693 sgd_solver.cpp:138] Iteration 123600, lr = 5e-05
I0331 17:11:39.073489  2693 solver.cpp:243] Iteration 123700, loss = 3.71189
I0331 17:11:39.073732  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.75098 (* 1 = 3.75098 loss)
I0331 17:11:39.073760  2693 sgd_solver.cpp:138] Iteration 123700, lr = 5e-05
I0331 17:13:51.317086  2693 solver.cpp:243] Iteration 123800, loss = 3.66045
I0331 17:13:51.317347  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.45158 (* 1 = 3.45158 loss)
I0331 17:13:51.317373  2693 sgd_solver.cpp:138] Iteration 123800, lr = 5e-05
I0331 17:16:02.900893  2693 solver.cpp:243] Iteration 123900, loss = 3.60914
I0331 17:16:02.901141  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.31138 (* 1 = 3.31138 loss)
I0331 17:16:02.901165  2693 sgd_solver.cpp:138] Iteration 123900, lr = 5e-05
I0331 17:18:15.031246  2693 solver.cpp:243] Iteration 124000, loss = 3.55719
I0331 17:18:15.031509  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.97801 (* 1 = 5.97801 loss)
I0331 17:18:15.031536  2693 sgd_solver.cpp:138] Iteration 124000, lr = 5e-05
I0331 17:20:26.341207  2693 solver.cpp:243] Iteration 124100, loss = 3.67111
I0331 17:20:26.341408  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.0119 (* 1 = 2.0119 loss)
I0331 17:20:26.341425  2693 sgd_solver.cpp:138] Iteration 124100, lr = 5e-05
I0331 17:22:36.733008  2693 solver.cpp:243] Iteration 124200, loss = 3.58944
I0331 17:22:36.733188  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.21676 (* 1 = 3.21676 loss)
I0331 17:22:36.733204  2693 sgd_solver.cpp:138] Iteration 124200, lr = 5e-05
I0331 17:24:50.584949  2693 solver.cpp:243] Iteration 124300, loss = 3.80786
I0331 17:24:50.590220  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.65151 (* 1 = 3.65151 loss)
I0331 17:24:50.590248  2693 sgd_solver.cpp:138] Iteration 124300, lr = 5e-05
I0331 17:27:02.448038  2693 solver.cpp:243] Iteration 124400, loss = 3.8498
I0331 17:27:02.454773  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.92289 (* 1 = 2.92289 loss)
I0331 17:27:02.454882  2693 sgd_solver.cpp:138] Iteration 124400, lr = 5e-05
I0331 17:29:14.699506  2693 solver.cpp:243] Iteration 124500, loss = 3.61467
I0331 17:29:14.721352  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.03484 (* 1 = 4.03484 loss)
I0331 17:29:14.721374  2693 sgd_solver.cpp:138] Iteration 124500, lr = 5e-05
I0331 17:31:24.827020  2693 solver.cpp:243] Iteration 124600, loss = 3.61677
I0331 17:31:24.827222  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.94233 (* 1 = 3.94233 loss)
I0331 17:31:24.827239  2693 sgd_solver.cpp:138] Iteration 124600, lr = 5e-05
I0331 17:33:36.446018  2693 solver.cpp:243] Iteration 124700, loss = 3.6836
I0331 17:33:36.446254  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.63262 (* 1 = 3.63262 loss)
I0331 17:33:36.446276  2693 sgd_solver.cpp:138] Iteration 124700, lr = 5e-05
I0331 17:35:48.450482  2693 solver.cpp:243] Iteration 124800, loss = 3.62111
I0331 17:35:48.450639  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.64972 (* 1 = 5.64972 loss)
I0331 17:35:48.450651  2693 sgd_solver.cpp:138] Iteration 124800, lr = 5e-05
I0331 17:38:00.380215  2693 solver.cpp:243] Iteration 124900, loss = 3.82844
I0331 17:38:00.381650  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.75979 (* 1 = 2.75979 loss)
I0331 17:38:00.381680  2693 sgd_solver.cpp:138] Iteration 124900, lr = 5e-05
I0331 17:40:09.566782  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_125000.caffemodel
I0331 17:40:10.503332  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_125000.solverstate
I0331 17:40:10.618890  2693 solver.cpp:433] Iteration 125000, Testing net (#0)
I0331 17:40:10.618983  2693 net.cpp:693] Ignoring source layer mbox_loss
I0331 17:41:40.593771  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.617279
I0331 17:41:41.870740  2693 solver.cpp:243] Iteration 125000, loss = 3.65626
I0331 17:41:41.870813  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.83316 (* 1 = 4.83316 loss)
I0331 17:41:41.870829  2693 sgd_solver.cpp:138] Iteration 125000, lr = 5e-05
I0331 17:44:40.558154  2693 solver.cpp:243] Iteration 125100, loss = 3.75768
I0331 17:44:40.558395  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.66709 (* 1 = 4.66709 loss)
I0331 17:44:40.558431  2693 sgd_solver.cpp:138] Iteration 125100, lr = 5e-05
I0331 17:47:37.075853  2693 solver.cpp:243] Iteration 125200, loss = 3.74029
I0331 17:47:37.076100  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.63083 (* 1 = 3.63083 loss)
I0331 17:47:37.076130  2693 sgd_solver.cpp:138] Iteration 125200, lr = 5e-05
I0331 17:50:16.438972  2693 solver.cpp:243] Iteration 125300, loss = 3.59958
I0331 17:50:16.439195  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.83849 (* 1 = 3.83849 loss)
I0331 17:50:16.439214  2693 sgd_solver.cpp:138] Iteration 125300, lr = 5e-05
I0331 17:52:27.066766  2693 solver.cpp:243] Iteration 125400, loss = 3.72036
I0331 17:52:27.068653  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.09032 (* 1 = 3.09032 loss)
I0331 17:52:27.068719  2693 sgd_solver.cpp:138] Iteration 125400, lr = 5e-05
I0331 17:54:39.313194  2693 solver.cpp:243] Iteration 125500, loss = 3.58672
I0331 17:54:39.313513  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.09297 (* 1 = 3.09297 loss)
I0331 17:54:39.313549  2693 sgd_solver.cpp:138] Iteration 125500, lr = 5e-05
I0331 17:56:49.762771  2693 solver.cpp:243] Iteration 125600, loss = 3.56251
I0331 17:56:49.763039  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.62891 (* 1 = 4.62891 loss)
I0331 17:56:49.763080  2693 sgd_solver.cpp:138] Iteration 125600, lr = 5e-05
I0331 17:59:02.607683  2693 solver.cpp:243] Iteration 125700, loss = 3.65062
I0331 17:59:02.607949  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.07709 (* 1 = 4.07709 loss)
I0331 17:59:02.607967  2693 sgd_solver.cpp:138] Iteration 125700, lr = 5e-05
I0331 18:01:14.633194  2693 solver.cpp:243] Iteration 125800, loss = 3.51279
I0331 18:01:14.633432  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.46847 (* 1 = 3.46847 loss)
I0331 18:01:14.633457  2693 sgd_solver.cpp:138] Iteration 125800, lr = 5e-05
I0331 18:03:25.337256  2693 solver.cpp:243] Iteration 125900, loss = 3.6112
I0331 18:03:25.337445  2693 solver.cpp:259]     Train net output #0: mbox_loss = 1.94864 (* 1 = 1.94864 loss)
I0331 18:03:25.337461  2693 sgd_solver.cpp:138] Iteration 125900, lr = 5e-05
I0331 18:05:35.294931  2693 solver.cpp:243] Iteration 126000, loss = 3.62564
I0331 18:05:35.295203  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.58538 (* 1 = 3.58538 loss)
I0331 18:05:35.295243  2693 sgd_solver.cpp:138] Iteration 126000, lr = 5e-05
I0331 18:07:47.606040  2693 solver.cpp:243] Iteration 126100, loss = 3.70527
I0331 18:07:47.612473  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.12142 (* 1 = 3.12142 loss)
I0331 18:07:47.612491  2693 sgd_solver.cpp:138] Iteration 126100, lr = 5e-05
I0331 18:10:00.769165  2693 solver.cpp:243] Iteration 126200, loss = 3.64224
I0331 18:10:00.769371  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.91963 (* 1 = 3.91963 loss)
I0331 18:10:00.769392  2693 sgd_solver.cpp:138] Iteration 126200, lr = 5e-05
I0331 18:12:11.478376  2693 solver.cpp:243] Iteration 126300, loss = 3.60197
I0331 18:12:11.478633  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.68062 (* 1 = 3.68062 loss)
I0331 18:12:11.478664  2693 sgd_solver.cpp:138] Iteration 126300, lr = 5e-05
I0331 18:14:22.123334  2693 solver.cpp:243] Iteration 126400, loss = 3.57428
I0331 18:14:22.123617  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.1037 (* 1 = 3.1037 loss)
I0331 18:14:22.123656  2693 sgd_solver.cpp:138] Iteration 126400, lr = 5e-05
I0331 18:16:34.318702  2693 solver.cpp:243] Iteration 126500, loss = 3.48011
I0331 18:16:34.318928  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.74251 (* 1 = 3.74251 loss)
I0331 18:16:34.318960  2693 sgd_solver.cpp:138] Iteration 126500, lr = 5e-05
I0331 18:18:45.851002  2693 solver.cpp:243] Iteration 126600, loss = 3.57983
I0331 18:18:45.851245  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.00445 (* 1 = 4.00445 loss)
I0331 18:18:45.851284  2693 sgd_solver.cpp:138] Iteration 126600, lr = 5e-05
I0331 18:20:57.029264  2693 solver.cpp:243] Iteration 126700, loss = 3.54179
I0331 18:20:57.036932  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.56173 (* 1 = 3.56173 loss)
I0331 18:20:57.036952  2693 sgd_solver.cpp:138] Iteration 126700, lr = 5e-05
I0331 18:23:07.938479  2693 solver.cpp:243] Iteration 126800, loss = 3.80945
I0331 18:23:07.938686  2693 solver.cpp:259]     Train net output #0: mbox_loss = 1.99776 (* 1 = 1.99776 loss)
I0331 18:23:07.938704  2693 sgd_solver.cpp:138] Iteration 126800, lr = 5e-05
I0331 18:25:21.154637  2693 solver.cpp:243] Iteration 126900, loss = 3.69098
I0331 18:25:21.154870  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.75355 (* 1 = 2.75355 loss)
I0331 18:25:21.154896  2693 sgd_solver.cpp:138] Iteration 126900, lr = 5e-05
I0331 18:27:32.863292  2693 solver.cpp:243] Iteration 127000, loss = 3.48914
I0331 18:27:32.863564  2693 solver.cpp:259]     Train net output #0: mbox_loss = 1.75494 (* 1 = 1.75494 loss)
I0331 18:27:32.863632  2693 sgd_solver.cpp:138] Iteration 127000, lr = 5e-05
I0331 18:29:45.005914  2693 solver.cpp:243] Iteration 127100, loss = 3.54012
I0331 18:29:45.012845  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.86379 (* 1 = 3.86379 loss)
I0331 18:29:45.012890  2693 sgd_solver.cpp:138] Iteration 127100, lr = 5e-05
I0331 18:31:57.010197  2693 solver.cpp:243] Iteration 127200, loss = 3.6919
I0331 18:31:57.010437  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.98494 (* 1 = 3.98494 loss)
I0331 18:31:57.010464  2693 sgd_solver.cpp:138] Iteration 127200, lr = 5e-05
I0331 18:34:07.961406  2693 solver.cpp:243] Iteration 127300, loss = 3.69447
I0331 18:34:07.968273  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.29646 (* 1 = 2.29646 loss)
I0331 18:34:07.968292  2693 sgd_solver.cpp:138] Iteration 127300, lr = 5e-05
I0331 18:36:20.117787  2693 solver.cpp:243] Iteration 127400, loss = 3.5748
I0331 18:36:20.118021  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.50386 (* 1 = 2.50386 loss)
I0331 18:36:20.118046  2693 sgd_solver.cpp:138] Iteration 127400, lr = 5e-05
I0331 18:38:31.728416  2693 solver.cpp:243] Iteration 127500, loss = 3.66193
I0331 18:38:31.728677  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.74679 (* 1 = 2.74679 loss)
I0331 18:38:31.728703  2693 sgd_solver.cpp:138] Iteration 127500, lr = 5e-05
I0331 18:40:43.684533  2693 solver.cpp:243] Iteration 127600, loss = 3.68609
I0331 18:40:43.684777  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.25029 (* 1 = 4.25029 loss)
I0331 18:40:43.684828  2693 sgd_solver.cpp:138] Iteration 127600, lr = 5e-05
I0331 18:42:54.950036  2693 solver.cpp:243] Iteration 127700, loss = 3.56319
I0331 18:42:54.950285  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.87972 (* 1 = 3.87972 loss)
I0331 18:42:54.950312  2693 sgd_solver.cpp:138] Iteration 127700, lr = 5e-05
I0331 18:45:06.987761  2693 solver.cpp:243] Iteration 127800, loss = 3.64186
I0331 18:45:06.987964  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.64352 (* 1 = 3.64352 loss)
I0331 18:45:06.987983  2693 sgd_solver.cpp:138] Iteration 127800, lr = 5e-05
I0331 18:47:19.058053  2693 solver.cpp:243] Iteration 127900, loss = 3.5295
I0331 18:47:19.058280  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.00871 (* 1 = 3.00871 loss)
I0331 18:47:19.058300  2693 sgd_solver.cpp:138] Iteration 127900, lr = 5e-05
I0331 18:49:29.605255  2693 solver.cpp:243] Iteration 128000, loss = 3.64355
I0331 18:49:29.605507  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.82306 (* 1 = 3.82306 loss)
I0331 18:49:29.605536  2693 sgd_solver.cpp:138] Iteration 128000, lr = 5e-05
I0331 18:51:40.844130  2693 solver.cpp:243] Iteration 128100, loss = 3.40735
I0331 18:51:40.844370  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.93745 (* 1 = 3.93745 loss)
I0331 18:51:40.844393  2693 sgd_solver.cpp:138] Iteration 128100, lr = 5e-05
I0331 18:53:52.786108  2693 solver.cpp:243] Iteration 128200, loss = 3.6121
I0331 18:53:52.786325  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.751 (* 1 = 3.751 loss)
I0331 18:53:52.786344  2693 sgd_solver.cpp:138] Iteration 128200, lr = 5e-05
I0331 18:56:04.273941  2693 solver.cpp:243] Iteration 128300, loss = 3.51988
I0331 18:56:04.274219  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.69952 (* 1 = 3.69952 loss)
I0331 18:56:04.274252  2693 sgd_solver.cpp:138] Iteration 128300, lr = 5e-05
I0331 18:58:18.015574  2693 solver.cpp:243] Iteration 128400, loss = 3.75528
I0331 18:58:18.015944  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.5048 (* 1 = 3.5048 loss)
I0331 18:58:18.015981  2693 sgd_solver.cpp:138] Iteration 128400, lr = 5e-05
I0331 19:00:32.506407  2693 solver.cpp:243] Iteration 128500, loss = 3.70444
I0331 19:00:32.506672  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.13548 (* 1 = 3.13548 loss)
I0331 19:00:32.506707  2693 sgd_solver.cpp:138] Iteration 128500, lr = 5e-05
I0331 19:02:46.461434  2693 solver.cpp:243] Iteration 128600, loss = 3.60452
I0331 19:02:46.461700  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.57738 (* 1 = 3.57738 loss)
I0331 19:02:46.461726  2693 sgd_solver.cpp:138] Iteration 128600, lr = 5e-05
I0331 19:04:58.699478  2693 solver.cpp:243] Iteration 128700, loss = 3.73166
I0331 19:04:58.699748  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.6952 (* 1 = 3.6952 loss)
I0331 19:04:58.699769  2693 sgd_solver.cpp:138] Iteration 128700, lr = 5e-05
I0331 19:07:10.373586  2693 solver.cpp:243] Iteration 128800, loss = 3.42001
I0331 19:07:10.373885  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.12479 (* 1 = 3.12479 loss)
I0331 19:07:10.373919  2693 sgd_solver.cpp:138] Iteration 128800, lr = 5e-05
I0331 19:09:22.952533  2693 solver.cpp:243] Iteration 128900, loss = 3.66258
I0331 19:09:22.952787  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.90521 (* 1 = 2.90521 loss)
I0331 19:09:22.952814  2693 sgd_solver.cpp:138] Iteration 128900, lr = 5e-05
I0331 19:11:35.871202  2693 solver.cpp:243] Iteration 129000, loss = 3.60492
I0331 19:11:35.871501  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.01132 (* 1 = 5.01132 loss)
I0331 19:11:35.871539  2693 sgd_solver.cpp:138] Iteration 129000, lr = 5e-05
I0331 19:13:49.699965  2693 solver.cpp:243] Iteration 129100, loss = 3.55952
I0331 19:13:49.700214  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.88387 (* 1 = 3.88387 loss)
I0331 19:13:49.700232  2693 sgd_solver.cpp:138] Iteration 129100, lr = 5e-05
I0331 19:16:03.750556  2693 solver.cpp:243] Iteration 129200, loss = 3.74305
I0331 19:16:03.750738  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.8476 (* 1 = 2.8476 loss)
I0331 19:16:03.750756  2693 sgd_solver.cpp:138] Iteration 129200, lr = 5e-05
I0331 19:18:16.292862  2693 solver.cpp:243] Iteration 129300, loss = 3.76779
I0331 19:18:16.293128  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.69978 (* 1 = 3.69978 loss)
I0331 19:18:16.293172  2693 sgd_solver.cpp:138] Iteration 129300, lr = 5e-05
I0331 19:20:28.224519  2693 solver.cpp:243] Iteration 129400, loss = 3.61481
I0331 19:20:28.224762  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.89018 (* 1 = 3.89018 loss)
I0331 19:20:28.224786  2693 sgd_solver.cpp:138] Iteration 129400, lr = 5e-05
I0331 19:22:44.130512  2693 solver.cpp:243] Iteration 129500, loss = 3.69532
I0331 19:22:44.130795  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.4449 (* 1 = 3.4449 loss)
I0331 19:22:44.130841  2693 sgd_solver.cpp:138] Iteration 129500, lr = 5e-05
I0331 19:24:55.986887  2693 solver.cpp:243] Iteration 129600, loss = 3.53609
I0331 19:24:55.987154  2693 solver.cpp:259]     Train net output #0: mbox_loss = 3.23629 (* 1 = 3.23629 loss)
I0331 19:24:55.987193  2693 sgd_solver.cpp:138] Iteration 129600, lr = 5e-05
I0331 19:27:07.330489  2693 solver.cpp:243] Iteration 129700, loss = 3.49002
I0331 19:27:07.343225  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.62917 (* 1 = 2.62917 loss)
I0331 19:27:07.343242  2693 sgd_solver.cpp:138] Iteration 129700, lr = 5e-05
I0331 19:29:18.538774  2693 solver.cpp:243] Iteration 129800, loss = 3.53932
I0331 19:29:18.541867  2693 solver.cpp:259]     Train net output #0: mbox_loss = 5.13891 (* 1 = 5.13891 loss)
I0331 19:29:18.541889  2693 sgd_solver.cpp:138] Iteration 129800, lr = 5e-05
I0331 19:31:32.000217  2693 solver.cpp:243] Iteration 129900, loss = 3.62188
I0331 19:31:32.000483  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.23101 (* 1 = 4.23101 loss)
I0331 19:31:32.000505  2693 sgd_solver.cpp:138] Iteration 129900, lr = 5e-05
I0331 19:33:43.169397  2693 solver.cpp:596] Snapshotting to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_130000.caffemodel
I0331 19:33:44.423883  2693 sgd_solver.cpp:307] Snapshotting solver state to binary proto file models/VGGNet/ssd_coco_part_clean/VGG_ssd_coco_part_clean_iter_130000.solverstate
I0331 19:33:44.548219  2693 solver.cpp:433] Iteration 130000, Testing net (#0)
I0331 19:33:44.548296  2693 net.cpp:693] Ignoring source layer mbox_loss
I0331 19:35:05.382815  2693 solver.cpp:546]     Test net output #0: detection_eval = 0.605822
I0331 19:35:06.093042  2693 solver.cpp:243] Iteration 130000, loss = 3.74117
I0331 19:35:06.093111  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.04686 (* 1 = 4.04686 loss)
I0331 19:35:06.093127  2693 sgd_solver.cpp:138] Iteration 130000, lr = 5e-05
I0331 19:37:20.462596  2693 solver.cpp:243] Iteration 130100, loss = 3.6621
I0331 19:37:20.462836  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.22951 (* 1 = 4.22951 loss)
I0331 19:37:20.462865  2693 sgd_solver.cpp:138] Iteration 130100, lr = 5e-05
I0331 19:39:32.931910  2693 solver.cpp:243] Iteration 130200, loss = 3.61298
I0331 19:39:32.932214  2693 solver.cpp:259]     Train net output #0: mbox_loss = 4.08352 (* 1 = 4.08352 loss)
I0331 19:39:32.932235  2693 sgd_solver.cpp:138] Iteration 130200, lr = 5e-05
I0331 19:41:44.872233  2693 solver.cpp:243] Iteration 130300, loss = 3.67179
I0331 19:41:44.872452  2693 solver.cpp:259]     Train net output #0: mbox_loss = 2.9982 (* 1 = 2.9982 loss)
I0331 19:41:44.872468  2693 sgd_solver.cpp:138] Iteration 130300, lr = 5e-05
*** Aborted at 1490960507 (unix time) try "date -d @1490960507" if you are using GNU date ***
PC: @     0x7faef820b9c1 caffe::MatchBBox()
*** SIGTERM (@0x3f700000d45) received by PID 2693 (TID 0x7faef8ecc9c0) from PID 3397; stack trace: ***
    @     0x7faef6b1bcb0 (unknown)
    @     0x7faef820b9c1 caffe::MatchBBox()
    @     0x7faef820da78 caffe::FindMatches()
    @     0x7faef83281ce caffe::MultiBoxLossLayer<>::Forward_cpu()
    @     0x7faef839b8e5 caffe::Net<>::ForwardFromTo()
    @     0x7faef839bc57 caffe::Net<>::Forward()
    @     0x7faef81ad818 caffe::Solver<>::Step()
    @     0x7faef81adf1e caffe::Solver<>::Solve()
    @           0x408e7a train()
    @           0x40627c main
    @     0x7faef6b06f45 (unknown)
    @           0x406aeb (unknown)
    @                0x0 (unknown)
Terminated
