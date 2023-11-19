base_ = [
    '../_base_/schedules/cyclic_40e.py', '../_base_/default_runtime.py',
    '../_base_/models/parta2.py'
]

point_cloud_range = [-6, -35, -3, 70.4, 35, 1]
# dataset settings
data_root = '/home/ubuntu/mmdetection3d/data/kitti/'
classes = ['Box', 'car', 'Gabelstapler']
dataset_type = 'Custom3DDataset'
modality = dict(use_lidar=True, use_camera=False)

#dateien
ann_file_test = 'test_crop.pkl'
ann_file_train = 'train_crop.pkl'
ann_file_val = 'val_crop.pkl'

box_type_3d = 'LiDAR'
filter_empty_gt = False

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=classes),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=classes),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='PointShuffle'),
    dict(type='MultiScaleFlipAug3D', img_scale=(1, 1), pts_scale_ratio=1, flip=False,
        transforms=[
            dict(type='GlobalRotScaleTrans', rot_range=[0, 0], scale_ratio_range=[1., 1.], translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='DefaultFormatBundle3D', class_names=classes, with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
    )
]


# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='DefaultFormatBundle3D',
        class_names=classes,
        with_label=False),
    dict(type='PointShuffle'),
    dict(type='Collect3D', keys=['points'])
]


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='Custom3DDataset',
            data_root=data_root,
            ann_file = data_root + ann_file_train,
            box_type_3d='LiDAR',
            classes=classes,
            modality=modality,
            pipeline = train_pipeline)),
    val = dict(
        type='Custom3DDataset',
        data_root=data_root,
        ann_file=data_root + ann_file_val,
        modality=modality,
        classes=classes,
        box_type_3d='LiDAR',
        test_mode=True,
        pipeline=eval_pipeline),
    test = dict(
        type='Custom3DDataset',
        data_root=data_root,
        ann_file=data_root + ann_file_test,
        pipeline=eval_pipeline,
        filter_empty_gt=False,
        classes=classes,
        test_mode=True,
        modality=modality,
        box_type_3d='LiDAR'))

# Part-A2 uses a different learning rate from what SECOND uses.
lr = 0.001
optimizer = dict(lr=lr)
evaluation = dict(pipeline=eval_pipeline)
find_unused_parameters = True