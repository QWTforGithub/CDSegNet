_base_ = ["../_base_/default_runtime.py"]

# ---- common ---
batch_size = 8 # bs=2 for 1 GPU, bs=4 for 2 GPUs, bs=8 for 4GPUs
num_worker = 16 # the num_worker is double batch_size.
mix_prob = 0.8

empty_cache = False
enable_amp = True

seed = 54421566 # 54421566, 42
gredient_clip = []
ignore_index = -1
# ---- common ---

# ---- Seg Model ----
condition = True
dm = True
dm_input = "xt"
dm_target = "noise"
dm_min_snr = None

T = 1000
T_dim = 128
beta_start = 0.002
beta_end = 0.003
noise_schedule = 'linear'

c_in_channels = 4
n_in_channels = 4
# ---- Seg Model ----

# ---- loss ----
loss_type = "GLS" # "EW", "GLS"
task_num = 2
# ---- loss ----

# --- backbone ---
enable_rpe = False
enable_flash = True

num_classes=  16
tm_bidirectional = False
tm_feat = 1.0 # "channel_scale", "b_channel_scale", "lr_scale", "b_lr_scale", 1.0
tm_restomer = False

skip_connection_mode = "cat" # "cat", "add", "cat_all"
b_factor = [1.0, 1.0, 1.0, 1.0]
s_factor = [1.0, 1.0, 1.0, 1.0]
skip_connection_scale = True
skip_connection_scale_i = False
# --- backbone ---

# model settings
model = dict(
    type="DefaultSegmentorV2",

    backbone=dict(
        type="PT-v3m1",
        c_in_channels=c_in_channels,
        n_in_channels=n_in_channels,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),

        c_stride=(4, 4),
        c_enc_depths=(2, 2, 2),
        c_enc_channels=(32, 64, 128),
        c_enc_num_head=(2, 4, 8),
        c_enc_patch_size=(1024, 1024, 1024),
        c_dec_depths=(2, 2),
        c_dec_channels=(64, 64),
        c_dec_num_head=(4, 4),
        c_dec_patch_size=(1024, 1024),

        n_stride=(2, 2, 2, 2),
        n_enc_depths=(2, 2, 2, 6, 2),
        n_enc_channels=(32, 64, 128, 256, 512),
        n_enc_num_head=(2, 4, 8, 16, 32),
        n_enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        n_dec_depths=(2, 2, 2, 2),
        n_dec_channels=(64, 64, 128, 256),
        n_dec_num_head=(4, 4, 8, 16),
        n_dec_patch_size=(1024, 1024, 1024, 1024),

        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=enable_rpe,
        enable_flash=enable_flash,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),

        num_classes=num_classes,
        T_dim=T_dim,
        tm_bidirectional=tm_bidirectional,
        tm_feat=tm_feat,
        tm_restomer=tm_restomer,
        condition=condition,

        skip_connection_mode=skip_connection_mode,
        b_factor=b_factor,
        s_factor=s_factor,
        skip_connection_scale=skip_connection_scale,
        skip_connection_scale_i=skip_connection_scale_i
    ),
    criteria=[
        dict(type="MSELoss", loss_weight=1.0, ignore_index=ignore_index, batch_sample_point=-1),
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
    ],

    loss_type=loss_type,
    task_num=task_num,

    num_classes=num_classes,
    T=T,
    beta_start=beta_start,
    beta_end=beta_end,
    noise_schedule=noise_schedule,
    T_dim=T_dim,
    dm=dm,
    dm_input=dm_input,
    dm_target=dm_target,
    dm_min_snr=dm_min_snr,
    condition=condition,
    c_in_channels=c_in_channels
)

# scheduler settings
epoch = 50
eval_epoch = 50

optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.50,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]


# dataset settings
dataset_type = "NuScenesDataset"
data_root = "data/nuscenes"
ignore_index = ignore_index
names = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            # dict(type="PointClip", point_cloud_range=(-51.2, -51.2, -4, 51.2, 51.2, 2.4)),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.025,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "strength"),
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
        ignore_index=ignore_index,
    ),
)
