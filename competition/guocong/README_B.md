# 郭聪（B）视觉感知交付说明

本目录按 2026 文旅搬运赛题修订稿、Q&A、README 和变更说明实现。正式比赛的 Server/Client 通过 ROS2 通信，客户端不能直接读取服务端 Python `obs`。纯 Python 算法层可在 Windows 离线测试，ROS2 适配层在 Linux 比赛镜像运行。

## 新赛题中 B 的职责

一次 600 秒运行内连续处理三项任务：

1. 桌边指定颜色包装盒 → 货架空层。
2. 货架指定颜色包装盒 → 任务一包装盒原来的桌边位置。
3. 白色立方体顶部指定颜色包装盒 → 货架旁白色长方体障碍物左侧。

移动物体为粉、黄、棕三种包装盒，尺寸为 `24 × 16 × 19 cm`。B 不硬编码颜色顺序、包装盒初始位置或货架层号；`/material/instruction` 给任务目标，包装盒当前坐标由实时视觉确定。

## 相机结论

| 相机 | 正式 ROS2 话题 | 数据 | 作用 |
|---|---|---|---|
| 头部相机 | `/head_camera/color/image_raw` | RGB，默认 640×480 | 大范围寻找桌边、货架和立方体上的包装盒 |
| 头部相机 | `/head_camera/aligned_depth_to_color/image_raw` | 对齐深度，`mono16`，毫米 | 2D→3D 世界坐标定位 |
| 左手眼 | `/left_camera/color/image_raw` | RGB | 左臂抓取前二维精定位 |
| 右手眼 | `/right_camera/color/image_raw` | RGB | 右臂抓取前二维精定位 |

验收问题的答案是：**头部 RGB-D 最适合寻找包装盒；执行抓取一侧的手眼 RGB 最适合最后精定位。** 左右手眼没有正式深度话题，不能直接把手眼像素反投影为三维位置。

旧版 DISCOVERSE 本地仿真的 `cam_id=0/1/2` 分别对应 `head_cam/lft_handeye/rgt_handeye`；正式新赛题客户端应使用上表的命名 ROS2 话题，不依赖数字 `cam_id`。

## 实现结构

```text
guocong/
├── config/perception.yaml
├── src/common/types.py
├── src/perception/
│   ├── color_detector.py          # 粉/黄/棕 HSV 检测
│   ├── reference_detector.py      # 白色参考物场景校验
│   ├── rgbd_localizer.py          # mono16 深度、2D→3D、世界坐标
│   ├── perception_pipeline.py     # 检测 + 定位 -> SceneState
│   ├── wrist_refiner.py           # 左右手眼二维居中误差
│   ├── ros_perception_node.py     # 头部 RGB-D 正式 ROS2 节点
│   └── ros_wrist_refiner_node.py  # 左右手眼精定位 ROS2 节点
├── tools/
│   ├── inspect_observation.py     # 三 RGB/头部 Depth 检查与保存
│   ├── detect_color.py            # 单张图片调阈值
│   └── evaluate_perception.py     # 成功率/误检率/失败样本
└── tests/
```

## 核心算法

### 1. HSV 颜色检测

ROS 图像通过 `cv_bridge` 转为 BGR。`color_detector.py` 根据 `perception.yaml` 的三组 HSV 阈值生成掩膜，做开/闭运算，最后使用连通域面积、填充率和面积上限过滤背景。参数只存放在 YAML，取得正式环境图片后可以调参而不改代码。

### 2. RGB-D 2D→3D

检测框中心 `(u,v)` 附近取有效深度中位数。正式深度单位是毫米，因此先乘 `0.001` 转为米，再用相机内参反投影：

```text
X = (u-cx)·Z/fx
Y = (v-cy)·Z/fy
Z = depth_mm·0.001
```

随后通过 ROS TF 的 `world <- head_camera_optical_frame` 变换得到世界坐标。深度为 0、超范围、RGB/Depth 时间差过大或 TF 不可用时，本帧不会发布错误三维坐标。

### 3. SceneState 与 ROS 输出

`PerceptionPipeline` 输出 `SceneState`，键包括 `pink_box`、`yellow_box`、`brown_box`、`white_cube` 和 `white_shelf_obstacle`。正式节点把完整状态作为 JSON 发布到 `/material/perception/scene_state`，并同时发布官方示例兼容的 `/material/detections`（`vision_msgs/Detection3DArray`）：

- `header.frame_id = world`
- 包装盒 `class_id = pink/yellow/brown`
- `pose.position` 为米制世界坐标
- 包装盒尺寸为 `0.24/0.16/0.19 m`

参考物 HSV 检测用于场景校验。正式任务的放置目标应优先采用指令里的 `place_world/place_type/place_radius`，不要用不稳定的白色检测覆盖裁判给出的目标坐标。

### 4. 手眼精定位

`WristRefiner` 在对应手眼 RGB 上重新检测目标颜色，输出相对图像中心的像素误差和归一化误差。C 模块可以据此做小步视觉伺服；手眼无深度，所以最终三维初值仍来自头部 RGB-D。

## Linux 正式环境运行

进入客户端容器并设置：

```bash
cd /workspace/competition/guocong
export PYTHONPATH="$PWD:$PYTHONPATH"
export ROS_DOMAIN_ID=99
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

先检查观测：

```bash
python3 tools/inspect_observation.py --output recordings/observation_inspection
```

输出：

```text
frame_rgb.png
frame_depth.npy
frame_depth_preview.png
left_rgb.png
right_rgb.png
observation_report.json
```

报告会打印 `obs.keys()`、各数组 shape/dtype/min/max。正式文档已确认头部 RGB 默认 640×480、头部深度存在且为毫米；实际 Depth 范围和包装盒完整可见性必须以这次运行保存的帧为准。

启动感知节点：

```bash
python3 -m src.perception.ros_perception_node --config config/perception.yaml
```

可选启动手眼精定位节点：

```bash
python3 -m src.perception.ros_wrist_refiner_node --config config/perception.yaml
```

它从 `/material/instruction` 更新目标颜色，并在 `/material/perception/wrist_alignment` 发布左右手眼的 `error_px`、`error_normalized`、`confidence` 和 `aligned` JSON。联调时也可以用 `--target-color pink` 手工固定颜色。

另开终端验收：

```bash
ros2 topic hz /material/detections
ros2 topic echo /material/detections
ros2 topic echo /material/perception/result_image --once
```

## Windows 离线验证

```powershell
cd E:\2026jiqiren\tourism-robot2026-main\tourism-robot2026-main\competition\guocong
D:\Anaconda\python.exe -m pytest -q -p no:cacheprovider
D:\Anaconda\python.exe tools\detect_color.py frame_rgb.png --output detections.png
```

Windows 没有 ROS2 Python 包时不能启动 `ros_perception_node.py`，但检测、深度定位、参考物、SceneState 和手眼误差可以完整测试。

Windows/PyCharm 中检查本地 DISCOVERSE 三相机：

```powershell
.\.venv\Scripts\python.exe competition\guocong\tools\inspect_sim_observation.py
```

该脚本是本地开发入口，会显式开启仿真三相机深度并输出 `recordings/sim_observation`；正式比赛不能直接读取这里的 `obs`，仍应使用前面的 ROS2 节点。

## 成功率、误检率与失败样本

标注 CSV 路径相对 CSV 文件：

```csv
image,pink,yellow,brown
frames/001.png,1,0,0
frames/002.png,0,1,1
```

运行：

```bash
python3 tools/evaluate_perception.py labels.csv
```

输出 `perception_metrics.json`，包含 TP/FP/FN/TN、召回成功率、精确率和误检率；误检或漏检图片复制到 `failure_cases/perception/`。需要分别覆盖桌边左右、三层货架、白色立方体顶部、远距离、小面积、遮挡、深度空洞和强光背景。

## A/C/D 集成方法

- A：把 `guocong` 挂载进客户端镜像，先运行观测检查，再常驻启动 ROS2 感知节点。
- C：从 `/material/detections` 取得目标颜色世界坐标；接近目标后使用对应手眼二维误差闭环。
- D：解析 `/material/instruction`，根据 `target_color` 选取最新、最高置信度且有三维坐标的目标；放置使用指令 `place_world`。

冻结参数前至少在多个随机种子下记录成功率与误检率。HSV 是不依赖权重的可靠基线；如果实际图片颜色漂移明显，可替换为官方 YOLO 示例，但保持 `SceneState` 和 `/material/detections` 接口不变。
