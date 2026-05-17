# Pull-up Pose Assessment

一个基于 Flask、MediaPipe 和 OpenCV 的引体向上视频测评系统：学生上传动作视频，系统提取人体关键点，与标准动作或历史动作对比，输出评分、差异关节、关键帧和训练建议。

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-web%20app-000000?logo=flask)
![MediaPipe](https://img.shields.io/badge/MediaPipe-pose%20landmarks-0097A7)
![License](https://img.shields.io/badge/license-MIT-green)

关键词：引体向上识别、姿态估计、动作评分、运动视频分析、MediaPipe Pose、pull-up counter、pose assessment。

## 🌱 这个 repo 是什么

前阵子做过一个引体向上动作分析系统。

一开始看起来只是识别一次动作有没有完成，但真做起来才发现，难点不止在“数次数”：

- 同一个动作从正面、侧面、斜侧面拍，能看到的问题完全不一样。
- 姿态关键点有置信度波动，不能只看单帧结果。
- 标准动作需要按视角管理，学生动作也要能和历史动作对比。
- 分析结果不能只给一个分数，要告诉用户差异最大的位置和对应关键帧。
- Web 端要让教师和学生都能自然使用，上传、等待、查看历史、看结果都要闭环。

所以我把它整理成一个可运行的本地 Web 应用。它保留了用户登录、视频上传、异步分析、标准视频管理、关键帧对照、历史记录和姿态分析 pipeline，适合作为运动分析和计算机视觉作品参考。

## 🚀 先跑一下

```bash
git clone https://github.com/Ce-Legend/pullup-pose-assessment.git
cd pullup-pose-assessment
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

初始化数据库并创建一个教师账号：

```bash
python -m server.cli init-db
python -m server.cli create-user --username teacher --password change-me --role teacher --display-name Teacher
```

用 mock 模式先跑通 Web 流程：

```bash
set MOCK_ANALYSIS=1
python -m server.app
```

另开一个终端启动 worker：

```bash
set MOCK_ANALYSIS=1
python -m server.worker
```

浏览器打开：

```text
http://127.0.0.1:5000
```

真实姿态分析需要本地准备标准动作视频。教师登录后，在标准视频页面分别上传正面、侧面或斜侧面的标准视频，再关闭 mock 模式启动 worker。

## 📦 里面有什么

```text
.
├── server/
│   ├── app.py                 # Flask 页面和 API
│   ├── worker.py              # 后台分析 worker
│   ├── cli.py                 # 初始化数据库、创建用户、配置标准视频
│   ├── schema.sql             # SQLite schema
│   ├── analysis/
│   │   ├── pipeline.py        # 姿态提取、动作分段、评分、关键帧生成
│   │   ├── mp4.py             # 轻量 MP4 时长解析
│   │   └── errors.py          # 分析错误类型
│   ├── templates/             # 登录、上传、历史、结果、标准视频页面
│   └── static/                # 页面样式
├── requirements.txt
├── requirements-analysis.txt
└── pyproject.toml
```

## 🧭 我最想留下来的几个经验

### 1. 动作识别不是只数次数

引体向上这种动作表面上很好判断：上去、下来，算一次。

但教学反馈需要的是更细的结果：上拉阶段是否充分、身体是否摆动、左右是否对称、顶点是否到位。只给一个完成次数，解释不了用户应该怎么改。

所以系统会输出总分、最大差异关节、关键帧对照和建议，把识别结果变成可讲解的反馈。

### 2. 视角要成为数据结构的一部分

正面适合看左右对称和握距，侧面更适合看摆动和蹬腿，斜侧面则是折中视角。

如果只维护一个标准视频，很多差异会被拍摄角度放大或掩盖。这个项目把 `front`、`side`、`angle` 放进数据库和分析记录里，每个视角可以有自己的标准动作。

### 3. 关键帧比平均分更容易被理解

平均分很适合排序，但不适合教学。

结果页里保留了起始、顶点、结束、最大差异几类关键帧。用户能看到标准动作和自己的动作并排对照，教师也能围绕某一帧讲动作问题。

### 4. 分析任务要异步化

视频分析可能需要十几秒，甚至更久。

这里没有让上传请求一直阻塞，而是先创建分析记录，再由 worker 领取队列任务，页面通过状态和进度查询结果。这个结构以后也方便换成 Celery、RQ 或别的任务队列。

### 5. Mock 模式很适合做演示和开发

真实姿态分析依赖视频质量、模型文件和标准动作配置。为了让 Web 流程先跑起来，项目保留了 `MOCK_ANALYSIS=1`。

这样可以先验证登录、上传、队列、结果页和关键帧展示，再切回真实 pipeline 排查姿态识别问题。

## 🧩 分析流程

```text
上传 MP4
  -> 写入 analyses 队列记录
  -> worker 领取任务
  -> 校验视频时长和格式
  -> 选择标准动作或历史动作
  -> MediaPipe 提取人体关键点
  -> 对齐动作阶段并计算差异
  -> 生成评分、建议、关键帧图片和 result.json
  -> Web 结果页展示
```

## ✅ 校验方式

```bash
python -m pytest -q
```

当前测试重点放在轻量、稳定的基础模块：

- MP4 时长解析失败路径。
- 设置项默认目录。
- mock 分析产物结构。

## 🙌 参考

- [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)：人体关键点模型。
- [OpenCV](https://opencv.org/)：视频读取和帧处理。
- [Flask](https://flask.palletsprojects.com/)：Web 应用框架。
- [SQLite](https://www.sqlite.org/)：本地分析记录和任务队列。

## 📄 License

MIT
