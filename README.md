
<div align="center">
    <h1>
        <img src="./assets/fire.svg" width=30, height=30> 
        Pic2LaTeX
        <img src="./assets/fire.svg" width=30, height=30>
    </h1>
    <!-- <p align="center">
        🤗 <a href="https://huggingface.co/OleehyO/Pic2LaTeX"> Hugging Face </a>
    </p> -->
</div>


Pic2LaTeX是类mathpix软件,可以把图片转换为对应的latex公式


---


## 🚀 开搞

1. 克隆本仓库:

   ```bash
   git clone https://github.com/abelxiaoxing/Pic2LaTeX.git
   ```

2. 安装本项目的依赖包:

   ```bash
   pip install -r requirement.txt
   ```

3. 进入`src/`目录，在终端运行以下命令进行推理:

   ```bash
    python inference.py -img "/path/to/image.{jpg,png}" 
    # use --inference-mode option to enable GPU(cuda or mps) inference
    #+e.g. python inference.py -img "img.jpg" --inference-mode cuda
   ```

   > 第一次运行时会在Hugging Face上下载所需要的权重

### 段落识别

如演示视频所示，Pic2LaTeX还可以识别整个文本段落。尽管Pic2LaTeX具备通用的文本OCR能力，但我们仍然建议使用段落识别来获得更好的效果：

1. 下载公式检测模型的权重到`src/models/det_model/model/`目录 [[链接](https://huggingface.co/TonyLee1256/Pic2LaTeX_det/resolve/main/rtdetr_r50vd_6x_coco.onnx?download=true)]

2. `src/`目录下运行`inference.py`并添加`-mix`选项，结果会以markdown的格式进行输出。

   ```bash
   python inference.py -img "/path/to/image.{jpg,png}" -mix
   ```

Pic2LaTeX默认使用轻量的[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)模型来识别中英文，可以尝试使用更大的模型来获取更好的中英文识别效果：

| 权重 | 描述 | 尺寸 |
|-------------|-------------------| ---- |
| [ch_PP-OCRv4_det.onnx](https://huggingface.co/OleehyO/paddleocrv4.onnx/resolve/main/ch_PP-OCRv4_det.onnx?download=true) | **默认的检测模型**，支持中英文检测 | 4.70M |
| [ch_PP-OCRv4_server_det.onnx](https://huggingface.co/OleehyO/paddleocrv4.onnx/resolve/main/ch_PP-OCRv4_server_det.onnx?download=true) | 高精度模型，支持中英文检测 | 115M |
| [ch_PP-OCRv4_rec.onnx](https://huggingface.co/OleehyO/paddleocrv4.onnx/resolve/main/ch_PP-OCRv4_rec.onnx?download=true) | **默认的识别模型**，支持中英文识别 | 10.80M |
| [ch_PP-OCRv4_server_rec.onnx](https://huggingface.co/OleehyO/paddleocrv4.onnx/resolve/main/ch_PP-OCRv4_server_rec.onnx?download=true) | 高精度模型，支持中英文识别 | 90.60M |

把识别/检测模型的权重放在`src/models/third_party/paddleocr/checkpoints/`
下的`det/`或`rec/`目录中，然后重命名为`default_model.onnx`。

> [!NOTE]
> 段落识别只能识别文档内容，无法还原文档的结构。

## ❓ 常见问题：无法连接到Hugging Face

默认情况下，会在Hugging Face中下载模型权重，**如果你的远端服务器无法连接到Hugging Face**，你可以通过以下命令进行加载：

1. 安装huggingface hub包

   ```bash
   pip install -U "huggingface_hub[cli]"
   ```

2. 在能连接Hugging Face的机器上下载模型权重:

   ```bash
   huggingface-cli download \
       OleehyO/Pic2LaTeX \
       --repo-type model \
       --local-dir "your/dir/path" \
       --local-dir-use-symlinks False
   ```

3. 把包含权重的目录上传远端服务器，然后把 `src/models/ocr_model/model/Pic2LaTeX.py`中的 `REPO_NAME = 'OleehyO/Pic2LaTeX'`修改为 `REPO_NAME = 'your/dir/path'`

<!-- 如果你还想在训练模型时开启evaluate，你需要提前下载metric脚本并上传远端服务器：

1. 在能连接Hugging Face的机器上下载metric脚本

   ```bash
   huggingface-cli download \
       evaluate-metric/google_bleu \
       --repo-type space \
       --local-dir "your/dir/path" \
       --local-dir-use-symlinks False
   ```

2. 把这个目录上传远端服务器，并在 `Pic2LaTeX/src/models/ocr_model/utils/metrics.py`中把 `evaluate.load('google_bleu')`改为 `evaluate.load('your/dir/path/google_bleu.py')` -->

## 🌐 网页演示

进入 `src/` 目录，运行以下命令

```bash
./start_web.sh
```

在浏览器里输入 `http://localhost:8501`就可以看到web demo

> [!NOTE]
> 1. 对于Windows用户, 请运行 `start_web.bat`文件。
> 2. 使用onnxruntime + gpu 推理时，需要安装onnxruntime-gpu

## 🔍 公式检测

Pic2LaTeX的公式检测模型在3415张中文教材数据(130+版式)和8272张[IBEM数据集](https://zenodo.org/records/4757865)上训练得到，支持对整张图片进行**公式检测**。

<div align="center">
    <img src="det_rec.png" width=250> 
</div>

1. 下载公式检测模型的权重到`src/models/det_model/model/`目录 [[链接](https://huggingface.co/TonyLee1256/Pic2LaTeX_det/resolve/main/rtdetr_r50vd_6x_coco.onnx?download=true)]

2. `src/`目录下运行以下命令，结果保存在`src/subimages/`

   ```bash
   python infer_det.py
   ```

<details>
<summary>更进一步：公式批识别</summary>

在进行**公式检测后**，`src/`目录下运行以下命令

```shell
python rec_infer_from_crop_imgs.py
```

会基于上一步公式检测的结果，对裁剪出的所有公式进行批量识别，将识别结果在 `src/results/`中保存为txt文件。
</details>

## 📡 API调用

我们使用[ray serve](https://github.com/ray-project/ray)来对外提供一个Pic2LaTeX的API接口，通过使用这个接口，你可以把Pic2LaTeX整合到自己的项目里。要想启动server，你需要先进入 `src/`目录然后运行以下命令:

```bash
python server.py 
```

| 参数 | 描述 |
| --- | --- |
| `-ckpt` | 权重文件的路径，*默认为Pic2LaTeX的预训练权重*。|
| `-tknz` | 分词器的路径，*默认为Pic2LaTeX的分词器*。|
| `-port` | 服务器的服务端口，*默认是8000*。|
| `--inference-mode` | 使用"cuda"或"mps"推理，*默认为"cpu"*。|
| `--num_beams` | beam search的beam数量，*默认是1*。|
| `--num_replicas` | 在服务器上运行的服务副本数量，*默认1个副本*。你可以使用更多的副本来获取更大的吞吐量。|
| `--ncpu_per_replica` | 每个服务副本所用的CPU核心数，*默认为1*。|
| `--ngpu_per_replica` | 每个服务副本所用的GPU数量，*默认为1*。你可以把这个值设置成 0~1之间的数，这样会在一个GPU上运行多个服务副本来共享GPU，从而提高GPU的利用率。(注意，如果 --num_replicas 2, --ngpu_per_replica 0.7, 那么就必须要有2个GPU可用) |
| `-onnx` | 使用Onnx Runtime进行推理，*默认不使用*。|

> [!NOTE]
> 一个客户端demo可以在 `Pic2LaTeX/client/demo.py`找到，你可以参考 `demo.py`来给server发送请求

## 🏋️‍♂️ 训练

### 数据集

我们在 `src/models/ocr_model/train/dataset/`目录中提供了一个数据集的例子，你可以把自己的图片放在 `images`目录然后在 `formulas.jsonl`中为每张图片标注对应的公式。

准备好数据集后，你需要在 `**/train/dataset/loader.py`中把 **`DIR_URL`变量改成你自己数据集的路径**

### 重新训练分词器

如果你使用了不一样的数据集，你可能需要重新训练tokenizer来得到一个不一样的词典。配置好数据集后，可以通过以下命令来训练自己的tokenizer：

1. 在`src/models/tokenizer/train.py`中，修改`new_tokenizer.save_pretrained('./your_dir_name')`为你自定义的输出目录

   > 注意：如果要用一个不一样大小的词典(默认1.5W个token)，你需要在`src/models/globals.py`中修改`VOCAB_SIZE`变量

2. **在`src/`目录下**运行以下命令:

   ```bash
   python -m models.tokenizer.train
   ```

### 训练模型

1. 修改`src/train_config.yaml`中的`num_processes`为训练用的显卡数(默认为1)

2. 在`src/`目录下运行以下命令：

   ```bash
   accelerate launch --config_file ./train_config.yaml -m models.ocr_model.train.train
   ```

你可以在`src/models/ocr_model/train/train.py`中设置自己的tokenizer和checkpoint路径（请参考`train.py`）。如果你使用了与Pic2LaTeX一样的架构和相同的词典，你还可以用自己的数据集来微调Pic2LaTeX的默认权重。

> [!NOTE]
> 我们的训练脚本使用了[Hugging Face Transformers](https://github.com/huggingface/transformers)库, 所以你可以参考他们提供的[文档](https://huggingface.co/docs/transformers/v4.32.1/main_classes/trainer#transformers.TrainingArguments)来获取更多训练参数的细节以及配置。

## 📅 计划

- [X] ~~使用更大的数据集来训练模型~~
- [X] ~~扫描图片识别~~
- [X] ~~中英文场景支持~~
- [X] ~~手写公式识别~~
- [ ] PDF文档识别
- [ ] 推理加速

## ⭐️ 观星曲线

[![Stargazers over time](https://starchart.cc/OleehyO/Pic2LaTeX.svg?variant=adaptive)](https://starchart.cc/OleehyO/Pic2LaTeX)

## 👥 贡献者

<a href="https://github.com/OleehyO/Pic2LaTeX/graphs/contributors">
   <a href="https://github.com/OleehyO/Pic2LaTeX/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=OleehyO/Pic2LaTeX" />
   </a>
</a>
