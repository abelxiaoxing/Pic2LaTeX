
<div align="center">
    <h1>
        Pic2LaTeX
    </h1>
</div>

Pic2LaTeX 是一款类似于 Mathpix 的工具，能够将图片中的数学公式转换为 LaTeX 代码。

## 功能

- **图片转 LaTeX**：自动识别图片中的数学公式并生成对应的 LaTeX 代码。
- **高效准确**：基于深度学习模型，提供高精度的公式识别。
- **简单易用**：提供清晰的安装和使用指南，适合初学者和专业用户。

## 安装步骤

1. **克隆仓库**：

   ```bash
   git clone https://github.com/abelxiaoxing/Pic2LaTeX.git
   ```

2. **安装依赖**：

   确保已安装 Python 和 pip，然后运行以下命令安装所需依赖：

   ```bash
   pip install -r requirements.txt
   ```

3. **下载权重文件**：

   - 通过以下网盘链接下载权重文件：
     - 链接: [Pic2LaTeX_checkpoints.7z](https://pan.baidu.com/s/1OXGvr7bbOpFOJCBVrrBeJw?pwd=abel)
     - 提取码: `abel`
   - 下载完成后，解压文件到项目根目录，确保权重文件夹命名为 `checkpoints`。

## 使用方法

1. 确保完成上述安装步骤。
2. 打开浏览器 [本地8501端口](http://127.0.0.1:8501)。
3. 程序将自动识别上传图片内容并输出对应的 LaTeX 代码。

## 依赖

- Python 3.6+
- 详见 `requirements.txt` 文件

## 许可证

本项目采用 [MIT 许可证](LICENSE)。