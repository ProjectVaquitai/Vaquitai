## 前置条件

* 推荐 Python==3.8
* gcc >= 5 (at least C++14 support)

## 安装

### 从源码安装

* 运行以下命令以安装最新版本

```shell
cd <path_to_data_juicer>
pip install -v -e .[all]
```

## 快速上手

### 数据处理

* 以配置文件路径作为参数来运行 `process_data.py` 或者 `dj-process` 命令行工具来处理数据集。

```shell
# 适用于从源码安装
python tools/process_data.py --config configs/demo/process.yaml

# 使用命令行工具
dj-process --config configs/demo/process.yaml
```

```shell
streamlit run app.py
```

### 构建配置文件

* 配置文件包含一系列全局参数和用于数据处理的算子列表。您需要设置:
  * 全局参数：输入/输出 数据集路径，worker 进程数量等。
  * 算子列表：列出用于处理数据集的算子及其参数。
* 您可以通过如下方式构建自己的配置文件:
  * ➖：修改我们的样例配置文件 [`config_all.yaml`](configs/config_all.yaml)。该文件包含了**所有**算子以及算子对应的默认参数。您只需要**移除**不需要的算子并重新设置部分算子的参数即可。
  * ➕：从头开始构建自己的配置文件。您可以参考我们提供的样例配置文件 [`config_all.yaml`](configs/config_all.yaml)，[算子文档](docs/Operators_ZH.md)，以及 [开发者指南](docs/DeveloperGuide_ZH.md#构建自己的算子).
  * 除了使用 yaml 文件外，您还可以在命令行上指定一个或多个参数，这些参数将覆盖 yaml 文件中的值。

```shell
python xxx.py --config configs/demo/process.yaml --language_id_score_filter.lang=en
```

* 基础的配置项格式及定义如下图所示

  ![基础配置项格式及定义样例](https://img.alicdn.com/imgextra/i4/O1CN01xPtU0t1YOwsZyuqCx_!!6000000003050-0-tps-1692-879.jpg "基础配置文件样例")

### 预处理原始数据（可选）

* 我们的 Formatter 目前支持一些常见的输入数据集格式：
  * 单个文件中包含多个样本：jsonl/json、parquet、csv/tsv 等。
  * 单个文件中包含单个样本：txt、code、docx、pdf 等。
* 但来自不同源的数据是复杂和多样化的，例如:
  * [从 S3 下载的 arXiv 原始数据](https://info.arxiv.org/help/bulk_data_s3.html) 包括数千个 tar 文件以及更多的 gzip 文件，并且所需的 tex 文件在 gzip 文件中，很难直接获取。
  * 一些爬取的数据包含不同类型的文件（pdf、html、docx 等），并且很难提取额外的信息，例如表格、图表等。
* Data-Juicer 不可能处理所有类型的数据，欢迎提 Issues/PRs，贡献对新数据类型的处理能力！
* 因此我们在 [`tools/preprocess`](tools/preprocess) 中提供了一些**常见的预处理工具**，用于预处理这些类型各异的数据。
  * 欢迎您为社区贡献新的预处理工具。
  * 我们**强烈建议**将复杂的数据预处理为 jsonl 或 parquet 文件。