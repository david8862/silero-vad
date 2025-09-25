# 微调Silero-VAD模型

> 此微调代码是在俄罗斯小型创新企业援助基金会 (FASIE) 的支持下创建的，是“俄罗斯联邦数字经济”国家计划下“人工智能”联邦项目的一部分。

此微调用于提升Silero-VAD模型在自定义数据上的语音检测质量

## 依赖项
微调VAD模型时使用以下依赖项:
- `torchaudio>=0.12.0`
- `omegaconf>=2.3.0`
- `sklearn>=1.2.0`
- `torch>=1.12.0`
- `pandas>=2.2.2`
- `tqdm`

## 数据准备

用于调优的数据帧必须以`.feather`格式准备并保存。训练和验证`.feather`文件中需要包含以下几列:
- **audio_path** - 磁盘上音频文件的绝对路径。音频文件应为`PCM`数据，最好为`.wav`或`.opus`格式(其他常用音频格式也支持)。为了加快再训练过程，建议将音频文件预采样(更改采样频率)至16000Hz
- **speech_ts** - 相应音频文件的标记。这是一个由{'start': START_SEC, 'end': 'END_SEC'}格式的字典组成的列表，其中`START_SEC`和`END_SEC`分别是语音片段的开始和结束时间(以秒为单位)。为了获得高质量的再训练，建议使用精度为30毫秒的标记。

再训练阶段使用的数据越多，适配后的模型在目标域上的有效性就越高。音频长度不受限制，因为每个音频样本在输入神经网络之前会被截断为`max_train_length_sec`秒。最好将较长的音频样本预先切割成`max_train_length_sec`长度的片段

`.feather`数据帧的示例可以在文件`example_dataframe.feather`中找到

## 配置文件`config.yml`

`config.yml`配置文件包含训练集和验证集的路径，以及用于重新训练的参数:
- `train_dataset_path` - `.feather`格式的训练数据框的绝对路径。它必须包含“数据准备”部分中描述的`audio_path`和`speech_ts`列。该数据框的示例可以在`example_dataframe.feather`中找到;
- `val_dataset_path` - `.feather`格式的验证数据框的绝对路径。它必须包含“数据准备”部分中描述的`audio_path`和`speech_ts`列。该数据框的示例可以在`example_dataframe.feather`中找到;
- `jit_model_path` - `.jit`格式的Silero-VAD模型的绝对路径。如果此字段留空，则将根据`use_torchhub`字段的值从代码库加载模型
- `use_torchhub` - 如果为`True`，则将使用torch.hub加载模型进行再训练。如果为`False`，则将使用silero-vad库（必须使用`pip install silero-vad`安装）加载模型进行再训练;
- `tune_8k` - 此参数指定要再训练哪个Silero-VAD头。如果为`True`，则将再训练采样率为8000Hz的头；否则，将再训练采样率为16000Hz的头:
- `model_save_path` - 保存再训练模型的路径;
- `noise_loss` - 应用于非语音音频窗口的损失系数;
- `max_train_length_sec` - 再训练阶段的最大音频长度（以秒为单位）。较长的音频将被截断为此长度;
- `aug_prob` - 在再训练阶段对音频文件应用增强的概率;
- `learning_rate` - 再训练速率;
- `batch_size` - 用于重新训练和验证的批次大小;
- `num_workers` - 用于加载数据的线程数;
- `num_epochs` - 重新训练的epoch数。所有训练数据在一个epoch中运行;
- `device` - `cpu`或`cuda`.

## 重新训练

使用以下命令启动重新训练

`python tune.py`

训练持续`num_epochs`个周期。验证集上ROC-AUC指标的最佳检查点将以jit格式保存在`model_save_path`中.

## 查找阈值

可以使用以下命令选择进入和退出阈值

`python search_thresholds`

此脚本使用上述配置文件。配置中指定的模型将用于在验证数据集上查找最佳阈值.

## 引文

```
@misc{Silero VAD,
  author = {Silero Team},
  title = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snakers4/silero-vad}},
  commit = {insert_some_commit_here},
  email = {hello@silero.ai}
}
```
