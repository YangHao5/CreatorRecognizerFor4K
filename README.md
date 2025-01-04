# Creator Recognizer For 4K

[Creator Recognizer For 4K](https://github.com/YangHao5/CreatorRecognizerFor4K) | [Yarn Yang](https://github.com/YangHao5) | `haoyang.official@gmail.com`

## Description

This project aims to identify the creator of a 4-key rhythm game chart.

The workflow of the code follows these steps: First, it converts each chart into a `.pt` file, which contains both the creator’s information and the note arrangement. Next, the dataset is split into training and validation sets. The model is then trained and validated before being tested on a given dataset (by default, the entire dataset).

The model supports various chart file formats. This project includes a Malody converter that transforms `.mc` (Malody chart files) into `.pt` files. Users can develop their own converters, such as an osu!mania chart converter for `.osz` files.

This work utilizes the Malody Stable Dataset for training, validation, and testing. The dataset comprises all 4-key stable charts up to October 7, 2022, as well as those released between July 16, 2024, and January 5, 2025. The charts before October 7, 2022, were provided by woc2006, while I contributed the remaining data. Their support is greatly appreciated.

Contributions of converters and datasets (chart files) from other rhythm games are highly encouraged and welcomed for public use.

## Usage

1. Download the Malody Stable Dataset or prepare your own dataset. I'll take Malody Stable Dataset as an example. Extract the packs to `beatmap`. Then use a converter to generate `.pt` files:
```
python mc2pt.py
```
After running this command, the following files will be created:
```
processed_tensors/5809.pt
processed_tensors/6349.pt
...
```
2. Train, validate and test the model. Simply run
```
python main.py
```
You can make custom settings in `main.py`. Testing result of a pre-trained model can be seen as `pretrained_result.csv`.

## Copyright
### For the Code

Modifications, developments, and publications are permitted; however, you must contact me in advance, cite this project in your work, and ensure that your work is publicly accessible. Commercial use is strictly prohibited.

### For the Dataset

Modification of any chart without the creator\'s permission is not allowed. However, contributions of new datasets are highly encouraged and greatly appreciated. Creators of the charts included in the dataset can be seen in `creators.csv`.

**Download Links**

[Baidu Drive](https://pan.baidu.com/s/1WQ9qTu14L48FeUK_Z4b1qw?pwd=7891) for users in China mainland. Whole dataset pack.

[lanzouyun](https://wwtb.lanzouv.com/b00y9rzyvc) (`atok`) for users in China mainland. Batched dataset.

[Google Drive](https://drive.google.com/file/d/1nScZVs2hEbMsjGOyGY-JKd9z7QmMDkIS/view?usp=sharing) for other users. Whole dataset pack.

## Contact

E-mail: `haoyang.official@gmail.com`

QQ: `2034329047`

If you do not wish for your chart to be used for analysis, please contact me, and I will remove it from the dataset.
