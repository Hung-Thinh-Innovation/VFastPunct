<h1 align="center">🛠️VFastPunct</h1>

Code by 🧑‍💻**Trong-Dat Ngo**.

Fast punctuation and capitalization restoration using Transformer Models for 🇻🇳Vietnamese

## Installation
This repository is tested on 🐍Python 3.7+ and 🔥PyTorch 1.8.2+, as well as it works fine on macOS, Windows, Linux.
```bash
pip install VFastPunct
```

## Run
```python
>>> from vfastpunct import VFastPunct
>>> punct = VFastPunct(model_name='mBertPuncCap', no_cuda=False)
>>> punct('việt nam quốc hiệu chính thức là cộng hòa xã hội chủ nghĩa việt nam là một quốc gia nằm ở cực đông của bán đảo đông dương thuộc khu vực đông nam á giáp với lào campuchia trung quốc biển đông và vịnh thái lan')
'Việt Nam quốc hiệu chính thức là Cộng hòa Xã hội chủ nghĩa Việt Nam, là một quốc gia nằm ở cực Đông của bán đảo Đông Dương, thuộc khu vực Đông Nam Á, giáp với Lào, Campuchia, Trung Quốc, Biển Đông và Vịnh Thái Lan.'
```
>Arguments:
> 
> + *model_name*: The name of the architectural model that was utilized to restore punctuation and capitalization. Valid model name can be [*`mBertPunct`*, *`mBertPuncCap`*]
> + *no_cuda*:  Whether to not use CUDA even when it is available or not.