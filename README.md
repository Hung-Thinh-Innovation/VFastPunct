<h1 align="center">🛠️VFastPunct</h1>

Code by **Trong-Dat Ngo**.

Fast Punctuation Restoration using Transformer Models for Vietnamese

## Installation
This repository is tested on Python 3.7+ and PyTorch 1.8.2+, as well as it works fine on macOS, Windows, Linux.
```bash
pip install VFastPunct
```

## Run
```python
>>> from vfastpunct import VFastPunct
>>> punct = VFastPunct('mBertLstmCrf', no_cuda=False)
>>> punct('việt nam quốc hiệu chính thức là cộng hòa xã hội chủ nghĩa việt nam là một quốc gia nằm ở cực đông của bán đảo đông dương thuộc khu vực đông nam á giáp với lào campuchia trung quốc biển đông và vịnh thái lan')
'việt nam, quốc hiệu chính thức là cộng hòa xã hội chủ nghĩa việt nam, là một quốc gia nằm ở cực đông của bán đảo đông dương thuộc khu vực đông nam á, giáp với lào, campuchia, trung quốc, biển đông và vịnh thái lan.'

```
