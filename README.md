# AI Just Got Five Times Cheaper. The Expensive Model Didn't Disappear. It Became a Teacher.

Code for the Medium article: an expensive teacher (3-MLP ensemble, 2.07M ops/prediction) and tiny NumPy students (8–128 units) on Fashion-MNIST, trained on human labels or by distillation (Hinton et al., 2015), plus 8/4-bit quantization.

| Result | Number |
|---|---|
| Teacher accuracy | 89.0% |
| Student, 128 units (1/20 compute), all labels | 88.5% |
| Student, 32 units: labels vs distillation (all labels) | 87.3% vs 86.0% |
| Student, 32 units, 500 labels: alone vs + teacher labels | 73.4% vs 85.4% |
| 8-bit / 4-bit (32-unit student) | 87.1% / 76.5% |

```bash
pip install -r requirements.txt
bash run_all.sh
```
License: MIT. Fashion-MNIST © Zalando SE, MIT License.
