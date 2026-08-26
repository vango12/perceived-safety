# Input data

The training workbook is not committed to this public repository. Copy the supplied `实验数据记录表.xlsx` here under a local name such as `experiment_data.xlsx`, or pass an absolute path to `scripts/train_and_evaluate.py`.

The loader reads the first five columns of the `experiment_data` sheet:

1. participant ID
2. experiment group
3. speed scale factor used by the experiment
4. human–robot distance in metres
5. perceived-safety score from 1 to 10

The code converts the source speed variable to actual robot motion speed with \(\omega=1.5v_s\) rad/s before fitting the final feature vector \([\omega,d,d^2,\omega d]\).
