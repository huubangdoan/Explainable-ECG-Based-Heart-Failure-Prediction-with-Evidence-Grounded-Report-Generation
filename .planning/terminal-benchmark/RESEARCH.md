# Primary evidence — checked 2026-11-01 for Line

## Benchmark

Pinned authors repository: https://github.com/helme/ecg_ptbxl_benchmarking/tree/cdbf4e66d7e57d9b6a2657b6024716212b8d0afa
Paper: https://arxiv.org/pdf/2004.13701
Dataset: https://physionet.org/content/ptb-xl/1.0.3/

Read upstream source with GitHub file-content API at that exact commit (not an unversioned secondary tutorial): code/experiments/scp_experiment.py, code/utils/utils.py, code/models/fastai_model.py, resnet1d.py, basic_conv1d.py, timeseries_utils.py, code/configs/fastai_configs.py.

- Default experiments use 100 Hz, folds1–8 train,9 validation,10 test. Diagnostic superclass is multilabel NORM,MI,STTC,CD,HYP; use diagnostic SCP code presence, including likelihood0 meaning unspecified certainty. It is not heart-failure ground truth.
- utils.py preprocess_signals fits StandardScaler on np.vstack(X_train).flatten()[:,None]: one global scalar, not per-lead. No mandatory extra bandpass in this reference path.
- fastai_model.py constructor defaults input_size=2.5 seconds, batch128, epochs50, lr.01, wd.01, chunkify_trainFalse/chunkify_validTrue, validation stride half window, aggregate_fn=max. Train random crop, validation record-level maximum of window probabilities. Paper text discusses5s crops, while pinned code defaults2.5: this package explicitly follows the pinned code.
- resnet1d_wang is ResNet1d(BasicBlock1d,[1,1,1]), stem k7,stride1,no maxpool,width128; fix_feature_dimTrue, block strides1,2,2. With caller kernel_size5, each block has two convolutions k5,k3; biasFalse, BN,ReLU. Projection1x1+BN for changed stride. Adaptive max/mean concat pooling. Head [256,128,5], BN before each linear, dropout.25 then.5, ReLU between the two linear layers, final logits. Do not substitute the different canonical three-convolution Wang architecture or resnet18.
- Upstream uses Fastai fit_one_cycle plus LR finder; this package independently implements modern AdamW/OneCycleLR with fixed hyperparameters and best validation checkpoint. This is benchmark-aligned, not an exact historical reproduction. Attribute upstream research; do not vendor old Fastai or copy its entire GPL implementation.
- Initialization verified separately: fastai_model.py:228 applies weight_init; basic_conv1d.py:73–81 initializes both Conv1d and Linear with kaiming_normal_ default fan_in, zeros any bias, BatchNorm unit weight/zero bias. The new port must preserve this; fan_out Conv plus default Linear would be an unnecessary departure.
- Original publication used an earlier dataset release. v1.0.3 differs (including duplicate removal). Our common cohort also excludes constant-lead train ECG12722. Historical headline scores are contextual, not directly comparable numbers on identical data.
- Paper appendix found no significant overall benefit from500Hz in its experiments. This does not prove higher frequency is universally useless. Retain frequency/filter/normalization as controlled empirical questions.

## Runtime and Windows

Primary commands checked at https://pytorch.org/get-started/previous-versions/ : torch==2.14.1 Windows/Linux CUDA12.6 index https://download.pytorch.org/whl/cu126 . Deliberately pin this verified version, do not call it latest. Version-specific PyPI metadata confirms torch2.14.1 requires Python>=3.10 and has a cp312-win_amd64 wheel; choose Python3.12. NumPy2.5.4 and SciPy1.18.1 require>=3.12 and have Windows312 wheels; pandas3.1.0 also has a Windows312 wheel; WFDB4.3.1 has a universal wheel. No torchvision needed. This is availability verification, not a performed installation on the target.

Current PyTorch docs checked through Context7 /pytorch/pytorch/v2.14.1: Windows DataLoader uses spawn, so entrypoint must have main guard and datasets/collate must be module-level/picklable. torch.amp.autocast('cuda') and torch.amp.GradScaler('cuda') are supported. torch.load defaults weights_onlyTrue: store checkpoint tensors/primitives and explicit RNG conversions; do not rely on unrestricted pickle for foreign artifacts.

Exact CUDA build additionally verified from the official https://download.pytorch.org/whl/cu126/torch/ wheel index: torch-2.14.1+cu126-cp312-cp312-win_amd64.whl, published SHA25697e3bc87ea74367518b740d912a7f5259c25a3c8115bf5916abee1d53ff115a4. Setup pins torch==2.14.1+cu126 so an existing CPU/other-CUDA installation is not silently treated as satisfying the requirement; preflight checks the exact build. No wheel was installed or executed on the laptop.

SciPy primary APIs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html and https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html . Full-record forward/backward SOS filtering has edge effects, no phase lag but not guaranteed morphology preservation; fourth-order design applied both directions. Polyphase resampling anti-aliases, specify time axis. Treat0.5–40Hz as an ablation, not universally best cleaning.

## Existing release and transfer

Existing manifest/audit are local primary evidence, with prior full audit in artifacts. Do not claim those old outputs executed new notebooks. New receiving validator independently verifies portable bytes without invoking producer runtime fingerprint guard. Numerical/preparation runtime must be recorded in new cache identity.

Source21799 / accepted21387 / train17083 / validation2146 / test2158. Source fingerprint678c7aaaeccbd4ff5bfef3e39b56a4daece41626102b24704675bd08823910a9. audit.json SHA256 fec4db02dc60fdb99093c0629cc73d96c6217748a647a05b5d14ee834ce1b2c6. Official SHA256SUMS.txt SHA256 b7224b92b341511ec3ceb13dc6652079b2c36a06504bcb49506f157f51dc695d. Official100 download base https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/ . records100 is absent locally and must be obtained on school computer with checksum verification; do not substitute downsampled500.

Release and raw together about7.83GB before caches/environment. Laptop has no Torch/sklearn; no installation/model execution/real-data preprocessing authorized here. Local evidence may be fixture tests, syntax, plan-only CLI, PowerShell parser and read-only archive creation. Every model/GPU claim remains Not verified until target execution.
