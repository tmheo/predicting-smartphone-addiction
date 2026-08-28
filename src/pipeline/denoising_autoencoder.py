"""분할 적합 8차원 비지도 잡음 제거 잠재 표현 제공자. (#483)

이슈 #476이 연 단일 고정 구현이다. 구조, 잡음, 학습률, 묶음 크기와 종료 규칙은
``DAE_SPEC``에 고정하며 결과를 본 뒤 바꾸지 않는다.

정보 경계:

- fold-fit 제공자이므로 각 바깥쪽 분할의 학습 행만 ``fit``에 들어온다.
  전처리(수치 최솟값·최댓값, 범주 어휘)와 자동부호화기 가중치는 그 행으로만 맞춘다.
- 목표값은 선언 입력이 아니므로 계획이 ``fit``에 넘기지 않는다(``uses_target = False``).
  ``fit``과 ``transform``은 선언한 열만 읽는다.
- 학습 종료 시점은 바깥쪽 학습 행 안의 고정 90:10 내부 분할에서 재구성 손실로 정한다.
- 전체 자료 재학습(``enter_full_data_fit``)에서는 내부 분할 대신 다섯 바깥쪽 분할의
  학습 횟수 중앙값을 사전 규칙으로 받은 ``full_data_epochs``만큼 학습한다.

입력 표현:

- 수치 열 블록은 ``[학습 행 최솟값·최댓값으로 0~1 변환한 값(결측은 0), 관측 표시]`` 두 칸이다.
  변환값은 학습 행 밖에서 0~1을 벗어날 수 있으며 자르지 않는다.
- 범주 열 블록은 학습 행 어휘의 원-핫이며 결측과 처음 보는 값은 전부 0이다.
- 잡음은 원래 열마다 독립적으로 ``mask_probability``로 그 열의 블록 전체를 0으로 가린다.

손실은 수치 열의 관측 셀 평균제곱오차와 범주 열의 알려진 값 교차엔트로피를
원래 열별로 같은 비중으로 평균한다.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import sys
from typing import Any

import numpy as np
import pandas as pd

from .data import ID, TARGET
from .features import PLACEBO

LATENT_DIM = 8
DAE_SPEC: dict[str, object] = {
    "latent_dim": LATENT_DIM,
    "hidden_rule": "min(256, max(64, 2d))",
    "bottleneck_hidden": 32,
    "hidden_activation": "relu",
    "bottleneck_activation": "linear",
    "mask_probability": 0.1,
    "optimizer": "adamw",
    "learning_rate": 0.001,
    "weight_decay": 0.00001,
    "batch_size": 4096,
    "max_epochs": 100,
    "internal_validation_fraction": 0.1,
    "patience": 10,
    "min_delta": 0.0,
    "numeric_scaling": "train_fold_min_max_unclipped",
    "numeric_block": ["scaled_value_or_zero", "observed_indicator"],
    "categorical_block": "train_fold_vocabulary_one_hot_unknown_all_zero",
    "loss": "mean_over_columns(numeric_mse_on_observed, categorical_cross_entropy_on_known)",
    "validation_noise": "fixed_mask_from_derived_seed",
    "selection": "best_internal_validation_loss_state",
}
_SEED_ROLES = ("split", "shuffle", "mask", "init")


def _derived_seed(seed: int, role: str) -> int:
    """후보 실행 난수에서 역할별 난수를 결정적으로 파생한다."""
    digest = hashlib.sha256(f"dae8:{int(seed)}:{role}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def _module_sha256() -> str:
    source = inspect.getsource(sys.modules[__name__]).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def hidden_width(input_dim: int) -> int:
    return min(256, max(64, 2 * input_dim))


def _cuda_execution_identity(torch: Any) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    driver = None
    try:
        import subprocess

        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - 드라이버 판본을 얻지 못하면 CUDA 재사용 정체성을 만들지 않는다.
        driver = None
    if not driver:
        raise RuntimeError("CUDA 실행의 드라이버 판본을 확인할 수 없어 재사용 정체성을 만들 수 없다.")
    return {
        "mode": "cuda",
        "gpu_model": str(properties.name),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "cuda_version": str(torch.version.cuda),
        "driver_version": str(driver),
    }


class DenoisingAutoencoderLatent:
    """fold-fit 제공자: 잡음 제거 자동부호화기의 8차원 선형 병목 잠재값 열. (#483)"""

    uses_target = False

    def __init__(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str],
        full_data_epochs: int | None = None,
    ) -> None:
        all_cols = [*numeric_cols, *categorical_cols]
        if not all_cols:
            raise ValueError("잠재 표현 입력 열이 하나 이상 필요하다.")
        forbidden = {ID, TARGET, PLACEBO} & set(all_cols)
        if forbidden:
            raise ValueError(f"자동부호화기 입력에 쓸 수 없는 열: {sorted(forbidden)}")
        duplicated = sorted({c for c in all_cols if all_cols.count(c) > 1})
        if duplicated:
            raise ValueError(f"numeric_cols/categorical_cols에 겹치는 열이 있다: {duplicated}")
        if full_data_epochs is not None and (
            isinstance(full_data_epochs, bool)
            or not isinstance(full_data_epochs, int)
            or full_data_epochs < 1
        ):
            raise ValueError(f"full_data_epochs는 1 이상의 정수여야 한다: {full_data_epochs!r}")
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.full_data_epochs = full_data_epochs
        self._full_data_mode = False
        self.fit_evidence_: dict[str, object] | None = None

    # ------------------------------------------------------------- 선언

    def columns(self) -> list[str]:
        return [f"dae8_z{i}" for i in range(LATENT_DIM)]

    def reuse_input_columns(self) -> list[str]:
        return [*self.numeric_cols, *self.categorical_cols]

    def reuse_settings(self) -> dict[str, object]:
        # full_data_epochs는 fold-fit 산출값에 영향을 주지 않으므로 재사용 정체성에서 뺀다.
        return {
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "spec": DAE_SPEC,
            "module_sha256": _module_sha256(),
        }

    def reuse_execution(self) -> dict[str, object]:
        import torch

        if torch.cuda.is_available():
            return _cuda_execution_identity(torch)
        return {"mode": "cpu"}

    def enter_full_data_fit(self) -> None:
        """전체 자료 재학습 경로임을 알린다. 이후 fit은 full_data_epochs를 요구한다."""
        self._full_data_mode = True

    def fit_evidence(self) -> dict[str, object] | None:
        return self.fit_evidence_

    # ------------------------------------------------------------- 전처리

    def _validate_inputs(self, df: pd.DataFrame) -> None:
        non_numeric = [
            c for c in self.numeric_cols if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric:
            raise ValueError(f"numeric_cols는 수치 열이어야 한다: {non_numeric}")

    def _fit_preprocessing(self, train_fold: pd.DataFrame) -> None:
        self.numeric_min_: dict[str, float] = {}
        self.numeric_max_: dict[str, float] = {}
        for c in self.numeric_cols:
            values = train_fold[c].astype("float64")
            if values.notna().sum() == 0:
                raise ValueError(f"학습 행에서 {c}의 관측값이 없어 눈금을 만들 수 없다.")
            self.numeric_min_[c] = float(values.min())
            self.numeric_max_[c] = float(values.max())
        self.vocab_: dict[str, list[str]] = {}
        for c in self.categorical_cols:
            observed = train_fold[c].dropna().astype(str)
            vocab = sorted(set(observed.tolist()))
            if not vocab:
                raise ValueError(f"학습 행에서 {c}의 관측 범주가 없어 어휘를 만들 수 없다.")
            self.vocab_[c] = vocab
        self.block_slices_: list[tuple[str, str, slice]] = []
        self.target_slices_: list[tuple[str, str, slice]] = []
        offset = 0
        target_offset = 0
        for c in self.numeric_cols:
            self.block_slices_.append((c, "numeric", slice(offset, offset + 2)))
            offset += 2
            self.target_slices_.append((c, "numeric", slice(target_offset, target_offset + 1)))
            target_offset += 1
        for c in self.categorical_cols:
            k = len(self.vocab_[c])
            self.block_slices_.append((c, "categorical", slice(offset, offset + k)))
            offset += k
            self.target_slices_.append(
                (c, "categorical", slice(target_offset, target_offset + k))
            )
            target_offset += k
        self.input_dim_ = offset
        self.output_dim_ = target_offset

    def _encode_inputs(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(입력 블록, 수치 목표, 수치 관측 표시, 범주 색인(-1은 미지)) 배열을 만든다."""
        n = len(df)
        x = np.zeros((n, self.input_dim_), dtype=np.float32)
        numeric_target = np.zeros((n, len(self.numeric_cols)), dtype=np.float32)
        numeric_observed = np.zeros((n, len(self.numeric_cols)), dtype=np.float32)
        categorical_index = np.full((n, len(self.categorical_cols)), -1, dtype=np.int64)
        for j, (c, _, block) in enumerate(self.block_slices_[: len(self.numeric_cols)]):
            values = df[c].to_numpy(dtype="float64")
            observed = np.isfinite(values)
            lo, hi = self.numeric_min_[c], self.numeric_max_[c]
            span = hi - lo
            scaled = np.zeros(n, dtype=np.float64)
            if span > 0:
                scaled[observed] = (values[observed] - lo) / span
            x[:, block.start] = scaled
            x[:, block.start + 1] = observed
            numeric_target[:, j] = scaled
            numeric_observed[:, j] = observed
        for j, (c, _, block) in enumerate(self.block_slices_[len(self.numeric_cols) :]):
            raw = df[c]
            present = raw.notna().to_numpy()
            codes = np.full(n, -1, dtype=np.int64)
            if present.any():
                # 어휘 밖 값(처음 보는 값)은 Categorical이 -1을 준다.
                codes[present] = pd.Categorical(
                    raw[present].astype(str).to_numpy(), categories=self.vocab_[c]
                ).codes.astype(np.int64)
            known = codes >= 0
            rows = np.nonzero(known)[0]
            x[rows, block.start + codes[known]] = 1.0
            categorical_index[:, j] = codes
        return x, numeric_target, numeric_observed, categorical_index

    # ------------------------------------------------------------- 학습

    def fit(self, train_fold: pd.DataFrame, seed: int) -> None:
        import torch

        cols = self.reuse_input_columns()
        frame = train_fold[cols]
        self._validate_inputs(frame)
        if self._full_data_mode and self.full_data_epochs is None:
            raise ValueError(
                "전체 자료 재학습에는 다섯 바깥쪽 분할 학습 횟수의 중앙값을 "
                "full_data_epochs로 미리 정해야 한다. (#483)"
            )
        self._fit_preprocessing(frame)
        x_all, num_target, num_observed, cat_index = self._encode_inputs(frame)
        seeds = {role: _derived_seed(seed, role) for role in _SEED_ROLES}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            # macOS에서 xgboost·lightgbm의 libomp와 torch의 libomp가 함께 적재되면
            # CPU 연산의 OpenMP fork가 교착한다. CPU 경로는 단일 스레드로 돈다.
            torch.set_num_threads(1)

        n = len(x_all)
        fixed_epochs = self.full_data_epochs if self._full_data_mode else None
        if fixed_epochs is None:
            split_generator = torch.Generator().manual_seed(seeds["split"])
            permutation = torch.randperm(n, generator=split_generator).numpy()
            n_valid = int(math.floor(n * float(DAE_SPEC["internal_validation_fraction"])))
            if n_valid < 1 or n - n_valid < 1:
                raise ValueError(f"내부 분할을 만들 수 없는 학습 행 수다: {n}")
            valid_rows = np.sort(permutation[:n_valid])
            train_rows = np.sort(permutation[n_valid:])
        else:
            valid_rows = np.zeros(0, dtype=np.int64)
            train_rows = np.arange(n)

        tensors = _to_device(
            torch, device, x_all, num_target, num_observed, cat_index
        )
        x_t, num_target_t, num_observed_t, cat_index_t = tensors
        train_rows_t = torch.as_tensor(train_rows, device=device)
        valid_rows_t = torch.as_tensor(valid_rows, device=device)
        column_kinds = [(kind, block) for _, kind, block in self.block_slices_]
        target_slices = [(kind, block) for _, kind, block in self.target_slices_]
        mask_generator = torch.Generator(device=device).manual_seed(seeds["mask"])
        shuffle_generator = torch.Generator(device=device).manual_seed(seeds["shuffle"])

        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(seeds["init"])
            model = _build_model(torch, self.input_dim_, self.output_dim_)
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(DAE_SPEC["learning_rate"]),
            weight_decay=float(DAE_SPEC["weight_decay"]),
        )
        batch_size = int(DAE_SPEC["batch_size"])
        mask_probability = float(DAE_SPEC["mask_probability"])
        max_epochs = int(DAE_SPEC["max_epochs"]) if fixed_epochs is None else int(fixed_epochs)
        patience = int(DAE_SPEC["patience"])
        min_delta = float(DAE_SPEC["min_delta"])

        def masked_input(rows: Any, generator: Any) -> Any:
            column_mask = (
                torch.rand(
                    (len(rows), len(column_kinds)), generator=generator, device=device
                )
                < mask_probability
            )
            keep = torch.ones((len(rows), self.input_dim_), device=device)
            for j, (_, block) in enumerate(column_kinds):
                keep[:, block] = keep[:, block] * (~column_mask[:, j : j + 1]).float()
            return x_t[rows] * keep

        def column_losses(output: Any, rows: Any) -> tuple[list[Any], list[Any]]:
            """열별 (손실 합, 분모) 목록. 합산은 호출자가 한다."""
            sums: list[Any] = []
            counts: list[Any] = []
            for j, (kind, block) in enumerate(target_slices):
                if kind == "numeric":
                    error = (output[:, block.start] - num_target_t[rows, j]) ** 2
                    observed = num_observed_t[rows, j]
                    sums.append((error * observed).sum())
                    counts.append(observed.sum())
                else:
                    codes = cat_index_t[rows, j - len(self.numeric_cols)]
                    known = codes >= 0
                    logits = output[:, block]
                    safe_codes = torch.where(known, codes, torch.zeros_like(codes))
                    ce = torch.nn.functional.cross_entropy(
                        logits, safe_codes, reduction="none"
                    )
                    sums.append((ce * known.float()).sum())
                    counts.append(known.float().sum())
            return sums, counts

        def batch_loss(output: Any, rows: Any) -> Any:
            sums, counts = column_losses(output, rows)
            per_column = [s / torch.clamp(c, min=1.0) for s, c in zip(sums, counts)]
            return torch.stack(per_column).mean()

        valid_input = None
        if fixed_epochs is None:
            valid_input = masked_input(valid_rows_t, mask_generator)

        def evaluate_validation() -> tuple[float, list[float]]:
            assert valid_input is not None
            model.eval()
            total_sums = [0.0] * len(target_slices)
            total_counts = [0.0] * len(target_slices)
            with torch.no_grad():
                for start in range(0, len(valid_rows_t), batch_size):
                    rows = valid_rows_t[start : start + batch_size]
                    output = model(valid_input[start : start + batch_size])
                    sums, counts = column_losses(output, rows)
                    for j in range(len(target_slices)):
                        total_sums[j] += float(sums[j])
                        total_counts[j] += float(counts[j])
            per_column = [s / max(c, 1.0) for s, c in zip(total_sums, total_counts)]
            return float(np.mean(per_column)), per_column

        baseline_loss, baseline_columns = (None, None)
        if fixed_epochs is None:
            baseline_loss, baseline_columns = self._mean_input_baseline(
                num_target[train_rows],
                num_observed[train_rows],
                cat_index[train_rows],
                num_target[valid_rows],
                num_observed[valid_rows],
                cat_index[valid_rows],
            )

        history: list[dict[str, float]] = []
        best_loss = math.inf
        best_epoch = 0
        best_state = None
        epochs_without_improvement = 0
        epochs_run = 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            order = train_rows_t[
                torch.randperm(len(train_rows_t), generator=shuffle_generator, device=device)
            ]
            train_loss_sum = 0.0
            train_batches = 0
            for start in range(0, len(order), batch_size):
                rows = order[start : start + batch_size]
                noisy = masked_input(rows, mask_generator)
                optimizer.zero_grad(set_to_none=True)
                output = model(noisy)
                loss = batch_loss(output, rows)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.detach())
                train_batches += 1
            epochs_run = epoch
            record = {"epoch": epoch, "train_loss": train_loss_sum / max(train_batches, 1)}
            if fixed_epochs is None:
                valid_loss, _ = evaluate_validation()
                record["valid_loss"] = valid_loss
                history.append(record)
                if valid_loss < best_loss - min_delta:
                    best_loss = valid_loss
                    best_epoch = epoch
                    best_state = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break
            else:
                history.append(record)
        if fixed_epochs is None:
            assert best_state is not None
            model.load_state_dict(best_state)
            selected_epochs = best_epoch
            final_valid_loss, final_valid_columns = evaluate_validation()
        else:
            selected_epochs = max_epochs
            final_valid_loss, final_valid_columns = (None, None)

        model.eval()
        self.encoder_ = model.encoder.to("cpu")
        self.device_ = device
        self.fit_evidence_ = {
            "schema_version": 1,
            "seed": int(seed),
            "derived_seeds": seeds,
            "device": device,
            "input_dim": int(self.input_dim_),
            "hidden_width": int(hidden_width(self.input_dim_)),
            "output_dim": int(self.output_dim_),
            "vocabulary_sizes": {c: len(v) for c, v in self.vocab_.items()},
            "numeric_min": dict(self.numeric_min_),
            "numeric_max": dict(self.numeric_max_),
            "rows_total": int(n),
            "rows_internal_train": int(len(train_rows)),
            "rows_internal_valid": int(len(valid_rows)),
            "mode": "full_data_fixed_epochs" if fixed_epochs is not None else "internal_early_stopping",
            "epochs_run": int(epochs_run),
            "selected_epochs": int(selected_epochs),
            "training_length_semantics": "one_based_count",
            "best_valid_loss": final_valid_loss,
            "best_valid_loss_per_column": (
                None
                if final_valid_columns is None
                else dict(zip([c for c, _, _ in self.target_slices_], final_valid_columns))
            ),
            "mean_input_baseline_valid_loss": baseline_loss,
            "mean_input_baseline_per_column": (
                None
                if baseline_columns is None
                else dict(zip([c for c, _, _ in self.target_slices_], baseline_columns))
            ),
            "history": history,
        }

    def _mean_input_baseline(
        self,
        train_num_target: np.ndarray,
        train_num_observed: np.ndarray,
        train_cat_index: np.ndarray,
        valid_num_target: np.ndarray,
        valid_num_observed: np.ndarray,
        valid_cat_index: np.ndarray,
    ) -> tuple[float, list[float]]:
        """평균 입력 복원: 수치는 학습 행 평균, 범주는 학습 행 빈도 분포로 예측한 손실."""
        per_column: list[float] = []
        for j in range(len(self.numeric_cols)):
            observed = train_num_observed[:, j] > 0
            mean = float(train_num_target[observed, j].mean()) if observed.any() else 0.0
            valid_observed = valid_num_observed[:, j] > 0
            if valid_observed.any():
                error = (valid_num_target[valid_observed, j] - mean) ** 2
                per_column.append(float(error.mean()))
            else:
                per_column.append(0.0)
        for j, c in enumerate(self.categorical_cols):
            k = len(self.vocab_[c])
            codes = train_cat_index[:, j]
            known = codes >= 0
            counts = np.bincount(codes[known], minlength=k).astype("float64")
            probabilities = (counts + 1e-12) / max(float(counts.sum()), 1e-12)
            valid_codes = valid_cat_index[:, j]
            valid_known = valid_codes >= 0
            if valid_known.any():
                per_column.append(float(-np.log(probabilities[valid_codes[valid_known]]).mean()))
            else:
                per_column.append(0.0)
        return float(np.mean(per_column)), per_column

    # ------------------------------------------------------------- 변환

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        import torch

        frame = df[self.reuse_input_columns()]
        self._validate_inputs(frame)
        x, _, _, _ = self._encode_inputs(frame)
        encoder = self.encoder_
        encoder.eval()
        if self.device_ == "cpu":
            torch.set_num_threads(1)
        outputs: list[np.ndarray] = []
        chunk = 65536
        with torch.no_grad():
            for start in range(0, len(x), chunk):
                batch = torch.as_tensor(x[start : start + chunk])
                outputs.append(encoder(batch).to(torch.float64).numpy())
        latent = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, LATENT_DIM))
        return pd.DataFrame(latent, index=df.index, columns=self.columns())


def _to_device(
    torch: Any,
    device: str,
    x: np.ndarray,
    num_target: np.ndarray,
    num_observed: np.ndarray,
    cat_index: np.ndarray,
) -> tuple[Any, Any, Any, Any]:
    return (
        torch.as_tensor(x, device=device),
        torch.as_tensor(num_target, device=device),
        torch.as_tensor(num_observed, device=device),
        torch.as_tensor(cat_index, device=device),
    )


def _build_model(torch: Any, input_dim: int, output_dim: int) -> Any:
    hidden = hidden_width(input_dim)
    bottleneck = int(DAE_SPEC["bottleneck_hidden"])
    nn = torch.nn

    class _Autoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, LATENT_DIM),
            )
            self.decoder = nn.Sequential(
                nn.Linear(LATENT_DIM, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, hidden),
                nn.ReLU(),
                nn.Linear(hidden, output_dim),
            )

        def forward(self, x: Any) -> Any:
            return self.decoder(self.encoder(x))

    return _Autoencoder()
