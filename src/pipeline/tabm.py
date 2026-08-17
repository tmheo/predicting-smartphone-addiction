"""TabM 학습기(pytabkit) fold 구현. (#61)

원문(szymonkapiski의 S6E8 TabM with constrained imputation, OOF 0.96867) 학습부의
재현: pytabkit TabM_D_Classifier(tabm-mini-normal, PWL 수치 embedding)를 fold 안에서
시드 평균(원문 N_SEEDS=3)해 잡음이 적은 예측 하나를 만든다. 앙상블에 모델 3개를
넣는 것이 아니라 fold를 떠나기 전에 평균하는 방식이라 후보 풀 규약과 충돌하지 않는다.

원문과 다른 점(티켓 #61의 누출 해소):
- 원문은 train+test 전체로 iterative imputer·중앙값 대체·격자 TE를 만들었다.
  여기서는 피처가 전부 파이프라인 provider(fold-fit)에서 오고, 이 모듈은 pytabkit이
  거부하는 수치 열의 남은 NaN 중앙값 대체만 학습 fold 통계로 수행한다.
- 원문의 자기 포함 평활 TE 대신 파이프라인의 내부 OOF TE를 쓴다(config 소관).
- 결측 지표·결측 개수 열(지도의 배제 경계)은 만들지 않는다.

pytabkit(torch)이 필요하므로 model.py의 adapter가 이 모듈을 lazy import한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# 원문 노트북의 TabM 하이퍼파라미터 그대로가 기본값이다.
_PYTABKIT_DEFAULTS = {
    "arch_type": "tabm-mini-normal",
    "tabm_k": 16,
    "num_emb_type": "pwl",
    "d_embedding": 16,
    "batch_size": 128,
    "lr": 5e-4,
    "n_epochs": 150,
    "dropout": 0.02,
    "d_block": 160,
    "n_blocks": 10,
    "weight_decay": 1e-2,
    "patience": 5,
    "num_emb_n_bins": 48,
    "share_training_batches": False,
    "gradient_clipping_norm": None,
    "tfms": ["quantile_tabr"],
}


class TabMFold:
    """fold 하나의 전처리·시드 평균 학습·예측·중요도 상태. adapter가 소유한다."""

    def __init__(self, params: dict, seed: int) -> None:
        params = dict(params)
        self._n_seed_avg = int(params.pop("n_seed_avg", 3))
        self._perm_repeats = int(params.pop("perm_repeats", 3))
        self._perm_sample = int(params.pop("perm_sample", 50_000))
        self._pytabkit = {
            k: params.pop(k, default) for k, default in _PYTABKIT_DEFAULTS.items()
        }
        if params:
            raise ValueError(f"tabm이 모르는 params: {sorted(params)}")
        if self._n_seed_avg < 1:
            raise ValueError(f"n_seed_avg는 1 이상이어야 한다: {self._n_seed_avg}")
        self._seed = seed
        self._models: list = []
        self._medians: dict[str, float] = {}
        self._columns: list[str] | None = None
        self._val: tuple[pd.DataFrame, np.ndarray] | None = None
        self._val_auc: float | None = None
        self._training_diagnostics: dict[str, object] | None = None

    # ---- 전처리: 중앙값은 학습 fold 통계만 쓴다(outer fold 규율) ----

    def _fit_medians(self, X_tr: pd.DataFrame) -> None:
        self._columns = list(X_tr.columns)
        self._medians = {}
        for col in self._columns:
            if isinstance(X_tr[col].dtype, pd.CategoricalDtype):
                continue  # 범주 결측은 pytabkit이 자체 카테고리로 다룬다.
            values = X_tr[col].replace([np.inf, -np.inf], np.nan)
            median = values.median()
            # 학습 fold가 전부 결측인 열은 원문(fillna(median) 후 fillna(0))처럼 0.
            self._medians[col] = float(median) if pd.notna(median) else 0.0

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert list(X.columns) == self._columns, "입력 컬럼이 학습 때와 다르다."
        out = X.copy()
        for col, median in self._medians.items():
            values = out[col].replace([np.inf, -np.inf], np.nan)
            out[col] = values.fillna(median).astype("float32")
        return out

    # ---- 학습: fold 안 시드 평균(원문의 SEED + 1000*s 파생 시드) ----

    def fit(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series
    ) -> np.ndarray:
        import torch
        from pytabkit import TabM_D_Classifier

        device = "cuda" if torch.cuda.is_available() else "cpu"
        extra: dict = {}
        if device == "cpu":
            # macOS에서 lightgbm 계열 libomp와 torch libomp가 같이 적재되면 CPU
            # 연산의 OpenMP 병렬이 충돌한다. pytabkit은 fit 안에서 스레드 수를
            # 물리 코어 수로 재설정하므로 n_threads=1을 명시해 단일 스레드로 돈다.
            extra["n_threads"] = 1
        self._fit_medians(X_tr)
        X_tr_t = self._transform(X_tr)
        X_va_t = self._transform(X_va)
        y_va_np = y_va.to_numpy(dtype="float64")

        self._models = []
        fit_parameters = []
        val_pred = np.zeros(len(X_va), dtype="float64")
        for s in range(self._n_seed_avg):
            model = TabM_D_Classifier(
                random_state=self._seed + 1000 * s,
                device=device,
                verbosity=0,
                **extra,
                **self._pytabkit,
            )
            model.fit(X_tr_t, y_tr, X_va_t, y_va)
            member_pred = model.predict_proba(X_va_t)[:, 1].astype("float64")
            val_pred += member_pred / self._n_seed_avg
            self._models.append(model)
            fit_parameters.append(_json_value(model.fit_params_))
            print(
                f"[tabm] member {s + 1}/{self._n_seed_avg} "
                f"valAUC={roc_auc_score(y_va_np, member_pred):.5f}",
                flush=True,
            )
        self._val = (X_va_t, y_va_np)
        self._val_auc = roc_auc_score(y_va_np, val_pred)
        self._training_diagnostics = {
            "members": fit_parameters,
            "validation_auc": float(self._val_auc),
        }
        print(f"[tabm] fold seed-avg valAUC={self._val_auc:.5f}", flush=True)
        return val_pred

    def fit_full(self, X: pd.DataFrame, y: pd.Series, epochs: int) -> None:
        """전체 자료를 검증 분할과 조기 종료 없이 고정 epoch 수로 학습한다."""
        import torch

        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("TabM 전체 자료 재학습 epoch 수는 양의 정수여야 한다.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._fit_medians(X)
        X_t = self._transform(X)
        params = dict(self._pytabkit)
        params["n_epochs"] = epochs
        self._models = []
        for s in range(self._n_seed_avg):
            member = _FixedEpochTabMMember(
                params=params,
                seed=self._seed + 1000 * s,
                device=device,
            )
            member.fit(X_t, y)
            self._models.append(member)
            print(
                f"[tabm] full member {s + 1}/{self._n_seed_avg} "
                f"epochs={epochs}",
                flush=True,
            )
        self._training_diagnostics = {
            "full_fit": True,
            "epochs": epochs,
            "members": [member.training_diagnostics() for member in self._models],
        }

    def _predict_transformed(self, X_t: pd.DataFrame) -> np.ndarray:
        pred = np.zeros(len(X_t), dtype="float64")
        for model in self._models:
            pred += model.predict_proba(X_t)[:, 1].astype("float64") / len(self._models)
        return pred

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_transformed(self._transform(X))

    def training_diagnostics(self) -> dict[str, object]:
        if self._training_diagnostics is None:
            raise RuntimeError("training_diagnostics는 fit 뒤에 호출해야 한다.")
        return self._training_diagnostics

    # ---- 중요도: 검증 fold permutation(AUC 하락 폭)을 gain 축으로 (#97 규약) ----
    # 정밀도 파라미터(표본 크기·반복 수)는 계열 소유다. 시드 평균 예측 전체를 다시
    # 만들면 fold당 수백 회 추론이라, 검증 fold를 결정적으로 표본추출해 잰다.

    def importance(self) -> pd.DataFrame:
        X_va, y_va = self._val
        if len(X_va) > self._perm_sample:
            keep = np.random.default_rng(self._seed).choice(
                len(X_va), size=self._perm_sample, replace=False
            )
            keep.sort()
            X_va = X_va.iloc[keep].reset_index(drop=True)
            y_va = y_va[keep]
        base = roc_auc_score(y_va, self._predict_transformed(X_va))
        gains = []
        for j, col in enumerate(self._columns):
            drops = []
            for r in range(self._perm_repeats):
                rng = np.random.default_rng(self._seed * 10007 + j * 101 + r)
                X_p = X_va.copy()
                # take는 category dtype을 보존한다(numpy 왕복은 object로 퇴화).
                X_p[col] = X_va[col].take(rng.permutation(len(X_va))).set_axis(X_p.index)
                drops.append(base - roc_auc_score(y_va, self._predict_transformed(X_p)))
            gains.append(float(np.mean(drops)))
        return pd.DataFrame({"feature": self._columns, "gain": gains})


def _json_value(value):
    """pytabkit 학습 파라미터를 유한한 JSON 기본형으로 바꾼다."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            raise ValueError("pytabkit 학습 파라미터에 유한하지 않은 수가 있다.")
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class _FixedEpochTabMMember:
    """pytabkit 전처리와 TabM 모형을 쓰되 검증 선택 없이 마지막 epoch를 보존한다.

    pytabkit 1.7.3의 TabM 공개 추정기는 검증 집합 없는 학습과 전체 자료 refit을
    구현하지 않았다. 전체 자료 규약에 필요한 차이는 학습 제어뿐이므로, 같은
    변환기와 모형 구현을 조립해 모든 행을 정확히 고정 epoch만큼 학습한다.
    """

    def __init__(self, *, params: dict, seed: int, device: str) -> None:
        self._params = dict(params)
        self._seed = seed
        self._device_name = device
        self._converter = None
        self._transform = None
        self._model = None
        self._num_col_mask = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        import torch
        from pytabkit.models.data.conversion import ToDictDatasetConverter
        from pytabkit.models.data.data import DictDataset, TensorInfo
        from pytabkit.models.nn_models import rtdl_num_embeddings
        from pytabkit.models.nn_models.models import PreprocessingFactory
        from pytabkit.models.nn_models.tabm import Model, make_parameter_groups

        training_seed = int(
            np.random.RandomState(self._seed).randint(0, 2**31 - 1)
        )
        torch.manual_seed(training_seed)
        np.random.seed(training_seed)
        device = torch.device(self._device_name)
        categorical = [isinstance(dtype, pd.CategoricalDtype) for dtype in X.dtypes]
        self._converter = ToDictDatasetConverter(cat_features=categorical, verbosity=0)
        x_ds = self._converter.fit_transform(X)
        y_values = y.to_numpy(dtype=np.int64).reshape(-1, 1)
        y_ds = DictDataset(
            tensors={"y": torch.as_tensor(y_values, dtype=torch.long)},
            tensor_infos={"y": TensorInfo(cat_sizes=[2])},
        )
        dataset = DictDataset.join(x_ds, y_ds)
        factory = PreprocessingFactory(**self._params)
        fitter = factory.create(dataset.tensor_infos)
        self._transform, transformed = fitter.fit_transform(dataset)
        transformed = transformed.to(device)
        x_cont = transformed.tensors["x_cont"]
        self._num_col_mask = ~torch.all(x_cont == x_cont[0:1, :], dim=0)
        transformed.tensors["x_cont"] = x_cont[:, self._num_col_mask]
        n_cont = int(transformed.tensors["x_cont"].shape[1])
        cat_sizes = dataset.tensor_infos["x_cat"].get_cat_sizes().numpy().tolist()
        bins = (
            rtdl_num_embeddings.compute_bins(
                transformed.tensors["x_cont"],
                n_bins=min(int(self._params["num_emb_n_bins"]), len(X) - 1),
            )
            if self._params["num_emb_type"] == "pwl" and n_cont > 0
            else None
        )
        n_blocks = self._params["n_blocks"]
        if n_blocks == "auto":
            n_blocks = 3 if bins is None else 2
        self._model = Model(
            n_num_features=n_cont,
            cat_cardinalities=cat_sizes,
            n_classes=2,
            backbone={
                "type": "MLP",
                "n_blocks": n_blocks,
                "d_block": int(self._params["d_block"]),
                "dropout": float(self._params["dropout"]),
            },
            bins=bins,
            num_embeddings=(
                None
                if bins is None
                else {
                    "type": "PiecewiseLinearEmbeddings",
                    "d_embedding": int(self._params["d_embedding"]),
                    "activation": False,
                    "version": "B",
                }
            ),
            arch_type=self._params["arch_type"],
            k=int(self._params["tabm_k"]),
            share_training_batches=bool(self._params["share_training_batches"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            make_parameter_groups(self._model),
            lr=float(self._params["lr"]),
            weight_decay=float(self._params["weight_decay"]),
        )
        n_train = len(X)
        batch_size = min(int(self._params["batch_size"]), n_train)
        y_train = transformed.tensors["y"]
        x_cat = transformed.tensors["x_cat"]
        has_categories = bool(cat_sizes)
        for _ in range(int(self._params["n_epochs"])):
            if self._model.share_training_batches:
                batches = torch.randperm(n_train, device=device).split(batch_size)
            else:
                batches = [
                    indexes.transpose(0, 1).flatten()
                    for indexes in torch.rand(
                        (self._model.k, n_train), device=device
                    ).argsort(dim=1).split(batch_size, dim=1)
                ]
            self._model.train()
            for indexes in batches:
                optimizer.zero_grad(set_to_none=True)
                logits = self._model(
                    transformed.tensors["x_cont"][indexes],
                    x_cat[indexes] if has_categories else None,
                )
                flat_logits = logits.flatten(0, 1)
                labels = y_train[indexes].squeeze(-1)
                if self._model.share_training_batches:
                    labels = labels.repeat_interleave(self._model.k)
                loss = torch.nn.functional.cross_entropy(flat_logits, labels)
                loss.backward()
                clip = self._params["gradient_clipping_norm"]
                if clip not in (None, "none"):
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), float(clip))
                optimizer.step()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        dataset = self._converter.transform(X)
        transformed = self._transform(dataset).to(self._device_name)
        transformed.tensors["x_cont"] = transformed.tensors["x_cont"][
            :, self._num_col_mask
        ]
        has_categories = transformed.tensor_infos["x_cat"].get_n_features() > 0
        outputs = []
        self._model.eval()
        with torch.inference_mode():
            for indexes in torch.arange(
                len(X), device=torch.device(self._device_name)
            ).split(1024):
                logits = self._model(
                    transformed.tensors["x_cont"][indexes],
                    transformed.tensors["x_cat"][indexes] if has_categories else None,
                )
                outputs.append(logits.softmax(dim=-1).mean(dim=1).cpu())
        return torch.cat(outputs).numpy()

    def training_diagnostics(self) -> dict[str, object]:
        training_seed = int(
            np.random.RandomState(self._seed).randint(0, 2**31 - 1)
        )
        return {
            "random_state": self._seed,
            "training_seed": training_seed,
            "epochs": int(self._params["n_epochs"]),
        }
