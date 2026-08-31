# 확장 스택용 외부 구성원 장부 (판본 3, 이슈 #484·#487)

## 결론

판본 3은 [판본 2 공개 노트북 재감사](https://github.com/tmheo/predicting-smartphone-addiction/issues/480) 통과 11개, [장부 밖 전수 조사](https://github.com/tmheo/predicting-smartphone-addiction/issues/479) 통과 8개와 [2026-08-30 증분 조사](https://github.com/tmheo/predicting-smartphone-addiction/issues/487)에서 발견한 6개 후보를 같은 계약으로 감사했다.
현행 후보 25개 전부 `감사 완료` 상태이며 그중 **24개가 `자격 있음`**, 0개가 `자격 없음`, 1개가 `근거 부족`이다.
증분 조사 후보 6개 가운데 5개는 `자격 있음`, 1개는 `근거 부족`이다.
정규화 예측 쌍 SHA-256은 25개가 조사 보고서의 값과 일치한다.
감사 진행 상태, 자격 판정, 후보 동결은 서로 다른 축이며 이 문서와 색인은 후보를 동결하지 않는다.

## 산출물

- 기계 판독 색인: `docs/research/external-member-ledger-v3/index.json` (`ledger_version: 3`, 현행 감사 기록만 가리키며 예측 배열을 담지 않는다)
- 감사 기록: `docs/research/external-member-ledger-v3/records/<감사 기록 식별자>.json` (변경 불가, `record_sha256`는 그 필드를 뺀 정규 JSON의 SHA-256)
- 근거 묶음 manifest: `docs/research/external-member-ledger-v3/evidence/<감사 기록 식별자>.json` (외부 파일의 경로·바이트·SHA-256·확보 시각)
- 반입 실행 기록: `docs/research/external-member-ledger-v3/ingest-run.json`
- 외부 파일: `data/external/ext484/<owner>_<slug>/` (소스 `.ipynb`, 출력 원문, `kernel-metadata.json`, `files.json`, `page-version.json`, `.download.log`, `normalized/` float64 npy). `data/`는 커밋 제외 경로다.
- 생성 도구: `scripts/build_external_member_ledger_v3.py` (`fetch` → `audit` → `verify`). 판본 1·2 장부와 `scripts/build_external_member_ledger.py`는 수정하지 않고 과거 기록으로 보존한다.
- 도구 커밋: `d43908e5a251856194c093a5d76cc14284636e06`

## 후보별 자격 판정

| 순서 | 구성원 | 고정 판본 | 자격 | 재채점 AUC | 독립 조사 AUC 차이 | 선언 AUC 차이 | 쌍 SHA-256 | 보고서 대조 | 주의 사항 |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | `nb_zhukov:cat_base` | [344661133](https://www.kaggle.com/code/zhukovoleksiy/ps6e8-eda-feature-engineering-pipeline?scriptVersionId=344661133) | 자격 있음 | 0.967992 | +0.0e+00 | +1.1e-16 | `ce0efc575d7b…` | 일치 | license_unknown_use_limited, naming_mismatch_lexD |
| 2 | `nb_zhukov:lgb02` | [344661133](https://www.kaggle.com/code/zhukovoleksiy/ps6e8-eda-feature-engineering-pipeline?scriptVersionId=344661133) | 자격 있음 | 0.968355 | +0.0e+00 | +1.1e-16 | `966f988ba097…` | 일치 | license_unknown_use_limited, naming_mismatch_lexD |
| 3 | `nb_zhukov:xgb_base` | [344661133](https://www.kaggle.com/code/zhukovoleksiy/ps6e8-eda-feature-engineering-pipeline?scriptVersionId=344661133) | 자격 있음 | 0.967859 | +0.0e+00 | -1.1e-16 | `2303df412878…` | 일치 | license_unknown_use_limited, naming_mismatch_lexD |
| 4 | `nb_reda_lgbm:lgbm` | [340546450](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965?scriptVersionId=340546450) | 자격 있음 | 0.968259 | +0.0e+00 | - | `ac29bdd21023…` | 일치 | license_unknown_use_limited |
| 5 | `nb_reda_hgb:hgb` | [340546105](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945?scriptVersionId=340546105) | 자격 있음 | 0.968026 | +0.0e+00 | - | `a7a46658f0c4…` | 일치 | license_unknown_use_limited |
| 6 | `nb_yekenot:trompt` | [344781653](https://www.kaggle.com/code/yekenot/ps-s6-e8-trompt-pytorch-frame?scriptVersionId=344781653) | 자격 있음 | 0.966671 | +0.0e+00 | - | `93411f087c5c…` | 일치 | license_unknown_use_limited |
| 7 | `nb_mohan_realmlp:realmlp` | [342288210](https://www.kaggle.com/code/mohankrishnathalla/s6e8-realmlp-oof-saver?scriptVersionId=342288210) | 자격 있음 | 0.958134 | +0.0e+00 | - | `5245ad2a78dc…` | 일치 | license_unknown_use_limited |
| 8 | `nb_lopure:linear_svm` | [342359513](https://www.kaggle.com/code/lopure/hdviz-pca-parallel-with-linear-svm?scriptVersionId=342359513) | 자격 있음 | 0.911345 | +0.0e+00 | - | `2900eb8774c7…` | 일치 | license_unknown_use_limited, decision_function_scores |
| 9 | `nb_lopure:poly_svm` | [342359513](https://www.kaggle.com/code/lopure/hdviz-pca-parallel-with-linear-svm?scriptVersionId=342359513) | 자격 있음 | 0.928795 | +0.0e+00 | - | `953e2da3c591…` | 일치 | license_unknown_use_limited, decision_function_scores |
| 10 | `nb_lopure:rbf_svm` | [342359513](https://www.kaggle.com/code/lopure/hdviz-pca-parallel-with-linear-svm?scriptVersionId=342359513) | 자격 있음 | 0.922167 | +0.0e+00 | - | `557a0da96ccc…` | 일치 | license_unknown_use_limited, decision_function_scores |
| 11 | `nb_shaman_baseline:lr` | [345105403](https://www.kaggle.com/code/shamanthakreddymallu/s6e8-baseline?scriptVersionId=345105403) | 자격 있음 | 0.936609 | +0.0e+00 | - | `e439bd0d7c03…` | 일치 | license_unknown_use_limited, fixed_subsample_training |
| 12 | `beicicc/s6e8-fold-safe-tabnet:tabnet` | [339872430](https://www.kaggle.com/code/beicicc/s6e8-fold-safe-tabnet?scriptVersionId=339872430) | 자격 있음 | 0.965657 | -4.9e-10 | +0.0e+00 | `b339d0b025bc…` | 일치 | license_unknown_use_limited, float32_storage |
| 13 | `beicicc/s6e8-fold-safe-realmlp:realmlp` | [339864149](https://www.kaggle.com/code/beicicc/s6e8-fold-safe-realmlp?scriptVersionId=339864149) | 자격 있음 | 0.968156 | -4.2e-10 | +0.0e+00 | `e21c22c3b241…` | 일치 | license_unknown_use_limited, float32_storage, near_duplicate_cluster |
| 14 | `busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:lgb` | [339485089](https://www.kaggle.com/code/busyaprime/s6e8-tabular-baseline-that-autodetects-the-task?scriptVersionId=339485089) | 자격 있음 | 0.962558 | -4.9e-10 | - | `ff58548f9868…` | 일치 | license_unknown_use_limited, near_duplicate_cluster |
| 15 | `busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:hgb` | [339485089](https://www.kaggle.com/code/busyaprime/s6e8-tabular-baseline-that-autodetects-the-task?scriptVersionId=339485089) | 자격 있음 | 0.962048 | +4.1e-10 | - | `b7b0afba77e4…` | 일치 | license_unknown_use_limited |
| 16 | `busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:xgb` | [339485089](https://www.kaggle.com/code/busyaprime/s6e8-tabular-baseline-that-autodetects-the-task?scriptVersionId=339485089) | 자격 있음 | 0.962314 | -1.5e-10 | - | `3f683ae1e737…` | 일치 | license_unknown_use_limited |
| 17 | `ravi20076/playgrounds6e8-public-baseline-v1:XGB1C` | [339444387](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v1?scriptVersionId=339444387) | 자격 있음 | 0.964201 | -3.8e-10 | - | `d795573efce0…` | 일치 | license_unknown_use_limited, float32_storage, rehosted_training_data_private_notebook, near_duplicate_cluster |
| 18 | `ravi20076/playgrounds6e8-public-baseline-v1:LGBM1C` | [339444387](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v1?scriptVersionId=339444387) | 자격 있음 | 0.964173 | -4.1e-11 | - | `15ea60831189…` | 일치 | license_unknown_use_limited, float32_storage, rehosted_training_data_private_notebook, near_duplicate_cluster |
| 19 | `ravi20076/playgrounds6e8-public-baseline-v1:CB1C` | [339444387](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-baseline-v1?scriptVersionId=339444387) | 자격 있음 | 0.963944 | -1.3e-10 | - | `f3e04b96a6bb…` | 일치 | license_unknown_use_limited, float32_storage, rehosted_training_data_private_notebook |
| 20 | `sometimessubodh/stacking-9-models-smartphone-addiction-prediction:cuml_rf` | [346039237](https://www.kaggle.com/code/sometimessubodh/stacking-9-models-smartphone-addiction-prediction?scriptVersionId=346039237) | 자격 있음 | 0.940504 | +2.1e-13 | - | `b3e6a5c21a00…` | 일치 | license_unknown_use_limited, full_feature_only_preprocessing |
| 21 | `sometimessubodh/stacking-9-models-smartphone-addiction-prediction:cuml_logreg` | [346039237](https://www.kaggle.com/code/sometimessubodh/stacking-9-models-smartphone-addiction-prediction?scriptVersionId=346039237) | 자격 있음 | 0.927827 | +3.4e-13 | - | `949716ab6998…` | 일치 | license_unknown_use_limited, full_feature_only_preprocessing |
| 22 | `sometimessubodh/stacking-9-models-smartphone-addiction-prediction:cuml_knn` | [346039237](https://www.kaggle.com/code/sometimessubodh/stacking-9-models-smartphone-addiction-prediction?scriptVersionId=346039237) | 자격 있음 | 0.929086 | -1.3e-13 | - | `6cc0389e39fd…` | 일치 | license_unknown_use_limited, full_feature_only_preprocessing |
| 23 | `sometimessubodh/stacking-9-models-smartphone-addiction-prediction:cuml_mbsgd` | [346039237](https://www.kaggle.com/code/sometimessubodh/stacking-9-models-smartphone-addiction-prediction?scriptVersionId=346039237) | 자격 있음 | 0.834174 | -4.1e-13 | - | `57c44070baf5…` | 일치 | license_unknown_use_limited, full_feature_only_preprocessing |
| 24 | `sometimessubodh/stacking-9-models-smartphone-addiction-prediction:torch_mlp` | [346039237](https://www.kaggle.com/code/sometimessubodh/stacking-9-models-smartphone-addiction-prediction?scriptVersionId=346039237) | 자격 있음 | 0.940874 | +4.6e-13 | - | `8f3d5ea1e94a…` | 일치 | license_unknown_use_limited, full_feature_only_preprocessing |
| 25 | `michaelqiu0606/s6e8-depth9-pair-te-inputs:depth9_pair_te` | [1](https://www.kaggle.com/datasets/michaelqiu0606/s6e8-depth9-pair-te-inputs/versions/1) | 근거 부족 | 0.970517 | +4.3e-13 | - | `1d85e728c61c…` | 일치 | lineage_source_missing |

모든 후보의 감사 기록 식별자, 제외 사유와 근거 부족 사유는 색인의 `current_records`에 있다.

## 원본·정규화 해시 대조표

| 구성원 | OOF 원본 파일 | OOF 원본 SHA-256 | 시험 원본 파일 | 시험 원본 SHA-256 | 정규화 OOF SHA-256 | 정규화 시험 SHA-256 | 쌍 SHA-256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nb_zhukov:cat_base` | `oof/oof_lexD_cat_base.npy` | `bb5fd7c2ed8816296b269c8009acd015eb6bc321ddb2d6c5d502319a117ab27e` | `oof/test_lexD_cat_base.npy` | `f859ee550e3a4fc7caae0444355a2b7f8a3439793012dc62532997e5975d12c2` | `f377d747e13f56299387ef9f67e4fb825b775a0aa5fe1d4c2279ff9b349fd2cd` | `2644329f7f45128b2e89042bf30d99bfc76708d8a8e304ea236c8456b45eeb79` | `ce0efc575d7b2ba918ed0046d29e1ff2ad22cb2c465412a57c217759d54b4b96` |
| `nb_zhukov:lgb02` | `oof/oof_lexD_lgb02.npy` | `137f9d932baa333e25a094b4f5be902639dc580b6818ca5a02c40d3804e3f93b` | `oof/test_lexD_lgb02.npy` | `fbc4f5e840395990b9f3caa9746a053acb448f37b02c17749df370aedd174c89` | `b76f37824cb6127822b7d783b9e5f158d89fd51c1db8d7c7187d3f50df387029` | `276bf96ed9857f50747a218ad40b3adaa9bbc8ffae553c8bf84b2a1d85487658` | `966f988ba097508f291356c049903b27e31a560b768238e12172621531419197` |
| `nb_zhukov:xgb_base` | `oof/oof_lexD_xgb_base.npy` | `340ea93444a56f2f7970a099a97cfbb4966ac9ee23bfbd6273b917431eb5f37b` | `oof/test_lexD_xgb_base.npy` | `6660350c1acb32c56dba1d66132d79f625b86652919fc77afc69b8726a324bb5` | `676c1026487ec6ff3fb9a7fbf3bd16c20cf8570c3f9d8da218df30f1ba9952bf` | `03e8ae76dcf2d2b50a3232b4074292d4417a9036cc7ea1b6a10bc544937fe855` | `2303df4128785ac3f417f0cc3258f04d2e3643a9c3735c53798192228a6732a3` |
| `nb_reda_lgbm:lgbm` | `lgbm_oof_preds.csv` | `1f9cf0d5f7eba3f7f678cad0535a8d76b318c672b51e3a6d957a9fafa3aed9e8` | `lgbm_test_preds.csv` | `4fc616a2fe45adf27e8393d9f08312f6910ba9f86943c861777e82e5b379ad33` | `73d31cc13e644c4d80dc23efbe51df93d38744572b823237ea9676fe27d7a120` | `d37e5ce46961317ba8c871b60a84eacc20822bf673fc58686211f7adcc7f22f5` | `ac29bdd21023f2dc401b4354d3f7900a02d2b1dfdc5aacf0809adc58b21ebe1a` |
| `nb_reda_hgb:hgb` | `tehgbc_oof_preds.csv` | `888ba086c2a2fdeeca170fc1cb9a236bb1d54bec600c61773c130c6908d9ef2f` | `tehgbc_test_preds.csv` | `0cbf0db8d52ad3cf8f84d7e835a9f004b30579653575c80b60080f544f7d94d5` | `aae047c93e3b3bd37b3dffa22387d40292da69021bdcd9b91af660bf5416d434` | `044ea70171384cdfbc7505f3e67fc72ca076b372fec96cebbb7abfcd4327eaf9` | `a7a46658f0c4fe1e530881f872610ba2920fd07ac8feecade52df9a4a5fe1739` |
| `nb_yekenot:trompt` | `oof_preds.csv` | `d71fb8ad39568df9713846e1934d8f90c3918f9da434a95a86e95f2160fbf812` | `submission.csv` | `d5a854f4eee4c459dbe0299696caa77b9b71c413541cf92bac2a8d99ffb323af` | `19f738c44dd9a59ad0f2386122ac9405fd5c4742997c90a14d2e65278442f897` | `5922d81c654b177de87090ebb7719f96ccb9d9d1bb41906168c7807222c79c2f` | `93411f087c5c7e3a7071fa1d63ca2779455c0ab29709cd0ed8ffc092e0b7eb3c` |
| `nb_mohan_realmlp:realmlp` | `oof_realmlp.npy` | `d680494d813119cf52a36550f33d26321143a29ded3630d47773196c33899acd` | `test_realmlp.npy` | `cb1168e04c06a7aa821ed787ec078ae9446ff90a456bfff2d01bac51139805a9` | `b5b627a719d24f50ba6ecc13f4b9359ef9a6339574de02acb83fd5a7c59b9ccc` | `3a6e87ac2e4463d2d00fa480e03b075a221b389ae7081603e14baf2859e10271` | `5245ad2a78dcb5982153ccbfde5a14fd53eaf6c0ff98faac3ffcae294e7f625d` |
| `nb_lopure:linear_svm` | `oof_linear_svm_gpu.csv` | `2ebd3580e65efb2bfbe1efe132916a7bbb161080e9eba22ba059e385b1d64725` | `submission_linear_svm_gpu.csv` | `8cc06c356e438f438372d72774414286aa1f22f072d0e884563c1e6a9e715567` | `37544a2ff95135c536104e6e1f8f88b8c3fab5bf3d95a479a947ec4caa50e944` | `cd6cde4c3f16df52fe8392f75289707d0e1d3629761316962a947ddd430f9d34` | `2900eb8774c78852b7c38d442027f0bbbce8a1e3b949a0ab394bcaa59295adcb` |
| `nb_lopure:poly_svm` | `oof_poly_svm_gpu.csv` | `44f30ad81f2346da6af27333588ba375be8b4b7969dbde42eacd5db818a984e0` | `submission_poly_svm_gpu.csv` | `50eb7bd9ecc922fa7fd9f00d3e0775afd3a5a832a65ad73a28fd22fbd7b0182e` | `bbdad48fb393889182b6b2597862f4a49be7e47de54c0cf50467ee3bebdefa4b` | `c3683dce40db2dfe3ad21eba5a39ee3182691c145b28f0f1410008f6ba42d6ca` | `953e2da3c591335868bf5d8d665307a0b578c7e153ae6ad448dd8d360711c350` |
| `nb_lopure:rbf_svm` | `oof_rbf_svm_gpu.csv` | `9ab7783d0b43ec97094da1f8c1801ba07d32d37b5fdf1bf388f374afa2c8f509` | `submission_rbf_svm_gpu.csv` | `fac0b74f354547c4487bed62371be63c26c4611894b618413d76e67decd8b654` | `8dbb5932bc10126d6ff4046f25a3b677a86dc61c5cb9b21213bb6577f532c1aa` | `d7730b920956c645d4fb45ac65909303a2adfa6f50ebb676ea1a497e391df013` | `557a0da96cccaa7d336e1b75fbec0bb9049e6e8227924903ae421c13baaa24b2` |
| `nb_shaman_baseline:lr` | `oof_lr.npy` | `41ce2986050cc13edb2f29b78822221a16dbe21b140130cf66924381db152caa` | `pred_lr.npy` | `b52b70af1c881788bf7f86329cf91090a76c2c03dd663b56e20a4aa704c25104` | `4a6940a4ff56714a8b12bd03ef3c3ca85086ac08e6c5f85acb470eabfaf92888` | `59f764a64440dee9216a7c8a941a0a500b110916abd1532e6362fec5e9538115` | `e439bd0d7c03aad70bc027fcf66e5fb9036b0f826d13f127ea205a3fd6cb15dc` |
| `beicicc/s6e8-fold-safe-tabnet:tabnet` | `tabnet_fold_safe_oof.csv` | `e93038b7859d5da0f0410e62a177a6965c9c83933e840bf0b79ce3d5b12d7451` | `tabnet_fold_safe_test.csv` | `84254c4ea4e0bf27a05b3fcabc4bec906c8be5fd71e6d8e751e458c5138b230d` | `e3cf1629a2f8bde8458697f0d3e7ec20bb4857b4400cef921cb9553f9cdeff3f` | `ffe5fe78ca589cfa5044c8054d3f629a339162499e522fb1a1cc658dca48dbce` | `b339d0b025bc3989e2e87c0c092b1e11d3ceb7df9ca792bfd9e4b9b645535722` |
| `beicicc/s6e8-fold-safe-realmlp:realmlp` | `realmlp_fold_safe_oof.csv` | `ba5440a6ebe836d57a0720f9c92847ffe55cb4cb049163c356ab94aeb7a03933` | `realmlp_fold_safe_test.csv` | `2a8b8891a7a2fce02b96d72aad72f5a65fd9affaed7bcf369016e581dbc12076` | `52cdbe666954fb48ca966802742e103abbf3d9dbb13043cdfc4bc02ebf5b8d7b` | `54c83127bdcd6ec74e7e6e0a5b4458304333c9892ada1df8faa5f4471fccc7e4` | `e21c22c3b2416598bd2bdc198cbbbbb2e8cdedd14f3434daa751282b97784665` |
| `busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:lgb` | `oof_lgb.npy` | `025869d2ea30905a2599a69fcae6b4c7d67bf5eb74d0c96926cbb803d6021bd4` | `test_lgb.npy` | `4fe3bccea74736787e9d53aa8b357a97718c6d5b411041abce6c056d6ad2eec2` | `7d194ada845939542011d31a8e86b8f4b3144ea54c44053cdd67a3dd7ff09e9b` | `5a74c2ba29a77d1ba40bfa04fb7095f93b75de6156199d12ea820a5f2d95469d` | `ff58548f9868bdd4a5dd3fe330060b39ad21f18f232a9a776f7a7ecdf20e618f` |
| `busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:hgb` | `oof_hgb.npy` | `fad2e34d2108497f5f4d91f0dc123a9d44e6e4e424fb29008f3045af8e0898fe` | `test_hgb.npy` | `8c16745ba49754a18683e55a3571f0c35fa0b995d4343dec6fa631b4af0129ae` | `ed05f442dc0c8c61140bef6d67404fcda4d65580e9ed0ff76218d14fd97517b0` | `d7f4e3e8adf958e2a4804dd9f0da9a3b236eb7fdb58fab971077cee3956a385e` | `b7b0afba77e4c3352a3c03b555c5c68fdf5fd9d6c234e4b5b00a402a5f02564a` |
| `busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:xgb` | `oof_xgb.npy` | `8caca844702b44ea99fb29a447fa7a32c0f4ed3a4f436238b0a7c09b8d3638aa` | `test_xgb.npy` | `46268e9e123bbdf907f20e655691e85564d435e36a3a3ab5d16de61b4ad9a65d` | `dff61fdc57727feaa4ce9016a9cd9092dae5980607603f3b84c2d518e2ef7f91` | `749246d4e517f7a749f2720ae9efb1167311162449f744e5982d40348cf776e4` | `3f683ae1e737a53a2c220103b6c31375685f705030998628ecf2090c7e3d8351` |
| `ravi20076/playgrounds6e8-public-baseline-v1:XGB1C` | `OOF_Preds_MLV1_1.parquet` | `474c9ae81249bfaf026dd8e716ea79a4e4514f8f1e1d0cbd4fece758867879d9` | `Mdl_Preds_MLV1_1.parquet` | `3ded242bf1efefc5acbeeeb340a0fb62fe116633d36d1a5fb62b24c38befdd8c` | `bdf0a479cd53068f555e2c28c9ca5bbfb1cb9116a5e06807985d07f3dffde6f3` | `56e609f55ff864c59c8b56f21f8d159ebe3653a9318f4a52c6eef50371537a82` | `d795573efce0daf7fa1f87e82bd0843f1e12960bc621e033e9d93c207be822ab` |
| `ravi20076/playgrounds6e8-public-baseline-v1:LGBM1C` | `OOF_Preds_MLV1_1.parquet` | `474c9ae81249bfaf026dd8e716ea79a4e4514f8f1e1d0cbd4fece758867879d9` | `Mdl_Preds_MLV1_1.parquet` | `3ded242bf1efefc5acbeeeb340a0fb62fe116633d36d1a5fb62b24c38befdd8c` | `d1dd6681d86cfce1b9a1bca1faa8f0014f4dfb91d16bb386f172efd1a6d459c1` | `2bfa9f0a1b082b734035ede8211e6afd069c9dbbcb93e1f9903a53350228853f` | `15ea60831189c09204e17cdefbaa8e262cee346fab45ce9f7f97e32870446b66` |
| `ravi20076/playgrounds6e8-public-baseline-v1:CB1C` | `OOF_Preds_MLV1_1.parquet` | `474c9ae81249bfaf026dd8e716ea79a4e4514f8f1e1d0cbd4fece758867879d9` | `Mdl_Preds_MLV1_1.parquet` | `3ded242bf1efefc5acbeeeb340a0fb62fe116633d36d1a5fb62b24c38befdd8c` | `da5db269a883d4a093e11165a051b52d65430654031a80ab986d65e35ac7cb9b` | `4de941e94c9a3931a9af611ba98c5b953b1808c66d643005593750674dc7835c` | `f3e04b96a6bb416cab11bf092570657e9bc6d74c7446ab8bf7f97815f17e80a0` |
| `sometimessubodh/stacking-9-models-smartphone-addiction-prediction:cuml_rf` | `stacking_matrices.pkl` | `8c1f7d3e761c1cf66f3095cda530b05f64d8de11de757e87166421a33116c5ad` | `stacking_matrices.pkl` | `8c1f7d3e761c1cf66f3095cda530b05f64d8de11de757e87166421a33116c5ad` | `89151a02d81ac6898e9f6e48e67e565b636c845ddeaf02bff79fd20009e1cd9d` | `9f27567de1b1a840b7a55807bea6608eee0b1b1c5be88833da2cb6724b22b250` | `b3e6a5c21a006b6f730170e55a3a37a3f8795b62734d7f12679b4919f9bc8ed4` |
| `sometimessubodh/stacking-9-models-smartphone-addiction-prediction:cuml_logreg` | `stacking_matrices.pkl` | `8c1f7d3e761c1cf66f3095cda530b05f64d8de11de757e87166421a33116c5ad` | `stacking_matrices.pkl` | `8c1f7d3e761c1cf66f3095cda530b05f64d8de11de757e87166421a33116c5ad` | `760c2dbc37f9dceb30d83bdc4605c335106688a724e8457594b131280bc78f37` | `3bd4ef7763e903edfc9a7f06c9749fca3e11abc47cd452776c80ca104b192286` | `949716ab6998c5f084a3ea9318a0f103b35c48d27f22bfdfb41cae070e2153a2` |
| `sometimessubodh/stacking-9-models-smartphone-addiction-prediction:cuml_knn` | `stacking_matrices.pkl` | `8c1f7d3e761c1cf66f3095cda530b05f64d8de11de757e87166421a33116c5ad` | `stacking_matrices.pkl` | `8c1f7d3e761c1cf66f3095cda530b05f64d8de11de757e87166421a33116c5ad` | `f926de18987ce6467e2781e88ce5c404ab030ef95c7e251a2b07ec8b84753d95` | `2b0b92306fcc710e67917c78b82fa7fe1ad7e2421df01dbacfd1162b0f08c328` | `6cc0389e39fd8ec6658322cf524d37d6a0b77bf5a86de587be074f77a841a90b` |
| `sometimessubodh/stacking-9-models-smartphone-addiction-prediction:cuml_mbsgd` | `stacking_matrices.pkl` | `8c1f7d3e761c1cf66f3095cda530b05f64d8de11de757e87166421a33116c5ad` | `stacking_matrices.pkl` | `8c1f7d3e761c1cf66f3095cda530b05f64d8de11de757e87166421a33116c5ad` | `55c9c62f31a2b0dfef1ac920bc9dea749093fe9ec8eb73ac44eec11eb0f89922` | `620f1c39d6a86527b032a216453a5caf4c86266b697857766c0e087dc4288c0f` | `57c44070baf52a346fe78c881f04d47952ed9e1bf29f5736e1f754d01caa85c6` |
| `sometimessubodh/stacking-9-models-smartphone-addiction-prediction:torch_mlp` | `stacking_matrices.pkl` | `8c1f7d3e761c1cf66f3095cda530b05f64d8de11de757e87166421a33116c5ad` | `stacking_matrices.pkl` | `8c1f7d3e761c1cf66f3095cda530b05f64d8de11de757e87166421a33116c5ad` | `cedd6cd7d824195328e0abb9f6a2a1035d9febe3a1906231ce3222c069b6f308` | `eb8c90fbcb015e9e3c94bc36808723d4a19fb55e7f2389f5e3e577f488057a17` | `8f3d5ea1e94adbeb6162ec49995803c9f8d88b9d30325dbd343c68374842ca8f` |
| `michaelqiu0606/s6e8-depth9-pair-te-inputs:depth9_pair_te` | `base_oof.npy` | `b181851f29c67b14ee1012a723d0456725da77c51b782ec814537925f596e0f1` | `base_test.npy` | `7859efd3e05190f89d868cac5a5c226a453d11aaddad9f9452c1ace82b24b4a3` | `eff3d9e138a7b8703afed2a63688001c65e72b673a2155f374e1594f461eaeea` | `0908a1e481b119a9f6ad309ad57e8c8c4fa352679dc7127996e2b736a7b62281` | `1d85e728c61ce6c177c90183b97b77e2bf20ff231a57dac6f9fd8b9bb93462d3` |

## 고정 공개 판본과 확보

| 공개 자료 | 판본 | 소스 SHA-256 | 페이지 판본 일치 | 소스 확보 시각(UTC) | 출력 확보 시각(UTC) |
| --- | ---: | --- | --- | --- | --- |
| `zhukovoleksiy/ps6e8-eda-feature-engineering-pipeline` | 344661133 | `a1e1f488bb4d68de5c4533df8f3816517d5ca65ceb2bb904f9a389360970d9c5` | 예 | 2026-08-28T02:58:36Z | 2026-08-28T02:59:00Z |
| `redamountassir/s6e8-lgbm-lb-0-96965` | 340546450 | `c5d72447072a783ce85950c261d61273e1b918872c8f72565471a72431960cbb` | 예 | 2026-08-28T02:59:06Z | 2026-08-28T02:59:19Z |
| `redamountassir/s6e8-histgradientboosting-lb-0-96945` | 340546105 | `33a5fe33bbf21ea3afe9a1e8eea5d8a6cc02054c053e06958f9dcc1c5e1d6110` | 예 | 2026-08-28T02:59:25Z | 2026-08-28T02:59:37Z |
| `yekenot/ps-s6-e8-trompt-pytorch-frame` | 344781653 | `a743e3ad4ac8c9916e94364ce77c8a8933319b31d7b050b928d97f0467c4bc37` | 예 | 2026-08-28T02:59:43Z | 2026-08-28T02:59:53Z |
| `mohankrishnathalla/s6e8-realmlp-oof-saver` | 342288210 | `17e2015025d753537023f1f2f90f8ec13adacad7e36f4d7b2bf5a326d028558a` | 예 | 2026-08-28T02:59:59Z | 2026-08-28T03:00:21Z |
| `lopure/hdviz-pca-parallel-with-linear-svm` | 342359513 | `a6cccd0afd1c8d46d88367d0d50d8386768be99af1bd9953800f7c0c79504b96` | 예 | 2026-08-28T03:00:27Z | 2026-08-28T03:00:48Z |
| `shamanthakreddymallu/s6e8-baseline` | 345105403 | `1647535be4fafd0c8bea4b32e7968fe91bfe3e5a6efd3e147e84e68d07a35aa7` | 예 | 2026-08-28T03:00:54Z | 2026-08-28T03:01:12Z |
| `beicicc/s6e8-fold-safe-tabnet` | 339872430 | `3f97a7351a649a6a583edf3492fe4db190d926ef94778c74db3bba41c8abfeb7` | 예 | 2026-08-28T03:01:19Z | 2026-08-28T03:01:31Z |
| `beicicc/s6e8-fold-safe-realmlp` | 339864149 | `60a0bd05332e8932468d9cc796855013be3c3798344fd75c15c016764eba58ef` | 예 | 2026-08-28T03:01:37Z | 2026-08-28T03:01:49Z |
| `busyaprime/s6e8-tabular-baseline-that-autodetects-the-task` | 339485089 | `52c509d1b726d2ddeaddd0e07ada8c345a21483c09f0128e0070359552025235` | 예 | 2026-08-28T03:01:55Z | 2026-08-28T03:02:15Z |
| `ravi20076/playgrounds6e8-public-baseline-v1` | 339444387 | `186d26a1aba7549fd182ed89322daff43f6083d8f9275175215c7c0207d31c30` | 예 | 2026-08-28T03:02:21Z | 2026-08-28T03:02:33Z |
| `ravi20076/playgrounds6e8-public-imports-v1` (보조 코드) | 339439580 | `26504708be69444a8df97ac7b0ecc95e788340b88b5cfaef823dfe2c9d8a1405`, `f5ab8352cc2c8b3233f07a0c35a833009466a5eb53739ec440c477f47c3fa98b`, `2a98eef32eca0a035473609fe7f37531f6c463dfed411dd62351a7f5216150fa` | 예 | 2026-08-28T03:02:39Z | 2026-08-28T03:02:46Z |
| `sometimessubodh/stacking-9-models-smartphone-addiction-prediction` | 346039237 | `691100dcf6f0b365e4c1a5902e52218797cfe00c73dca19b8e6a2b19087473bb` | 예 | 2026-08-31T12:01:21Z | 2026-08-31T12:01:40Z |
| `michaelqiu0606/s6e8-depth9-pair-te-inputs` | 1 | `ec1ff5cf164c212335cfa0c748ada21c4ed4f473315524e9e5bccc371e195ee9` | 예 | 2026-08-31T12:01:52Z | 2026-08-31T12:01:52Z |

노트북 고정 판본은 `kaggle kernels pull`이 받은 공개 소스 SHA-256과 공개 페이지의 `scriptVersionId`를 함께 대조했다.
자료 고정 판본은 Kaggle 자료 판본 번호와 내려받은 README.md SHA-256을 함께 대조했다.
Kaggle CLI 2.2.4는 특정 판본 내려받기를 거부(403)하므로 위 두 대조가 판본 고정의 근거다.

## 검증 항목

- 행 수 691,369(OOF)와 296,302(시험), 유한값.
- 원래 행 순서: id 열이 있는 CSV는 train.csv·test.csv의 id 순서와 정확히 일치해야 하고, npy·parquet는 위치 정렬이며 저장 코드가 원본 순서로 채우는지 소스에서 확인한다.
- 고정 5분할: 소스 안의 분할 코드 위치를 감사 기록에 남긴다(`fold_contract.evidence`).
- 학습 격리 주장 10개: 각각 `확인됨`, `위반 확인`, `알 수 없음`, `해당 없음`과 근거 종류, 고정 판본 안의 셀·줄 위치를 기록한다. 근거 조각을 소스에서 찾지 못하면 자동으로 `알 수 없음`이 된다.
- 외부 입력·의사 목표값·증류 패턴 자동 검색(`absence_scan`): 설명되지 않은 일치가 있으면 `근거 부족`.
- 재채점 AUC는 독립 조사 AUC, 선언 AUC와 1e-5 안에서 맞아야 하며 산출물 동일성 확인에만 쓴다.
- 정확 중복은 제외 사유, 근접 중복(스피어만 0.998 이상)은 주의 사항이며 자격을 바꾸지 않는다.
- 사용 조건 미표시는 `license_unknown_use_limited` 주의 사항과 결합 입력 전용 범위로 기록하며 자격 제외 사유가 아니다.

## 재현

```
uv run python scripts/build_external_member_ledger_v3.py fetch
uv run python scripts/build_external_member_ledger_v3.py audit
uv run python scripts/build_external_member_ledger_v3.py verify
```

`audit`는 같은 입력에서 같은 내용 지문을 얻으면 기존 감사 기록을 그대로 두고(감사 시각도 유지), 고정 판본·예측·근거·판정이 달라진 후보만 새 기록을 만들어 `supersedes_audit_record_id`로 잇는다.
`verify`는 모든 기록의 `record_sha256`와 근거 manifest 해시를 다시 계산하고, 커밋된 기록에 예측 배열이 없는지 확인한다.

## 범위 밖

- 외부 후보 동결 명세 생성, 중첩 선별 판정, 확장 스택 조립과 제출.
