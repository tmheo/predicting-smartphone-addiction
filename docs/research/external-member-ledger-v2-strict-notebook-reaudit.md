# 판본 2 공개 노트북 구성원 엄격 재감사

이 문서는 GitHub 이슈 [판본 2의 공개 노트북 구성원 45개를 엄격 기준으로 재감사한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/480)의 조사 결과다.

## 결론

판본 2 장부에서 통과했던 공개 노트북 구성원 45개를 지도 [#477](https://github.com/tmheo/predicting-smartphone-addiction/issues/477)의 `검증 가능한 개별 외부 구성원` 기준으로 다시 판정하면 **11개가 통과하고 34개가 제외된다**.
통과 11개는 모두 공식 훈련 자료의 커뮤니티 5분할을 쓰며, 바깥 검증 행의 목표값을 학습, 목표값 기반 전처리, 학습 시점 결정이나 설정 선택에 쓰지 않은 하나의 모형 계보다.
제외 34개는 모두 바깥 검증 목표값을 조기 종료, 최적 가중치 복원, 최적 반복 수 결정 또는 검증 자료를 받는 학습 호출에 사용했다.
그 가운데 yaminh 3개는 전체 OOF 결측 처리 비교로 설정을 고르고, Danush 3개는 첫 바깥 검증 결과로 특성 공학 사용 여부도 고르므로 설정 선택 누출까지 겹친다.
따라서 판본 3에 그대로 편입할 수 있는 공개 노트북 구성원 집합은 아래 11개뿐이다.

## 판정 계약과 자료

판정 단위는 공식 훈련 자료에서 직접 학습한 하나의 모형 계보가 만든 OOF와 시험 예측 한 쌍이다.
고정 설정의 여러 난수 초기값이나 분할 예측 평균은 허용하지만, 바깥 검증을 이용한 조기 종료, 서로 다른 모형이나 설정의 결합, 외부 예측 재학습, 의사 목표값 학습과 결합 예측 교사 증류는 제외했다.
공개 전 탐색 이력은 고정 공개 판본만으로 보증할 수 없으므로 그 사실만으로 제외하지 않았다.
행 순서와 산출물 검사는 판본 2의 [기계 판독 장부](https://github.com/tmheo/predicting-smartphone-addiction/blob/96ec136/docs/research/external-member-ledger.json)를 원자료로 다시 확인했다.
45개 모두 OOF 691,369행과 시험 예측 296,302행이고 모든 값이 유한하며, 장부가 가리키는 90개 로컬 파일도 전부 존재했다.
표의 산출물 SHA-256은 장부가 OOF와 시험 예측을 float64로 정규화해 이어 붙인 내용 식별자다.
코드 판정은 각 표에 연결한 `scriptVersionId` 고정 판본을 Kaggle 명령줄 도구로 내려받아 저장 출력과 전체 코드 셀을 함께 읽어 내렸다.
공개 노트북 소스의 이용 조건은 Kaggle의 [Meta Kaggle Code 설명](https://www.kaggle.com/datasets/kaggle/meta-kaggle-code)과 [Apache License 2.0 원문](https://www.apache.org/licenses/LICENSE-2.0)에 따르지만, 예측 배열에는 별도 이용 조건 표시가 없으므로 결합기 입력으로만 사용하고 재배포하지 않는다.

## 통과 11개

| 고정 공개 판본 | 구성원과 OOF AUC | 산출물 SHA-256 | 통과 근거 |
| --- | --- | --- | --- |
| [zhukovoleksiy, 판본 344661133](https://www.kaggle.com/code/zhukovoleksiy/ps6e8-eda-feature-engineering-pipeline?scriptVersionId=344661133) | `nb_zhukov:cat_base` 0.967992, `nb_zhukov:lgb02` 0.968355, `nb_zhukov:xgb_base` 0.967859 | `ce0efc575d7b2ba918ed0046d29e1ff2ad22cb2c465412a57c217759d54b4b96`, `966f988ba097508f291356c049903b27e31a560b768238e12172621531419197`, `2303df4128785ac3f417f0cc3258f04d2e3643a9c3735c53798192228a6732a3` | 난수 초기값 42의 커뮤니티 5분할이며, 학습행 목표값 인코딩은 안쪽 교차 적합을 하고 검증행에는 바깥 학습 부분의 대응표만 적용하며, 세 모형 모두 고정 3,333회 설정으로 바깥 검증 자료 없이 적합한다. |
| [redamountassir LGBM, 판본 340546450](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965?scriptVersionId=340546450) | `nb_reda_lgbm:lgbm` 0.968259 | `ac29bdd21023f2dc401b4354d3f7900a02d2b1dfdc5aacf0809adc58b21ebe1a` | 난수 초기값 42의 커뮤니티 5분할이며, 목표값 인코딩은 바깥 학습 부분 안에서 5분할 교차 적합되고 모형은 바깥 검증 자료 없이 고정 780회 학습한다. |
| [redamountassir HGB, 판본 340546105](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945?scriptVersionId=340546105) | `nb_reda_hgb:hgb` 0.968026 | `a7a46658f0c4fe1e530881f872610ba2920fd07ac8feecade52df9a4a5fe1739` | 같은 목표값 인코딩과 분할을 쓰며, `max_iter=780`과 `early_stopping=False`가 고정돼 있다. |
| [yekenot, 판본 344781653](https://www.kaggle.com/code/yekenot/ps-s6-e8-trompt-pytorch-frame?scriptVersionId=344781653) | `nb_yekenot:trompt` 0.966671 | `93411f087c5c7e3a7071fa1d63ca2779455c0ab29709cd0ed8ffc092e0b7eb3c` | 난수 초기값 42의 커뮤니티 5분할에서 매 분할을 고정 8회 학습하며, 검증 AUC는 기록만 하고 중단이나 가중치 선택에 쓰지 않는다. |
| [mohankrishnathalla RealMLP, 판본 342288210](https://www.kaggle.com/code/mohankrishnathalla/s6e8-realmlp-oof-saver?scriptVersionId=342288210) | `nb_mohan_realmlp:realmlp` 0.958134 | `5245ad2a78dcb5982153ccbfde5a14fd53eaf6c0ff98faac3ffcae294e7f625d` | 난수 초기값 42의 커뮤니티 5분할이며, 바깥 학습 부분에서만 결측 대치와 목표값 인코딩을 맞춘 뒤 `n_epochs=512`로 고정하고 바깥 검증 자료를 `fit`에 넘기지 않는다. |
| [lopure, 판본 342359513](https://www.kaggle.com/code/lopure/hdviz-pca-parallel-with-linear-svm?scriptVersionId=342359513) | `nb_lopure:linear_svm` 0.911345, `nb_lopure:poly_svm` 0.928795, `nb_lopure:rbf_svm` 0.922167 | `2900eb8774c78852b7c38d442027f0bbbce8a1e3b949a0ab394bcaa59295adcb`, `953e2da3c591335868bf5d8d665307a0b578c7e153ae6ad448dd8d360711c350`, `557a0da96cccaa7d336e1b75fbec0bb9049e6e8227924903ae421c13baaa24b2` | 난수 초기값 42의 커뮤니티 5분할에서 대치, 표준화와 범주 부호화를 분할마다 학습 부분에만 맞추며, 세 SVM 설정은 고정돼 있고 검증 목표값을 학습 호출에 넘기지 않는다. |
| [shamanthakreddymallu, 판본 345105403](https://www.kaggle.com/code/shamanthakreddymallu/s6e8-baseline?scriptVersionId=345105403) | `nb_shaman_baseline:lr` 0.936609 | `e439bd0d7c03aad70bc027fcf66e5fb9036b0f826d13f127ea205a3fd6cb15dc` | 공개 코드에 `SEED=42`와 커뮤니티 5분할이 있으며, 고정 로지스틱 회귀 절차를 각 바깥 학습 부분에만 맞추므로 판본 2의 `fold_evidence_none` 기록을 바로잡을 수 있다. |

zhukovoleksiy 코드가 `VARIANT='C'`를 계산하고도 `lexD_*` 파일명으로 저장하는 이름 불일치는 확인했지만, 실제 부호기 설정과 저장 배열 계보가 일치하므로 누출 사유는 아니다.
mohankrishnathalla RealMLP가 내부 검증을 자동 구성하더라도 바깥 학습 부분만 전달받으므로 각 바깥 검증 행의 목표값 격리는 유지된다.

## 제외 34개

| 고정 공개 판본 | 제외 구성원 | 수 | 결정적 제외 사유 |
| --- | --- | ---: | --- |
| [kodaifukuda0311, 판본 345260655](https://www.kaggle.com/code/kodaifukuda0311/s6e8-how-to-achieve-0-97-with-realmlp-only?scriptVersionId=345260655) | `nb_kodaifukuda:realmlp` | 1 | `use_early_stopping=True`인 RealMLP의 `fit`에 바깥 검증 입력과 목표값을 직접 넘긴다. |
| [omid FT-Transformer, 판본 339707060](https://www.kaggle.com/code/omidbaghchehsaraei/ft-transformer-for-predicting-smartphone-addiction?scriptVersionId=339707060), [CNN, 판본 342747549](https://www.kaggle.com/code/omidbaghchehsaraei/cnn-for-predicting-smartphone-addiction?scriptVersionId=342747549), [TabTransformer, 판본 342815072](https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction?scriptVersionId=342815072), [FastAI, 판본 344317824](https://www.kaggle.com/code/omidbaghchehsaraei/fastai-for-predicting-smartphone-addiction?scriptVersionId=344317824), [XGBoost v2, 판본 343324186](https://www.kaggle.com/code/omidbaghchehsaraei/xgboost-v2-for-predicting-smartphone-addiction?scriptVersionId=343324186), [CatBoost, 판본 339750720](https://www.kaggle.com/code/omidbaghchehsaraei/catboost-for-predicting-smartphone-addiction?scriptVersionId=339750720) | `nb_omid_ft:ft_transformer`, `nb_omid_cnn:cnn`, `nb_omid_tabtr:tabtransformer`, `nb_omid_fastai:fastai`, `nb_omid_xgb2:xgboost_v2`, `nb_omid_cat:catboost` | 6 | FT-Transformer와 CatBoost는 바깥 검증 자료를 학습 호출에 넘기고, CNN과 TabTransformer는 검증 AUC로 최적 가중치와 중단 시점을 고르며, FastAI는 검증 ROC AUC 최적 모형을 저장하고, XGBoost는 바깥 검증 집합으로 조기 종료한다. |
| [yaminh, 판본 342906969](https://www.kaggle.com/code/yaminh/smartphone-addiction-prediction-strong-eda-cv-eble?scriptVersionId=342906969) | `nb_yaminh:lgbm_te`, `nb_yaminh:xgb_te`, `nb_yaminh:catboost` | 3 | 세 모형이 모두 바깥 검증 집합으로 조기 종료하며, 전체 OOF 결측 처리 비교의 최고 점수로 설정까지 고른다. |
| [sidhaarthshree, 판본 340068345](https://www.kaggle.com/code/sidhaarthshree/lightgbm-ensemble-based-on-eda?scriptVersionId=340068345) | `nb_sidhaarth:lgb_a`, `nb_sidhaarth:lgb_b`, `nb_sidhaarth:xgb` | 3 | 두 LightGBM과 XGBoost가 모두 바깥 검증 집합으로 조기 종료한다. |
| [lucymlai32 XGBoost, 판본 344000843](https://www.kaggle.com/code/lucymlai32/phase-2-xgboost-and-model-blending?scriptVersionId=344000843), [CatBoost, 판본 343943487](https://www.kaggle.com/code/lucymlai32/smartphone-addiction-prediction?scriptVersionId=343943487) | `nb_lucy_xgb:xgboost`, `nb_lucy_cat:catboost_v2` | 2 | 두 모형 모두 바깥 검증 집합으로 조기 종료한다. |
| [cdeotte XGBoost, 판본 343747866](https://www.kaggle.com/code/cdeotte/simple-xgb-starter?scriptVersionId=343747866), [CatBoost, 판본 343961092](https://www.kaggle.com/code/cdeotte/simple-cat-starter?scriptVersionId=343961092), [신경망, 판본 343957076](https://www.kaggle.com/code/cdeotte/simple-nn-starter?scriptVersionId=343957076) | `nb_cdeotte_xgb:xgb`, `nb_cdeotte_cat:cat`, `nb_cdeotte_nn:nn` | 3 | XGBoost와 CatBoost는 바깥 검증 집합으로 조기 종료하고, 신경망은 바깥 검증 AUC로 최적 세대와 가중치를 고른다. |
| [dariushafshar, 판본 345135659](https://www.kaggle.com/code/dariushafshar/0-97184-leader-xgb-feature-ablation?scriptVersionId=345135659) | `nb_darius_ablation:xgb` | 1 | 바깥 검증 집합으로 XGBoost를 조기 종료한다. |
| [rv1922, 판본 341656424](https://www.kaggle.com/code/rv1922/smartphone-addiction?scriptVersionId=341656424) | `nb_rv1922:lgbm_v1_seed42`, `nb_rv1922:lgbm_v2_seed42`, `nb_rv1922:lgbm_v3_seed42`, `nb_rv1922:xgb_seed42` | 4 | 세 LightGBM 설정과 XGBoost가 모두 바깥 검증 집합으로 조기 종료한다. |
| [yadoy666, 판본 342773026](https://www.kaggle.com/code/yadoy666/predicting-smartphone-addiction?scriptVersionId=342773026) | `nb_yadoy:catboost`, `nb_yadoy:xgboost` | 2 | CatBoost와 XGBoost가 모두 바깥 검증 집합으로 조기 종료하고 최적 반복 수를 쓴다. |
| [danushkumarv, 판본 340123043](https://www.kaggle.com/code/danushkumarv/smartphone-addiction-gbm-rank-blend-nb01?scriptVersionId=340123043) | `nb_danush:lgb`, `nb_danush:xgb`, `nb_danush:cb` | 3 | 세 모형이 모두 바깥 검증 집합으로 조기 종료하며, 첫 분할의 검증 점수로 특성 공학 사용 여부도 고른다. |
| [harwindersingh766, 판본 340654922](https://www.kaggle.com/code/harwindersingh766/ps-s6e8-xgboost-te-lb-0-96548?scriptVersionId=340654922) | `nb_harwinder:xgb` | 1 | 바깥 검증 집합으로 XGBoost를 조기 종료한다. |
| [dynamo14324, 판본 344573682](https://www.kaggle.com/code/dynamo14324/smartphone-addiction-championship-v11?scriptVersionId=344573682) | `nb_dynamo:lgb_v11`, `nb_dynamo:xgb_v11` | 2 | LightGBM과 XGBoost가 모두 바깥 검증 집합으로 조기 종료한다. |
| [mohankrishnathalla MLP, 판본 341905997](https://www.kaggle.com/code/mohankrishnathalla/s6e8-tabm-oof-saver?scriptVersionId=341905997) | `nb_mohan_tabm:mlp` | 1 | 바깥 검증 AUC로 최적 가중치를 복원하고 인내 횟수 15의 조기 종료를 한다. |
| [kava1, 판본 343708558](https://www.kaggle.com/code/kava1/predicting-smartphone-addiction-resnet-fe?scriptVersionId=343708558) | `nb_kava1:resnet` | 1 | 바깥 검증 AUC로 최적 가중치를 복원하고 인내 횟수 15의 조기 종료를 한다. |
| [shamanthakreddymallu, 판본 345105403](https://www.kaggle.com/code/shamanthakreddymallu/s6e8-baseline?scriptVersionId=345105403) | `nb_shaman_baseline:lgb_fe` | 1 | 바깥 검증 집합과 인내 횟수 150으로 LightGBM을 조기 종료한다. |
| **합계** | | **34** | 34개 모두 바깥 검증 목표값 격리 기준을 어긴다. |

Omid 여섯 판본은 판본 2에서 형제 노트북 코드에 의존했던 근거 부족을 이번에 각 고정 판본의 실제 코드로 해소했지만, 직접 코드가 오히려 바깥 검증 목표값 사용을 확인하므로 결론은 제외다.
Shaman의 두 구성원은 공개 코드에서 커뮤니티 5분할을 확인했으므로 판본 2의 `fold_evidence_none`은 둘 다 `published_code`로 고칠 수 있지만, 그중 LightGBM만 조기 종료 때문에 제외된다.

## 고정 판본 원문 식별자

다음 SHA-256은 이번 감사에서 내려받은 `.ipynb` 원문 파일의 식별자이며, 같은 행의 고정 판본 링크와 한 쌍으로 보관해야 한다.

| 노트북 | `scriptVersionId` | 내려받은 `.ipynb` SHA-256 |
| --- | ---: | --- |
| kodaifukuda0311/s6e8-how-to-achieve-0-97-with-realmlp-only | 345260655 | `1208c5741d457025a89f961f85ff1f03698e93479ae1a2428c4fc6b23d05e017` |
| omidbaghchehsaraei/ft-transformer-for-predicting-smartphone-addiction | 339707060 | `2dfda354ae3c7c5b01b25eb5614ef556c5e860a8d5c26f23c93ee9ac81b39f97` |
| omidbaghchehsaraei/cnn-for-predicting-smartphone-addiction | 342747549 | `2310c4fa1b98230989f8e3bcf3f9661985a2c30df90597786e739cd34321f4dc` |
| omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction | 342815072 | `eeb3e1cccbaab29c71ef946876f7042509f6ef537df4a9b04ced36e3c424e46c` |
| omidbaghchehsaraei/fastai-for-predicting-smartphone-addiction | 344317824 | `09c13b3895353b28a8c16bd172c84f9f3359d97c5583704c5067b5c9c672ed32` |
| omidbaghchehsaraei/xgboost-v2-for-predicting-smartphone-addiction | 343324186 | `56bb9ed8b6fbce927a77e19ab1f60f3e0f2c46f2ef37153f8cb8ec47a707ebbc` |
| omidbaghchehsaraei/catboost-for-predicting-smartphone-addiction | 339750720 | `b05ec5f3a44ef0f65bf394937cf111068e4c234d16088d345538adc726fc3e6e` |
| zhukovoleksiy/ps6e8-eda-feature-engineering-pipeline | 344661133 | `a1e1f488bb4d68de5c4533df8f3816517d5ca65ceb2bb904f9a389360970d9c5` |
| redamountassir/s6e8-lgbm-lb-0-96965 | 340546450 | `c5d72447072a783ce85950c261d61273e1b918872c8f72565471a72431960cbb` |
| redamountassir/s6e8-histgradientboosting-lb-0-96945 | 340546105 | `33a5fe33bbf21ea3afe9a1e8eea5d8a6cc02054c053e06958f9dcc1c5e1d6110` |
| yaminh/smartphone-addiction-prediction-strong-eda-cv-eble | 342906969 | `881334cd7030462594ff9432ceea1eec4e429756999c8a5aa65b6da6d451ecd1` |
| sidhaarthshree/lightgbm-ensemble-based-on-eda | 340068345 | `a12918ab88f65e7e6284be7b6a8f34f6fa06e767e0b7c861e1137fe7c98c779e` |
| yekenot/ps-s6-e8-trompt-pytorch-frame | 344781653 | `a743e3ad4ac8c9916e94364ce77c8a8933319b31d7b050b928d97f0467c4bc37` |
| lucymlai32/phase-2-xgboost-and-model-blending | 344000843 | `b383a4eb53b0cb9e926cccd3a937bef7c01393dda5bc8537ee50b921b912c815` |
| lucymlai32/smartphone-addiction-prediction | 343943487 | `d6ee697d0bfc97a91ea4ca5f20aa88dfce257270297cae5acb2a8ea8284bbf9a` |
| cdeotte/simple-xgb-starter | 343747866 | `958d80eab63f9ac627fdecbaab50b45a117cac4b459ab5865a7cb3a30e6aaefb` |
| cdeotte/simple-cat-starter | 343961092 | `434e805cee86d2e16ebdaa131b1cad6b5f51cf53e152aa667d6922c69ccdba80` |
| cdeotte/simple-nn-starter | 343957076 | `2f6e2ff792d3207d42bf9f13aa017cf0298ba3486e3d26267a2becc0fdeec41d` |
| [lavanyabacche/xgb-starter-01](https://www.kaggle.com/code/lavanyabacche/xgb-starter-01?scriptVersionId=343891260) | 343891260 | `f7ececb3a43d3a899b61131fe16d058da1cbaa443286b58d9997a8bec813be44` |
| dariushafshar/0-97184-leader-xgb-feature-ablation | 345135659 | `28ccf573709939ae63765ea67b7a7af9d547a729cfc62974dd6bc9ce8f568be2` |
| rv1922/smartphone-addiction | 341656424 | `62fe4cbab7b08672ff9c4d869f723aa4d01148c837a895cc22b9855226a6ddd1` |
| yadoy666/predicting-smartphone-addiction | 342773026 | `c3bde7c040169d9636743523886f80b506bc283376dff42fd086bfa793020dd3` |
| danushkumarv/smartphone-addiction-gbm-rank-blend-nb01 | 340123043 | `c05eaf580ea07a2e87ffb37c4bfaf72a616b41280b3965e0a92449fa67fc6554` |
| harwindersingh766/ps-s6e8-xgboost-te-lb-0-96548 | 340654922 | `be58b091cad5a234b564049682663f7233b510faa2c3ade9759774d856f12292` |
| dynamo14324/smartphone-addiction-championship-v11 | 344573682 | `95d74da6bd06878ead1e252802eb7f23c19d9f7abdf99e847b4b779707f6cb37` |
| mohankrishnathalla/s6e8-realmlp-oof-saver | 342288210 | `17e2015025d753537023f1f2f90f8ec13adacad7e36f4d7b2bf5a326d028558a` |
| mohankrishnathalla/s6e8-tabm-oof-saver | 341905997 | `590b996fcbfa435ff7257f8bce9e3a50c99ba2049fd080e4ce26119f6d03ac91` |
| kava1/predicting-smartphone-addiction-resnet-fe | 343708558 | `3e81ca720dcaef7a3bf3dab5af83559b8b559ca5bc8e56ad42eecc656c820081` |
| lopure/hdviz-pca-parallel-with-linear-svm | 342359513 | `a6cccd0afd1c8d46d88367d0d50d8386768be99af1bd9953800f7c0c79504b96` |
| shamanthakreddymallu/s6e8-baseline | 345105403 | `1647535be4fafd0c8bea4b32e7968fe91bfe3e5a6efd3e147e84e68d07a35aa7` |

lavanyabacche 판본은 공개 노트북 후보 46번째 항목이지만 cdeotte XGBoost와 예측 배열이 바이트 단위로 같아서 판본 2부터 이미 제외됐으며, 이번 45개 재감사 분모에는 들어가지 않는다.

## 판본 3 반영 사항

판본 3은 위 통과 11개만 공개 노트북 계열의 검증 가능한 개별 외부 구성원으로 편입해야 한다.
현재 판본 2의 `accepted` 표시나 `published_code` 표시는 행 수와 분할 모양의 1차 검증일 뿐, 바깥 검증 목표값 격리를 보증하지 않으므로 이번 판정을 우선해야 한다.
Omid 여섯 항목의 분할 근거는 `sibling_code`에서 직접 공개 코드로 승격할 수 있지만 모두 제외 상태여야 한다.
Shaman 로지스틱 회귀의 분할 근거는 `none`에서 직접 공개 코드로 승격하고 통과 상태로 유지해야 하며, 같은 노트북의 LightGBM도 분할 근거만 승격한 뒤 제외해야 한다.
공개 이전의 실험 탐색, 노트북 내부 라이브러리의 바깥 학습 부분 안쪽 자동 분할, 이용 조건이 표시되지 않은 예측 배열의 권리 범위는 이번 코드 감사만으로 더 넓게 보증하지 않는다.
