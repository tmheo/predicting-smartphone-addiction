# OOF 후보 풀 품질·다양성 감사

## 결론

현재 장부 16개 중 16개가 계보, 정렬, fold, float64와 유한성 검사를 통과했다.
시드별 OOF 평균은 6개에서 재계산해 확인했고, #98 이전 실행 10개는 시드별 파일이 없어 기존 기록 부분 확인으로 남는다.
정확·순위 중복 제거 뒤 16개가 남았고, 모두 nested OOF 평가 후보로 유지한다.
전체 후보의 균등 순위 평균 OOF AUC는 `0.968677978702`다.
난수 64개와 구성원 복제 대조의 영점 대역은 `-0.000837324462`에서 `+0.000123276030`다.
균등 순위 평균의 제외 기여와 영점 대역은 참고값이며, 후보 진입이나 제거에 쓰지 않는다.

무결성 실패 후보가 없다.

## 무결성 및 배열 해시

| 후보 | run | 판정 | 시드 평균 | OOF 결측 | 시험 결측 | OOF SHA-256 | 시험 SHA-256 |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| exp006_te_drop_gaming | `4aaddd505ade428588cf83659f205cf6` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `6582c852dee84acc1cae9b5231c3e4b223f2849338286cb195bcc6a8bda100f6` | `407bf10dc247cbe51e9d4a638e4c415186e2467d653715a8e54259fc3a73b6a1` |
| exp011_resid_pair | `e21d19af37bb461db5dc1a59b66411a6` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `15988c7117300ed4be52bee52e0fe065f63f2799448969bef4cc07cab65bd6df` | `63f0e1b7612ed426698fdb4cf7a88ec1bc6320d7360063a31ebbbc5d1175e9bf` |
| exp022_orig_knn | `52e9c12bcfb94ee9912e4bb7f43e3f09` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `f66f0895d1e8f9fd86420c02c669b319144d5f9d8bb306a48416fb2968cb71de` | `d802a2ccb39c54b37c931d2389e3d925ed016a0de119031b19679b293b0fa4ca` |
| exp023_orig_proxy_residual | `202b7d47c3bd4b8e8904fa5c005c7423` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `585fd50e1eb834a0854434d803022f10c088324cd47d752096312aeda23f283b` | `ac21067ffd3e782c70df32d52d0c9f00a030d4d35792f0d2d43615b4faa5f8c4` |
| exp026_constrained_impute_nowidth | `62f57ea7478f4de4b4fecdfb4baa3a35` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `9d49671e47d140a48f2a32e1134d2bbf371b2e831d988b15e5379629115ce6e6` | `c7d5cd9c0a358cbd32e7165d1b509c990620d5f402ebf1a0ad95ae5a76f289e3` |
| exp032_recon_orig_mean_top3 | `b1bd4b08386946c9a4e05b755dbd5f26` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `542114610c478fed58b4bb1bb35fee50ad704a01b78cc2f763eab0096f31f064` | `347326a0698770e3d153b12ee475c42d0508c2c4588f2af1020c5e56331d722d` |
| exp033_recon_orig_mean_top3_raw | `c34f1da180544c3ba022140d394cb055` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `9bb907c98aec08e3a6c401a866b67c563b5290b9656d8b3f6b2ebd878d067e81` | `f5bfa6174b00e80875f7ca08f1e04dba8e65997dd71de24b5179dfc06ef8ec56` |
| exp035_lattice_te | `c62a9ad334bf486cb46d738474ccd767` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `490f5939bed0d2ebb4192da172dc5b8ecfd3960b2fee2dc2b51ac9dd2f3c443b` | `7d7a4c7ab6c7815e6ff7a12f071d2d75e5a6448758129001eb9ee913cf7abffe` |
| exp045_xgb_depth8 | `e2c432b487564df990a4cc3baf3d6fc0` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `63ce3b70609ea7aaec4e199aa79020a2ea6f08985df97cb7e868835f389e4d9e` | `ad1e3bfe801e9357e0b6e67ad9f7d79c6adaa1d35941807f9f8a86d8958f262a` |
| exp058_logreg_onehot | `e2b76edd9d204290810e39a7457c4c48` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `f3d479df12e2fa16a33b8d188dafa8faa3159458606f3dbdefe5a41365082769` | `4dd874272dfd305ea41ba6a8ddb017fe5e1ca0987d9f19271aa7e6fa7a192398` |
| exp059_lookup_transformer | `b951fac51b6b44298f7fdb0b543caba6` | 통과 | 완전 확인 | 0 | 0 | `9407248f85db87e1825f140a8c3927d8d296c8daddf80215d929148da6ccf3d1` | `c23651771024fa66f9065dc14bf93a694517f280df6e360f19fc2f030e0c4f25` |
| exp065_tabm | `df2023d4bd71429b88a8026d52f5258c` | 통과 | 완전 확인 | 0 | 0 | `565d10dda8054e9bf988d3417adda1ae939a1d1fed64970c2ec8f8b8bb1b1fb2` | `045c9e67ad6d4857d1aef978175e109bcc20561f123bcc102516d80906271a2d` |
| exp070_cat_exact_cats | `6238d8c58beb42ee8102209ded92516e` | 통과 | 완전 확인 | 0 | 0 | `64039d4b52ce4dbf5ff690de19d92346d0e6c6a695337c1f34ac82cba65a1c30` | `e3b2cb7a9db28990e0adb82e95775b0761971cf18d6a7a954f3b0b6f1d59ac77` |
| exp067_tabpfn3 | `85b09132aae543aab16e69dea5906a30` | 통과 | 완전 확인 | 0 | 0 | `59a742f0a4ad336826d87491a01f92b1f23bb2ba23ba6c0380d3eede34c070f7` | `cea3b01adf1cd3c0ad2c02b57cb3adcb5afe540d14e129763cb65ae7ca7040f7` |
| exp074_lgb_kitopl_d2_bundle | `446a90bedd2a4d8bafad56f308b2b999` | 통과 | 완전 확인 | 0 | 0 | `6a6738cedaa80e844e4940c700ab56faae77dbeea48e962530c87ae24870bc96` | `345ab8301d30abc8b51f0f180cc678aa81466a26d683ec2cacc70c39aa6a3a44` |
| exp081_lookup_fold_initialization_avg3 | `d55d1cd49c194eb8bf7b5128e548df81` | 통과 | 완전 확인 | 0 | 0 | `92d2188043561b57cc937800e54026d7cea97a8ef8d6ee6f1947d7e6007e18a0` | `6ea966a3116e4fff05b5726656ad852ef2298b1d5650dec7a139e9318de770ad` |

기존 기록 부분 확인 사유:

- `exp006_te_drop_gaming`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.
- `exp011_resid_pair`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.
- `exp022_orig_knn`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.
- `exp023_orig_proxy_residual`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.
- `exp026_constrained_impute_nowidth`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.
- `exp032_recon_orig_mean_top3`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.
- `exp033_recon_orig_mean_top3_raw`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.
- `exp035_lattice_te`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.
- `exp045_xgb_depth8`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.
- `exp058_logreg_onehot`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.

## 중복 판정

OOF와 시험 예측이 모두 같은 정확 중복 및 스피어만 0.998 이상 중복이 없다.

## 후보 유지 정책

무결성과 중복 검사를 통과한 후보는 균등 순위 평균 기여의 부호와 관계없이 모두 유지한다.
구성원 선택과 가중치는 outer fold 안에서 학습하는 nested OOF 평가가 결정한다.

고정점 후보: `exp006_te_drop_gaming`, `exp011_resid_pair`, `exp022_orig_knn`, `exp023_orig_proxy_residual`, `exp026_constrained_impute_nowidth`, `exp032_recon_orig_mean_top3`, `exp033_recon_orig_mean_top3_raw`, `exp035_lattice_te`, `exp045_xgb_depth8`, `exp058_logreg_onehot`, `exp059_lookup_transformer`, `exp065_tabm`, `exp070_cat_exact_cats`, `exp067_tabpfn3`, `exp074_lgb_kitopl_d2_bundle`, `exp081_lookup_fold_initialization_avg3`.

## 품질과 다양성

제외 기여는 전체 후보의 균등 순위 평균에서 각 후보 하나를 제외한 참고값이다.
잔차 상관은 최근접 순위 상관 후보와의 피어슨 상관이다.

| 후보 | 단독 OOF | 최근접 후보 | 스피어만 | 잔차 상관 | 제외 기여 | 판정 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| exp006_te_drop_gaming | 0.966592431674 | exp022_orig_knn | 0.996466394406 | 0.989395144148 | -0.000058415123 | 유지 |
| exp011_resid_pair | 0.967401071703 | exp022_orig_knn | 0.997394283622 | 0.994927156725 | -0.000044430872 | 유지 |
| exp022_orig_knn | 0.967331098842 | exp011_resid_pair | 0.997394283622 | 0.994927156725 | -0.000044009052 | 유지 |
| exp023_orig_proxy_residual | 0.967373954625 | exp022_orig_knn | 0.996264613586 | 0.993491611494 | -0.000038664473 | 유지 |
| exp026_constrained_impute_nowidth | 0.967547642359 | exp032_recon_orig_mean_top3 | 0.997904795064 | 0.996384321945 | -0.000041376767 | 유지 |
| exp032_recon_orig_mean_top3 | 0.967647221038 | exp033_recon_orig_mean_top3_raw | 0.997946130512 | 0.997585682744 | -0.000029928560 | 유지 |
| exp033_recon_orig_mean_top3_raw | 0.967620218499 | exp032_recon_orig_mean_top3 | 0.997946130512 | 0.997585682744 | -0.000032093904 | 유지 |
| exp035_lattice_te | 0.967289719640 | exp026_constrained_impute_nowidth | 0.994121633305 | 0.987930021405 | -0.000012255864 | 유지 |
| exp045_xgb_depth8 | 0.967936049520 | exp074_lgb_kitopl_d2_bundle | 0.997376668651 | 0.994948173361 | -0.000008482929 | 유지 |
| exp058_logreg_onehot | 0.959658396372 | exp070_cat_exact_cats | 0.970424166807 | 0.925194891283 | -0.000002848186 | 유지 |
| exp059_lookup_transformer | 0.968922178533 | exp081_lookup_fold_initialization_avg3 | 0.991421686832 | 0.993383341501 | 0.000142295082 | 유지 |
| exp065_tabm | 0.968326118177 | exp067_tabpfn3 | 0.995049391943 | 0.987721774047 | 0.000040621691 | 유지 |
| exp070_cat_exact_cats | 0.968579346887 | exp074_lgb_kitopl_d2_bundle | 0.994448538521 | 0.992618197628 | 0.000048526828 | 유지 |
| exp067_tabpfn3 | 0.967243226668 | exp065_tabm | 0.995049391943 | 0.987721774047 | 0.000012019995 | 유지 |
| exp074_lgb_kitopl_d2_bundle | 0.968398563697 | exp045_xgb_depth8 | 0.997376668651 | 0.994948173361 | 0.000010446180 | 유지 |
| exp081_lookup_fold_initialization_avg3 | 0.969195761811 | exp059_lookup_transformer | 0.991421686832 | 0.993383341501 | 0.000154043292 | 유지 |

## 주요 신호 구간

구간은 원시 12개 입력 열의 행별 결측 개수로 고정했다.

| 후보 | 결측 0 | 결측 1-2 | 결측 3-5 | 결측 6+ |
| --- | ---: | ---: | ---: | ---: |
| exp006_te_drop_gaming | 0.97440398 | 0.96754041 | 0.94433577 | 0.89338750 |
| exp011_resid_pair | 0.97520321 | 0.96839084 | 0.94485759 | 0.89422938 |
| exp022_orig_knn | 0.97520779 | 0.96829514 | 0.94467163 | 0.89268989 |
| exp023_orig_proxy_residual | 0.97516736 | 0.96835635 | 0.94492516 | 0.89391942 |
| exp026_constrained_impute_nowidth | 0.97516025 | 0.96865917 | 0.94522375 | 0.89376859 |
| exp032_recon_orig_mean_top3 | 0.97530471 | 0.96874769 | 0.94514546 | 0.89438160 |
| exp033_recon_orig_mean_top3_raw | 0.97528202 | 0.96872004 | 0.94514706 | 0.89403262 |
| exp035_lattice_te | 0.97510206 | 0.96835540 | 0.94447211 | 0.89284176 |
| exp045_xgb_depth8 | 0.97553163 | 0.96905308 | 0.94553096 | 0.89507554 |
| exp058_logreg_onehot | 0.96997799 | 0.96069055 | 0.93043234 | 0.86141438 |
| exp059_lookup_transformer | 0.97651387 | 0.97012161 | 0.94615749 | 0.89313528 |
| exp065_tabm | 0.97573665 | 0.96954762 | 0.94618701 | 0.89412606 |
| exp070_cat_exact_cats | 0.97603705 | 0.96974465 | 0.94645033 | 0.89476252 |
| exp067_tabpfn3 | 0.97461068 | 0.96850555 | 0.94527462 | 0.89484965 |
| exp074_lgb_kitopl_d2_bundle | 0.97599234 | 0.96947168 | 0.94610565 | 0.89473741 |
| exp081_lookup_fold_initialization_avg3 | 0.97672649 | 0.97044694 | 0.94639048 | 0.89313388 |

## 기여 영점 대조

난수 대조는 고정 seed `630063`로 독립 순위 열 64개를 각각 하나씩 추가했다.
난수 변화는 최소 `-0.000837324462`, 중앙값 `-0.000788448259`, 95백분위 `-0.000750378758`, 최대 `-0.000710299332`다.
단독 OOF 최고 후보 `exp081_lookup_fold_initialization_avg3`의 정확 복제 변화는 `+0.000123276030`다.

| 복제 후보 | 기여 변화 |
| --- | ---: |
| exp006_te_drop_gaming | -0.000059661375 |
| exp011_resid_pair | -0.000043424887 |
| exp022_orig_knn | -0.000043597092 |
| exp023_orig_proxy_residual | -0.000039186056 |
| exp026_constrained_impute_nowidth | -0.000040156135 |
| exp032_recon_orig_mean_top3 | -0.000030508844 |
| exp033_recon_orig_mean_top3_raw | -0.000032433549 |
| exp035_lattice_te | -0.000019202817 |
| exp045_xgb_depth8 | -0.000011944640 |
| exp058_logreg_onehot | -0.000061778715 |
| exp059_lookup_transformer | +0.000112112995 |
| exp065_tabm | +0.000029057082 |
| exp070_cat_exact_cats | +0.000036951721 |
| exp067_tabpfn3 | -0.000000714790 |
| exp074_lgb_kitopl_d2_bundle | +0.000006099406 |
| exp081_lookup_fold_initialization_avg3 | +0.000123276030 |

두 대조를 합친 영점 대역은 `-0.000837324462`에서 `+0.000123276030`다.

## 판정 경계

OOF와 시험 예측 양쪽 배열 해시가 같은 후보는 정확 중복으로 제거한다.
정확 중복 제거 뒤 OOF 스피어만 순위 상관이 0.998 이상인 후보끼리는 단독 OOF가 높은 후보만 유지한다.
균등 순위 평균의 제외 기여와 영점 대역은 후보 제거 기준이 아니다.
무결성과 중복 검사를 통과한 후보는 모두 nested OOF 평가에 넘긴다.
