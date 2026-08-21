# OOF 후보 풀 품질·다양성 감사

## 결론

현재 장부 35개 중 35개가 계보, 정렬, fold, float64와 유한성 검사를 통과했다.
시드별 OOF 평균은 27개에서 재계산해 확인했고, #98 이전 실행 8개는 시드별 파일이 없어 기존 기록 부분 확인으로 남는다.
정확·순위 중복 제거 뒤 35개가 남았고, 모두 nested OOF 평가 후보로 유지한다.
전체 후보의 균등 순위 평균 OOF AUC는 `0.969167893722`다.
난수 64개와 구성원 복제 대조의 영점 대역은 `-0.000187230055`에서 `+0.000056105904`다.
균등 순위 평균의 제외 기여와 영점 대역은 참고값이며, 후보 진입이나 제거에 쓰지 않는다.

무결성 실패 후보가 없다.

## 무결성 및 배열 해시

| 후보 | run | 판정 | 시드 평균 | OOF 결측 | 시험 결측 | OOF SHA-256 | 시험 SHA-256 |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| exp006_te_drop_gaming | `4aaddd505ade428588cf83659f205cf6` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `6582c852dee84acc1cae9b5231c3e4b223f2849338286cb195bcc6a8bda100f6` | `407bf10dc247cbe51e9d4a638e4c415186e2467d653715a8e54259fc3a73b6a1` |
| exp011_resid_pair | `e21d19af37bb461db5dc1a59b66411a6` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `15988c7117300ed4be52bee52e0fe065f63f2799448969bef4cc07cab65bd6df` | `63f0e1b7612ed426698fdb4cf7a88ec1bc6320d7360063a31ebbbc5d1175e9bf` |
| exp022_orig_knn | `52e9c12bcfb94ee9912e4bb7f43e3f09` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `f66f0895d1e8f9fd86420c02c669b319144d5f9d8bb306a48416fb2968cb71de` | `d802a2ccb39c54b37c931d2389e3d925ed016a0de119031b19679b293b0fa4ca` |
| exp023_orig_proxy_residual | `202b7d47c3bd4b8e8904fa5c005c7423` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `585fd50e1eb834a0854434d803022f10c088324cd47d752096312aeda23f283b` | `ac21067ffd3e782c70df32d52d0c9f00a030d4d35792f0d2d43615b4faa5f8c4` |
| exp025_constrained_impute | `9804689a5e3642e586abb972b522f3d4` | 통과 | 완전 확인 | 0 | 0 | `28ea5e1056ab90765220431169ea5bd67cfad682b2e4f41274cfd24e38b4329f` | `fd9b43be0aaba86fe90eb1020f7936d558cbe37f3c843207d867d24971cdb517` |
| exp032_recon_orig_mean_top3 | `b1bd4b08386946c9a4e05b755dbd5f26` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `542114610c478fed58b4bb1bb35fee50ad704a01b78cc2f763eab0096f31f064` | `347326a0698770e3d153b12ee475c42d0508c2c4588f2af1020c5e56331d722d` |
| exp033_recon_orig_mean_top3_raw | `c34f1da180544c3ba022140d394cb055` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `9bb907c98aec08e3a6c401a866b67c563b5290b9656d8b3f6b2ebd878d067e81` | `f5bfa6174b00e80875f7ca08f1e04dba8e65997dd71de24b5179dfc06ef8ec56` |
| exp035_lattice_te | `c62a9ad334bf486cb46d738474ccd767` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `490f5939bed0d2ebb4192da172dc5b8ecfd3960b2fee2dc2b51ac9dd2f3c443b` | `7d7a4c7ab6c7815e6ff7a12f071d2d75e5a6448758129001eb9ee913cf7abffe` |
| exp058_logreg_onehot | `e2b76edd9d204290810e39a7457c4c48` | 통과 | 기존 기록 부분 확인 | 0 | 0 | `f3d479df12e2fa16a33b8d188dafa8faa3159458606f3dbdefe5a41365082769` | `4dd874272dfd305ea41ba6a8ddb017fe5e1ca0987d9f19271aa7e6fa7a192398` |
| exp059_lookup_transformer | `b951fac51b6b44298f7fdb0b543caba6` | 통과 | 완전 확인 | 0 | 0 | `9407248f85db87e1825f140a8c3927d8d296c8daddf80215d929148da6ccf3d1` | `c23651771024fa66f9065dc14bf93a694517f280df6e360f19fc2f030e0c4f25` |
| exp070_cat_exact_cats | `6238d8c58beb42ee8102209ded92516e` | 통과 | 완전 확인 | 0 | 0 | `64039d4b52ce4dbf5ff690de19d92346d0e6c6a695337c1f34ac82cba65a1c30` | `e3b2cb7a9db28990e0adb82e95775b0761971cf18d6a7a954f3b0b6f1d59ac77` |
| exp067_tabpfn3 | `85b09132aae543aab16e69dea5906a30` | 통과 | 완전 확인 | 0 | 0 | `59a742f0a4ad336826d87491a01f92b1f23bb2ba23ba6c0380d3eede34c070f7` | `cea3b01adf1cd3c0ad2c02b57cb3adcb5afe540d14e129763cb65ae7ca7040f7` |
| exp081_lookup_fold_initialization_avg3 | `d55d1cd49c194eb8bf7b5128e548df81` | 통과 | 완전 확인 | 0 | 0 | `92d2188043561b57cc937800e54026d7cea97a8ef8d6ee6f1947d7e6007e18a0` | `6ea966a3116e4fff05b5726656ad852ef2298b1d5650dec7a139e9318de770ad` |
| exp110_lgb_kitopl_no_te | `ae829ae3d886402ab100a8e966c3131a` | 통과 | 완전 확인 | 0 | 0 | `740fe8ef20d6864bc38fb975521c606249490d43b617d9197fb1277060a459f0` | `ffe237cde45ad5f9ae3d58e04cba7f4e0c39a0a0d0f071122036ec9d11c0310c` |
| exp111_xgb_depth8_no_te | `3cbc2ccce9eb46458d884006a33e516e` | 통과 | 완전 확인 | 0 | 0 | `f1f3d1fb61785b5ec78647c73eabdbb0128057a5b1fb9b0eead26d377cb31a05` | `9c49e51d029866802e0c58e29024b480ee1fe9c732400118e4f040f0fca12c8b` |
| exp071_cat_exact_no_te | `521b4924b2ef4ba1830724333b83deb2` | 통과 | 완전 확인 | 0 | 0 | `e6611866d54b6cf384690e539ec4ced26ee266c62d02fc4206137ae63c572029` | `29701c540e24cd2f33cd9ff4dfadb69411c1eb1758face814a5b999630dd4a01` |
| exp106_lookup_fixed24_train_test_preprocessing | `547e7bc90b9f4c448307d406f053629a` | 통과 | 완전 확인 | 0 | 0 | `a149a194f0ed082887cdbffa2b7791a61ab5e4b90c86f9769f2ee4ae7713942e` | `eca0d5764ed538aa6d943f55bdaa27c2f62ca12b679f3dc1b67907c157fe8c03` |
| exp107_logreg_onehot_nn10 | `c4c4c780acab4b83baf10d67b98cc85a` | 통과 | 완전 확인 | 0 | 0 | `025a572e1183fb022cdb00345e4b0776e6e3eba5c76d574c133b1989b794e949` | `96ea2c1e07bf0668889c2ca5cc3c86f11819aca0756a68678dd809a668b916af` |
| exp108_logreg_onehot_nn10_l1 | `83f7977a87ef46679c26824c5b85f928` | 통과 | 완전 확인 | 0 | 0 | `349098fb8300f07af39f41eb2cc487f376569a190f8032724121af3bad8fdb69` | `b47ed2d2946108a67a5243729862650354e5be76556e681c460cba05e4a65f6e` |
| exp117_ag25_gbm_r21 | `d107ea874ebe4dbe8094694141a162b6` | 통과 | 완전 확인 | 0 | 0 | `1d8617af4a8d2cb21e898f1e33e805e6f1d351e96be64bbe70008fe94749494b` | `0c84bc4ae72a46d74ef4287e41f72207856c96dc2019c08d9828aba53653c5f1` |
| exp113_tab_cnn_m0 | `b47a4184249746cdb1d062aa98364437` | 통과 | 완전 확인 | 0 | 0 | `97721e2bf2254933057ee0512d76cdbd8b091a60684d22e3396eeacae35b277c` | `63b42e1a968e976bfd1ce440fa7b45c5fcae01714f3c1ca00d85c1caaddd6794` |
| exp085_contextual_spline_m0 | `e84d97e875754af1b073b399db2a0616` | 통과 | 완전 확인 | 0 | 0 | `04b2a8f637a978ed3d428f28d1c2e7bec51f856cdf844aee434d63dc239f6671` | `be72f7a7879548c075361e5caf3d9fdc6ae47073df386222fd771ce02f76f392` |
| exp124_realmlp_dtype_fix | `c41c6a4deae04e1fbd8a75193eaaa32c` | 통과 | 완전 확인 | 0 | 0 | `c468867d7e18f9db383922296331d1034769742479526d3d793ffbd78ee13315` | `942acb79afaa1a8bdd65ab52958911950a2ae6e2e8944508fd93135e516b152a` |
| exp127_lookup_muon | `7124425b5b51421dbbeba597229554da` | 통과 | 완전 확인 | 0 | 0 | `7634d94dcdf5d8c987b6b3f7e8bd4403e621ef97296958422f119a51366ccf5b` | `be2d33bedc9f022c0c55fadccf658c393e2223e8438de9e7b8e0b4fe00058c58` |
| exp027_recon_ce | `e4e81bdce7df4a72bf250ef842fd8781` | 통과 | 완전 확인 | 0 | 0 | `82d4ece1e6d7d6dd20df27a0d48bac1ff99303a17e7900f9d7e6daa86820e1e2` | `800ef0598b6796a6ec844407456f0e8255e4c96b7d4901421bd2d58db8dc18f9` |
| exp048_lgb_orig_cdf_diff | `2b5fda6d3ea94ff886c5fe616376f0f0` | 통과 | 완전 확인 | 0 | 0 | `2bc251ba0f9005dcfddf35aad26f0b380e9d3538b43fe25b875d722a894fc5d5` | `fff2c201e3b736f5c9d3f04cae8df9fb52193916e3088a190146f2c754eb0610` |
| exp134_realmlp_muon | `f72eeee10b1b4947af79228f888e7a81` | 통과 | 완전 확인 | 0 | 0 | `29f30a098fb818dc859ea2adad44e1ec9f8e797cefcfd4d05d85e4d937d0cb80` | `b54d9be9d087353eb68c601a08070b225ecfa2208927815416e5ed660a9ece9f` |
| exp135_xgb_hpo_trial30 | `119b9c4e44674d38a6f758b65a68593e` | 통과 | 완전 확인 | 0 | 0 | `b4a601c3a41137a0319be1dd8ff3a4a9ccc2aeaf68a1037ae171f78637085299` | `8052c82fbac39a96a636fb6ac53aa709aca0eb5ee018c0c5d481f3039110b0a6` |
| exp131_lookup_bivariate_plr5 | `54acd002d3c04ec3a74c09871e83da50` | 통과 | 완전 확인 | 0 | 0 | `3735b1ba771c32cc3feea9bb325e8da5d04e8474589d52995b2126e9f7a0e8ad` | `e46bff92c8046c495266a5b7339ddfa69af0c64c23817944667fb552fbf57fc6` |
| exp136_realmlp_muon_recon_widths | `bd2bd1154e6343f68b364714e2fedcc4` | 통과 | 완전 확인 | 0 | 0 | `61490376cc8f8a7d49e57c4ee614950a140746838b118e8919213b3eb9f792f9` | `21f4f333b205bdd0cf64f9d8e8c5b090022ff6060e45d2f01c66d6233c473845` |
| exp137_tabm_recon_widths | `e7682b186be14dea822d41d7d411bb98` | 통과 | 완전 확인 | 0 | 0 | `895eace593d884dc911aa1d2c6090aba8d2791ce068e28f4f136d6fe4898946c` | `1680157e667eb64edcbb6604a88210c6d07f7158e84271cd08740dc508e41b77` |
| exp133_scalar_token_transformer_oof_te | `f16dbc88860a41f39194da2f581f057d` | 통과 | 완전 확인 | 0 | 0 | `9528f1971b3d8d453f567358b94f18836b9bccd3b9bbfcd54a1615fbea9e5d86` | `63410f9ed2fe6f53ad7f695ede661218d7837332f1902a39050ccbc79a9d9e01` |
| exp131_tab_cnn_oof_target_mean | `ea78fe0dbe2442788580dec56cf7f3ad` | 통과 | 완전 확인 | 0 | 0 | `fc02cb262dd32ec69e3ae776b45997a452ea72a09186e2214c9e637c34ef0505` | `e0cc675ffa36e1e1eb55ade768ca241c8d203fbfab2e0a5af3989ebd4c7a775d` |
| exp132_tab_cnn_epochs100 | `09d17ca4091e45118d201ba7a321f956` | 통과 | 완전 확인 | 0 | 0 | `ed6ef11b04834abf1b3da83964e1c3e8cf98bb741ec8b211cc4b8f3d4a92c3c8` | `19848eaa7221e0fa2566c8d9b2ff1f359777d7c791f49042b658b036541eb8f5` |
| exp139_realmlp_reference_qnormal_train_test | `1af9442e67c14a96a5256251f23a1b9d` | 통과 | 완전 확인 | 0 | 0 | `93b787afe2f8212865a6975e3bc64a1d68c7efdb14c15c8b0178f2f0f39aeea6` | `c68c546f8223fbbf3abc647e67f848e406ffdf333d452e36f183b45e03becd3b` |

기존 기록 부분 확인 사유:

- `exp006_te_drop_gaming`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.
- `exp011_resid_pair`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.
- `exp022_orig_knn`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.
- `exp023_orig_proxy_residual`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.
- `exp032_recon_orig_mean_top3`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.
- `exp033_recon_orig_mean_top3_raw`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.
- `exp035_lattice_te`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.
- `exp058_logreg_onehot`: #98 이전 실행이라 시드별 OOF 산출물이 없어 평균을 독립 재계산할 수 없음.

## 중복 판정

OOF와 시험 예측이 모두 같은 정확 중복 및 스피어만 0.998 이상 중복이 없다.

## 후보 유지 정책

무결성과 중복 검사를 통과한 후보는 균등 순위 평균 기여의 부호와 관계없이 모두 유지한다.
구성원 선택과 가중치는 outer fold 안에서 학습하는 nested OOF 평가가 결정한다.

고정점 후보: `exp006_te_drop_gaming`, `exp011_resid_pair`, `exp022_orig_knn`, `exp023_orig_proxy_residual`, `exp025_constrained_impute`, `exp032_recon_orig_mean_top3`, `exp033_recon_orig_mean_top3_raw`, `exp035_lattice_te`, `exp058_logreg_onehot`, `exp059_lookup_transformer`, `exp070_cat_exact_cats`, `exp067_tabpfn3`, `exp081_lookup_fold_initialization_avg3`, `exp110_lgb_kitopl_no_te`, `exp111_xgb_depth8_no_te`, `exp071_cat_exact_no_te`, `exp106_lookup_fixed24_train_test_preprocessing`, `exp107_logreg_onehot_nn10`, `exp108_logreg_onehot_nn10_l1`, `exp117_ag25_gbm_r21`, `exp113_tab_cnn_m0`, `exp085_contextual_spline_m0`, `exp124_realmlp_dtype_fix`, `exp127_lookup_muon`, `exp027_recon_ce`, `exp048_lgb_orig_cdf_diff`, `exp134_realmlp_muon`, `exp135_xgb_hpo_trial30`, `exp131_lookup_bivariate_plr5`, `exp136_realmlp_muon_recon_widths`, `exp137_tabm_recon_widths`, `exp133_scalar_token_transformer_oof_te`, `exp131_tab_cnn_oof_target_mean`, `exp132_tab_cnn_epochs100`, `exp139_realmlp_reference_qnormal_train_test`.

## 품질과 다양성

제외 기여는 전체 후보의 균등 순위 평균에서 각 후보 하나를 제외한 참고값이다.
잔차 상관은 최근접 순위 상관 후보와의 피어슨 상관이다.

| 후보 | 단독 OOF | 최근접 후보 | 스피어만 | 잔차 상관 | 제외 기여 | 판정 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| exp006_te_drop_gaming | 0.966592431674 | exp022_orig_knn | 0.996466394406 | 0.989395144148 | -0.000028069000 | 유지 |
| exp011_resid_pair | 0.967401071703 | exp022_orig_knn | 0.997394283622 | 0.994927156725 | -0.000020664564 | 유지 |
| exp022_orig_knn | 0.967331098842 | exp011_resid_pair | 0.997394283622 | 0.994927156725 | -0.000021193276 | 유지 |
| exp023_orig_proxy_residual | 0.967373954625 | exp022_orig_knn | 0.996264613586 | 0.993491611494 | -0.000019081483 | 유지 |
| exp025_constrained_impute | 0.967574689811 | exp048_lgb_orig_cdf_diff | 0.997669033947 | 0.997012493679 | -0.000019874870 | 유지 |
| exp032_recon_orig_mean_top3 | 0.967647221038 | exp033_recon_orig_mean_top3_raw | 0.997946130512 | 0.997585682744 | -0.000016921442 | 유지 |
| exp033_recon_orig_mean_top3_raw | 0.967620218499 | exp032_recon_orig_mean_top3 | 0.997946130512 | 0.997585682744 | -0.000017740090 | 유지 |
| exp035_lattice_te | 0.967289719640 | exp027_recon_ce | 0.993714565205 | 0.987496930766 | -0.000011300536 | 유지 |
| exp058_logreg_onehot | 0.959658396372 | exp107_logreg_onehot_nn10 | 0.996470705220 | 0.995746533313 | -0.000029035537 | 유지 |
| exp059_lookup_transformer | 0.968922178533 | exp081_lookup_fold_initialization_avg3 | 0.991421686832 | 0.993383341501 | 0.000032871683 | 유지 |
| exp070_cat_exact_cats | 0.968579346887 | exp071_cat_exact_no_te | 0.995426104991 | 0.994778976067 | 0.000005768119 | 유지 |
| exp067_tabpfn3 | 0.967243226668 | exp117_ag25_gbm_r21 | 0.994683170717 | 0.986234185258 | -0.000005237641 | 유지 |
| exp081_lookup_fold_initialization_avg3 | 0.969195761811 | exp131_lookup_bivariate_plr5 | 0.994813331402 | 0.997599826141 | 0.000036246100 | 유지 |
| exp110_lgb_kitopl_no_te | 0.967332310630 | exp111_xgb_depth8_no_te | 0.994981325422 | 0.985886361752 | 0.000006511260 | 유지 |
| exp111_xgb_depth8_no_te | 0.964826294866 | exp110_lgb_kitopl_no_te | 0.994981325422 | 0.985886361752 | 0.000000528159 | 유지 |
| exp071_cat_exact_no_te | 0.968159940645 | exp070_cat_exact_cats | 0.995426104991 | 0.994778976067 | 0.000006071568 | 유지 |
| exp106_lookup_fixed24_train_test_preprocessing | 0.967869802093 | exp059_lookup_transformer | 0.982890024491 | 0.980720396112 | 0.000065434492 | 유지 |
| exp107_logreg_onehot_nn10 | 0.959990699262 | exp058_logreg_onehot | 0.996470705220 | 0.995746533313 | -0.000030254408 | 유지 |
| exp108_logreg_onehot_nn10_l1 | 0.960229396949 | exp058_logreg_onehot | 0.992732550247 | 0.995704171076 | -0.000030257189 | 유지 |
| exp117_ag25_gbm_r21 | 0.968715815637 | exp135_xgb_hpo_trial30 | 0.997998904392 | 0.995171743900 | 0.000003199268 | 유지 |
| exp113_tab_cnn_m0 | 0.962074833927 | exp132_tab_cnn_epochs100 | 0.976731872101 | 0.987449757009 | -0.000002070823 | 유지 |
| exp085_contextual_spline_m0 | 0.968142275138 | exp081_lookup_fold_initialization_avg3 | 0.977042023847 | 0.982907743957 | 0.000030533338 | 유지 |
| exp124_realmlp_dtype_fix | 0.968322345786 | exp136_realmlp_muon_recon_widths | 0.987068855663 | 0.997453611772 | 0.000013377895 | 유지 |
| exp127_lookup_muon | 0.969284045012 | exp131_lookup_bivariate_plr5 | 0.997917259329 | 0.998688793573 | 0.000043677283 | 유지 |
| exp027_recon_ce | 0.967717165004 | exp025_constrained_impute | 0.997578586697 | 0.996363082815 | -0.000016356757 | 유지 |
| exp048_lgb_orig_cdf_diff | 0.967609718237 | exp032_recon_orig_mean_top3 | 0.997702406876 | 0.995958122099 | -0.000019149901 | 유지 |
| exp134_realmlp_muon | 0.968426063131 | exp136_realmlp_muon_recon_widths | 0.997210654816 | 0.999068775039 | 0.000017341922 | 유지 |
| exp135_xgb_hpo_trial30 | 0.968330716027 | exp117_ag25_gbm_r21 | 0.997998904392 | 0.995171743900 | -0.000004001948 | 유지 |
| exp131_lookup_bivariate_plr5 | 0.969339732964 | exp127_lookup_muon | 0.997917259329 | 0.998688793573 | 0.000045975753 | 유지 |
| exp136_realmlp_muon_recon_widths | 0.968476504445 | exp139_realmlp_reference_qnormal_train_test | 0.997363479614 | 0.999441721178 | 0.000019066914 | 유지 |
| exp137_tabm_recon_widths | 0.968384668812 | exp117_ag25_gbm_r21 | 0.995959005683 | 0.993882826217 | 0.000005448335 | 유지 |
| exp133_scalar_token_transformer_oof_te | 0.967815517727 | exp067_tabpfn3 | 0.970180178538 | 0.986078639246 | 0.000001254971 | 유지 |
| exp131_tab_cnn_oof_target_mean | 0.967951848534 | exp136_realmlp_muon_recon_widths | 0.968875111726 | 0.933514322875 | 0.000003798474 | 유지 |
| exp132_tab_cnn_epochs100 | 0.962787914650 | exp113_tab_cnn_m0 | 0.976731872101 | 0.987449757009 | 0.000001267337 | 유지 |
| exp139_realmlp_reference_qnormal_train_test | 0.968545629088 | exp136_realmlp_muon_recon_widths | 0.997363479614 | 0.999441721178 | 0.000020403410 | 유지 |

## 주요 신호 구간

구간은 원시 12개 입력 열의 행별 결측 개수로 고정했다.

| 후보 | 결측 0 | 결측 1-2 | 결측 3-5 | 결측 6+ |
| --- | ---: | ---: | ---: | ---: |
| exp006_te_drop_gaming | 0.97440398 | 0.96754041 | 0.94433577 | 0.89338750 |
| exp011_resid_pair | 0.97520321 | 0.96839084 | 0.94485759 | 0.89422938 |
| exp022_orig_knn | 0.97520779 | 0.96829514 | 0.94467163 | 0.89268989 |
| exp023_orig_proxy_residual | 0.97516736 | 0.96835635 | 0.94492516 | 0.89391942 |
| exp025_constrained_impute | 0.97515333 | 0.96870965 | 0.94527309 | 0.89428331 |
| exp032_recon_orig_mean_top3 | 0.97530471 | 0.96874769 | 0.94514546 | 0.89438160 |
| exp033_recon_orig_mean_top3_raw | 0.97528202 | 0.96872004 | 0.94514706 | 0.89403262 |
| exp035_lattice_te | 0.97510206 | 0.96835540 | 0.94447211 | 0.89284176 |
| exp058_logreg_onehot | 0.96997799 | 0.96069055 | 0.93043234 | 0.86141438 |
| exp059_lookup_transformer | 0.97651387 | 0.97012161 | 0.94615749 | 0.89313528 |
| exp070_cat_exact_cats | 0.97603705 | 0.96974465 | 0.94645033 | 0.89476252 |
| exp067_tabpfn3 | 0.97461068 | 0.96850555 | 0.94527462 | 0.89484965 |
| exp081_lookup_fold_initialization_avg3 | 0.97672649 | 0.97044694 | 0.94639048 | 0.89313388 |
| exp110_lgb_kitopl_no_te | 0.97524084 | 0.96850363 | 0.94383513 | 0.89031465 |
| exp111_xgb_depth8_no_te | 0.97298649 | 0.96605942 | 0.94075801 | 0.88606992 |
| exp071_cat_exact_no_te | 0.97570364 | 0.96928932 | 0.94592506 | 0.89404653 |
| exp106_lookup_fixed24_train_test_preprocessing | 0.97577528 | 0.96911712 | 0.94401716 | 0.89020809 |
| exp107_logreg_onehot_nn10 | 0.96993089 | 0.96092976 | 0.93210100 | 0.86230368 |
| exp108_logreg_onehot_nn10_l1 | 0.97011761 | 0.96118971 | 0.93241596 | 0.86309455 |
| exp117_ag25_gbm_r21 | 0.97615746 | 0.96987292 | 0.94663581 | 0.89483837 |
| exp113_tab_cnn_m0 | 0.97032809 | 0.96341578 | 0.93782030 | 0.88200322 |
| exp085_contextual_spline_m0 | 0.97592192 | 0.96935638 | 0.94479012 | 0.89130954 |
| exp124_realmlp_dtype_fix | 0.97576702 | 0.96945221 | 0.94632151 | 0.89551603 |
| exp127_lookup_muon | 0.97678041 | 0.97054789 | 0.94652473 | 0.89369589 |
| exp027_recon_ce | 0.97536301 | 0.96881435 | 0.94528004 | 0.89419237 |
| exp048_lgb_orig_cdf_diff | 0.97525003 | 0.96871066 | 0.94518497 | 0.89433755 |
| exp134_realmlp_muon | 0.97584885 | 0.96957890 | 0.94640165 | 0.89542715 |
| exp135_xgb_hpo_trial30 | 0.97592026 | 0.96944038 | 0.94594218 | 0.89427977 |
| exp131_lookup_bivariate_plr5 | 0.97686863 | 0.97057425 | 0.94654843 | 0.89365923 |
| exp136_realmlp_muon_recon_widths | 0.97578165 | 0.96969398 | 0.94664419 | 0.89539723 |
| exp137_tabm_recon_widths | 0.97580317 | 0.96960302 | 0.94621585 | 0.89419710 |
| exp133_scalar_token_transformer_oof_te | 0.97527840 | 0.96902510 | 0.94560128 | 0.89413629 |
| exp131_tab_cnn_oof_target_mean | 0.97535157 | 0.96915234 | 0.94597184 | 0.89415834 |
| exp132_tab_cnn_epochs100 | 0.97101819 | 0.96410374 | 0.93854644 | 0.88450331 |
| exp139_realmlp_reference_qnormal_train_test | 0.97586036 | 0.96975034 | 0.94671668 | 0.89563869 |

## 기여 영점 대조

난수 대조는 고정 seed `630063`로 독립 순위 열 64개를 각각 하나씩 추가했다.
난수 변화는 최소 `-0.000187230055`, 중앙값 `-0.000164826245`, 95백분위 `-0.000148092394`, 최대 `-0.000128063627`다.
단독 OOF 최고 후보 `exp131_lookup_bivariate_plr5`의 정확 복제 변화는 `+0.000041137417`다.

| 복제 후보 | 기여 변화 |
| --- | ---: |
| exp006_te_drop_gaming | -0.000029013224 |
| exp011_resid_pair | -0.000021266525 |
| exp022_orig_knn | -0.000021788930 |
| exp023_orig_proxy_residual | -0.000019894219 |
| exp025_constrained_impute | -0.000020373197 |
| exp032_recon_orig_mean_top3 | -0.000017556818 |
| exp033_recon_orig_mean_top3_raw | -0.000018340539 |
| exp035_lattice_te | -0.000012981813 |
| exp058_logreg_onehot | -0.000039896523 |
| exp059_lookup_transformer | +0.000028797669 |
| exp070_cat_exact_cats | +0.000004089425 |
| exp067_tabpfn3 | -0.000007711691 |
| exp081_lookup_fold_initialization_avg3 | +0.000032241904 |
| exp110_lgb_kitopl_no_te | +0.000002851687 |
| exp111_xgb_depth8_no_te | -0.000006171955 |
| exp071_cat_exact_no_te | +0.000003783246 |
| exp106_lookup_fixed24_train_test_preprocessing | +0.000056105904 |
| exp107_logreg_onehot_nn10 | -0.000040542926 |
| exp108_logreg_onehot_nn10_l1 | -0.000040201830 |
| exp117_ag25_gbm_r21 | +0.000002048886 |
| exp113_tab_cnn_m0 | -0.000012576398 |
| exp085_contextual_spline_m0 | +0.000025596700 |
| exp124_realmlp_dtype_fix | +0.000010658269 |
| exp127_lookup_muon | +0.000038969141 |
| exp027_recon_ce | -0.000016972942 |
| exp048_lgb_orig_cdf_diff | -0.000019629432 |
| exp134_realmlp_muon | +0.000014305075 |
| exp135_xgb_hpo_trial30 | -0.000004957808 |
| exp131_lookup_bivariate_plr5 | +0.000041137417 |
| exp136_realmlp_muon_recon_widths | +0.000015907490 |
| exp137_tabm_recon_widths | +0.000003503494 |
| exp133_scalar_token_transformer_oof_te | -0.000001086232 |
| exp131_tab_cnn_oof_target_mean | +0.000001369475 |
| exp132_tab_cnn_epochs100 | -0.000008561364 |
| exp139_realmlp_reference_qnormal_train_test | +0.000017223714 |

두 대조를 합친 영점 대역은 `-0.000187230055`에서 `+0.000056105904`다.

## 판정 경계

OOF와 시험 예측 양쪽 배열 해시가 같은 후보는 정확 중복으로 제거한다.
정확 중복 제거 뒤 OOF 스피어만 순위 상관이 0.998 이상인 후보끼리는 단독 OOF가 높은 후보만 유지한다.
균등 순위 평균의 제외 기여와 영점 대역은 후보 제거 기준이 아니다.
무결성과 중복 검사를 통과한 후보는 모두 nested OOF 평가에 넘긴다.

