# 행별 OOF 난점 분석

[동결한 측정 계약으로 행별 OOF 난점 분석을 실행한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/459)의 실행 결과다.
[행별 OOF 난점 분석의 측정 계약을 완성한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/460)의 해답을 바꾸지 않고 구현했다.

## 결론

다섯 분할 확인 관문을 통과한 오류 조건 범위는 238개다.
이 조건들은 관찰된 연관성이며 단일 변화 대조가 전체 OOF를 개선하기 전에는 원인으로 부르지 않는다.

| 조건 | 목표값 범위 | 눈금 | 다섯 분할 최소 AUC 손실 여지 |
| --- | --- | --- | ---: |
| `raw_daily_screen_time_hours__q04_q05` | target_1 | both_scales | 0.00585578 |
| `raw_daily_screen_time_hours__q03_q04` | target_1 | both_scales | 0.00569144 |
| `derived_screen_minus_work__q03_q04` | target_1 | both_scales | 0.00562906 |
| `iqr_own_members__q04_q05` | target_1 | both_scales | 0.00550517 |
| `derived_sgw_sum__q04_q05` | target_1 | both_scales | 0.00517796 |
| `raw_weekend_screen_time__q04_q05` | target_1 | both_scales | 0.00505017 |
| `raw_weekend_screen_time__q03_q04` | target_1 | both_scales | 0.00494317 |
| `raw_social_media_hours__q04_q05` | target_1 | both_scales | 0.00487972 |
| `derived_screen_to_sleep__q03_q04` | target_1 | both_scales | 0.00481350 |
| `iqr_own_members__q03_q04` | target_1 | both_scales | 0.00470491 |
| `iqr_own_members__q04_q05` | balanced | both_scales | 0.00435078 |
| `raw_social_media_hours__q05_q06` | target_0 | both_scales | 0.00417338 |
| `raw_social_media_hours__q03_q04` | target_1 | both_scales | 0.00398638 |
| `raw_daily_screen_time_hours__q04_q05` | balanced | both_scales | 0.00387068 |
| `disagreement_abs_final_champion__q04_q05` | target_1 | both_scales | 0.00383194 |
| `derived_sgw_sum__q03_q04` | target_1 | both_scales | 0.00380432 |
| `derived_screen_minus_work__q04_q05` | target_1 | both_scales | 0.00374094 |
| `raw_social_media_hours__q04_q05` | balanced | both_scales | 0.00370527 |
| `iqr_final_members__q03_q04` | target_1 | both_scales | 0.00358661 |
| `derived_screen_minus_work__q04_q05` | balanced | both_scales | 0.00350351 |
| `raw_weekend_screen_time__q04_q05` | balanced | both_scales | 0.00333347 |
| `derived_wk_other__q04_q05` | target_1 | both_scales | 0.00319261 |
| `derived_screen_minus_work__q04_q05` | target_0 | both_scales | 0.00317299 |
| `iqr_own_members__q04_q05` | target_0 | both_scales | 0.00309163 |
| `iqr_external_members__q03_q04` | target_1 | both_scales | 0.00307136 |
| `derived_sgw_sum__q05_q06` | target_0 | both_scales | 0.00297290 |
| `derived_screen_slack__q03_q04` | target_1 | both_scales | 0.00281635 |
| `derived_screen_to_sleep__q04_q05` | target_1 | both_scales | 0.00279975 |
| `iqr_final_members__q04_q05` | target_1 | both_scales | 0.00273213 |
| `iqr_final_members__q04_q05` | balanced | both_scales | 0.00272745 |
| `iqr_final_members__q04_q05` | target_0 | both_scales | 0.00272277 |
| `iqr_own_members__q05_q06` | target_0 | both_scales | 0.00269021 |
| `raw_daily_screen_time_hours__q05_q06` | target_0 | both_scales | 0.00266494 |
| `iqr_final_members__q05_q06` | target_0 | both_scales | 0.00263301 |
| `raw_social_media_hours__q04_q05` | target_0 | both_scales | 0.00253082 |
| `derived_other_screen__q03_q04` | target_1 | both_scales | 0.00244775 |
| `disagreement_abs_final_champion__q03_q04` | target_1 | both_scales | 0.00238506 |
| `iqr_external_members__q05_q06` | target_0 | both_scales | 0.00238015 |
| `derived_wk_other__q03_q04` | target_1 | both_scales | 0.00237850 |
| `raw_weekend_screen_time__q05_q06` | target_0 | both_scales | 0.00232953 |
| `iqr_external_members__q04_q05` | target_0 | both_scales | 0.00222465 |
| `raw_work_study_hours__q04_q05` | target_1 | both_scales | 0.00220452 |
| `derived_social_share_screen__q05_q06` | target_0 | both_scales | 0.00208546 |
| `iqr_external_members__q04_q05` | balanced | both_scales | 0.00202786 |
| `derived_screen_minus_work__q05_q06` | target_0 | both_scales | 0.00201040 |
| `raw_gaming_hours__q04_q05` | target_1 | both_scales | 0.00199880 |
| `missing__weekend_screen_time` | target_0 | both_scales | 0.00195731 |
| `derived_social_share_screen__q06_q07` | target_0 | both_scales | 0.00190709 |
| `raw_daily_screen_time_hours__q04_q05` | target_0 | both_scales | 0.00188558 |
| `raw_work_study_hours__q03_q04` | target_1 | both_scales | 0.00187465 |
| `disagreement_abs_own_external__q06_q07` | target_0 | both_scales | 0.00183122 |
| `derived_screen_to_sleep__q04_q05` | balanced | both_scales | 0.00180431 |
| `iqr_external_members__q04_q05` | target_1 | both_scales | 0.00179171 |
| `disagreement_abs_final_champion__q05_q06` | target_0 | both_scales | 0.00177675 |
| `derived_gaming_minus_work__q05_q06` | target_1 | both_scales | 0.00173133 |
| `missing__weekend_screen_time` | balanced | both_scales | 0.00171434 |
| `derived_other_screen__q04_q05` | target_1 | both_scales | 0.00169991 |
| `derived_screen_to_sleep__q05_q06` | target_0 | both_scales | 0.00168280 |
| `derived_screen_to_sleep__q02_q03` | target_1 | both_scales | 0.00165168 |
| `derived_wk_other__q05_q06` | target_0 | both_scales | 0.00155165 |
| `derived_slack_frac__q04_q05` | target_1 | both_scales | 0.00155157 |
| `derived_sgw_frac__q05_q06` | target_1 | both_scales | 0.00155074 |
| `raw_daily_screen_time_hours__q02_q03` | target_1 | both_scales | 0.00154959 |
| `derived_screen_minus_work__q02_q03` | target_1 | both_scales | 0.00153979 |
| `raw_weekend_screen_time__q02_q03` | target_1 | both_scales | 0.00148227 |
| `iqr_own_members__q02_q03` | target_1 | both_scales | 0.00146895 |
| `iqr_external_members__q06_q07` | target_0 | both_scales | 0.00145705 |
| `derived_sgw_sum__q02_q03` | target_1 | both_scales | 0.00145228 |
| `raw_daily_screen_time_hours__q01_q02` | target_1 | both_scales | 0.00141386 |
| `raw_weekend_screen_time__q04_q05` | target_0 | both_scales | 0.00139478 |
| `derived_screen_slack__q02_q03` | target_1 | both_scales | 0.00135272 |
| `missing__weekend_screen_time` | target_1 | both_scales | 0.00132083 |
| `iqr_final_members__q06_q07` | target_0 | both_scales | 0.00130010 |
| `derived_wk_minus_sgw__q04_q05` | target_1 | both_scales | 0.00126610 |
| `disagreement_abs_own_external__q05_q06` | target_0 | both_scales | 0.00124420 |
| `raw_weekend_screen_time__q01_q02` | target_1 | both_scales | 0.00123639 |
| `derived_sgw_sum__q01_q02` | target_1 | both_scales | 0.00121585 |
| `derived_screen_to_sleep__q01_q02` | target_1 | both_scales | 0.00120513 |
| `derived_work_share_screen__q04_q05` | target_0 | both_scales | 0.00115685 |
| `missing_count__4plus` | target_0 | both_scales | 0.00112634 |
| `raw_gaming_hours__q03_q04` | target_1 | both_scales | 0.00109370 |
| `missing__gaming_hours` | balanced | both_scales | 0.00108622 |
| `derived_screen_minus_work__q01_q02` | target_1 | both_scales | 0.00107757 |
| `missing__gaming_hours` | target_0 | both_scales | 0.00106747 |
| `derived_sgw_sum__q06_q07` | target_0 | both_scales | 0.00102581 |
| `disagreement_abs_final_champion__q05_q06` | balanced | both_scales | 0.00101955 |
| `missing__gaming_hours` | target_1 | both_scales | 0.00100141 |
| `missing_count__4plus` | balanced | both_scales | 0.00099797 |
| `raw_app_opens_per_day__q04_q05` | target_1 | both_scales | 0.00098971 |
| `derived_screen_slack__q06_q07` | target_0 | both_scales | 0.00095821 |
| `iqr_final_members__q02_q03` | target_1 | both_scales | 0.00094390 |
| `derived_wk_minus_sgw__q03_q04` | target_1 | both_scales | 0.00093150 |
| `raw_social_media_hours__q06_q07` | target_0 | both_scales | 0.00092594 |
| `disagreement_abs_final_champion__q06_q07` | target_0 | both_scales | 0.00089836 |
| `derived_other_screen__q05_q06` | target_0 | both_scales | 0.00089087 |
| `derived_other_screen__q02_q03` | target_1 | both_scales | 0.00089024 |
| `derived_social_share_screen__q04_q05` | target_1 | both_scales | 0.00087748 |
| `derived_slack_frac__q03_q04` | target_1 | both_scales | 0.00086562 |
| `derived_sgw_frac__q06_q07` | target_1 | both_scales | 0.00086494 |
| `missing_count__4plus` | target_1 | both_scales | 0.00084825 |
| `iqr_own_members__q01_q02` | target_1 | both_scales | 0.00082104 |
| `derived_wk_other__q02_q03` | target_1 | both_scales | 0.00080566 |
| `raw_social_media_hours__q02_q03` | target_1 | both_scales | 0.00078229 |
| `derived_work_share_screen__q06_q07` | target_1 | both_scales | 0.00076904 |
| `derived_screen_slack__q01_q02` | target_1 | both_scales | 0.00074804 |
| `derived_gaming_share_screen__q04_q05` | target_0 | both_scales | 0.00073210 |
| `raw_work_study_hours__q01_q02` | target_1 | both_scales | 0.00073129 |
| `derived_wk_other__q06_q07` | target_0 | both_scales | 0.00072919 |
| `missing__daily_screen_time_hours` | target_0 | both_scales | 0.00072839 |
| `disagreement_abs_final_champion__q02_q03` | target_1 | both_scales | 0.00072425 |
| `iqr_external_members__q02_q03` | target_1 | both_scales | 0.00071363 |
| `disagreement_abs_own_external__q07_q08` | target_0 | both_scales | 0.00070042 |
| `missing__notifications_per_day` | balanced | both_scales | 0.00063533 |
| `disagreement_abs_own_external__q08_q09` | target_0 | both_scales | 0.00061286 |
| `disagreement_abs_final_champion__q01_q02` | target_1 | both_scales | 0.00061150 |
| `missing__notifications_per_day` | target_0 | both_scales | 0.00061006 |
| `derived_social_share_screen__q07_q08` | target_0 | both_scales | 0.00060969 |
| `missing_count__3` | target_0 | both_scales | 0.00058849 |
| `derived_screen_to_sleep__q06_q07` | target_0 | both_scales | 0.00058713 |
| `missing_count__3` | balanced | both_scales | 0.00057561 |
| `raw_work_study_hours__q02_q03` | target_1 | both_scales | 0.00055814 |
| `derived_gaming_share_screen__q08_q09` | target_1 | both_scales | 0.00053188 |
| `derived_social_share_screen__q08_q09` | target_0 | both_scales | 0.00052248 |
| `derived_wk_other__q01_q02` | target_1 | both_scales | 0.00051247 |
| `missing__daily_screen_time_hours` | balanced | both_scales | 0.00051127 |
| `derived_work_share_screen__q08_q09` | target_1 | both_scales | 0.00051067 |
| `raw_notifications_per_day__q06_q07` | target_1 | both_scales | 0.00050263 |
| `derived_other_screen__q01_q02` | target_1 | both_scales | 0.00049043 |
| `raw_social_media_hours__q01_q02` | target_1 | both_scales | 0.00048535 |
| `iqr_external_members__q07_q08` | target_0 | both_scales | 0.00047628 |
| `iqr_own_members__q06_q07` | target_0 | both_scales | 0.00046894 |
| `derived_work_share_screen__q07_q08` | target_1 | both_scales | 0.00046086 |
| `missing_count__3` | target_1 | both_scales | 0.00045790 |
| `derived_screen_minus_work__q06_q07` | target_0 | both_scales | 0.00045788 |
| `derived_wk_minus_sgw__q06_q07` | target_0 | both_scales | 0.00044434 |
| `raw_work_study_hours__q06_q07` | target_0 | both_scales | 0.00042739 |
| `iqr_external_members__q08_q09` | target_0 | both_scales | 0.00042222 |
| `missing__notifications_per_day` | target_1 | both_scales | 0.00040707 |
| `derived_wk_minus_sgw__q02_q03` | target_1 | both_scales | 0.00039709 |
| `iqr_final_members__q07_q08` | target_0 | both_scales | 0.00038495 |
| `iqr_final_members__q01_q02` | target_1 | both_scales | 0.00038236 |
| `raw_app_opens_per_day__q08_q09` | target_0 | both_scales | 0.00037524 |
| `derived_work_share_screen__q03_q04` | target_0 | both_scales | 0.00036801 |
| `derived_wk_minus_sgw__q01_q02` | target_1 | both_scales | 0.00034795 |
| `iqr_final_members__q08_q09` | target_0 | both_scales | 0.00032393 |
| `raw_daily_screen_time_hours__q00_q01` | target_1 | both_scales | 0.00032037 |
| `raw_app_opens_per_day__q01_q02` | target_1 | both_scales | 0.00031814 |
| `derived_gaming_share_screen__q07_q08` | target_1 | both_scales | 0.00030970 |
| `derived_screen_slack__q07_q08` | target_0 | both_scales | 0.00030572 |
| `derived_screen_to_sleep__q00_q01` | target_1 | both_scales | 0.00029715 |
| `raw_work_study_hours__q00_q01` | target_1 | both_scales | 0.00029604 |
| `derived_slack_frac__q02_q03` | target_1 | both_scales | 0.00027989 |
| `derived_sgw_frac__q07_q08` | target_1 | both_scales | 0.00027974 |
| `derived_wk_minus_sgw__q07_q08` | target_0 | both_scales | 0.00026215 |
| `raw_weekend_screen_time__q00_q01` | target_1 | both_scales | 0.00025596 |
| `raw_notifications_per_day__q07_q08` | target_0 | both_scales | 0.00023909 |
| `raw_gaming_hours__q02_q03` | target_1 | both_scales | 0.00023446 |
| `raw_gaming_hours__q01_q02` | target_1 | both_scales | 0.00023261 |
| `raw_age__q00_q03` | target_1 | both_scales | 0.00021598 |
| `raw_sleep_hours__q02_q03` | target_1 | both_scales | 0.00018962 |
| `raw_sleep_hours__q01_q02` | target_1 | both_scales | 0.00018319 |
| `iqr_external_members__q01_q02` | target_1 | both_scales | 0.00018248 |
| `raw_work_study_hours__q08_q09` | target_0 | both_scales | 0.00017815 |
| `derived_work_share_screen__q01_q02` | target_0 | both_scales | 0.00017781 |
| `derived_sgw_frac__q01_q02` | target_0 | both_scales | 0.00015696 |
| `derived_slack_frac__q08_q09` | target_0 | both_scales | 0.00015696 |
| `raw_work_study_hours__q07_q08` | target_0 | both_scales | 0.00014994 |
| `raw_notifications_per_day__q08_q09` | target_1 | both_scales | 0.00014493 |
| `derived_sgw_sum__q00_q01` | target_1 | both_scales | 0.00014238 |
| `disagreement_abs_own_external__q09_q10` | balanced | both_scales | 0.00014136 |
| `disagreement_abs_final_champion__q00_q01` | target_1 | both_scales | 0.00013935 |
| `disagreement_abs_own_external__q09_q10` | target_1 | both_scales | 0.00013463 |
| `derived_sgw_sum__q00_q01` | balanced | both_scales | 0.00011822 |
| `derived_wk_other__q00_q01` | target_1 | both_scales | 0.00011535 |
| `raw_gaming_hours__q00_q01` | target_1 | both_scales | 0.00011451 |
| `derived_gaming_share_screen__q01_q02` | target_0 | both_scales | 0.00011081 |
| `derived_work_share_screen__q02_q03` | target_0 | both_scales | 0.00010974 |
| `disagreement_abs_own_external__q09_q10` | target_0 | both_scales | 0.00009814 |
| `iqr_external_members__q09_q10` | target_0 | both_scales | 0.00008407 |
| `raw_notifications_per_day__q01_q02` | target_0 | both_scales | 0.00008033 |
| `derived_sgw_sum__q00_q01` | target_0 | both_scales | 0.00006594 |
| `derived_wk_minus_sgw__q00_q01` | target_1 | both_scales | 0.00006498 |
| `raw_notifications_per_day__q00_q01` | target_1 | both_scales | 0.00006000 |
| `missing_mask__106` | target_0 | both_scales | 0.00005946 |
| `raw_gaming_hours__q07_q08` | target_0 | both_scales | 0.00005407 |
| `iqr_final_members__q09_q10` | target_0 | both_scales | 0.00005317 |
| `derived_gaming_minus_work__q01_q02` | target_0 | both_scales | 0.00005284 |
| `missing_mask__10c` | balanced | both_scales | 0.00005114 |
| `missing__social_media_hours` | target_0 | standard_only | 0.00200864 |
| `missing__social_media_hours` | balanced | standard_only | 0.00196009 |
| `missing__social_media_hours` | target_1 | standard_only | 0.00164123 |
| `missing__daily_screen_time_hours` | target_1 | standard_only | 0.00157230 |
| `missing__app_opens_per_day` | target_0 | standard_only | 0.00061658 |
| `missing__app_opens_per_day` | balanced | standard_only | 0.00059667 |
| `missing__app_opens_per_day` | target_1 | standard_only | 0.00057676 |
| `derived_gaming_minus_work__q06_q07` | target_1 | standard_only | 0.00038854 |
| `raw_gaming_hours__q01_q02` | balanced | standard_only | 0.00014512 |
| `missing_mask__106` | balanced | standard_only | 0.00014360 |
| `raw_notifications_per_day__q02_q03` | target_0 | standard_only | 0.00013468 |
| `missing_mask__006` | balanced | standard_only | 0.00011092 |
| `missing_mask__006` | target_1 | standard_only | 0.00010757 |
| `missing_mask__006` | target_0 | standard_only | 0.00009620 |
| `missing_mask__106` | target_1 | standard_only | 0.00007810 |
| `missing_mask__102` | balanced | standard_only | 0.00007662 |
| `missing_mask__102` | target_0 | standard_only | 0.00006938 |
| `missing_mask__00e` | target_0 | standard_only | 0.00006923 |
| `missing_mask__00e` | balanced | standard_only | 0.00006223 |
| `raw_gaming_hours__q08_q09` | target_0 | standard_only | 0.00006161 |
| `derived_screen_to_sleep__q04_q05` | target_0 | weighted_only | 0.00121471 |
| `missing__work_study_hours` | target_1 | weighted_only | 0.00080621 |
| `missing__academic_work_impact` | target_0 | weighted_only | 0.00076074 |
| `missing__work_study_hours` | balanced | weighted_only | 0.00076022 |
| `missing__academic_work_impact` | balanced | weighted_only | 0.00071826 |
| `missing__work_study_hours` | target_0 | weighted_only | 0.00069431 |
| `missing__academic_work_impact` | target_1 | weighted_only | 0.00067579 |
| `derived_gaming_share_screen__q05_q06` | target_0 | weighted_only | 0.00060759 |
| `raw_app_opens_per_day__q06_q07` | target_0 | weighted_only | 0.00052987 |
| `raw_app_opens_per_day__q03_q04` | target_0 | weighted_only | 0.00048026 |
| `missing__age` | target_0 | weighted_only | 0.00042322 |
| `missing__age` | balanced | weighted_only | 0.00041501 |
| `missing__age` | target_1 | weighted_only | 0.00040680 |
| `missing__sleep_hours` | balanced | weighted_only | 0.00035477 |
| `missing__sleep_hours` | target_0 | weighted_only | 0.00033148 |
| `missing__sleep_hours` | target_1 | weighted_only | 0.00028849 |
| `missing__gender` | target_0 | weighted_only | 0.00018796 |
| `missing__gender` | balanced | weighted_only | 0.00018531 |
| `missing_mask__108` | balanced | weighted_only | 0.00017089 |
| `missing_mask__040` | target_1 | weighted_only | 0.00015383 |
| `missing_mask__040` | balanced | weighted_only | 0.00014755 |
| `missing_mask__040` | target_0 | weighted_only | 0.00014128 |
| `missing_mask__108` | target_1 | weighted_only | 0.00013110 |
| `missing_mask__108` | target_0 | weighted_only | 0.00012578 |
| `derived_sgw_frac__q02_q03` | target_0 | weighted_only | 0.00010778 |
| `derived_slack_frac__q07_q08` | target_0 | weighted_only | 0.00010778 |
| `missing_mask__010` | balanced | weighted_only | 0.00007414 |
| `derived_sgw_frac__q00_q01` | target_0 | weighted_only | 0.00007126 |
| `derived_slack_frac__q09_q10` | target_0 | weighted_only | 0.00007126 |
| `missing_mask__800` | target_1 | weighted_only | 0.00005062 |

## 계약 무결성

최종 242개 `shrunk_rank_logit_logistic` nested OOF AUC는 `0.9702876097776773`로 동결 기준값과 절대 차이 `0.000e+00`다.
입력 해시, 행 수, 식별자 순서와 유일성, 목표값, 분할, 구성원 242개의 순서와 예측 유한성 검사를 모두 통과했다.
자체 35개 OOF는 기존 감사 장부의 little-endian float64 배열 해시와 일치했고, 외부 207개는 동결 장부의 OOF와 시험 예측 결합 해시와 일치했다.
표준 및 가중 순위 손실 부담 항등식의 최대 절대 오차는 각각 `1.527e-16`, `1.442e-13`다.
계약 위반 수는 `0`이다.

## 후보 소거

| 단계 | 진입 범위 | 통과 범위 |
| --- | ---: | ---: |
| 탐색 | 1986 | 437 |
| 정제 | 437 | 432 |
| 확인 | 432 | 426 |

한 범위는 조건, 목표값 범위와 손실 눈금의 조합 하나다.
탐색은 분할 0, 1, 2 각각과 합친 자료를 모두 통과해야 하며 정제는 분할 3, 확인은 분할 4를 사용했다.

## 산출물

| 파일 | 행 수 | SHA-256 |
| --- | ---: | --- |
| `run-logs/issue459/row-metrics.parquet` | 691369 | `bfcb488a8bf205140fc0b032f48d67081dcff6e919648a83bf1f2f4902f859ee` |
| `run-logs/issue459/condition-results.parquet` | 8813 | `e530c06e23fe9dcfd0976adbae98e4ecfa177ca94fdd3763dacae29ad6316dea` |
| `run-logs/issue459/manifest.json` | - | `428456e9d8fecd5d95b92a8afe4172129aa983ad1020bea30e348eeb3ffe10cc` |

분석 실행 자체는 `1737.0`초였고 전체 명령은 `1744.1`초였다.
대용량 행별 결과는 커밋하지 않고 내용 해시로 계보를 남긴다.
