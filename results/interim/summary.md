# Interim backbone summary

> **INTERIM / EXPLORATORY — DO NOT USE FOR INFERENCE**

Backbones with manifest `step=12_record`: **10**

| backbone | family | family_label | gate1_pass | selected_r | selected_lambda_proj | w_raw_norm | embed_dim | domain_probe_alpha0 | id_bal_acc_alpha0 | lid_mean_alpha0 | lid_mean_alpha1 | spectral_slope_alpha0 | spectral_slope_alpha1 | maha_auroc_alpha1 | cosine_auroc_alpha1 | ea03_pass_alpha025 | step12_timestamp | commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| biomedclip | medical_vlm | Medical VLM | True | 128 | 0.1000 | 40.8095 | 768 | 0.9937 | 0.4491 | 19.3412 | 19.6616 | 0.0069 | 0.0071 | nan | nan | True | 2026-08-13T07:30:48.842155+00:00 | f3b9ada90008aa0d3cc2cee0ab286af5e0c022db |
| dinov3 | general_ssl | General SSL | True | 16 | 0.1000 | 8.3685 | 1024 | 0.9991 | 0.5178 | 17.4074 | 16.7522 | 0.0053 | 0.0053 | 0.9832 | 0.7617 | True | 2026-08-13T11:09:26.319245+00:00 | f3b9ada90008aa0d3cc2cee0ab286af5e0c022db |
| efficientnet_b3 | cnn | CNN | True | 16 | 0.1000 | 4.6559 | 1536 | 0.9768 | 0.4061 | 20.7816 | 20.1550 | 0.0035 | 0.0035 | 0.8191 | 0.6966 | True | 2026-08-12T01:39:08.181666+00:00 | 039cc7eaba7cfe62c289981fd63c8bae52959263 |
| mocov3 | general_ssl | General SSL | True | 16 | 0.1000 | 0.5658 | 768 | 0.9974 | 0.4645 | 20.6172 | 14.8285 | 0.0058 | 0.0062 | nan | nan | True | 2026-08-13T08:02:15.980389+00:00 | f3b9ada90008aa0d3cc2cee0ab286af5e0c022db |
| monet | medical_vlm | Medical VLM | True | 16 | 0.1000 | 5.3593 | 1024 | 0.9994 | 0.4899 | 18.2192 | 17.8247 | 0.0049 | 0.0050 | nan | nan | True | 2026-08-13T07:43:01.391134+00:00 | f3b9ada90008aa0d3cc2cee0ab286af5e0c022db |
| openclip | general_vlm | General VLM | True | 16 | 0.1000 | 12.5477 | 768 | 0.9987 | 0.4447 | 19.0757 | 18.8612 | 0.0072 | 0.0073 | 0.9786 | 0.8381 | True | 2026-08-13T08:33:53.411855+00:00 | f3b9ada90008aa0d3cc2cee0ab286af5e0c022db |
| panderm | medical_ssl | Medical SSL | True | 16 | 0.1000 | 9.0150 | 1024 | 0.9990 | 0.5358 | 15.0279 | 14.6819 | 0.0070 | 0.0068 | 0.9984 | 0.9503 | True | 2026-08-11T17:15:08.348931+00:00 | 2573166a41b2ff273d641cef331379f298e24674 |
| resnet50 | cnn | CNN | True | 16 | 0.1000 | 2.2128 | 2048 | 0.9827 | 0.3760 | 18.3457 | 16.5082 | 0.0034 | 0.0035 | 0.7791 | 0.7125 | True | 2026-08-12T14:07:06.134595+00:00 | d6a4cbf87f3786dea3640b21b40165aa9296c5a5 |
| siglip | general_vlm | General VLM | True | 16 | 0.1000 | 8.5050 | 1024 | 0.9986 | 0.4710 | 17.3195 | 16.9567 | 0.0131 | 0.0130 | 0.9663 | 0.7993 | True | 2026-08-13T13:48:13.882217+00:00 | f3b9ada90008aa0d3cc2cee0ab286af5e0c022db |
| uni | medical_ssl | Medical SSL | True | 16 | 0.1000 | 16.6812 | 1024 | 0.9957 | 0.4679 | 20.5289 | 20.0465 | 0.0053 | 0.0053 | 0.9725 | 0.8613 | True | 2026-08-12T20:19:32.777085+00:00 | 21c1bdfdb764c2f0c8d1109b5046a58566f4822e |
