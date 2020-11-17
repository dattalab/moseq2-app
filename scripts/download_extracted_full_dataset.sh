#!/bin/bash
mkdir full_extracted_test_ds
cp Main-MoSeq2-Notebook.ipynb full_extracted_test_ds
cp Interactive-Model-Results-Exploration.ipynb full_extracted_test_ds
cd full_extracted_test_ds

# amphetamine dataset
curl -L -o amphetamine_example_00.zip https://www.dropbox.com/sh/drmaofh8m2dd5hi/AADMCt5cijqWFDmw5TVNscdka?dl=0 && unzip amphetamine_example_00.zip -d amphetamine_example_00
rm amphetamine_example_00.zip
curl -L -o amphetamine_example_01.zip https://www.dropbox.com/sh/o9q48trj26a0i07/AADHjoyJR0cetgR8yAE_evCca?dl=0 && unzip amphetamine_example_01.zip -d amphetamine_example_01
rm amphetamine_example_01.zip
curl -L -o amphetamine_example_02.zip https://www.dropbox.com/sh/acwe2efswd6qs15/AADUlrMZ2yEPdWfkJaemM9-ga?dl=0 && unzip amphetamine_example_02.zip -d amphetamine_example_02
rm amphetamine_example_02.zip
curl -L -o amphetamine_example_03.zip https://www.dropbox.com/sh/ofmqjm5jjzmawe0/AACfiBqAAntf1Xts_lOuJZqLa?dl=0 && unzip amphetamine_example_03.zip -d amphetamine_example_03
rm amphetamine_example_03.zip
curl -L -o amphetamine_example_04.zip https://www.dropbox.com/sh/wj0xi4cnz7hyqm7/AABOwjLRcR7Lit04NQMG4baCa?dl=0 && unzip amphetamine_example_04.zip -d amphetamine_example_04
rm amphetamine_example_04.zip
curl -L -o amphetamine_example_05.zip https://www.dropbox.com/sh/eep9ia6x3efsmvk/AACMGl9Er-n11g2IC2bjyO4ma?dl=0 && unzip amphetamine_example_05.zip -d amphetamine_example_05
rm amphetamine_example_05.zip
curl -L -o amphetamine_example_06.zip https://www.dropbox.com/sh/8aww55u5b5mq604/AADRsIQ5j3SAK5VEFH_omElra?dl=0 && unzip amphetamine_example_06.zip -d amphetamine_example_06
rm amphetamine_example_06.zip
curl -L -o amphetamine_example_07.zip https://www.dropbox.com/sh/u6lbhdqyfn6dug6/AACHev7SUwPelB403b__uCxla?dl=0 && unzip amphetamine_example_07.zip -d amphetamine_example_07
rm amphetamine_example_07.zip
curl -L -o amphetamine_example_08.zip https://www.dropbox.com/sh/3b60oag1xz13epf/AAAjICwg3wdWIYQMRBH0bqFWa?dl=0 && unzip amphetamine_example_08.zip -d amphetamine_example_08
rm amphetamine_example_08.zip
curl -L -o amphetamine_example_09.zip https://www.dropbox.com/sh/7cldri3t6f592yd/AABK76J46lKanNegTnIQO1AIa?dl=0 && unzip amphetamine_example_09.zip -d amphetamine_example_09
rm amphetamine_example_09.zip
curl -L -o amphetamine_example_10.zip https://www.dropbox.com/sh/h2vxosn8eaflaya/AAC48gg6CRal9A9QjpiSTYNXa?dl=0 && unzip amphetamine_example_10.zip -d amphetamine_example_10
rm amphetamine_example_10.zip
curl -L -o amphetamine_example_11.zip https://www.dropbox.com/sh/hadwviajpz6vl81/AABSTYozm7sZORM9bm3ukV7Aa?dl=0 && unzip amphetamine_example_11.zip -d amphetamine_example_11
rm amphetamine_example_11.zip
curl -L -o amphetamine_example_12.zip https://www.dropbox.com/sh/usrx6zmb4dh6hy7/AACDRE9rOzD-YeHbS3_PLQNna?dl=0 && unzip amphetamine_example_12.zip -d amphetamine_example_12
rm amphetamine_example_12.zip
curl -L -o amphetamine_example_13.zip https://www.dropbox.com/sh/s3ps0rnueww90q0/AABPjsisbjvP1Gte9Wkqg5FYa?dl=0 && unzip amphetamine_example_13.zip -d amphetamine_example_13
rm amphetamine_example_13.zip
curl -L -o amphetamine_example_14.zip https://www.dropbox.com/sh/5knt6k3yl6rssih/AADclJrP-vIRGvp_XLWXg1Rxa?dl=0 && unzip amphetamine_example_14.zip -d amphetamine_example_14
rm amphetamine_example_14.zip
curl -L -o amphetamine_example_15.zip https://www.dropbox.com/sh/o4mmkno7wazebol/AACOQ8KxsZ62ci59N9okB1Wpa?dl=0 && unzip amphetamine_example_15.zip -d amphetamine_example_15
rm amphetamine_example_15.zip
curl -L -o amphetamine_example_16.zip https://www.dropbox.com/sh/hsrs5cmilre675f/AAA8ql5P4Lx5-6nQu4p5ZdLsa?dl=0 && unzip amphetamine_example_16.zip -d amphetamine_example_16
rm amphetamine_example_16.zip
curl -L -o amphetamine_example_17.zip https://www.dropbox.com/sh/5okozzcedttdl2v/AADjkXy_GFm-eItK6fWzxSVza?dl=0 && unzip amphetamine_example_17.zip -d amphetamine_example_17
rm amphetamine_example_17.zip
curl -L -o amphetamine_example_18.zip https://www.dropbox.com/sh/26d7uiko5d69dlm/AACRoLMh1cJCIYPwP1emfs3Sa?dl=0 && unzip amphetamine_example_18.zip -d amphetamine_example_18
rm amphetamine_example_18.zip
curl -L -o amphetamine_example_19.zip https://www.dropbox.com/sh/0htondgsgcu2xbb/AAD3HGMPeZFmNx0fQcb0BJfNa?dl=0 && unzip amphetamine_example_19.zip -d amphetamine_example_19
rm amphetamine_example_19.zip
curl -L -o amphetamine_example_20.zip https://www.dropbox.com/sh/qfwbupgyjqp3ane/AACxO4KDXiyq_ZAFK5tN1DdMa?dl=0 && unzip amphetamine_example_20.zip -d amphetamine_example_20
rm amphetamine_example_20.zip
curl -L -o amphetamine_example_21.zip https://www.dropbox.com/sh/dlmo871xk17fz2h/AAAYDSo7R066Q4FdmTVxtQqXa?dl=0 && unzip amphetamine_example_21.zip -d amphetamine_example_21
rm amphetamine_example_21.zip
curl -L -o amphetamine_example_22.zip https://www.dropbox.com/sh/evou4ncsi0cgzb9/AAA4zCd3nDJHdEKOIS1yR66Fa?dl=0 && unzip amphetamine_example_22.zip -d amphetamine_example_22
rm amphetamine_example_22.zip
curl -L -o amphetamine_example_23.zip https://www.dropbox.com/sh/eq9cit74r0orc5n/AAAS3Xf-jvp76N70UIcYAJtya?dl=0 && unzip amphetamine_example_23.zip -d amphetamine_example_23
rm amphetamine_example_23.zip

# saline dataset
curl -L -o saline_example_00.zip https://www.dropbox.com/sh/r2jxghghqfogk1x/AADOZ5z7B-7W_gLUnIQt1SoUa?dl=0 && unzip saline_example_00.zip -d saline_example_00
rm saline_example_00.zip
curl -L -o saline_example_01.zip https://www.dropbox.com/sh/z7shmmje9e4zsuf/AADGRm7JctJLlyFMLAHpJwJia?dl=0 && unzip saline_example_01.zip -d saline_example_01
rm saline_example_01.zip
curl -L -o saline_example_02.zip https://www.dropbox.com/sh/n7n6jpuqvnd2lcz/AACgN-f0Vf2LUBnPv527QEH5a?dl=0 && unzip saline_example_02.zip -d saline_example_02
rm saline_example_02.zip
curl -L -o saline_example_03.zip https://www.dropbox.com/sh/1a0foelj4rvntq9/AADPrvdGNni-jXfuAV326MjEa?dl=0 && unzip saline_example_03.zip -d saline_example_03
rm saline_example_03.zip
curl -L -o saline_example_04.zip https://www.dropbox.com/sh/y2ni57giz2uh99z/AACZpAAbZXz_tqVE7e5G4alsa?dl=0 && unzip saline_example_04.zip -d saline_example_04
rm saline_example_04.zip
curl -L -o saline_example_05.zip https://www.dropbox.com/sh/41l73kw6p5i3pxm/AACS76pvnsuDPvfVKb30pcK5a?dl=0 && unzip saline_example_05.zip -d saline_example_05
rm saline_example_05.zip
curl -L -o saline_example_06.zip https://www.dropbox.com/sh/vxhz6jlsehz1kiu/AAAhlmlBG1lrb9t2s9kDr1S8a?dl=0 && unzip saline_example_06.zip -d saline_example_06
rm saline_example_06.zip
curl -L -o saline_example_07.zip https://www.dropbox.com/sh/50n9m6gi03pdhvn/AAAR4WpPckPuRN6M03tGfUdca?dl=0 && unzip saline_example_07.zip -d saline_example_07
rm saline_example_07.zip
curl -L -o saline_example_08.zip https://www.dropbox.com/sh/halc7atxvxeziqy/AAA_QwTAbT4lFpRSzZv0dxQWa?dl=0 && unzip saline_example_08.zip -d saline_example_08
rm saline_example_08.zip
curl -L -o saline_example_09.zip https://www.dropbox.com/sh/8zdzft44e2a7xwe/AAB1qkMK_3-RXZupQUa5G5Wfa?dl=0 && unzip saline_example_09.zip -d saline_example_09
rm saline_example_09.zip
curl -L -o saline_example_10.zip https://www.dropbox.com/sh/hzq15yw2hzdowor/AAAXgMyjWGONHvX1ytFVQ8_ia?dl=0 && unzip saline_example_10.zip -d saline_example_10
rm saline_example_10.zip
curl -L -o saline_example_11.zip https://www.dropbox.com/sh/2g4fidkon0y94l3/AACVhZrpkWBhz5wNU2Btzay6a?dl=0 && unzip saline_example_11.zip -d saline_example_11
rm saline_example_11.zip
curl -L -o saline_example_12.zip https://www.dropbox.com/sh/b8bb8ionulz4q06/AAAbQ50Fzt2ND1y3iBxxMHc0a?dl=0 && unzip saline_example_12.zip -d saline_example_12
rm saline_example_12.zip
curl -L -o saline_example_13.zip https://www.dropbox.com/sh/nx1uqlszf99x59v/AAD3G-1kvlBZazUYgODP2HeMa?dl=0 && unzip saline_example_13.zip -d saline_example_13
rm saline_example_13.zip
curl -L -o saline_example_14.zip https://www.dropbox.com/sh/2uea7v788h7e7h7/AABMq1kVNBTXHscYZZeY43txa?dl=0 && unzip saline_example_14.zip -d saline_example_14
rm saline_example_14.zip
curl -L -o saline_example_15.zip https://www.dropbox.com/sh/7f3eyw77h2ztd8j/AAD3W6Fo-oV73nQ7Hb4tlJ3ga?dl=0 && unzip saline_example_15.zip -d saline_example_15
rm saline_example_15.zip
curl -L -o saline_example_16.zip https://www.dropbox.com/sh/v8y1fowmfkq3l2u/AACSTqofKhhs3FxdlHxbPjaAa?dl=0 && unzip saline_example_16.zip -d saline_example_16
rm saline_example_16.zip
curl -L -o saline_example_17.zip https://www.dropbox.com/sh/1ff51ixir9yf2uo/AACh0CCYQ_VTe_Pa0dicJVepa?dl=0 && unzip saline_example_17.zip -d saline_example_17
rm saline_example_17.zip
curl -L -o saline_example_18.zip https://www.dropbox.com/sh/tt6f47kfs4rol3y/AACZwyCqXg4W_wsERS0JsDiqa?dl=0 && unzip saline_example_18.zip -d saline_example_18
rm saline_example_18.zip
curl -L -o saline_example_19.zip https://www.dropbox.com/sh/1q1q1a65l0mx7th/AAC1Nsc_-cxKvzN4YT5wTU6Ta?dl=0 && unzip saline_example_19.zip -d saline_example_19
rm saline_example_19.zip
curl -L -o saline_example_20.zip https://www.dropbox.com/sh/fautnfscsvj1u7j/AABhL7WBkbeRvBpCB_Yr2Nzga?dl=0 && unzip saline_example_20.zip -d saline_example_20
rm saline_example_20.zip
curl -L -o saline_example_21.zip https://www.dropbox.com/sh/bj76z7evdzovjv5/AADhr4uwhHR20Oxw64yCKeRsa?dl=0 && unzip saline_example_21.zip -d saline_example_21
rm saline_example_21.zip
curl -L -o saline_example_22.zip https://www.dropbox.com/sh/umdkmpvdniyco3q/AADTNfr9kuBm_2ftLpSen5xra?dl=0 && unzip saline_example_22.zip -d saline_example_22
rm saline_example_22.zip
curl -L -o saline_example_23.zip https://www.dropbox.com/sh/ggs7zcsj5mmtcua/AABlGhfGxJRDBw_G2n3DNnOva?dl=0 && unzip saline_example_23.zip -d saline_example_23
rm saline_example_23.zip

# PCA Files
curl -L -o _pca.zip https://www.dropbox.com/sh/m4fw0uuoqzhnk3q/AAA97ol_QiT2YGXto50hvWpUa?dl=0 && unzip _pca.zip -d _pca
rm _pca.zip

# Modeling Files
curl -L -o saline-amphetamine.zip https://www.dropbox.com/sh/qnsrbpqrbot0lw2/AAAB3HBUGLVAY1HxWztY6RKla?dl=0  && unzip saline-amphetamine.zip -d saline-amphetamine
rm saline-amphetamine.zip

# Progress and Index Files
curl -L -o progress.yaml https://www.dropbox.com/s/49445asfydjpkvd/progress.yaml?dl=0

# Config File
curl -L -o config.yaml https://www.dropbox.com/s/3n10dtwo5rnta2u/config.yaml?dl=0

