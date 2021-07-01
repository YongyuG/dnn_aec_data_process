
Timit data process for dnn acoustic echo cancellation experiments
==============================
This repo is following the data setup from [Deep Learning for Acoustic Echo Cancellation in Noisy and Double-TalkScenarios](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1484.pdf).

It' a draft script, I will modify it and put all changeable configurations into a json so that it can be used more friendly.

By the way, if you want to do some work in deep learning aec, I recommend using farend data from AEC-challenge and mix with other clean open source datasets.

Notification
============

References:

Paper: [Deep Learning for Acoustic Echo Cancellation in Noisy and Double-TalkScenarios](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1484.pdf)  

DNS-CHALLENGE: [INTERSPEECH 2021 Deep Noise Suppression Challenge](https://arxiv.org/pdf/2101.01902.pdf)  
DNS-CHALLENGE CODE: [INTERSPEECH 2021 Deep Noise Suppression Challenge](https://github.com/microsoft/DNS-Challenge)  

AEC-CHALLENGE:[ICASSP 2021 ACOUSTIC ECHO CANCELLATION CHALLENGE: DATASETS, TESTINGFRAMEWORK, AND RESULTS](https://arxiv.org/pdf/2009.04972.pdf)  
AEC-CHALLENGE CODE:[ICASSP 2021 ACOUSTIC ECHO CANCELLATION CHALLENGE: DATASETS, TESTINGFRAMEWORK, AND RESULTS](https://github.com/microsoft/AEC-Challenge)  


How to use
==========
1. change __dataPath__, __noisePath__, __outPath__ and __rirPath__ according to your setups, p.s. __rirPath__ is provided from DNS-CHALLENGE where you can review above

2. python timit_pre_process.py

Last Modification
============

1. add json
2. randomly pad signal to certain length
