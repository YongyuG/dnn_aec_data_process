import numpy as np
import soundfile as sf
import os

def ERLE(nearend_mic_signal, error_signal):
    erle = 10 * np.log10(
        np.mean(nearend_mic_signal**2) / np.mean( error_signal **2)
    )
    return erle

def SER(nearend_speech, far_echo):
    return 10 * np.log10(((nearend_speech ** 2 ).mean()**0.5) / (far_echo **2).mean()**0.5)

if __name__ == "__main__":

    fileid = 9999
    nearend_mic_path = "/home/yongyug/data/aec_challenge/datasets/synthetic/nearend_mic_signal/nearend_mic_fileid_{}.wav".format(fileid)
    nearend_speech_path = "/home/yongyug/data/aec_challenge/datasets/synthetic/nearend_speech/nearend_speech_fileid_{}.wav".format(fileid)
    error_path = "/home/yongyug/data/aec_challenge/datasets/synthetic/filter_out/mixdata_fileid_{}.wav".format(fileid)
    echo_path = "/home/yongyug/data/aec_challenge/datasets/synthetic/echo_signal/echo_fileid_{}.wav".format(fileid)

    nlp_path = "/home/yongyug/data/aec_challenge/datasets/synthetic/nearend_mic_mix_farend_speech_signal/mixdata_fileid_{}_aec_native.wav".format(fileid)

    nearend_mic_signal, sr = sf.read(nearend_mic_path)
    error_signal, _ = sf.read(error_path)
    echo_signal, _ = sf.read(echo_path)
    nearend_speech, _ = sf.read(nearend_speech_path)


    nlp_signal, _ = sf.read(nlp_path)

    erle_nonlp = ERLE(nearend_mic_signal, error_signal)
    erle_nlp = ERLE(nearend_mic_signal, nlp_signal)
    print(erle_nonlp)
    print(erle_nlp)

    ser = SER(nearend_speech, echo_signal)
    print(ser)