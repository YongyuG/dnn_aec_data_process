'''
Author: Yongyu Gao
Email: yongyugao@hotmail.com

This script is strictly following experimental data setup from paper:

<<Deep Learning for Acoustic Echo Cancellation in Noisy and Double-Talk Scenarios>>

'''
import glob
import librosa
import os
import random
import pandas as pd
import numpy as np
from random import shuffle
import soundfile as sf
from scipy import signal
from evaluation import SER

MAXTRIES = 50

np.random.seed(9999)
random.seed(9999)
EPS = np.finfo(float).eps

def get_single_gender_index_list(data_list, repeat_enable=False, num_pair=30):    #获取不重复的单性别说话人对
    index_list = []
    seen_list = []
    i = 0
    while i < num_pair:
        index_set = list(np.random.randint(0, len(data_list), 2))
        if repeat_enable:
            if index_set not in index_list and index_set[::-1] not in index_list:
                index_list.append(index_set)
                i += 1
        else:
            if index_set[0] not in seen_list and index_set[1] not in seen_list:
                index_list.append(index_set)
                seen_list.append(index_set[0])
                seen_list.append(index_set[1])

                i += 1

    return index_list

def get_double_gender_index_list(male_list, female_list, repeat_enable=False, num_pair=40):
    male_female_index_list = []
    seen_list = []

    i = 0
    while i < num_pair:
        male_index = np.random.randint(0, len(male_list))
        female_index = np.random.randint(0, len(female_list))
        index_set = [male_index, female_index]
        if repeat_enable:
            if index_set not in male_female_index_list:
                male_female_index_list.append(index_set)
                i += 1
        else:
            if male_index not in seen_list and female_index not in seen_list:
                male_female_index_list.append(index_set)
                seen_list.append(male_index)
                seen_list.append(female_index)
                i += 1
    return male_female_index_list


def random_three_nonrepeat_sample(data_len):
    three_sample_list = []
    for i in range(data_len):
        for j in range(i + 1, data_len):
            for k in range(j + 1, data_len):
                three_sample_list.append([i, j, k])

    return three_sample_list

def add_pyreverb(clean_speech, rir):
    predelay = 50
    early_delay_samples = (predelay * 16000) // 1000
    early_rir = rir[:early_delay_samples]

    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")
    noreverb_speech = signal.fftconvolve(clean_speech, early_rir, mode="full")

    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[0 : clean_speech.shape[0]]
    noreverb_speech = noreverb_speech[0 : clean_speech.shape[0]]

    return reverb_speech, noreverb_speech

def signal_pad(signal, audio_sample_length):
    if len(signal) < audio_sample_length:  # 设定一个统一的长度,如果长度不够, 则前后补零
        if len(signal) % 2 == 0:
            signal = np.pad(signal, ((audio_sample_length - len(signal)) // 2,
                                                     (audio_sample_length - len(signal)) // 2), 'constant',
                                    constant_values=(0, 0))
        elif len(signal) % 2 != 0:
            signal = np.pad(signal, ((audio_sample_length - len(signal)) // 2,
                                                     (audio_sample_length - len(signal)) // 2 + len(
                                                         signal) % 2), 'constant',
                                    constant_values=(0, 0))  # 无法被2整除则把余数补零至最后

    elif len(signal) >= audio_sample_length:
        signal = signal[:audio_sample_length]
    return signal

# def generate_single_gender_wav_pair(nearend_data_list, farend_data_list, data_dict, pairname):
def generate_gender_wav_pair(nearend_data_list, farend_data_list, data_dict1, data_dict2, pairname):

    farend_three_sample_index = random_three_nonrepeat_sample(10)  # 将所有farend不重复的组合list列出来


    train_res_list = []
    validate_res_list = []
    count = 0
    for i in range(len(nearend_data_list)):
        nearend_spk = nearend_data_list[i]
        farend_spk = farend_data_list[i]
        if nearend_spk[0] == 'M':
            nearend_spk_wav = data_dict1[nearend_spk]
        else:
            nearend_spk_wav = data_dict2[nearend_spk]
        if farend_spk[0] == 'M':
            farend_spk_wav = data_dict1[farend_spk]
        else:
            farend_spk_wav = data_dict2[farend_spk]


        nearend_select_index = np.arange(10)  ## 每个人10条语音, 做一个随机
        np.random.shuffle(nearend_select_index)  # for nearend_spk in nearend_spk_list:

        nearend_wav_pick = np.array(nearend_spk_wav)[nearend_select_index]
        farend_group = [[i for i in range(j * 5, (j + 1) * 5)] for j in range(len(nearend_select_index))]  # 这里是对于nearend来说, 渠道每个wav对应farend的index
        random.shuffle(farend_three_sample_index)  # 把farend的三元list随机一下
        farend_select_index = np.array(farend_three_sample_index)[np.array(farend_group)]  # 把每个nearend选择的farend取出来
        farend_wav_pick = np.array(farend_spk_wav)[farend_select_index]

        for k in range(len(nearend_wav_pick)):
            if k < 7:
                count += 1
                #print(nearend_wav_pick[k], farend_wav_pick[k])
                train_res_list.append((nearend_wav_pick[k], farend_wav_pick[k]))

            else:
                validate_res_list.append((nearend_wav_pick[k], farend_wav_pick[k]))
    #print(validate_res_list)
    return train_res_list, validate_res_list



def get_data_pair(dataPath, repeat_enable=True, samle_gender_pair=30, diff_gender_pair=40):

    male_dict = {}
    female_dict = {}
    count = 0

    #分别将timit中男女相关的
    for root, _, files in os.walk(dataPath):
        for file in files:
            if file.endswith('WAV'):
                count += 1
                dataType, spk = root.split(os.path.sep)[-3], root.split(os.path.sep)[-1]
                gender = spk[0]
                if gender == "M":
                    if spk not in male_dict.keys():
                        male_dict[spk] = []
                    male_dict[spk].append(os.path.join(root, file))
                elif gender == "F":
                    if spk not in female_dict.keys():
                        female_dict[spk] = []
                    female_dict[spk].append(os.path.join(root, file))



    male_name_list = list(male_dict.keys())
    female_name_list = list(female_dict.keys())


    #Randomize coresponding data-pairs, get the speaker idx from the list
    male_male_index_list = get_single_gender_index_list(male_name_list, repeat_enable=repeat_enable, num_pair=samle_gender_pair)

    female_female_index_list = get_single_gender_index_list(female_name_list, repeat_enable=repeat_enable, num_pair=samle_gender_pair)
    male_female_index_list = get_double_gender_index_list(male_name_list, female_name_list, repeat_enable=repeat_enable, num_pair=diff_gender_pair)

    #Randomize which speaker as for farend spk
    male_female_farend_choice = np.random.randint(0, 2, len(male_female_index_list)) #这里是随机选择pair中哪一个spk当作farend
    male_male_farend_choice = np.random.randint(0, 2, len(male_male_index_list))
    female_female_farend_choice = np.random.randint(0, 2, len(female_female_index_list))

    #Get the speaker name from data-pair
    male_name_arr = np.array(male_name_list)
    male_name_arr = male_name_arr[np.array(male_male_index_list)]
    female_name_arr = np.array(female_name_list)
    female_name_arr = female_name_arr[np.array(female_female_index_list)]
    male_female_name_arr = np.array([np.array(male_name_list)[np.array(male_female_index_list).T[0]],
                        np.array(female_name_list)[np.array(male_female_index_list).T[1]]]).T  #转置是为了把male和female分开, 因为male_female_index_list是 [male, female顺序排列的]


    #Get specific farend and nearend speaker key
    male_male_nearend_spk_list = [male_name_arr[i][male_male_farend_choice[i] ^ 1] for i in range(len(male_name_arr))]
    male_male_farend_spk_list = [male_name_arr[i][male_male_farend_choice[i]] for i in range(len(male_name_arr))]
    female_female_nearend_spk_list = [female_name_arr[i][female_female_farend_choice[i] ^ 1] for i in range(len(female_name_arr))]
    female_female_farend_spk_list = [female_name_arr[i][female_female_farend_choice[i]] for i in range(len(female_name_arr))]
    male_female_nearend_spk_list = [male_female_name_arr[i][male_female_farend_choice[i] ^ 1] for i in range(len(male_female_name_arr))]
    male_female_farend_spk_list = [male_female_name_arr[i][male_female_farend_choice[i]] for i in range(len(male_female_name_arr))]


    #Generate specifc wav_pair for each data_pair
    male_male_train, male_male_validate = generate_gender_wav_pair(male_male_nearend_spk_list, male_male_farend_spk_list, male_dict, male_dict, 'male_male')
    female_female_train, female_female_validate = generate_gender_wav_pair(female_female_nearend_spk_list, female_female_farend_spk_list , female_dict, female_dict,'female_female')
    male_female_train, male_female_validate = generate_gender_wav_pair(male_female_nearend_spk_list, male_female_farend_spk_list, male_dict, female_dict, 'male_female')

    train_dataset = male_male_train + female_female_train + male_female_train
    validate_dataset = male_male_validate + female_female_validate + male_female_validate
    return train_dataset, validate_dataset

    #all 3-type pairs will be merged in male_male_final_data with both train and validate

    # train_dict = {}
    # train_dict.update(male_male_final_data['train'])
    # train_dict.update(female_female_final_data['train'])
    # train_dict.update(male_female_final_data['train'])
    #
    # validate_dict = {}
    # validate_dict.update(male_male_final_data['validate'])
    # validate_dict.update(female_female_final_data['validate'])
    # validate_dict.update(male_female_final_data['validate'])

    # male_male_final_data['train'].update(female_female_final_data['train'])
    # male_male_final_data['train'].update(male_female_final_data['train'])
    # male_male_final_data['validate'].update(female_female_final_data['validate'])
    # male_male_final_data['validate'].update(male_female_final_data['validate'])
    # res_dict = male_male_final_data
    # return res_dict

def get_rir_dict(rir_csv_path):
    temp = pd.read_csv(rir_csv_path, skiprows=[1], sep=',', header=None,
                       names=['wavfile', 'channel', 'T60_WB', 'C50_WB', 'isRealRIR'])
    #temp.keys()

    rir_wav = temp['wavfile'][1:]  # 115413
    rir_channel = temp['channel'][1:]
    rir_t60 = temp['T60_WB'][1:]
    rir_isreal = temp['isRealRIR'][1:]

    rir_wav2 = [w.replace('\\', '/') for w in rir_wav]
    rir_channel2 = [w for w in rir_channel]
    rir_t60_2 = [w for w in rir_t60]
    rir_isreal2 = [w for w in rir_isreal]

    myrir = []
    mychannel = []
    myt60 = []

    lower_t60 = 0.3
    upper_t60 = 1.3

    all_indices = [i for i, x in enumerate(rir_isreal2)]

    chosen_i = []
    for i in all_indices:
        if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
            chosen_i.append(i)

    myrir = [rir_wav2[i] for i in chosen_i]
    mychannel = [rir_channel2[i] for i in chosen_i]
    myt60 = [rir_t60_2[i] for i in chosen_i]

    rir_dict = {"myrir":myrir, 'mychannel':mychannel, 'myt60':myt60}
    return rir_dict

def get_rir_samples(rir_dict):
    myrir = rir_dict['myrir']
    mychannel = rir_dict['mychannel']
    myt60 = rir_dict['myt60']
    #
    #
    rir_index = random.randint(0, len(myrir) - 1)
    my_rir = myrir[rir_index]

    while not os.path.exists(my_rir):
        rir_index = random.randint(0, len(myrir) - 1)
        my_rir = myrir[rir_index]

    samples_rir, fs_rir = sf.read(my_rir)

    my_channel = int(mychannel[rir_index])

    if samples_rir.ndim == 1:
        samples_rir_ch = np.array(samples_rir)
    elif my_channel > 1:
        samples_rir_ch = samples_rir[:, my_channel - 1]
    else:
        samples_rir_ch = samples_rir[:, my_channel - 1]

    return samples_rir_ch

def get_noise_files(noise_path):
    sources_files_names = glob.glob(os.path.join(noise_path, "*.wav"))
    shuffle(sources_files_names)
    return sources_files_names

def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)

def build_noise_audio(noise_path, fs=16000, audio_length=8, audio_samples_length=-1):
    '''Construct an audio signal from source files'''

    fs_output = fs
    silence_length = 0.2
    if audio_samples_length == -1:
        audio_samples_length = int(audio_length*fs)

    output_audio = np.zeros(0)
    remaining_length = audio_samples_length
    files_used = []
    clipped_files = []

    source_files = glob.glob(os.path.join(noise_path,
                                          "*.wav"))
    shuffle(source_files)
    # pick a noise source file index randomly
    idx = np.random.randint(0, np.size(source_files))

    # initialize silence
    silence = np.zeros(int(fs_output*silence_length))

    # iterate through multiple clips until we have a long enough signal
    tries_left = MAXTRIES
    while remaining_length > 0 and tries_left > 0:

        # read next audio file and resample if necessary

        idx = (idx + 1) % np.size(source_files) #这里有种shift的感觉, 第0个是最后的时候才process的
        input_audio, fs_input = sf.read(source_files[idx])
        if fs_input != fs_output:
            input_audio = librosa.resample(input_audio, fs_input, fs_output)

        # if current file is longer than remaining desired length, and this is
        # noise generation or this is training set, subsample it randomly
        if len(input_audio) > remaining_length:
            idx_seg = np.random.randint(0, len(input_audio)-remaining_length)
            input_audio = input_audio[idx_seg:idx_seg+remaining_length]

        # check for clipping, and if found move onto next file
        if is_clipped(input_audio):
            clipped_files.append(source_files[idx])
            tries_left -= 1
            continue

        # concatenate current input audio to output audio stream
        files_used.append(source_files[idx])
        output_audio = np.append(output_audio, input_audio)
        remaining_length -= len(input_audio)

        # add some silence if we have not reached desired audio length
        if remaining_length > 0:
            silence_len = min(remaining_length, len(silence))
            output_audio = np.append(output_audio, silence[:silence_len])
            remaining_length -= silence_len

    return output_audio, files_used, clipped_files

def normalize(audio, target_level=-25):
    '''Normalize the signal to the target level'''
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    audio = audio * scalar
    return audio

def normalize_segmental_rms(audio, rms, target_level=-25):
    '''Normalize the signal to the target level
    based on segmental RMS'''
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    audio = audio * scalar
    return audio


def active_rms(clean, noise, fs=16000, energy_thresh=-50):
    '''Returns the clean and noise RMS of the noise calculated only in the active portions'''
    window_size = 100  # in ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    noise_active_segs = []
    clean_active_segs = []

    while sample_start < len(noise):
        sample_end = min(sample_start + window_samples, len(noise))
        noise_win = noise[sample_start:sample_end]
        clean_win = clean[sample_start:sample_end]
        noise_seg_rms = 20 * np.log10((noise_win ** 2).mean() + EPS)
        # Considering frames with energy
        if noise_seg_rms > energy_thresh:
            noise_active_segs = np.append(noise_active_segs, noise_win)
            clean_active_segs = np.append(clean_active_segs, clean_win)
        sample_start += window_samples

    if len(noise_active_segs) != 0:
        noise_rms = (noise_active_segs ** 2).mean() ** 0.5
    else:
        noise_rms = EPS

    if len(clean_active_segs) != 0:
        clean_rms = (clean_active_segs ** 2).mean() ** 0.5
    else:
        clean_rms = EPS

    return clean_rms, noise_rms


def segmental_snr_mixer(clean, noise, snr, target_level=-25, clipping_threshold=0.99):
    '''Function to mix clean speech and noise at various segmental SNR levels'''
    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean)-len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise)-len(clean)))
    clean = clean/(max(abs(clean))+EPS)
    noise = noise/(max(abs(noise))+EPS)
    rmsclean, rmsnoise = active_rms(clean=clean, noise=noise)
    clean = normalize_segmental_rms(clean, rms=rmsclean, target_level=target_level)
    noise = normalize_segmental_rms(noise, rms=rmsnoise, target_level=target_level)
    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10**(snr/20)) / (rmsnoise+EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = np.random.randint(-35, -15)
    rmsnoisy = (noisyspeech**2).mean()**0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy+EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy
    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech))/(clipping_threshold-EPS)
        noisyspeech = noisyspeech/noisyspeech_maxamplevel
        clean = clean/noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel/noisyspeech_maxamplevel
        noisy_rms_level = int(20*np.log10(scalarnoisy/noisyspeech_maxamplevel*(rmsnoisy+EPS)))

    return clean, noisenewlevel, noisyspeech, noisy_rms_level

def nearend_farend_mixer(nearend_data, echo_data, lowerbound_ser=-10, upperbound_ser=13, clipping_threshold=0.99):
    ser = np.random.uniform(lowerbound_ser, upperbound_ser)
    ser = float(format(ser, '.4f'))

    local_ser = SER(nearend_data, echo_data)
    # print(nearend[nearend_index], farend_dict[key], local_ser)
    # print(len(echo_data), len(farend_speech), len(nearend_data))

    nearend_data = nearend_data / max(abs(nearend_data))
    echo_data = echo_data / max(abs(echo_data))

    nearend_rms = np.mean(nearend_data ** 2) ** 0.5
    echo_rms = np.mean(echo_data ** 2) ** 0.5

    echocalar = nearend_rms / (10 ** (ser / 10)) / (echo_rms + EPS)
    new_echo = echo_data * echocalar
    new_ser = SER(nearend_data, new_echo)

    nearend_mic = nearend_data + new_echo
    if is_clipped(nearend_mic):
        nearnendmic_maxamplevel = max(abs(nearend_mic)) / (clipping_threshold - EPS)
        nearend_mic = nearend_mic / nearnendmic_maxamplevel
        nearend_data = nearend_data / nearnendmic_maxamplevel
        new_echo = new_echo / nearnendmic_maxamplevel
        #echo_rms_level = int(10 * np.log10(echocalar / nearnendmic_maxamplevel * (echo_rms + EPS)))
    return nearend_mic, nearend_data, new_echo, ser


'''
We follow the dataset allocation from AEC-CHALLENGE,
The validation data are placed at first 300 fileid
the remaining file are training sets
我们数据分布方式根据aec-challenge来,即前N个,在汪德凉老师的paper中为300,是验证集, 剩余的是训练集
'''
def generate_pair_audio(train_dataset, validate_dataset, rir_path, noise_dataset, outputPath, use_reverb=True, sample_rate=16000, audio_length=8):

    csv_path = os.path.join(outputPath, "csv")
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    nearend_speech_out = os.path.join(outputPath, "nearend_speech")
    farend_speech_out = os.path.join(outputPath, "farend_speech")
    echo_out = os.path.join(outputPath, "echo_signal")
    nearend_mic_out = os.path.join(outputPath, "nearend_mic")

    subdir_list = [nearend_speech_out, echo_out, nearend_mic_out, farend_speech_out]
    for i in subdir_list:
        if not os.path.exists(i):
            os.makedirs(i)


    count = 0
    audio_sample_length = sample_rate * audio_length

    farend_speech_name_1 = []
    farend_speech_name_2 = []
    farend_speech_name_3 = []
    nearend_speech_name = []
    filed_id = []
    noise_file_name = []

    noise_clipped_files = []
    noise_source_files = []
    split = []
    ser_list = []
    snr_list = []



    rir_dict = get_rir_dict(rir_path)

    #TODO: clean it up, make it in a function
    for i in range(len(validate_dataset)):
        print("validate = ", count)
        validate_data_pair = validate_dataset[i]
        validate_nearend_path = validate_data_pair[0]
        validate_farend_path_sets = validate_data_pair[1][np.random.randint(len(validate_data_pair[1]))]
        validate_nearend_speech, validate_sr = sf.read(validate_nearend_path)
        assert validate_sr == sample_rate
        validate_nearend_speech = signal_pad(validate_nearend_speech, audio_sample_length)

        validate_farend_speech = np.concatenate([sf.read(wav)[0] for wav in validate_farend_path_sets])
        validate_farend_speech = signal_pad(validate_farend_speech, audio_sample_length)
        samples_rir_ch = get_rir_samples(rir_dict)
        validate_reverb_farend, validate_noreverb_farend = add_pyreverb(validate_farend_speech, samples_rir_ch)

        #TODO 加噪,但有些部分我有争议, 可能不需要对输入做那么多的归一和scaling这个到时候实验看看
        validate_noise_audio, validate_noise_file, validate_noise_cf = build_noise_audio(noise_dataset, fs=sample_rate, audio_length=audio_length, audio_samples_length=-1)


        snr = np.random.randint(-5, 20)
        snr_list.append(snr)
        if use_reverb:
            validate_farend_snr, validate_noise_snr, validate_echo_signal, target_level = segmental_snr_mixer(clean=validate_reverb_farend,
                                                                                   noise=validate_noise_audio, snr=snr)
        else:
            validate_farend_snr, validate_noise_snr, validate_echo_signal, target_level = segmental_snr_mixer(clean=validate_noreverb_farend,
                                                                                   noise=validate_noise_audio, snr=snr)

        validate_nearend_mic, validate_nearend_speech2, validate_echo_signal, validate_ser= nearend_farend_mixer(validate_nearend_speech, validate_echo_signal,
                                                                                                    lowerbound_ser=-10, upperbound_ser=13, clipping_threshold=0.99)
        ser_list.append(validate_ser)
        nearend_data_path = os.path.join(nearend_speech_out, "nearend_speech_fileid_{}.wav".format(count))
        farend_data_path = os.path.join(farend_speech_out, "farend_speech_fileid_{}.wav".format(count))
        nearend_mic_data_path = os.path.join(nearend_mic_out, "nearend_mic_fileid_{}.wav".format(count))
        echo_data_path = os.path.join(echo_out, "echo_fileid_{}.wav".format(count))
        if use_reverb:
            audio_signals = [validate_nearend_speech2, validate_farend_speech, validate_nearend_mic, validate_echo_signal]
        else:
            audio_signals = [validate_nearend_speech2, validate_farend_speech, validate_nearend_mic, validate_echo_signal]
        file_paths = [nearend_data_path, farend_data_path, nearend_mic_data_path, echo_data_path]
        for k in range(len(audio_signals)):
            try:
                pass
                sf.write(file_paths[k], audio_signals[k], sample_rate)
            except Exception as e:
                print(str(e))

        noise_clipped_files += validate_noise_cf
        noise_source_files += validate_noise_file
        hyphen = '-'
        noise_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in validate_noise_file]

        noise_file_name.append(hyphen.join(noise_source_filenamesonly)[:50])
        farend_speech_name_1.append(validate_farend_path_sets[0])
        farend_speech_name_2.append(validate_farend_path_sets[1])
        farend_speech_name_3.append(validate_farend_path_sets[2])
        nearend_speech_name.append(validate_nearend_path)
        split.append('validate')
        filed_id.append(count)
        count += 1


    for i in range(len(train_dataset)):  ##TODO  为了方便就直接复制下来了,后期要改成函数
        data_pair = train_dataset[i]

        nearend_speech_path = data_pair[0]
        farend_speech_list = data_pair[1]
        nearend_speech, sr = sf.read(nearend_speech_path)
        assert sr == sample_rate
        nearend_speech = signal_pad(nearend_speech, audio_sample_length)
        for j in range(len(farend_speech_list)):

            three_farend_sets = farend_speech_list[j]

            farend_speech = np.concatenate([sf.read(wav)[0] for wav in three_farend_sets])
            #print(len(farend_speech), [len(sf.read(k)[0]) for k in three_farend_sets], np.array([len(sf.read(k)[0]) for k in three_farend_sets]).sum()) 检查是否被concate到一起了

            farend_speech = signal_pad(farend_speech, audio_sample_length)
            samples_rir_ch = get_rir_samples(rir_dict)
            reverb_farend, noreverb_farend = add_pyreverb(farend_speech, samples_rir_ch)
            #noise_sample, noise_sr = sf.read(noise_files[np.random.randint(0, np.size(noise_files))])
            noise_audio, noise_file, noise_cf = build_noise_audio(noise_dataset, fs=sample_rate, audio_length=audio_length, audio_samples_length=-1)


            snr = np.random.randint(-5, 20)
            snr_list.append(snr)
            if use_reverb:
                farend_snr, noise_snr, echo_signal, target_level = segmental_snr_mixer(clean=reverb_farend, noise=noise_audio, snr=snr)
            else:
                farend_snr, noise_snr, echo_signal, target_level = segmental_snr_mixer(clean=noreverb_farend, noise=noise_audio, snr=snr)

            nearend_mic, nearend_speech2, echo_signal, ser = nearend_farend_mixer(nearend_speech, echo_signal, lowerbound_ser=-10, upperbound_ser=13, clipping_threshold=0.99)
            ser_list.append(ser)
            #print("%%%%%%%%%%%Processing fileid%%%%%%%%%%%: {}".format(count))

            nearend_data_path = os.path.join(nearend_speech_out, "nearend_speech_fileid_{}.wav".format(count))
            farend_data_path = os.path.join(farend_speech_out, "farend_speech_fileid_{}.wav".format(count))
            nearend_mic_data_path = os.path.join(nearend_mic_out, "nearend_mic_fileid_{}.wav".format(count))
            echo_data_path = os.path.join(echo_out, "echo_fileid_{}.wav".format(count))
            if use_reverb:
                audio_signals = [nearend_speech2, farend_speech, nearend_mic, echo_signal]
            else:
                audio_signals = [nearend_speech2, noreverb_farend, nearend_mic, echo_signal]
            file_paths = [nearend_data_path, farend_data_path, nearend_mic_data_path, echo_data_path]
            for k in range(len(audio_signals)):
                try:

                    sf.write(file_paths[k], audio_signals[k], sample_rate)
                except Exception as e:
                    print(str(e))
            print("train = ", count)
            noise_clipped_files += noise_cf
            noise_source_files += noise_file
            hyphen = '-'
            noise_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in noise_file]

            noise_file_name.append(hyphen.join(noise_source_filenamesonly)[:50])
            farend_speech_name_1.append(three_farend_sets[0])
            farend_speech_name_2.append(three_farend_sets[1])
            farend_speech_name_3.append(three_farend_sets[2])
            nearend_speech_name.append(nearend_speech_path)
            filed_id.append(count)
            split.append('train')
            count += 1
    print(len(nearend_speech_name), len(farend_speech_name_1), len(farend_speech_name_2), len(farend_speech_name_3), len(noise_file_name), len(filed_id), len(snr_list), len(ser_list))
    dataFrame = pd.DataFrame({'nearend_speech_path': nearend_speech_name, 'farend_speech_path_1':farend_speech_name_1, 'farend_speech_path_2': farend_speech_name_2,
                              'farend_speech_path_3': farend_speech_name_3, 'noise_file_path':noise_file_name, 'filed_id':filed_id, 'snr':snr_list, 'ser':ser_list})
    dataFrame.to_csv(os.path.join(csv_path, 'train.csv'), index=False, sep=',')


def main():
    #TODO: This is a draft procesing script, will update and make it clean after
    #

    dataPath = "/home/yongyug/data/timit/TIMIT"
    noisePath = "/home/yongyug/data/aec_challenge/datasets/noise"                               ##noise from DNS-challenge
    outPath = "/home/yongyug/data/timit_aec_output"                                             ##OUTPUT path for saving the generated datasets
    rirPath = "/home/yongyug/data/aec_challenge/datasets/acoustic_params/RIR_table_simple.csv"  ##using rir from DNS-challenge datasets

    train_dataset, validate_dataset = get_data_pair(dataPath, repeat_enable=True, samle_gender_pair=30, diff_gender_pair=40)
    generate_pair_audio(train_dataset, validate_dataset, rirPath, noisePath, outPath, sample_rate=16000, audio_length=8)

    #output_audio, files_used, clipped_files = build_noise_audio(noisePath, fs=16000, audio_length=8, audio_samples_length=-1)
    #print(files_used)
    #print(output_audio)

main()
