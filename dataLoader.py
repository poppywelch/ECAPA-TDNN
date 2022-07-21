'''
DataLoader for training
'''
import glob, os, random, soundfile, torch
import numpy as np
from scipy import signal
from torch import Tensor
import soundfile as sf


class train_loader(object):
	def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
		self.train_path = train_path
		self.num_frames = num_frames
		# Load and configure augmentation files
		self.noisetypes = ['noise','speech','music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-4] not in self.noiselist:
				self.noiselist[file.split('/')[-4]] = []
			self.noiselist[file.split('/')[-4]].append(file)
		self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
		# Load data & labels
		self.data_list  = []
		self.data_label = []
		lines = open(train_list).read().splitlines() #read in list (id and filepath per line, add to list)
		dictkeys = list(set([x.split()[0] for x in lines])) # have ids as a set -> dict keys
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys) } # set dict values as 0,1,2... using enumerate index (each speaker has own number as value)
		for index, line in enumerate(lines):
			speaker_label = dictkeys[line.split()[0]] # split original line to get speaker id label
			file_name     = os.path.join(train_path, line.split()[1]) # join the training path and the path for the wav file (e.g. id00012/21Uxsk56VDQ/00002.wav)
			self.data_label.append(speaker_label) # add the speaker ids to the data label
			self.data_list.append(file_name) # adds each wav filename to the data list

	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		audio, sr = soundfile.read(self.data_list[index]) # for each wav file: returns audio as numpy array for each file?
		length = self.num_frames * 160 + 240
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			audio = np.pad(audio, (0, shortage), 'wrap')
		start_frame = np.int64(random.random()*(audio.shape[0]-length))
		audio = audio[start_frame:start_frame + length]
		audio = np.stack([audio],axis=0)
		# Data Augmentation
		augtype = random.randint(0,5)
		if augtype == 0:   # Original
			audio = audio
		elif augtype == 1: # Reverberation
			audio = self.add_rev(audio)
		elif augtype == 2: # Babble
			audio = self.add_noise(audio, 'speech')
		elif augtype == 3: # Music
			audio = self.add_noise(audio, 'music')
		elif augtype == 4: # Noise
			audio = self.add_noise(audio, 'noise')
		elif augtype == 5: # Television noise
			audio = self.add_noise(audio, 'speech')
			audio = self.add_noise(audio, 'music')
		return torch.FloatTensor(audio[0]), self.data_label[index] # return audio files as data form, with corresponding speaker ids

	def __len__(self):
		return len(self.data_list) # number of wav files

	def add_rev(self, audio):
		rir_file    = random.choice(self.rir_files)
		rir, sr     = soundfile.read(rir_file)
		rir         = np.expand_dims(rir.astype(np.float),0)
		rir         = rir / np.sqrt(np.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]

	def add_noise(self, audio, noisecat):
		clean_db    = 10 * np.log10(np.mean(audio ** 2)+1e-4)
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []
		for noise in noiselist:
			noiseaudio, sr = soundfile.read(noise)
			length = self.num_frames * 160 + 240
			if noiseaudio.shape[0] <= length:
				shortage = length - noiseaudio.shape[0]
				noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
			start_frame = np.int64(random.random()*(noiseaudio.shape[0]-length))
			noiseaudio = noiseaudio[start_frame:start_frame + length]
			noiseaudio = np.stack([noiseaudio],axis=0)
			noise_db = 10 * np.log10(np.mean(noiseaudio ** 2)+1e-4)
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = np.sum(np.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio



class LA_loader(object):
	def __init__(self, train_list, train_path, cm_eval_list, eval_path, num_frames, eval_pad=False, **kwargs):
		self.d_meta_asv = {}
		self.d_meta_cm = {}
		self.eval_pad = eval_pad
		if self.eval_pad:
			self.data_list = train_list # path to protocol file
			self.data_path = train_path # path to audio files
		else:
			self.data_list = cm_eval_list
			self.data_path = eval_path
		self.track_ids = [] # list of track ids
		self.num_frames = num_frames
		self.cut = 64600
		lines = open(self.data_list).read().splitlines()
		for line in lines:
			spk_id, track_id, _, _, label = line.split(" ")
			if label == "bonafide":
				self.d_meta_asv[track_id] = spk_id
				self.track_ids.append(track_id)
			self.d_meta_cm[track_id] = 1 if label == "bonafide" else 0
		print(f"length of asv d_meta: {len(self.d_meta_asv)}, length of cm d_meta: {len(self.d_meta_cm)}\
			length of track ids: {len(self.track_ids)}")


	def __getitem__(self, index):
		length = self.num_frames * 160 + 240
		track_id = self.track_ids[index]
		X, _ = sf.read(str(self.data_path + f"/flac/{track_id}.flac"))
		if self.eval_pad: # take audio from start of clip, asv and cm take different lengths
			X_asv_pad = self.pad(X, length)
			X_cm_pad = self.pad(X, self.cut)
		else: # randomly select audio clip in training, diff lengths for asv and cm
			X_asv_pad = self.pad_random(X, length)
			X_cm_pad = self.pad_random(X, self.cut)
		x_asv_inp = Tensor(X_asv_pad)
		x_cm_inp = Tensor(X_cm_pad)
		y_asv = self.d_meta_asv[track_id]
		y_cm = self.d_meta_cm[track_id]
		print(f"ASV: x inp {x_inp.shape} and y {y_asv}")
		print(f"CM: x inp {x_inp.shape} and y {y_cm}")
		return (x_asv_inp, y_asv), (x_cm_inp, y_cm)

	def __len__(self):
		return len(self.track_ids) # number of files in data

	def pad_random(self, x: np.ndarray, max_len: int):
		x_len = x.shape[0]
		# if duration is already long enough
		if x_len >= max_len:
			stt = np.random.randint(x_len - max_len)
			return x[stt:stt + max_len]
		# if too short
		num_repeats = int(max_len / x_len) + 1
		padded_x = np.tile(x, (num_repeats))[:max_len]
		return padded_x

	def pad(self, x, max_len):
		x_len = x.shape[0]
		if x_len >= max_len:
			return x[:max_len]
		# need to pad
		num_repeats = int(max_len / x_len) + 1
		padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
		return padded_x













