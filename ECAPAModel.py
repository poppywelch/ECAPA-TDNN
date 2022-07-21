'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN

from pathlib import Path
from aasist_utils import get_model, aasist_config, aasist_paths, set_seed, create_optimizer
from torchcontrib.optim import SWA
from aasist_evaluation import calculate_tDCF_EER

import json


class CombModel(nn.Module):
	def __init__(self, loader_len, lr, lr_decay, C , n_class, m, s, test_step, config, save_path, seed, eval, **kwargs):
		# super(ECAPAModel, self).__init__()
		super(CombModel, self).__init__()

		self.eval_mode = eval

		# AASIST
		# model configuration, optimizer configuration, device (cpu/gpu)
		self.config, self.aasist_model_config, self.aasist_optim_config, self.device = aasist_config(config, loader_len)
		set_seed(seed, self.config)
		# model tag name, path to save model to, path for evaluation score file, summary writer
		self.aasist_model_tag, self.aasist_model_save_path, self.aasist_eval_score_path, self.database_path = aasist_paths(save_path, self.config, config)

		self.aasist_metric_path = self.aasist_model_tag / "metrics"
		os.makedirs(self.aasist_metric_path, exist_ok=True)

		# AASIST model
		self.aasist_model = get_model(self.aasist_model_config, self.device)
		# AASIST optimizer and scheduler
		self.aasist_optim, self.aasist_scheduler = create_optimizer(self.aasist_model.parameters(), self.aasist_optim_config)
		self.aasist_optim_swa = SWA(self.aasist_optim)

		## ECAPA-TDNN
		self.speaker_encoder = ECAPA_TDNN(C = C).cuda() # IMPORT AASIST HERE TOO
		## Classifier
		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).cuda()

		self.ECAPA_optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.ECAPA_scheduler       = torch.optim.lr_scheduler.StepLR(self.ECAPA_optim, step_size = test_step, gamma=lr_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

	def train_network(self, epoch, loader):
		self.train()
		self.aasist_optim_config["steps_per_epoch"] = len(loader)
		## Update the learning rate based on the current epcoh
		self.ECAPA_scheduler.step(epoch - 1)
		asv_index, asv_top1, asv_loss = 0, 0, 0
		asv_lr = self.ECAPA_optim.param_groups[0]['lr']

		# cm_running_loss = 0
		cm_num_total, cm_ii = 0.0, 0
		cm_weight = torch.FloatTensor([0.1, 0.9]).to(device)
		cm_criterion = nn.CrossEntropyLoss(weight=cm_weight)

		joint_run_loss = 0

		for num, (asv, cm) in enumerate(loader, start=1): # (x_asv_inp, y_asv), (x_cm_inp, y_cm)
			self.zero_grad()

			cm_batch_size = cm[0].size(0)
			cm_num_total += cm_batch_size
			cm_ii += 1

			asv_batch_size = asv[0].size(0)
			asv_num_total += asv_batch_size

			asv_labels = torch.LongTensor(asv[1]).cuda()
			asv_spk_embedding = self.speaker_encoder.forward(asv[0].cuda(), aug=True)
			asv_nloss, asv_prec = self.speaker_loss.forward(asv_spk_embedding, asv_labels)
			asv_index += len(asv_labels)
			asv_top1 += asv_prec

			# asv_loss += nloss.detach().cpu().numpy()
			print(f"ASV LOSS: {asv_nloss}") # save

			cm_input = cm[0].to(self.device)
			cm_labels = cm[1].view(-1).type(torch.int64).to(self.device)
			_, cm_pred = self.aasist_model(cm_input, Freq_aug=str_to_bool(self.config["freq_aug"])) # what is freq_aug, IMPORT STR_TO_BOOL !!!!!
			cm_batch_loss = cm_criterion(cm_pred, cm_labels) # loss function (prediction, ground truth)
			# cm_running_loss += cm_batch_loss.item() * cm_batch_size
			print(f"CM LOSS: {cm_batch_loss}")

			joint_loss = asv_nloss + cm_batch_loss
			joint_loss.backward()
			joint_run_loss += joint_loss

			# step optimizer and scheduler
			self.ECAPA_optim.step()
			self.aasist_optim.step()

			if self.aasist_model_config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
				self.aasist_scheduler.step()
			elif self.aasist_scheduler is None:
				pass

			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, asv_lr, 100 * (num / loader.__len__())) + \
			" Loss: %.5f, ACC: %2.2f%% \r"        %(joint_loss/(num), asv_top1/asv_index*len(asv_labels)))
			sys.stderr.flush()
		sys.stdout.write("\n")

		# multiply by batch loss and divide by running total num
		return joint_run_loss/num, asv_lr, cm_lr, asv_top1/asv_index*len(asv_labels) #joint loss, asv lr, cm lr, asv precision


	def ECAPA_eval_network(self, asv_eval_list, eval_path):
		""" eval list = trials file for evaluation
			eval path = path to data files
		used to evaluate model, either using dev set or eval set depending on arguments passed when
		the function is called
		"""
		self.eval()
		files = []
		embeddings = {}
		lines = open(asv_eval_list).read().splitlines()
		for line in lines:
			print(f"line: {line}")
			files.append(line.split()[1])
			print(f"evaluation split line [1]: {line.split()[1]}")
			files.append(line.split()[2])
			print(f"evaluation split line [2]: {line.split()[2]}")
		setfiles = list(set(files))
		setfiles.sort()
		print(f"setfiles: {setfiles}")

		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			audio, _  = soundfile.read(os.path.join(eval_path, file))
			# Full utterance
			data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()

			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			for asf in startframe:
				feats.append(audio[int(asf):int(asf)+max_audio])
			feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			data_2 = torch.FloatTensor(feats).cuda()
			# Speaker embeddings
			with torch.no_grad():
				embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2 = self.speaker_encoder.forward(data_2, aug = False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			embeddings[file] = [embedding_1, embedding_2]
		scores, labels  = [], []

		for line in lines:			
			embedding_11, embedding_12 = embeddings[line.split()[1]]
			embedding_21, embedding_22 = embeddings[line.split()[2]]
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			score = (score_1 + score_2) / 2
			score = score.detach().cpu().numpy()
			scores.append(score)
			labels.append(int(line.split()[0]))
			
		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

		return EER, minDCF

	def AASIST_eval_network(self, best_eer, n_swa_update, epoch, cm_eval_list, eval_loader):
		""" best eer = returned by model and taken in each iteration
			n_swa_update = returned by model and taken in each following iteration
			epoch
			cm_eval_list = trials file for evaluation
			eval_loader = gives data for evaluation in correct format
		"""
		self.eval()

		# path to the default asv scores used for tdcf evaluation
		database_path = Path(self.config["database_path"])
		asv_score_file = database_path / self.config["asv_score_path"]

		use_type = "eval" if self.eval_mode else "dev"

		# gives asv score path and output path depending on evaluation using dev or final eval
		if self.eval_mode():
			score_save_path = self.aasist_metric_path/"eval_scores_using_best_dev_model.txt"
			output_file = self.aasist_model_tag / "t-DCF_EER.txt"
		else:
			score_save_path = self.aasist_metric_path/"dev_scores.txt"
			output_file = self.aasist_metric_path / f"dev_t-DCF_EER_{epoch:03d}epo.txt"

		# best_dev_eer = 1.
		# best_eval_eer = 100.
		# best_dev_tdcf = 0.05
		# best_eval_tdcf = 1.
		# n_swa_update = 0  # number of snapshots of model to use in SWA
		# f_log = open(model_tag / "metric_log.txt", "a")
		# f_log.write("=" * 5 + "\n")

		# open protocol file, run model, add filenames and predictions each to a list
		with open(cm_eval_list, "r") as f_trl:
			trial_lines = f_trl.readlines()
		fname_list = []
		score_list = []
		for _, (cm) in eval_loader: #cm[0] = input, cm[1]=label
			cm_inp = cm[0].to(self.device)
			with torch.no_grad():
				_, cm_pred = self.aasist_model(cm_inp)
				cm_batch_score = (cm_pred[:, 1]).data.cpu().numpy().ravel()
			fname_list.extend(cm[1])
			score_list.extend(cm_batch_score.tolist())
		# write out to the score file the utt id, the source (spooftype), bonafide/spoof and the score
		assert len(trial_lines) == len(fname_list) == len(score_list)
		with open(score_save_path, "w") as fh:
			for fn, sco, trl in zip(fname_list, score_list, trial_lines):
				_, utt_id, _, src, key = trl.strip().split(" ")
				assert fn == utt_id
				fh.write(f"{utt_id} {src} {key} {sco}")

		# calculate the evaluation metrics using the cm scores calculated above, default asv scores and specified output file
		# should it print out for dev ???
		eer, tdcf = calculate_tDCF_EER(cm_scores_file=score_save_path, asv_score_file=asv_score_file, output_file=output_file, printout=self.eval_mode)

		# if doing dev - DOES MAKING IT SELF MEAN IT'LL USE SAME IN EACH ITERATRION
		# if declared in init, wont it update back to 0 before final evaluation?

		if not self.eval_mode():
			if self.best_eer >= eer:
				self.best_eer = dev_eer
				self.aasist_optim_swa.update_swa()
				self.n_swa_update += 1

		# print(f"{use_type.upper()} DONE.\n Loss: {running_loss:.5f}, {use_type}_eer{eer: .5f}, {use_type}:{tdcf: .5f}")

		return eer, tdcf

	def do_swa_update(self, trn_loader):
		if self.n_swa_update > 0:
			self.aasist_optim_swa.swap_swa_sgd()
			self.aasist_optim_swa.bn_update(trn_loader, self.aasist_model, self.device)
			torch.save(model.state_dict(),
					   model_save_path / "swa.pth")


	def save_parameters(self, path):
		torch.save(self.state_dict(), path) # is this state_dict() doing for both? config file?

	# load separately first , save together then together

	def load_parameters(self, path):
		self_state = self.state_dict()
		print(f"getting self_state")
		print(self_state.keys())
		print(f"length self state {len(self_state)}")
		print(f"initial self_state done")
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)
		print("done self state")
		print(self_state.keys())
		print(f"length self state {len(self_state)}")
		print(f"done loading")


	def load_sep_state_dicts(self, aasist_config, ecapa_model):
		""" load in state dicts from two pre-trained subsystems, return one large state dict """
		comb_state_dict = {}
		ecapa_list = []
		aasist_list = []

		with open(aasist_config, "r") as f_json:
			aasist_config = json.loads(f_json.read())
		aasist_state_dict = torch.load(aasist_config["model_path"], map_location=self.device)

		ecapa_state_dict = torch.load(ecapa_model)

		for k, v in aasist_state_dict.items():
			comb_state_dict[k] = v
		for k, v in ecapa_state_dict.items():
			comb_state_dict[k] = v

		for name in ecapa_state_dict.keys():
			ecapa_list.append(name)
		for name in aasist_state_dict.keys():
			aasist_list.append(name)
		print(f"len ecapa {len(ecapa_list)}, len aasist {len(aasist_list)}")

		# ecapa_set = set(ecapa_list)
		# aasist_set = set(aasist_list)
		# print(f"len ecapa set {len(ecapa_set)}, len aasist set {len(aasist_set)}")
		# overlap = ecapa_set.intersection(aasist_set)
		# print(f"overlap: {overlap}")

		print(f" combined dict len {len(comb_state_dict)}")

