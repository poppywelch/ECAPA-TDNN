'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, os, torch, warnings, time
import numpy as np
from tools import *
from dataLoader import train_loader, LA_loader
from ECAPAModel import CombModel

# python trainECAPAModel.py --batch_size=20, --train_list /LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt --train_path /LA/ASVspoof2019_LA_trai
# /LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
# /LA/ASVspoof2019_LA_train

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
## ECAPA
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=400,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=4,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')
## AASIST
parser.add_argument('--config', default="config/AASIST.conf", dest="config", type=str, help="configuration file")


## Training and evaluation path/lists, save path
parser.add_argument('--train_list', type=str,   default="LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",     help='The path of the training list, eg:"/data08/VoxCeleb2/train_list.txt" in my case, which contains 1092009 lins')
parser.add_argument('--train_path', type=str,   default="LA/ASVspoof2019_LA_train/",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')

parser.add_argument('--asv_eval_list',  type=str,   default="LA/ASVspoof2019_LA_asv_protocols/dev_nospoof.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--cm_eval_list',  type=str,   default="LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--eval_path',  type=str,   default="LA/ASVspoof2019_LA_dev/",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')

parser.add_argument('--musan_path', type=str,   default="/data08/Others/musan_split",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--rir_path',   type=str,   default="/data08/Others/RIRS_NOISES/simulated_rirs",     help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case')

parser.add_argument('--save_path',  type=str,   default="exps/exp1",                                     help='Path to save the score.txt and models for ECAPA')
parser.add_argument('--initial_model',  type=str,   default="exps/pretrain.model",                                          help='Path of the initial_model')
## AASIST
# parser.add_argument('--output_dir', dest="output_dir", type=str, help="output directory for aasist results", default="./exp_result")

## Model and Loss settings
parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=5994,   help='Number of speakers')
## from AASIST
parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')

## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')



## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)
print("DONE INITIALIZATION")

## Define the data loader
# trainloader = train_loader(**vars(args))
trainloader = LA_loader(**vars(args))
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)
loader_len = len(trainLoader)
print("LOADED TRAIN LOADER")

evalloader = LA_loader(eval_pad=True, **vars(args))
evalLoader = torch.utils.data.DataLoader(evalloader, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)
print("LOADED EVAL LOADER")

## Search for the exist models
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()
print("SEARCHED FOR EXISTING MODEL")
## check state dict assist
s = CombModel(loader_len, **(vars(args)))
s.load_parameters(args.initial_model)
# s.load_sep_state_dicts(args.config, args.initial_model)
quit()

## Only do evaluation, the initial_model is necessary
if args.eval == True:
	s = CombModel(loader_len, **vars(args))
	print("Model %s loaded from previous state!"%args.initial_model)
	s.load_parameters(args.initial_model)

	ecapa_EER, ecapa_minDCF = s.ECAPA_eval_network(eval_list = args.asv_eval_list, eval_path = args.eval_path)
	aasist_EER, aasist_minDCF = s.aasist_eval_network(config=args.config, cm_eval_list=args.cm_eval_list, loader=evalLoader)

	print("ECAPA EER %2.2f%%, ECAPA minDCF %.4f%%"%(ecapa_EER, ecapa_minDCF))
	print("AASIST EER %2.2f%%, AASIST minDCF %.4f%%"%(aasist_EER, aasist_minDCF))

	quit()

## If initial_model is exist, system will train from the initial_model
if args.initial_model != "":
	print("Model %s loaded from previous state!"%args.initial_model)
	s = CombModel(loader_len, **vars(args))
	s.load_parameters(args.initial_model)
	epoch = 1

## Otherwise, system will try to start from the saved model&epoch
elif len(modelfiles) >= 1:
	print("Model %s loaded from previous state!"%modelfiles[-1])
	epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
	s = CombModel(loader_len, **vars(args))
	s.load_parameters(modelfiles[-1])
## Otherwise, system will train from scratch
else:
	epoch = 1
	s = CombModel(loader_len, **vars(args))
ecapa_EERs = []
aasist_EERs = []
ecapa_score_file = open(args.score_save_path, "a+")
best_aasist_tdcf = 0

while(1):
	## Training for one epoch
	joint_loss, asv_lr, cm_lr, asv_acc = s.train_network(epoch = epoch, loader = trainLoader)

	## Evaluation every [test_step] epochs
	if epoch % args.test_step == 0:
		s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)

		ecapa_EERs.append(s.ECAPA_eval_network(asv_eval_list = args.asv_eval_list, eval_path = args.eval_path)[0])
		aasist_EERs.append(s.aasist_eval_network(best_aasist_tdcf, config=args.config, cm_eval_list=args.cm_eval_list, loader=evalLoader)[0])

		print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"%(epoch, asv_acc, EERs[-1], min(EERs)))
		ecapa_score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"%(epoch, asv_lr, joint_loss, asv_acc, EERs[-1], min(EERs)))
		ecapa_score_file.flush()

	if epoch >= args.max_epoch:
		s.do_swa_update(trainLoader)
		quit()

	epoch += 1
