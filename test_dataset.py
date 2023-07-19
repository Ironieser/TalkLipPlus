
from fairseq import checkpoint_utils, utils, tasks
import sys
def readtext(path,label_proc):
    with open(path, "r") as f:
        trgt = f.readline()[7:]
    trgt = label_proc(trgt)
    return trgt
def get_avhubert(hubert_root, ckptpath):
    sys.path.append(hubert_root)
    from avhubert.hubert_pretraining import LabelEncoderS2SToken
    from fairseq.dataclass.utils import DictConfig
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckptpath])
    dictionaries = [task.target_dictionary]
    bpe_tokenizer = task.s2s_tokenizer
    # procs = []
    # for dictionary in dictionaries:
    #     temp = LabelEncoderS2SToken(dictionary, bpe_tokenizer)
    #     procs.append(temp)
    procs = [LabelEncoderS2SToken(dictionary, bpe_tokenizer) for dictionary in dictionaries]
    # procs1 = LabelEncoderS2SToken(task.target_dictionary, bpe_tokenizer)
    return procs[0]
hubert_root = r'/workdir/TalkLip/av_hubert'  # args.avhubert_root
hubert_ckpt = r'/workdir/TalkLip/ckpt/lip_reading_expert.pt'  # args.avhubert_path
text_root = r'/workdir/TalkLip/datalist/'
word_path = r'/workdir/TalkLip/datalist/text.txt'
label_proc = get_avhubert(hubert_root, hubert_ckpt)

# word_path = '{}/{}.txt'.format(text_root, sample)
trgt = readtext(word_path,label_proc)


def parse_filelist(file_list, save_root, check):

    with open(file_list) as f:
        lines = f.readlines()

    if check:
        sample_paths = []
        for line in lines:
            line = line.strip().split()[0]
            if not os.path.exists('{}/{}.mp4'.format(save_root, line)):
                sample_paths.append(line)
    else:
        sample_paths = [line.strip().split()[0] for line in lines]

    return sample_paths
_ = parse_filelist(r'/workdir/TalkLip/finetune_data/datalist/train.txt',None,False)

