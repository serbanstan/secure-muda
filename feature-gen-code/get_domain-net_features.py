import os
from multiprocessing import Process


dataset   = 'domain-net'
domains   = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
n_classes = 345
n_epochs  = 100

def run_cmd(src, trgt, gpu_id):
    cmd = "python main.py --dataset {} --dataset_path '../../data/{}' --model_name resnet50 --source {} --target {} --num_class {} --epoch {} --gpu {}".format(\
        dataset, dataset, src, trgt, n_classes, n_epochs, gpu_id)
    print(cmd)
    os.system(cmd)

    if not os.path.exists("../../data/pretrained-features/{}".format(dataset)):
        os.mkdir("../../data/pretrained-features/{}".format(dataset))
    for t in trgt.split(','):
        os.system("cp ./save_model_{}/{}_{}_resnet50_train.csv ../../data/pretrained-features/{}/{}_{}_train.csv".format(dataset, src, t, dataset, src, t))
        os.system("cp ./save_model_{}/{}_{}_resnet50_test.csv ../../data/pretrained-features/{}/{}_{}_test.csv".format(dataset, src, t, dataset, src, t))

def get_target_feature_for_fixed_src(src):
    gpu_id = domains.index(src) % 2

    target = ""

    for d in domains:
        target += d + ","
    target = target[:-1]

    run_cmd(src, target, gpu_id)


proc = []
for d1 in domains:
    # first four
    # if domains.index(d1) not in [0,1,2,3]:
    # last two
    if domains.index(d1) not in [4,5]:
        continue
    proc.append(Process(target=get_target_feature_for_fixed_src, args=([d1])))
    proc[-1].start()

for p in proc:
    p.join()


# python main.py --dataset domain-net --dataset_path '../../data/domain-net/' --model_name resnet50 --source clipart --target painting --num_class 345 --epoch 100 --gpu 0