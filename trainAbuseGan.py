import argparse
import os
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import json
import torch.nn as nn
import tensorboard_logger as tf_log
import logging

from utils import *
from src import AbuseGanTransform

cudnn.benchmark = True

if not os.path.isdir('./tmp'):
    os.mkdir('./tmp')


def configure_log(log_dir, task_name):
    log_path = os.path.join(log_dir, task_name + '.log')
    if not os.path.isdir(os.path.join(log_dir, 'tf_log/')):
        os.mkdir(os.path.join(log_dir, 'tf_log/'))
    tf_log_path = os.path.join(log_dir, 'tf_log/' + task_name)
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )
    tf_log.configure(tf_log_path)


def train_gan(config):
    set_random_seed(333)
    saveDir = "AbuseGanModel/"
    logDir = "AbuseGanLog/"
    resDir = "AbuseGanRes/"

    max_patient_epoch = 20
    if not os.path.isdir(saveDir):
        os.mkdir(saveDir)
    if not os.path.isdir(logDir):
        os.mkdir(logDir)
    if not os.path.isdir(resDir):
        os.mkdir(resDir)

    model_type, dataset_name, device, batch_size, \
    attack_norm, pert_lambda, max_iter \
        = load_configure(config)
    if attack_norm == 'inf':
        per_size_list = [0.03, 0.06, 0.09, 0.12, 0.15]
    else:
        per_size_list = [10, 15, 20, 25, 30]

    device = torch.device('cuda')

    task_name = model_type + '_' + dataset_name + '_' + attack_norm
    configure_log(logDir, task_name)

    model, trainSet, testSet = load_model_data(model_type, dataset_name)
    model = model.to(device).train()
    # model = nn.DataParallel(model)
    # model = model.cuda()

    print('----------   config  ---------------')
    logging.info('load model %s, load dataset %s' % (model_type, dataset_name))
    logging.info('----------   config  ---------------')
    for k in config:
        logging.info(k + ":" + str(config[k]))
    train_loader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testSet, batch_size=500, shuffle=False)

    for per_size in per_size_list:
        logging.info('----------   per_size: %.3f  ---------------' % per_size)
        logging.info('-------------------------------------------')

        energy_gan = AbuseGanTransform(
            model, image_nc=3, pert_lambda=pert_lambda, per_size=per_size,
            adv_opt=1, attack_norm=float(attack_norm), device=device
        )

        sub_dir = task_name + '_hyper' + str(pert_lambda) + '_' + str(per_size)
        sub_dir = os.path.join(saveDir, sub_dir)

        ori_metric = test_performance(test_loader, energy_gan.model, device, float(attack_norm), None)
        ori_res = ori_metric.dump('ori')
        print(ori_res)
        logging.info(ori_res)
        ori_mask = ori_metric.result['mask']['avg']
        best_mask = float('-inf')

        current_patient = 0
        best_adv_metric = None
        for i in range(max_iter):

            energy_gan.train(train_loader, 1, save_path=sub_dir)

            adv_metric = test_performance(test_loader, energy_gan.model, device, float(attack_norm), energy_gan)
            adv_res = adv_metric.dump('Attack ' + str(i))
            if adv_metric.result['mask']['avg'] > best_mask:
                best_mask = adv_metric.result['mask']['avg']
                best_adv_metric = adv_metric
                torch.save((energy_gan, adv_metric), os.path.join(sub_dir, 'best.tar'))
                logging.info('save best check point')
                print('save best check point')
                current_patient = 0
            else:
                current_patient += 1
            current_mask = adv_metric.result['mask']['avg']
            current_inc = (current_mask / ori_mask - 1) * 100
            best_inc = (best_mask / ori_mask - 1) * 100
            logging.info(adv_res)
            logging.info('current inc: %.2f, best inc %.2f' % (current_inc, best_inc))
            print(adv_res)
            print('current inc: %.2f, best inc %.2f' % (current_inc, best_inc))

            if current_patient > max_patient_epoch:
                logging.info('************* current epoch %d achieve best patient ****************' % i)
                print('************* current epoch %d achieve best patient ****************' % i)
                break
        save_file = os.path.join(resDir, task_name + '_hyper' + str(pert_lambda) + '_' + str(per_size) + '.res')
        torch.save([ori_metric, best_adv_metric], save_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/l2/cifar100_DeepShallow_l2.json',
                        help='configuration path')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configs = json.load(f)
    train_gan(configs)
