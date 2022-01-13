from evaluator0918 import Evaluator
import json
import torch
import torch.utils.data as Data
import modelMe0918
import dataset_load0916
import numpy as np
import os
import logging
import traceback
import loss_func
import random
import time
import argparse
import sys
from tensorboardX import SummaryWriter
from Fed0918 import traditionalFedAvg, FedAvg
from sklearn.mixture import GaussianMixture


# Hyper Parameters
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

hyper_params = {
    # dataset to use (datasets/) [ml-1m, ml-latest, ml-10m] provided
    'dataset_path': 'ml-1m',
    # alpha
    'kl_weight': 0.05,
    # beta
    'contrast_weight': 0.1,
    # Total epochs
    'epochs': 400,
    # Set the number of users in evalutaion during training to speed up
    # If set to None, all of the users will be evaluated
    'evaluate_users': None,
    'item_embed_size': 128,
    'rnn_size': 100,
    'hidden_size': 100,
    'latent_size': 64,
    'timesteps': 5,
    'test_prop': 0.2,
    'batch_size': 1,
    # 'batch_size': 64,
    'anneal': False,
    'time_split': True,
    # 'model_func': 'fc_cnn',
    'model_func': 'fc_att',
    'add_eps': True,
    # 'device': 'cuda',
    'device': 'cpu',  # Add
    'check_freq': 1,
    'cluster_num': 8,  # GMM cluster
    'Ks': [1, 5, 10, 20, 50, 100],
    # 'lr_primal': 1e-3,
    # 'lr_dual': 3e-4,
    # 'lr_prior': 5e-4,
    'lr_primal': 0.5e-3,
    'lr_dual': 1.5e-4,
    'lr_prior': 2.5e-4,
    'l2_regular': 1e-2,
    'l2_adver': 1e-1,
    'total_step': 2000000,
    'total_users': 6034  #6034
}
dataset_info = json.load(open('./dataset_info.json', 'rt')
                         )[hyper_params['dataset_path']]
hyper_params['total_users'] = dataset_info[0]
hyper_params['total_items'] = dataset_info[1]
hyper_params['seq_len'] = dataset_info[2]

info_str = 'dataset:' + hyper_params['dataset_path'] + ' lr1:' + str(hyper_params["lr_primal"]) + ' lr2:' + str(
    hyper_params['lr_dual']) + ' kl:' + str(hyper_params['kl_weight']) + ' contrast:' + str(hyper_params['contrast_weight']) + ' batch:' + str(hyper_params['batch_size']) + ' model_func:' + hyper_params['model_func']
path_str = f'{hyper_params["model_func"]}_{hyper_params["dataset_path"]}_kl_{hyper_params["kl_weight"]}_contrast_{hyper_params["contrast_weight"]}_dropout_0.5_addeps_{hyper_params["add_eps"]}'


# Parser
parser = argparse.ArgumentParser(prog='train')
parser.add_argument("-m", "--msg", default="no description")
args = parser.parse_args()
train_msg = args.msg


# Setup Seed.
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1111)


# Config logging module.
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
local_time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
handler = logging.FileHandler(
    "model_log/log_" + local_time_str + '_' + train_msg.replace(' ', '_') + ".txt")

handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info(train_msg)
logger.info(info_str)
logger.info('Using CUDA:' + os.environ['CUDA_VISIBLE_DEVICES'])

# Accelerate with CuDNN.
# torch.backends.cudnn.benchmark = True

# Load data at first.
dataset_load0916.load_data(hyper_params)
user_dataset, train_dict = dataset_load0916.generate_train_data(hyper_params)
user_dataloader = Data.DataLoader(
    user_dataset, batch_size=hyper_params['batch_size'], shuffle=True)

test_dataset, _ = dataset_load0916.generate_test_data(hyper_params)
test_dataloader = Data.DataLoader(
    test_dataset, batch_size=hyper_params['batch_size'], shuffle=True)

# Generate validate dataset.
val_dataset, _ = dataset_load0916.generate_validate_data(hyper_params)
val_dataloader = Data.DataLoader(
    val_dataset, batch_size=hyper_params['batch_size'], shuffle=True)

# Build the model.
print('Building net...')
logger.info('Building net...')

net = modelMe0918.Model(hyper_params).to(hyper_params['device'])
adversary = modelMe0918.Adversary(hyper_params).to(hyper_params['device'])
contrast_adversary = modelMe0918.GRUAdversary(
    hyper_params).to(hyper_params['device'])

print(net)
print('Net build finished.')
logger.info('Net build finished.')

# ---------------------------------------Optimizer-----------------------------------------
optimizer_primal = torch.optim.AdamW([{
    'params': net.parameters(),
    'lr': hyper_params['lr_primal'],
    'weight_decay': hyper_params['l2_regular']}
])
optimizer_dual = torch.optim.SGD([{
    'params': net.encoder.parameters(),
    'lr': hyper_params['lr_dual'],
    'weight_decay': hyper_params['l2_adver']
}, {
    'params': contrast_adversary.parameters(),
    'lr': hyper_params['lr_dual'],
    'weight_decay': hyper_params['l2_adver']
}
])
optimizer_prior = torch.optim.SGD([{
    'params': adversary.parameters(),
    'lr': hyper_params['lr_prior'],
    'weight_decay': hyper_params['l2_adver']}
])


print('User datasets loaded and saved.')
logger.info('User datasets loaded and saved.')


# Evaluator
evaluator = Evaluator(hyper_params=hyper_params, logger=logger)


def train():
    writer = SummaryWriter(f'./runs/{path_str}')
    print('Start training...')
    logger.info('Start training...')

    global_step = 0
    mebank = loss_func.MetricShower()

    latent_z_inferred = {}  # key:0~6033
    
    net.train()
    net_global = net.state_dict()
    adversary_global = adversary.state_dict()
    contrast_adversary_global = contrast_adversary.state_dict()  
    net_locals_1 = []
    adversary_locals_1 = []
    contrast_adversary_locals_1 = []
    for i in range(hyper_params['total_users']):
        batchx, batchy, padding, user_id, cur_cnt = train_dict[i]
        batchx = batchx.unsqueeze(0)
        batchy = batchy.unsqueeze(0)
        padding = padding.unsqueeze(0)
        padding_z = (1.0 - padding.float()).unsqueeze(2)
        user_id = torch.from_numpy(np.array(user_id)).unsqueeze(0)
        cur_cnt = torch.from_numpy(np.array(cur_cnt)).unsqueeze(0)
        batchx = batchx.to(hyper_params['device'])   # print(batchx.shape)  [64,200]
        batchy = batchy.to(hyper_params['device'])
        padding = padding.to(hyper_params['device'])
        user_id = user_id.to(hyper_params['device'])
        cur_cnt = cur_cnt.to(hyper_params['device'])
        # Forward.
        optimizer_primal.zero_grad()
        optimizer_dual.zero_grad()
        pred, x_real, z_inferred, out_embed = net(batchx)
        
        latent_z_inferred[i] = z_inferred*padding_z  # z_inferred: [1, 200, 64]
        # --------------------------VAE---------------------------
        multi_loss = loss_func.vae_loss(pred, batchy, cur_cnt, padding, hyper_params)
        if hyper_params['anneal']:
            anneal = global_step / \
                    hyper_params['total_step'] * hyper_params['kl_weight']
        else:
            anneal = hyper_params['kl_weight']
        kl_loss = loss_func.kl_loss(adversary, x_real, z_inferred, padding, KL_WEIGHT=anneal)
        adver_loss = loss_func.adversary_loss(
                contrast_adversary, x_real, z_inferred, padding, CONTRAST_WEIGHT=hyper_params['contrast_weight'])
        # loss
        loss = multi_loss + adver_loss + kl_loss
        loss.backward()
        optimizer_primal.step()
        optimizer_dual.step()            
        net_local = net.state_dict()
        contrast_adversary_local = contrast_adversary.state_dict()            
        net_locals_1.append(net_local)
        contrast_adversary_locals_1.append(contrast_adversary_local)
        # --------------------------ADVER------------------------------
        optimizer_prior.zero_grad()
        adver_kl_loss = loss_func.adversary_kl_loss(
            adversary, x_real.detach(), z_inferred.detach(), padding)
        adver_kl_loss.backward()
        optimizer_prior.step()            
        adversary_local = adversary.state_dict()
        adversary_locals_1.append(adversary_local)
    net_global = traditionalFedAvg(net_locals_1)
    contrast_adversary_global = traditionalFedAvg(contrast_adversary_locals_1)     
    adversary_global = traditionalFedAvg(adversary_locals_1)
        
    net.load_state_dict(net_global)
    contrast_adversary.load_state_dict(contrast_adversary_global)
    adversary.load_state_dict(adversary_global)  
         
    # --------------------------GMM cluster------------------------------
    latent_z_tmp = sorted(latent_z_inferred.items(), key=lambda item:item[0])
    latent_z_ = {}
    for i in range(len(latent_z_tmp)):
        latent_z_[latent_z_tmp[i][0]] = latent_z_tmp[i][1]
    tmp = []  
    for i in latent_z_.keys():
        tmp.append(latent_z_[i].detach().numpy().tolist())
    latent_z_ = torch.tensor(tmp)   # [6034, 1, 200, 64] 
    latent_z = torch.mean(torch.squeeze(latent_z_), dim=1)  # [6034, 64] 
    gmm = GaussianMixture(n_components=hyper_params['cluster_num'], covariance_type='spherical', tol=0.001, init_params='random', random_state=3)
    gmm.fit(latent_z)
    cluster = gmm.predict(latent_z)
    z_cluster_tmp = []
    for i in range(len(cluster)):
        tmp = {}
        tmp[cluster[i]] = i   # {cluster K: client ID}
        z_cluster_tmp.append(tmp)   
    z_cluster_t = {}
    for _ in z_cluster_tmp:
        for k, v in _.items():
            z_cluster_t.setdefault(k, []).append(v)  # {cluster K: [client ID1, client ID2,...]}
    cluster_result_tmp = sorted(z_cluster_t.items(), key=lambda item:item[0])
    z_cluster = {}
    for i in range(len(cluster_result_tmp)):
        z_cluster[cluster_result_tmp[i][0]] = cluster_result_tmp[i][1]    
    # --------------------------train loop------------------------------
    for epoch in range(hyper_params['epochs']):
        net.train()
        
        net_global = net.state_dict()
        adversary_global = adversary.state_dict()
        contrast_adversary_global = contrast_adversary.state_dict()
        
        latent_x_inferred = {}  # key:0~6033
        
        net_weight_cluster = {}
        adversary_weight_cluster = {}
        contrast_adversary_weight_cluster = {}
        
        net_locals = []
        adversary_locals = []
        contrast_adversary_locals = []
        
        for k in z_cluster.keys():                                 
            vae_locals = []
            kl_locals = []
            adver_locals = []
            prior_locals = []
            # client train        
            for v in z_cluster[k]:  # v is client ID
                batchx, batchy, padding, user_id, cur_cnt = train_dict[v]            
                batchx = batchx.unsqueeze(0)
                batchy = batchy.unsqueeze(0)                
                padding = padding.unsqueeze(0)
                padding_z = (1.0 - padding.float()).unsqueeze(2)
                user_id = torch.from_numpy(np.array(user_id)).unsqueeze(0)
                cur_cnt = torch.from_numpy(np.array(cur_cnt)).unsqueeze(0)           
                batchx = batchx.to(hyper_params['device'])   # print(batchx.shape)  [64,200]
                batchy = batchy.to(hyper_params['device'])
                padding = padding.to(hyper_params['device'])
                user_id = user_id.to(hyper_params['device'])
                cur_cnt = cur_cnt.to(hyper_params['device'])
                # Forward.
                optimizer_primal.zero_grad()
                optimizer_dual.zero_grad()
                pred, x_real, z_inferred, out_embed = net(batchx)
                
                # Update z_inferred to server for GMM clustering.
                latent_x_inferred[v] = z_inferred*padding_z  # z_inferred: [1, 200, 64]
                # --------------------------VAE---------------------------
                multi_loss = loss_func.vae_loss(pred, batchy, cur_cnt, padding, hyper_params)
                if hyper_params['anneal']:
                    anneal = global_step / \
                        hyper_params['total_step'] * hyper_params['kl_weight']
                else:
                    anneal = hyper_params['kl_weight']
                kl_loss = loss_func.kl_loss(adversary, x_real, z_inferred, padding, KL_WEIGHT=anneal)
                adver_loss = loss_func.adversary_loss(
                    contrast_adversary, x_real, z_inferred, padding, CONTRAST_WEIGHT=hyper_params['contrast_weight'])
                loss = multi_loss + adver_loss + kl_loss
                loss.backward()
                optimizer_primal.step()
                optimizer_dual.step()            
                # --------------------------ADVER------------------------------
                optimizer_prior.zero_grad()
                adver_kl_loss = loss_func.adversary_kl_loss(adversary, x_real.detach(), z_inferred.detach(), padding)
                adver_kl_loss.backward()
                optimizer_prior.step()
            # Parameters of each cluster           
            net_local_cluster = net.state_dict()
            contrast_adversary_cluster = contrast_adversary.state_dict()
            adversary_cluster = adversary.state_dict()

            # net_weight_cluster[k] = net.load_state_dict(net_local_cluster)
            # adversary_weight_cluster[k] = adversary.load_state_dict(adversary_cluster)
            # contrast_adversary_weight_cluster[k] = contrast_adversary.load_state_dict(contrast_adversary_cluster)
        
            # update parameters of each cluster to server        
            net_locals.append(net_local_cluster)
            adversary_locals.append(adversary_cluster)
            contrast_adversary_locals.append(contrast_adversary_cluster)            
            vae_locals.append(multi_loss.item())
            kl_locals.append(kl_loss.item())
            adver_locals.append(adver_loss.item())
            prior_locals.append(adver_kl_loss.item())
            
            net_weight_cluster[k] = net_local_cluster
            adversary_weight_cluster[k] = adversary_cluster
            contrast_adversary_weight_cluster[k] = contrast_adversary_cluster
            
        # print(len(net_weight_cluster))  # 8
        # print(net_weight_cluster)

        # server aggregate different parameters     
        # net_global = FedAvg(net_locals, z_cluster)
        # contrast_adversary_global = FedAvg(contrast_adversary_locals, z_cluster)
        net_global = traditionalFedAvg(net_locals)
        contrast_adversary_global = traditionalFedAvg(contrast_adversary_locals)
        net.load_state_dict(net_global)
        contrast_adversary.load_state_dict(contrast_adversary_global)     
        # adversary_global = FedAvg(adversary_locals, z_cluster)
        adversary_global = traditionalFedAvg(adversary_locals)
        adversary.load_state_dict(adversary_global)
        
        # server compute current iterative loss 
        vae = np.array(vae_locals).mean()
        kl = np.array(kl_locals).mean()
        adver = np.array(adver_locals).mean()
        prior = np.array(prior_locals).mean()
        mebank.store({'vae': vae, 'kl': kl, 'adver': adver, 'prior': prior}) 
                
        global_step += 1   

        # Show Loss
        print(
            f'EPOCH:({epoch}/{hyper_params["epochs"]}),STEP:{global_step}/{hyper_params["total_step"]},vae:{mebank.get("vae")},kl:{mebank.get("kl")},contrast:{mebank.get("adver")},prior:{mebank.get("prior")}')
        logger.info(
            f'EPOCH:({epoch}/{hyper_params["epochs"]}),STEP:{global_step}/{hyper_params["total_step"]},vae:{mebank.get("vae")},kl:{mebank.get("kl")},contrast:{mebank.get("adver")},prior:{mebank.get("prior")}')

        writer.add_scalar('loss', mebank.get("vae"), global_step=epoch)
        writer.flush()

        mebank.clear()

        # Check (These codes are just to monitor the results of training)
        # Here val_dataloader and test_dataloader are the same
        if epoch % hyper_params['check_freq'] == 0:
            hr, _, _ = evaluator.evaluate(net, contrast_adversary, adversary, dataloader=val_dataloader,
                                          validate=True, evaluate_users=hyper_params['evaluate_users'])
            writer.add_scalar('hr10', hr[2], global_step=epoch)
            writer.flush()

            hr, _, _ = evaluator.evaluateClient(train_dict,z_cluster,net,contrast_adversary, adversary, net_weight_cluster,contrast_adversary_weight_cluster, adversary_weight_cluster,dataloader=val_dataloader,validate=True, evaluate_users=hyper_params['evaluate_users'])
            net.train()
            adversary.train()

        if global_step >= hyper_params['total_step']:
            break
              
        # --------------------------GMM cluster------------------------------
        latent_z_t = sorted(latent_x_inferred.items(), key=lambda item:item[0])
        latent_zz = {}
        for i in range(len(latent_z_t)):
            latent_zz[latent_z_t[i][0]] = latent_z_t[i][1]
        tmpp = []  
        for i in latent_zz.keys():
            tmpp.append(latent_zz[i].detach().numpy().tolist())
        latent_zz = torch.tensor(tmpp)   # [6034, 1, 200, 64] 
        latent_zz = torch.mean(torch.squeeze(latent_zz), dim=1)  # [6034, 64] 
        gmm = GaussianMixture(n_components=hyper_params['cluster_num'], covariance_type='spherical', tol=0.001, init_params='random', random_state=3)
        gmm.fit(latent_zz)
        cluster = gmm.predict(latent_zz)
        z_cluster_t = []
        for i in range(len(cluster)):
            tmp = {}
            tmp[cluster[i]] = i   # {cluster K: client ID}
            z_cluster_t.append(tmp)   
        z_each_cluster_t = {}
        for _ in z_cluster_t:
            for k, v in _.items():
                z_each_cluster_t.setdefault(k, []).append(v)  # {cluster K: [client ID1, client ID2,...]}
        z_each_cluster_tmp = sorted(z_each_cluster_t.items(), key=lambda item:item[0])
        z_each_cluster = {}
        for i in range(len(z_each_cluster_tmp)):
            z_each_cluster[z_each_cluster_tmp[i][0]] = z_each_cluster_tmp[i][1]                   
        z_cluster = z_each_cluster

    evaluator.evaluate(net, contrast_adversary, adversary, dataloader=test_dataloader,
                       validate=False)
    writer.close()


# Main
if __name__ == '__main__':
    # Train the model.
    try:
        train()
        logger.info('Finished.')
    except Exception as err:
        err_info = traceback.format_exc()
        print(err_info)
        logger.info(err_info)
        logger.info('Error.')
