import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import numpy as np
import json

class Evaluator:
    """
        Evaluator for model.
    """

    def __init__(self, hyper_params, logger):
        self.hyper_params = hyper_params
        self.Ks = hyper_params['Ks']
        self.logger = logger

        self.high_hr = [0.0 for _ in range(len(self.Ks))]
        self.high_ndcg = [0.0 for _ in range(len(self.Ks))]
        self.high_mrr = [0.0 for _ in range(len(self.Ks))]
        
        self.high_hr_client = [0.0 for _ in range(len(self.Ks))]
        self.high_ndcg_client = [0.0 for _ in range(len(self.Ks))]
        self.high_mrr_client = [0.0 for _ in range(len(self.Ks))]

    def generate_result_str(self, hr, ndcg, mrr):
        s = ''
        for i in range(len(self.Ks)):
            s += 'HR@' + str(self.Ks[i]) + ':' + str(hr[i]) + ' '
        s += '\n'

        for i in range(len(self.Ks)):
            s += 'NDCG@' + str(self.Ks[i]) + ':' + str(ndcg[i]) + ' '
        s += '\n'

        for i in range(len(self.Ks)):
            s += 'MRR@' + str(self.Ks[i]) + ':' + str(mrr[i]) + ' '
        return s

    def evaluate(self, net, contrast_adversary, adversary, dataloader, validate=True, evaluate_users=None):
        if validate:
            print('Start validating...')
            self.logger.info('Start validating...')
        else:
            print('Start checking...')
            self.logger.info('Start checking...')
        net.eval()
        contrast_adversary.eval()
        adversary.eval()
        with torch.no_grad():
            # Parameters
            checked_users = 0
            cur_hr = [0.0 for _ in range(len(self.Ks))]
            cur_ndcg = [0.0 for _ in range(len(self.Ks))]
            cur_mrr = [0.0 for _ in range(len(self.Ks))]

            for _, (batchx, batchy, padding, user_id, cur_cnt) in enumerate(tqdm(dataloader)):
                
                batchx = batchx.to(self.hyper_params['device'])
                batchy = batchy.to(self.hyper_params['device'])
                padding = padding.to(self.hyper_params['device'])
                user_id = user_id.to(self.hyper_params['device'])
                cur_cnt = cur_cnt.to(self.hyper_params['device'])

                # Forward.
                pred, _, _, _ = net(batchx)

                flag = False
                for i in range(batchy.shape[0]):
                    for j, k in enumerate(self.Ks):
                        best, now_at, dcg, hits = 0.0, 0.0, 0.0, 0.0
                        last_pred = pred[i, cur_cnt[i] - 1].scatter_(
                            dim=0, index=batchx[i], value=-10000000.0)
                        last_pred[0] = -10000000.0

                        rec_list = list(torch.argsort(
                            last_pred, descending=True)[:k].cpu().numpy())

                        first_correct = sys.maxsize
                        for movie in batchy[i]:
                            movie = movie.item()
                            if movie == 0:
                                break
                            now_at += 1.0
                            if now_at <= k:
                                best += 1.0 / float(np.log2(now_at + 1))

                            if movie not in rec_list:
                                continue

                            hits += 1.0
                            dcg += 1.0 / \
                                float(np.log2(float(rec_list.index(movie) + 2)))
                            first_correct = min(
                                first_correct, rec_list.index(movie))

                        cur_hr[j] += float(hits) / float(now_at)
                        cur_ndcg[j] += float(dcg) / float(best)
                        cur_mrr[j] += 1.0 / (first_correct + 1)

                    checked_users += 1
                    if evaluate_users is not None and checked_users >= evaluate_users:
                        flag = True
                        break

                if flag:
                    break

            for i in range(len(self.Ks)):
                cur_hr[i] /= checked_users
                cur_ndcg[i] /= checked_users
                cur_mrr[i] /= checked_users
                if validate:
                    self.high_hr[i] = max(self.high_hr[i], cur_hr[i])
                    self.high_ndcg[i] = max(self.high_ndcg[i], cur_ndcg[i])
                    self.high_mrr[i] = max(self.high_mrr[i], cur_mrr[i])

            cur_str = self.generate_result_str(cur_hr, cur_ndcg, cur_mrr)

            print('Global_cur_str:',cur_str)
            if validate:
                self.logger.info(
                    f'--------------------VALIDATE RESULT--------------------')
            else:
                self.logger.info(
                    f'----------------------TEST RESULT----------------------')
                # with open('fc_att_(x+y)_ML-1m_0812_30.json','a') as f:
                with open('fc_att_Fed_ML-1m_0918(print1)Global_23.json','a') as f:
                    s = json.dumps(cur_str) + '\n'
                    f.write(s)
            self.logger.info(cur_str)

            if validate:
                high_str = self.generate_result_str(
                    self.high_hr, self.high_ndcg, self.high_mrr)
                print("------------------Global HIGHEST RESULT---------------------")
                print('Global_high_str:',high_str)
                # with open('fc_att_(x+y)_ML-1m_0812_30.json','a') as f:
                with open('fc_att_Fed_ML-1m_0918(print1)Global_23.json','a') as f:
                    s = json.dumps(high_str) + '\n'
                    f.write(s)
                self.logger.info(
                    '------------------------HIGHEST RESULT----------------------')
                self.logger.info(high_str)

            return cur_hr, cur_ndcg, cur_mrr
    
       
    def evaluateClient(self, train_dict, z_cluster, net, contrast_adversary, adversary, net_weight_cluster, contrast_adversary_weight_cluster, adversary_weight_cluster, dataloader, validate=True, evaluate_users=None):
        if validate:
            print('Start client validating...')
            self.logger.info('Start client validating...')
        else:
            print('Start client checking...')
            self.logger.info('Start client checking...')
        net.eval()
        contrast_adversary.eval()
        adversary.eval()
        with torch.no_grad():
            # Parameters
            checked_users = 0
            cur_hr = [0.0 for _ in range(len(self.Ks))]
            cur_ndcg = [0.0 for _ in range(len(self.Ks))]
            cur_mrr = [0.0 for _ in range(len(self.Ks))]
            # print(net_weight_cluster)            
            for k in z_cluster.keys():
                net_local = net_weight_cluster[k]
                net.load_state_dict(net_local)
                # contrast_adversary.load_state_dict(contrast_adversary_weight_cluster[k])
                # adversary.load_state_dict(adversary_weight_cluster[k])
                for v in z_cluster[k]:                                      
                    batchx, batchy, padding, user_id, cur_cnt = train_dict[v]                    
                    batchx = batchx.unsqueeze(0)
                    batchy = batchy.unsqueeze(0)
                    padding = padding.unsqueeze(0)
                    user_id = torch.from_numpy(np.array(user_id)).unsqueeze(0)
                    cur_cnt = torch.from_numpy(np.array(cur_cnt)).unsqueeze(0)           
                    batchx = batchx.to(self.hyper_params['device'])   # print(batchx.shape)  [64,200]
                    batchy = batchy.to(self.hyper_params['device'])
                    padding = padding.to(self.hyper_params['device'])
                    user_id = user_id.to(self.hyper_params['device'])
                    cur_cnt = cur_cnt.to(self.hyper_params['device'])
                    
                    # Forward.
                    pred, _, _, _ = net(batchx)                    

                    flag = False
                    for i in range(batchy.shape[0]):
                        for j, k in enumerate(self.Ks):
                            best, now_at, dcg, hits = 0.0, 0.0, 0.0, 0.0
                            last_pred = pred[i, cur_cnt[i] - 1].scatter_(dim=0, index=batchx[i], value=-10000000.0)
                            last_pred[0] = -10000000.0

                            rec_list = list(torch.argsort(last_pred, descending=True)[:k].cpu().numpy())

                            first_correct = sys.maxsize
                            for movie in batchy[i]:
                                movie = movie.item()
                                if movie == 0:
                                    break
                                now_at += 1.0
                                if now_at <= k:
                                    best += 1.0 / float(np.log2(now_at + 1))

                                if movie not in rec_list:
                                    continue

                                hits += 1.0
                                dcg += 1.0 / \
                                    float(np.log2(float(rec_list.index(movie) + 2)))
                                first_correct = min(first_correct, rec_list.index(movie))

                            cur_hr[j] += float(hits) / float(now_at)
                            cur_ndcg[j] += float(dcg) / float(best)
                            cur_mrr[j] += 1.0 / (first_correct + 1)

                        checked_users += 1
                        if evaluate_users is not None and checked_users >= evaluate_users:
                            flag = True
                            break

                    if flag:
                        break

            for i in range(len(self.Ks)):
                cur_hr[i] /= checked_users
                cur_ndcg[i] /= checked_users
                cur_mrr[i] /= checked_users
                if validate:
                    self.high_hr_client[i] = max(self.high_hr_client[i], cur_hr[i])
                    self.high_ndcg_client[i] = max(self.high_ndcg_client[i], cur_ndcg[i])
                    self.high_mrr_client[i] = max(self.high_mrr_client[i], cur_mrr[i])

            cur_str = self.generate_result_str(cur_hr, cur_ndcg, cur_mrr)

            print('Client_cur_str:',cur_str)
            if validate:
                self.logger.info(
                    f'--------------------VALIDATE RESULT--------------------')
            else:
                self.logger.info(
                    f'----------------------TEST RESULT----------------------')
                # with open('fc_att_(x+y)_ML-1m_0812_30.json','a') as f:
                with open('fc_att_Fed_ML-1m_0918(print1)Client_23.json','a') as f:
                    s = json.dumps(cur_str) + '\n'
                    f.write(s)
            self.logger.info(cur_str)

            if validate:
                high_str = self.generate_result_str(
                    self.high_hr_client, self.high_ndcg_client, self.high_mrr_client)
                print("------------------Client HIGHEST RESULT---------------------")
                print('Client_high_str:',high_str)
                # with open('fc_att_(x+y)_ML-1m_0812_30.json','a') as f:
                with open('fc_att_Fed_ML-1m_0918(print1)Client_23.json','a') as f:
                    s = json.dumps(high_str) + '\n'
                    f.write(s)
                self.logger.info(
                    '------------------------HIGHEST RESULT----------------------')
                self.logger.info(high_str)

            return cur_hr, cur_ndcg, cur_mrr
