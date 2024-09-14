import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torchvision.transforms as transforms
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from loguru import logger
import torch.nn.functional as F
from model_loader import load_model
from evaluate import mean_average_precision
from torch.nn import Parameter
from sklearn.cluster import KMeans

def train(train_s_dataloader,
          train_t_dataloader,
          query_dataloader,
          retrieval_dataloader,
          code_length,
          max_iter,
          arch,
          lr,
          device,
          verbose,
          topk,
          num_class,
          evaluate_interval,
          tag,
          training_source,
          training_target,
          num_features,
          max_iter_target,
          gpu_number,
          thresh,
          rho,
          ):
    if training_source:
        logger.info('Training on source starts')
        model = load_model(arch, code_length).to(device)
        parameter_list = model.parameters()
        optimizer = optim.SGD(parameter_list, lr=lr, momentum=0.9, weight_decay=1e-5)
        criterion_s = Source_Loss()
        source_labels = extract_source_labels(train_s_dataloader, num_class, verbose)
        S_s = torch.mm(source_labels, source_labels.t())
        S_s[S_s == 0] = -1
        S_s = S_s.to(device)
        model.train()
        for epoch in range(max_iter):
            for i, (data_s,  _, index) in enumerate(train_s_dataloader):
                data_s = data_s.to(device)
                optimizer.zero_grad()
                _, code_s = model(data_s)
                H_s = code_s @ code_s.t() / code_length
                source_targets = S_s[index, :][:, index]
                loss_s = criterion_s(H_s, source_targets)
                total_loss = loss_s
                total_loss.backward()
                optimizer.step()
            # Print log
            logger.info('[Epoch:{}/{}][loss:{:.4f}]'.format(epoch+1, max_iter, total_loss.item()))
            # Evaluate
            if epoch == 49:
                mAP = evaluate(model,
                                query_dataloader,
                                retrieval_dataloader,
                                code_length,
                                device,
                                topk,
                                save = False,
                                )
                logger.info('[iter:{}/{}][map:{:.4f}]'.format(
                    epoch+1,
                    max_iter,
                    mAP,
                ))
                if epoch == 49:
                    if not os.path.exists("./checkpoint"):
                        os.makedirs("./checkpoint")
                    torch.save({'iteration': epoch + 1,
                                'model_state_dict': model.state_dict(),
                                }, os.path.join('checkpoint', 'resume_{}.t'.format(epoch + 1)))
        # Evaluate and save
        mAP = evaluate(model,
                    query_dataloader,
                    retrieval_dataloader,
                    code_length,
                    device,
                    topk,
                    save=False,
                    )

        logger.info('Training on source finished, [iteration:{}][map:{:.4f}]'.format(epoch+1, mAP))

    if training_target:
        logger.info('Test-time adaptation on target starts')
        model_target = load_model(arch, code_length).to(device)
        saved_state_dict = torch.load('checkpoint/resume_{}.t'.format(50), map_location=torch.device("cuda:%d" % gpu_number))
        model_target.load_state_dict(saved_state_dict['model_state_dict'])
        initial_state_dict = {name: param.clone().detach() for name, param in model_target.named_parameters()}
        optimizer_target = optim.SGD(model_target.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
        logger.info('Construct pseudo labels')
        features0, train_codes = extract_features(model_target, train_t_dataloader,num_features, device,verbose,code_length)
        features_1 = F.normalize(features0, dim=-1).to(device)
        train_codes_1 = F.normalize(train_codes, dim=-1).to(device)
        p_labels, sim_p_begin, sim_t, original_centers = generate_labels_sim_p(features0,num_class)
        logger.info('done')

        logger.info('Construct the sets O and E, Calculate similarity probabilities P_ij')
        S1 = features_1 @ features_1.t()
        threshold_s1 = torch.kthvalue(S1.flatten(), int(S1.numel() * (1-thresh))).values
        Mask = (S1>=threshold_s1)*1.0 + (S1<threshold_s1)*0.0
        Mask = Mask.to(device)
        p_labels1 = p_labels.unsqueeze(0)
        p_labels2 = p_labels.unsqueeze(1)
        targets2 = (p_labels2 == p_labels1).float().squeeze(-1)
        targets2 = targets2.to(device)
        N = Mask.size(0)
        extended_similarity = torch.zeros((N, N)).to(device)
        weight_matrix = torch.zeros((N, N)).to(device)
        weight_matrix_neg = torch.zeros((N, N)).to(device)
        for i in range(N):
            for j in range(N):
                if Mask[i, j] == 1:
                    extended_similarity[i, j] = 1
                    weight_matrix[i, j] = 1
                    sim_i = torch.where(Mask[i] == 1)[0]
                    sim_j = torch.where(Mask[j] == 1)[0]
                    common_sim = len(set(sim_i.tolist()).intersection(sim_j.tolist()))
                    if common_sim > 0:
                        a = len(sim_i)
                        similarity = common_sim / a
                        extended_similarity[i, j] = 1
                        weight_matrix_neg[i, j] = similarity
                elif i != j and Mask[i, j] == 0:
                    sim_i = torch.where(Mask[i] == 1)[0]
                    sim_j = torch.where(Mask[j] == 1)[0]
                    common_sim = len(set(sim_i.tolist()).intersection(sim_j.tolist()))
                    if common_sim > 0:
                        a = len(sim_i)
                        similarity = common_sim / a
                        extended_similarity[i, j] = 1
                        weight_matrix[i, j] = similarity
                        weight_matrix_neg[i, j] = similarity
        targets_pos = (targets2.to(torch.bool) & extended_similarity.to(torch.bool)).to(torch.float)
        logger.info('done')
        logger.info('Test-time training starts')

        p_labels = p_labels.to(device)
        original_centers = original_centers.to(device)
        torch.autograd.set_detect_anomaly(True)
        model_target.train()
        indexs = torch.arange(features_1.size(0)).to(device)
        memory_bank = MemoryBank(features_1, train_codes_1, p_labels, indexs, original_centers)
        threshold_list = []
        for epoch in range(max_iter_target):
            running_loss = 0.0
            if (epoch == 0):
                mAP = evaluate(model_target,
                               query_dataloader,
                               retrieval_dataloader,
                               code_length,
                               device,
                               topk,
                               save=False,
                               )
                logger.info('[iter:{}/{}][map:{:.4f}]'.format(
                    epoch + 1,
                    max_iter_target,
                    mAP,
                ))
            for i, (data1,data2, _, index) in enumerate(train_t_dataloader):
                data1 = data1.to(device)
                index = index.to(device)
                optimizer_target.zero_grad()
                f,v = model_target(data1)
                pseudo_label = p_labels[index, :]
                r_percentile = 5
                cluster_center_feature = memory_bank.get_cluster_center_features()
                cluster_center_feature = cluster_center_feature.to(device)
                f1 = F.normalize(f)
                v1 = F.normalize(v)
                cluster_center_feature1 = cluster_center_feature.to(torch.float32)
                sim_t1 = torch.mm(f1, cluster_center_feature1.t())
                sim_t2 = sim_t1.detach()
                f2 = f1.detach()
                v2 = v1.detach()
                sim_p = F.softmax(sim_t2, dim=1)
                confident_mask, threshold_list = extract_confident_samples(sim_p, pseudo_label, r_percentile, threshold_list)

                memory_features,memory_codes, memory_labels, memory_indexs = memory_bank.get_all_samples()
                memory_codes = memory_codes.to(device)
                targets_p = targets_pos[index, :][:, memory_indexs]
                weights = weight_matrix[index, :][:, memory_indexs]
                weights_neg = weight_matrix_neg[index, :][:, memory_indexs]
                theta_exp = torch.exp(v1.mm(memory_codes.t()) / 0.1)
                the_frac = ((1-targets_p) * (1-weights_neg)  * theta_exp).sum(1).view(-1, 1) + 0.00001
                pro_loss = - (torch.log(theta_exp / the_frac) * targets_p *weights).sum() / targets_p.sum()

                quan_loss = - torch.mean(torch.sum(F.normalize(v, dim=-1) * F.normalize(torch.sign(v), dim=-1), dim=1))
                loss_target = pro_loss + quan_loss
                loss_target = separate_param(loss_target, model_target, optimizer_target, rho, initial_state_dict)

                running_loss += loss_target.item()
                confident_mask2 = torch.tensor(confident_mask).to(device)
                memory_bank.update_memory(f2, v2, pseudo_label,index, confident_mask2.unsqueeze(1))

            n_batch = len(train_t_dataloader)
            epoch_loss = running_loss / n_batch

            # Print log
            logger.info('[Epoch:{}/{}][loss:{:.4f}]'.format(epoch + 1, max_iter_target, epoch_loss))

            # Evaluate
            if (epoch % evaluate_interval == evaluate_interval - 1) or (epoch==4):
                mAP = evaluate(model_target,
                               query_dataloader,
                               retrieval_dataloader,
                               code_length,
                               device,
                               topk,
                               save=True,
                               )

                logger.info('[iter:{}/{}][map:{:.4f}]'.format(
                    epoch + 1,
                    max_iter_target,
                    mAP,
                ))
        # Evaluate and save
        mAP = evaluate(model_target,
                    query_dataloader,
                    retrieval_dataloader,
                    code_length,
                    device,
                    topk,
                    save=True,
                    )

        logger.info('Training on target finished, [iteration:{}][map:{:.4f}]'.format(epoch+1, mAP))

class MemoryBank:
    def __init__(self, train_features, train_codes, train_labels, indexs, cluster_center_features):
        self.max_size = len(train_features)
        self.feature_dim = train_features.size(1)
        self.memory = train_features.clone()
        self.memory_code = train_codes.clone()
        self.labels = train_labels.clone()
        self.indexs = indexs.clone()
        self.current_size = len(train_features)
        self.cluster_center_features = cluster_center_features.clone()

    def update_memory(self, features,codes, labels,indexes, confident_mask):
        for i in range(features.size(0)):
            if confident_mask[i]:
                self.memory[indexes[i]] = features[i]
                self.memory_code[indexes[i]] = codes[i]
                self.labels[indexes[i]] = labels[i]
                self.indexs[indexes[i]] = indexes[i]
        # Update cluster center features related to the new confident samples
        memory_label = self.labels.squeeze(-1)
        for label in labels[confident_mask].unique():
            self.cluster_center_features[label] = self.memory[memory_label == label].mean(dim=0)

    def get_all_samples(self):
        return self.memory[:self.current_size], self.memory_code[:self.current_size], self.labels[:self.current_size], self.indexs[:self.current_size]

    def get_cluster_center_features(self):
        return self.cluster_center_features


def evaluate(model,query_dataloader, retrieval_dataloader, code_length, device, topk, save):
    model.eval()
    # Generate hash code
    query_code = generate_code(model, query_dataloader, code_length, device)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)
    # One-hot encode targets
    onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
    onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)
    # Calculate mean average precision
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )
    if save:
        np.save("code/query_code_{}_mAP_{}".format(code_length, mAP), query_code.cpu().detach().numpy())
        np.save("code/retrieval_code_{}_mAP_{}".format(code_length, mAP), retrieval_code.cpu().detach().numpy())
        np.save("code/query_target_{}_mAP_{}".format(code_length, mAP), onehot_query_targets.cpu().detach().numpy())
        np.save("code/retrieval_target_{}_mAP_{}".format(code_length, mAP), onehot_retrieval_targets.cpu().detach().numpy())
    model.train()
    return mAP

def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.

    Returns
        code(torch.Tensor): Hash code.
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data,  _,index in dataloader:
            data = data.to(device)
            _,outputs= model(data)
            code[index, :] = outputs.sign().cpu()

    return code

def extract_features(model, dataloader, num_features, device, verbose, code_length):
    """
    Extract features.
    """
    model.eval()
    model.set_extract_features(True)
    features_1 = torch.zeros(dataloader.dataset.data.shape[0], num_features)
    continous_codes = torch.zeros(dataloader.dataset.data.shape[0], code_length)
    with torch.no_grad():
        N = len(dataloader)
        for i, (_,data_1,  _, index) in enumerate(dataloader):
            if verbose:
                logger.debug('[Batch:{}/{}]'.format(i + 1, N))
            data_1 = data_1.to(device)
            features_1[index, :], continous_codes[index, :] = model(data_1)[0].cpu(), model(data_1)[1].cpu()
    model.set_extract_features(False)
    model.train()
    return features_1, continous_codes

def extract_source_labels(dataloader, num_class, verbose):
    """
    Extract features.
    """
    labels = torch.zeros(dataloader.dataset.data.shape[0], num_class)
    with torch.no_grad():
        N = len(dataloader)
        for i, (_, label, index) in enumerate(dataloader):
            if verbose:
                logger.debug('[Batch:{}/{}]'.format(i+1, N))
            labels[index, :] = label.float()
    return labels


class Source_Loss(nn.Module):
    def __init__(self):
        super(Source_Loss, self).__init__()

    def forward(self, H, S):
        loss = ( S.abs() * (H - S).pow(2)).sum() / (H.shape[0] ** 2)
        return loss

def generate_labels_sim_p(features, Classes):
    features_norm = (features.T / np.linalg.norm(features, axis=1)).T
    kmeans = KMeans(n_clusters=Classes, random_state=0, init='k-means++').fit(features_norm)
    labels = kmeans.labels_.reshape(-1, 1)
    centers = kmeans.cluster_centers_
    centers = torch.tensor(centers)
    centers_norm = (centers.T / np.linalg.norm(centers, axis=1)).T
    features_norm_tensor = torch.tensor(features_norm, dtype=torch.float32)
    centers_norm_tensor = torch.tensor(centers_norm, dtype=torch.float32)
    sim_t = torch.exp(torch.mm(torch.FloatTensor(features_norm_tensor), torch.FloatTensor(centers_norm_tensor).t()))
    sim_p_clean_i = F.softmax(sim_t, dim=1)
    return torch.tensor(labels).int(), sim_p_clean_i, sim_t, centers_norm_tensor

def extract_confident_samples(sim_p_clean_i, labels, r_percentile, threshold_list):
    """
    Extract confident samples based on pseudo-labels and probability matrix.

    Args:
        sim_p_clean_i (torch.Tensor): Probability matrix (shape: num_samples x num_clusters).
        labels (torch.Tensor): Pseudo-labels (shape: num_samples).
        r_percentile (float): R-th percentile value (e.g., 5 for 5th percentile).

    Returns:
        torch.Tensor: Boolean mask indicating confident samples (shape: num_samples).
    """
    num_samples, num_clusters = sim_p_clean_i.shape
    labels = labels.squeeze().long()
    # Extract probabilities corresponding to pseudo-labels
    prob_for_labels = sim_p_clean_i[range(num_samples), labels]
    prob_for_labels = prob_for_labels.detach().cpu().numpy()
    # Calculate the R-th percentile threshold
    threshold = np.percentile(prob_for_labels, 100-r_percentile)
    threshold_list.append(threshold)
    if len(threshold_list) > 20:
        del threshold_list[0]
        threshold = sum(threshold_list)/20
    confident_mask = prob_for_labels >= threshold
    return confident_mask, threshold_list

def separate_param(loss_target_, net, optimizer, rho, initial_state_dict):
    net.train()
    loss_target_.backward()
    nonzero_ratio = rho
    to_concat_v = []
    to_concat_v_o = []
    to_concat_g = []
    for name, param in net.named_parameters():
        if param.dim() == 4:
            to_concat_v.append(param.data.view(-1))
            to_concat_g.append(param.grad.data.view(-1))
            to_concat_v_o.append(initial_state_dict[name].data.view(-1))

    all_v = torch.cat(to_concat_v)
    all_v_o = torch.cat(to_concat_v_o)
    all_g = torch.cat(to_concat_g)
    metric = torch.abs(all_v-all_v_o)
    metric2 = torch.abs(all_g * all_v)
    num_params = all_v.size(0)
    nz = int((1-nonzero_ratio) * num_params)
    top_values, _ = torch.topk(metric, nz)
    thresh = top_values[-1]
    nz2 = int(nonzero_ratio * num_params)
    top_values2, _ = torch.topk(metric2, nz2)
    thresh2 = top_values2[-1]
    for name, param in net.named_parameters():
        if param.dim() ==4:
            mask = (torch.abs(param.data - initial_state_dict[name].data) <= thresh).type(torch.cuda.FloatTensor)
            mask2 = (torch.abs(param.data * param.grad.data) >= thresh2).type(torch.cuda.FloatTensor)
            param.grad.data = mask * mask2 * param.grad.data
            param.grad.data.add_(1e-5 * torch.sign(param.data))
    optimizer.step()
    optimizer.zero_grad()
    return loss_target_


