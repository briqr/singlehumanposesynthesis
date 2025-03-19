
# evaluation with respect to the validation set

# generate SMPL shapes based on the captions from the validation set, then calculate their distance
# to the GT SMPL shapes in the UV atlas space.To do the matching between two samples from the generated sample and the GT
# sample, we match them by solving the assignment problem
import torch
from hungarian import *

hungarian = Hungarian()

# the size of gt samples should look like N*p*res*res*3, and of generated samples M*p*res*res*3
def calc_min_distance(gt_samples, generated_samples, eval_embeddings):
    total_dist = 0
    total_embed_dist = 0
    for gen_smp_key, gen_smp in generated_samples.items():
        min_dist = 1000000
        min_im_id = -1
        for gt_smp_key, gt_smp in gt_samples.items():
            pairwise_dist = calc_pairwise_dist(gen_smp, gt_smp)
            assignment = calc_assignment(pairwise_dist.cpu())
            current_assignment_cost = pairwise_dist.cpu()[assignment[:, 0].cpu(), assignment[:, 1].cpu()].sum()
            current_assignment_cost /= len(assignment)
            if current_assignment_cost < min_dist:
                min_dist = current_assignment_cost
                min_im_id = gt_smp_key
        min_embed = torch.norm(eval_embeddings[min_im_id] - eval_embeddings[gen_smp_key], dim=1)[0]
        total_embed_dist += min_embed
        total_dist += min_dist
    total_embed_dist /= len(generated_samples)
    total_dist /= len(generated_samples)
    return total_dist, total_embed_dist

def calc_min_distance_param(gt_samples, generated_samples, eval_embeddings):
    total_dist = 0
    total_embed_dist = 0
    for gen_smp_key, gen_smp in generated_samples.items():
        min_dist = 1000000
        min_im_id = -1
        for gt_smp_key, gt_smp in gt_samples.items():
            pairwise_dist = calc_pairwise_dist_param(gen_smp, gt_smp)
            assignment = calc_assignment(pairwise_dist.cpu())
            current_assignment_cost = pairwise_dist.cpu()[assignment[:, 0].cpu(), assignment[:, 1].cpu()].sum()
            current_assignment_cost /= len(assignment)
            if current_assignment_cost < min_dist:
                min_dist = current_assignment_cost
                min_im_id = gt_smp_key
        min_embed = torch.norm(eval_embeddings[min_im_id] - eval_embeddings[gen_smp_key], dim=1)[0]
        total_embed_dist += min_embed
        total_dist += min_dist
    total_embed_dist /= len(generated_samples)
    total_dist /= len(generated_samples)
    return total_dist, total_embed_dist

def calc_distance_to_gt(gt_samples, generated_samples):
    total_dist = 0
    for gen_smp_key, gen_smp in generated_samples.items():
        gt_smp = gt_samples[gen_smp_key]
        pairwise_dist = calc_pairwise_dist(gen_smp, gt_smp)
        assignment = calc_assignment(pairwise_dist.cpu())
        current_assignment_cost = pairwise_dist.cpu()[assignment[:, 0].cpu(), assignment[:, 1].cpu()].sum()
        current_assignment_cost /= len(assignment)
        total_dist += current_assignment_cost
    total_dist /= len(generated_samples)
    return total_dist

def calc_distance_to_gt_param(gt_samples, generated_samples):
    total_dist = 0
    for gen_smp_key, gen_smp in generated_samples.items():
        gt_smp = gt_samples[gen_smp_key]
        pairwise_dist = calc_pairwise_dist_param(gen_smp, gt_smp)
        assignment = calc_assignment(pairwise_dist.cpu())
        current_assignment_cost = pairwise_dist.cpu()[assignment[:, 0].cpu(), assignment[:, 1].cpu()].sum()
        current_assignment_cost /= len(assignment)
        total_dist += current_assignment_cost
    total_dist /= len(generated_samples)
    return total_dist

def calc_distance_all_pairs(gt_samples, generated_samples):
    total_dist = 0
    num_comp = 0
    for gen_smp in generated_samples:
        for gt_smp in gt_samples:
            pairwise_dist = calc_pairwise_dist(gen_smp, gt_smp)
            assignment = calc_assignment(pairwise_dist.cpu())
            current_assignment_cost = pairwise_dist.cpu()[assignment[:, 0].cpu(), assignment[:, 1].cpu()].sum()
            current_assignment_cost /= len(assignment)
            total_dist += current_assignment_cost
            num_comp += 1
    #total_dist /= len(generated_samples)
    total_dist /= num_comp
    return total_dist

def calc_distance_all_pairs_param(gt_samples, generated_samples):
    total_dist = 0
    num_comp = 0
    for gen_smp in generated_samples:
        for gt_smp in gt_samples:
            pairwise_dist = calc_pairwise_dist_param(gen_smp, gt_smp)
            assignment = calc_assignment(pairwise_dist.cpu())
            current_assignment_cost = pairwise_dist.cpu()[assignment[:, 0].cpu(), assignment[:, 1].cpu()].sum()
            current_assignment_cost /= len(assignment)
            total_dist += current_assignment_cost
            num_comp += 1
    #total_dist /= len(generated_samples)
    total_dist /= num_comp
    return total_dist
# def calc_min_distance(out_smp, gt_smp):
#     for l in range(len(out_smp)):
#         for k in range(len(gt_smp)):
#             pairwise_dist = calc_pairwise_dist(out_smp[l], gt_smp[k])
#             assignment = calc_assignment(pairwise_dist)
#             assignment_cost = pairwise_dist[assignment].sum()


def calc_pairwise_dist(hm_out, hm_gt):
    nu_persons = hm_gt.shape[0]
    nu_persons_hat = len(hm_out)
    nu_joints = hm_gt.shape[1]
    hm_out = hm_out.view(nu_persons_hat, hm_out[0].shape[0], hm_out[0].shape[1], hm_out[0].shape[2])
    hm_gt = hm_gt.unsqueeze(1).expand(-1,nu_persons_hat, -1, -1, -1)
    hm_out = hm_out.unsqueeze(0).expand(nu_persons,-1, -1, -1, -1)
    hm_out = hm_out.contiguous().view(nu_persons,  nu_persons_hat, nu_joints, hm_out.shape[3]* hm_out.shape[4])
    hm_gt = hm_gt.contiguous().view(nu_persons, nu_persons_hat, nu_joints, hm_gt.shape[3]* hm_gt.shape[4])
    dist = torch.norm(hm_gt-hm_out,dim=-1).mean(dim=2)
    return dist

def calc_pairwise_dist_param(hm_out, hm_gt):
    nu_persons = hm_gt.shape[0]
    nu_persons_hat = len(hm_out)
    nu_joints = hm_gt.shape[1]
    hm_out = hm_out.view(nu_persons_hat, hm_out[0].shape[0])
    hm_gt = hm_gt.unsqueeze(1).expand(-1,nu_persons_hat, -1)
    hm_out = hm_out.unsqueeze(0).expand(nu_persons,-1, -1,)
    hm_out = hm_out.contiguous().view(nu_persons,  nu_persons_hat, nu_joints)
    hm_gt = hm_gt.contiguous().view(nu_persons, nu_persons_hat, nu_joints)
    dist = torch.norm(hm_gt-hm_out,dim=-1)#.mean(dim=-1)
    return dist
def calc_assignment(pairwise_dist):
    #print(pairwise_dist.shape)
    nu_persons = max(pairwise_dist.shape[0], pairwise_dist.shape[1])
    hungarian.calculate(pairwise_dist)
    res = hungarian.get_results()
    if res is None:
        return None
    assignments = torch.from_numpy(np.asarray(res))
    return assignments


def calc_embed_pairwise_dist(all_embeddings):
    all_embeddings = list(all_embeddings)
    total_dist = 0
    num_comparison = 0
    for i in range(len(all_embeddings)):
        for j in range(i+1, len(all_embeddings)):
            current_dist = torch.norm(all_embeddings[i] - all_embeddings[j], dim=1)[0]
            total_dist += current_dist
            num_comparison += 1
    average_dist = total_dist/num_comparison
    return average_dist
