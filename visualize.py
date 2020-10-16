import visdom
import h5py
import torch
import matplotlib as plt
#from metrics.evaluation_metrics import compute_all_metrics, EMD_CD

f = h5py.File("./results/image_prior_rnvp_with_whole_pc_2500_2500_predicting_segs_clouds.h5", "r")
#f = h5py.File("/srv/beegfs-benderdata/scratch/density_estimation/data/3DMultiView/ShapeNetCore_points_images/chair/saved_model_from_train_all_svr/recon_results/all_svr_model_train_2500_2048_clouds_predicting.h5", "r")
#gt_clouds = torch.from_numpy(f['gt_clouds'])
#sampled_clouds = torch.from_numpy(f['sampled_clouds'])
gt_clouds = torch.from_numpy(f['gt_clouds'][:])
sampled_clouds = torch.from_numpy(f['sampled_clouds'][:])
images = torch.from_numpy(f['images'][:])
print("data loaded.")

'''
gt_clouds = torch.transpose(gt_clouds, 1, 2).cuda()
sampled_clouds = torch.transpose(sampled_clouds, 1, 2).cuda()      # N * npoints * 3
cd, emd = EMD_CD(sampled_clouds, gt_clouds, batch_size=60, accelerated_cd=True, reduced=True)
print("cd: ", cd)
print("emd: ", emd)
res_output= compute_all_metrics(sampled_clouds, gt_clouds, batch_size=60, accelerated_cd=True)
print(res_output)
'''

vis = visdom.Visdom()
for i in range(len(sampled_clouds)):
    vis_sampled = torch.transpose(sampled_clouds[i], 0, 1)
    vis_gt = torch.transpose(gt_clouds[i], 0, 1)
    vis.scatter(vis_sampled)
    vis.scatter(vis_gt)
    vis_image = images[i]
    vis.image(vis_image)
    if i == 100:
        break