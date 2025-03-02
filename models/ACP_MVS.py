import torch
import torch.nn as nn
import torch.nn.functional as F
from .CWF import *
from .CGA import CoarsestUpdateBlockDepth
from .module import *
from .GRU_regularization import BasicUpdateBlockDepth
from functools import partial

Align_Corners_Range = False


class CoarsestNet(nn.Module):
    def __init__(self):
        super(CoarsestNet, self).__init__()
        self.L_3DCNN = L_3DCNN(32, 1, kernel_size=3, stride=1, padding=1)
        self.context_feature_att = ContextAtt(1, 64)

    def forward(self, context, features, proj_matrices, depth_values, num_depth, cost_regularization):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(
            depth_values.shape[1], num_depth)
        num_views = len(features)

        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = torch.zeros_like(ref_volume)

        for src_fea, src_proj in zip(src_features, src_projs):

            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)

            if self.training:
                two_view_volume = (warped_volume - ref_volume) ** 2
                init_weight = self.L_3DCNN(two_view_volume)
                consti_weight = self.context_feature_att(init_weight, context.detach())
                cwf_volume = consti_weight * two_view_volume
                volume_sum = volume_sum + cwf_volume

            else:
                two_view_volume = (warped_volume - ref_volume).pow_(2)
                init_weight = self.L_3DCNN(two_view_volume)
                consti_weight = self.context_feature_att(init_weight, context.detach())
                cwf_volume = consti_weight * two_view_volume
                volume_sum = volume_sum + cwf_volume
            del warped_volume, two_view_volume, cwf_volume
        del ref_volume

        volume_variance = volume_sum.div_(num_views - 1)

        prob_volume_pre = cost_regularization(volume_variance).squeeze(1)

        prob_volume = F.softmax(prob_volume_pre, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)
        with torch.no_grad():
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            depth_index = depth_index.clamp(min=0, max=num_depth-1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
        return {"depth": depth,  "photometric_confidence": photometric_confidence}


def disp_to_depth(disp, min_depth, max_depth):


    min_disp = 1 / max_depth

    max_disp = 1 / min_depth

    scaled_disp = min_disp + (max_disp - min_disp) * disp

    scaled_disp = scaled_disp.clamp(min = 1e-4)
    depth = 1 / scaled_disp
    return scaled_disp, depth


def depth_to_disp(depth, min_depth, max_depth):

    scaled_disp = 1 / depth

    min_disp = 1 / max_depth

    max_disp = 1 / min_depth

    disp = (scaled_disp - min_disp) / ((max_disp - min_disp))

    return disp


def upsample_depth(depth, mask, ratio=8):
    """ Upsample depth field [H/ratio, W/ratio, 2] -> [H, W, 2] using convex combination """
    N, _, H, W = depth.shape
    mask = mask.view(N, 1, 9, ratio, ratio, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(depth, [3, 3], padding=1)
    up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, ratio * H, ratio * W)


class GetCost(nn.Module):
    def __init__(self):
        super(GetCost, self).__init__()

    def forward(self, depth_values, features, proj_matrices, depth_interval, depth_max, depth_min, CostNum=4):
        proj_matrices = torch.unbind(proj_matrices, 1)
        num_views = len(features)

        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        depth_values = 1./depth_values

        if CostNum > 1:
            depth_range_samples = get_depth_range_samples(cur_depth=depth_values.squeeze(1),
                                                          ndepth=CostNum,
                                                          depth_inteval_pixel=depth_interval.squeeze(1),
                                                          dtype=ref_feature[0].dtype,
                                                          device=ref_feature[0].device,
                                                          shape=[ref_feature.shape[0], ref_feature.shape[2],
                                                                 ref_feature.shape[3]],
                                                          max_depth=depth_max,
                                                          min_depth=depth_min)
        else:
            depth_range_samples = depth_values
        depth_range_samples = 1./depth_range_samples

        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, depth_range_samples.shape[1], 1, 1)
        volume_sum = torch.zeros_like(ref_volume)

        for src_fea, src_proj in zip(src_features, src_projs):
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_range_samples)
            if self.training:
                volume_sum = volume_sum + (warped_volume - ref_volume) ** 2
            else:
                volume_sum = volume_sum + (warped_volume - ref_volume).pow_(2)
            del warped_volume
        del ref_volume
        volume_variance = volume_sum.div_(num_views - 1)
        b,c,d,h,w = volume_variance.shape
        volume_variance = volume_variance.view(b, c*d, h, w)
        return volume_variance

class ACP_MVS(nn.Module):
    def __init__(self, args, depth_interals_ratio=[4, 2, 1], stage_channel=True):
        super(ACP_MVS, self).__init__()
        self.ndepths = args.ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.stage_channel = stage_channel
        seq_len = [int(e) for e in args.GRUiters.split(",")]
        self.seq_len = seq_len
        self.args = args
        self.num_stage = 3
        self.CostNum = args.CostNum
        self.GetCost = GetCost()
        self.hdim_stage = [64, 32, 16]
        self.cdim_stage = [64, 32, 16]
        self.context_feature = [128, 64, 32]
        self.feat_ratio = 2
        self.cost_dim_stage = [32, 16, 8]
        print("**********netphs:{}, depth_intervals_ratio:{}, hdim_stage:{}, cdim_stage:{}, context_feature:{}, cost_dim_stage:{}************".format(
            self.ndepths, depth_interals_ratio, self.hdim_stage, self.cdim_stage, self.context_feature, self.cost_dim_stage))
        self.featureNet = P_1to8_FeatureNet(base_channels=8, out_channel=self.cost_dim_stage, stage_channel=self.stage_channel)
        self.context_featureNet = P_1to8_FeatureNet(base_channels=8, out_channel=self.context_feature, stage_channel=self.stage_channel)
        self.gru_regularization1 = CoarsestUpdateBlockDepth(hidden_dim=self.hdim_stage[0], cost_dim=self.cost_dim_stage[0]*self.CostNum,
                                                        ratio=self.feat_ratio, context_dim=self.cdim_stage[0], UpMask=True)
        self.gru_regularization2 = BasicUpdateBlockDepth(hidden_dim=self.hdim_stage[1], cost_dim=self.cost_dim_stage[1]*self.CostNum,
                                                        ratio=self.feat_ratio, context_dim=self.cdim_stage[1], UpMask=True)

        self.gru_regularization3 = BasicUpdateBlockDepth(hidden_dim=self.hdim_stage[2], cost_dim=self.cost_dim_stage[2]*self.CostNum,
                                                        ratio=self.feat_ratio, context_dim=self.cdim_stage[2], UpMask=True)
        self.gru_regularization = nn.ModuleList([self.gru_regularization1, self.gru_regularization2, self.gru_regularization3])
        self.coarsestnet = CoarsestNet()
        self.cost_regularization = CostRegNet_small(in_channels=self.cost_dim_stage[0], base_channels=8)


    def forward(self, imgs, proj_matrices, depth_values):
        disp_min = depth_values[:, 0, None, None, None]
        disp_max = depth_values[:, -1, None, None, None]
        depth_max_ = 1. / disp_min
        depth_min_ = 1. / disp_max

        self.scale_inv_depth = partial(disp_to_depth, min_depth=depth_min_, max_depth=depth_max_)

        depth_interval = (disp_max - disp_min) / depth_values.size(1)
        features = []
        depth_predictions = []
        for nview_idx in range(imgs.size(1)):
            img = imgs[:, nview_idx]
            features.append(self.featureNet(img))
        context_features = self.context_featureNet(imgs[:, 0])
        for stage_idx in range(self.num_stage):

            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            ref_feature = features_stage[0]

            context_features_stage = context_features["stage{}".format(stage_idx + 1)]

            hidden_c, context = torch.split(context_features_stage, [self.hdim_stage[stage_idx], self.cdim_stage[stage_idx]], dim=1)

            current_hidden_d = torch.tanh(hidden_c)

            inp_c = torch.relu(context)

            if stage_idx == 0:
                depth_range_samples = get_depth_range_samples(cur_depth=depth_values,
                                                              ndepth=self.ndepths,
                                                              depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                                                              dtype=ref_feature[0].dtype,
                                                              device=ref_feature[0].device,
                                                              shape=[ref_feature.shape[0], ref_feature.shape[2],
                                                                     ref_feature.shape[3]],
                                                              max_depth=disp_max,
                                                              min_depth=disp_min)
                depth_range_samples = 1./depth_range_samples

                init_depth = self.coarsestnet(context, features_stage, proj_matrices_stage, depth_values=depth_range_samples,
                                           num_depth=self.ndepths, cost_regularization=self.cost_regularization)

                photometric_confidence = init_depth["photometric_confidence"]
                photometric_confidence = F.interpolate(photometric_confidence.unsqueeze(1),[ref_feature.shape[2]*8, ref_feature.shape[3]*8], mode='nearest')
                photometric_confidence = photometric_confidence.squeeze(1)
                init_depth = init_depth['depth']
                cur_depth = init_depth.unsqueeze(1)

                depth_predictions = [init_depth.squeeze(1)]
            else:
                cur_depth = depth_predictions[-1].unsqueeze(1)
            inv_cur_depth = depth_to_disp(cur_depth, depth_min_, depth_max_)
            
            depth_cost_func = partial(self.GetCost, features=features_stage, proj_matrices=proj_matrices_stage,
                                      depth_interval=depth_interval*self.depth_interals_ratio[stage_idx], depth_max=disp_max, depth_min=disp_min,
                                      CostNum=self.CostNum)

            current_hidden_d, up_mask_seqs, inv_depth_seqs = self.gru_regularization[stage_idx](current_hidden_d, depth_cost_func,
                                                                             inv_cur_depth,
                                                                             inp_c, seq_len=self.seq_len[stage_idx],
                                                                             scale_inv_depth=self.scale_inv_depth)

            for up_mask_i, inv_depth_i in zip(up_mask_seqs, inv_depth_seqs):
                depth_predictions.append(self.scale_inv_depth(inv_depth_i)[1].squeeze(1))
            last_mask = up_mask_seqs[-1]
            last_inv_depth = inv_depth_seqs[-1]
            inv_depth_up = upsample_depth(last_inv_depth, last_mask, ratio=self.feat_ratio).unsqueeze(1)
            final_depth = self.scale_inv_depth(inv_depth_up)[1].squeeze(1)
            depth_predictions.append(final_depth)

        return {"depth": depth_predictions, "photometric_confidence": photometric_confidence}