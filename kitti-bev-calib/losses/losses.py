import torch
import torch.nn as nn
import torch.nn.functional as F
from .quat_tools import quaternion_distance, batch_quat2mat, batch_tvector2mat, quaternion_from_matrix

class quat_norm_loss(nn.Module):
    def __init__(self):
        super(quat_norm_loss, self).__init__()
    
    def forward(self, pred_rotation):
        """
        Args:
            pred_rotation: (B, 4)
        Returns:
            loss: (1), rad
        """
        norm_square = torch.sum(pred_rotation ** 2, dim = 1) # (B,)
        loss = (norm_square - 1) ** 2 # (B,)
        return loss.mean()

class translation_loss(nn.Module):
    def __init__(self, l1 = True):
        super(translation_loss, self).__init__()
        self.l1 = l1
        if l1:
            self.criterion = nn.SmoothL1Loss(reduction='none')
        else:
            self.criterion = nn.MSELoss(reduction='none')
    
    def forward(self, pred_translation, gt_translation):
        """
        Args:
            pred_translation: (B, 3)
            gt_translation: (B, 3)
        """
        if self.l1:
            loss = self.criterion(pred_translation, gt_translation)
            loss = loss.sum(dim = 1).mean()
        else:
            loss = self.criterion(pred_translation, gt_translation)
            loss = loss.sum(dim = 1)
            loss = torch.sqrt(loss).mean()
        return loss
    
class rotation_loss(nn.Module):
    def __init__(self):
        super(rotation_loss, self).__init__()
    
    def forward(self, pred_rotation, gt_rotation):
        """
        Args:
            pred_rotation: (B, 3, 3)
            gt_rotation: (B, 3, 3)
        Returns:
            loss: (1), rad
        """
        B, _, _ = pred_rotation.shape
        pred_rot = torch.zeros(B, 4, device = pred_rotation.device)
        gt_rot = torch.zeros(B, 4, device = pred_rotation.device)
        for i in range(B):
            pred_rot[i] = quaternion_from_matrix(pred_rotation[i])
            gt_rot[i] = quaternion_from_matrix(gt_rotation[i])
        loss = quaternion_distance(pred_rot, gt_rot, device = pred_rot.device)
        return loss.mean()

class axis_rotation_loss(nn.Module):
    """Per-axis (Roll/Pitch/Yaw) rotation loss via Euler angle decomposition.
    Provides explicit supervision on each rotation axis independently,
    preventing the network from compensating errors across axes."""

    def __init__(self):
        super(axis_rotation_loss, self).__init__()

    @staticmethod
    def _rotation_matrix_to_euler(R):
        """Extract Roll (X), Pitch (Y), Yaw (Z) from rotation matrix.
        Uses ZYX convention: R = Rz @ Ry @ Rx.
        Args:
            R: (B, 3, 3)
        Returns:
            euler: (B, 3) — [roll, pitch, yaw] in radians
        """
        sy = torch.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
        singular = sy < 1e-6

        roll  = torch.where(singular, torch.atan2(-R[:, 1, 2], R[:, 1, 1]),
                            torch.atan2(R[:, 2, 1], R[:, 2, 2]))
        pitch = torch.where(singular, torch.atan2(-R[:, 2, 0], sy),
                            torch.atan2(-R[:, 2, 0], sy))
        yaw   = torch.where(singular, torch.zeros_like(sy),
                            torch.atan2(R[:, 1, 0], R[:, 0, 0]))
        return torch.stack([roll, pitch, yaw], dim=1)

    def forward(self, pred_rotation, gt_rotation):
        """
        Args:
            pred_rotation: (B, 3, 3)
            gt_rotation:   (B, 3, 3)
        Returns:
            loss: scalar (mean absolute Euler-angle error in radians)
            per_axis: dict with 'roll', 'pitch', 'yaw' losses (radians)
        """
        pred_euler = self._rotation_matrix_to_euler(pred_rotation)
        gt_euler   = self._rotation_matrix_to_euler(gt_rotation)

        diff = torch.abs(pred_euler - gt_euler)
        diff = torch.min(diff, 2 * torch.pi - diff)

        per_axis = {
            'roll':  diff[:, 0].mean(),
            'pitch': diff[:, 1].mean(),
            'yaw':   diff[:, 2].mean(),
        }
        loss = diff.mean()
        return loss, per_axis


class AdaptiveAxisRotationLoss(nn.Module):
    """Per-axis weighted mean rotation loss.

    Applies static (roll, pitch, yaw) weights to Euler-angle errors before
    averaging, emphasizing axes that matter more for the task (e.g. roll for
    BEV-based calibration).

    Args:
        axis_weights: (roll, pitch, yaw) static weights. Higher = more emphasis.
    """

    def __init__(self, axis_weights=(3.0, 1.5, 1.0)):
        super().__init__()
        self.register_buffer('axis_weights',
                             torch.tensor(axis_weights, dtype=torch.float32))

    @staticmethod
    def _rotation_matrix_to_euler(R):
        sy = torch.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
        singular = sy < 1e-6
        roll  = torch.where(singular, torch.atan2(-R[:, 1, 2], R[:, 1, 1]),
                            torch.atan2(R[:, 2, 1], R[:, 2, 2]))
        pitch = torch.where(singular, torch.atan2(-R[:, 2, 0], sy),
                            torch.atan2(-R[:, 2, 0], sy))
        yaw   = torch.where(singular, torch.zeros_like(sy),
                            torch.atan2(R[:, 1, 0], R[:, 0, 0]))
        return torch.stack([roll, pitch, yaw], dim=1)

    def forward(self, pred_rotation, gt_rotation):
        pred_euler = self._rotation_matrix_to_euler(pred_rotation)
        gt_euler   = self._rotation_matrix_to_euler(gt_rotation)

        diff = torch.abs(pred_euler - gt_euler)
        diff = torch.min(diff, 2 * torch.pi - diff)

        per_axis = {
            'roll':  diff[:, 0].mean(),
            'pitch': diff[:, 1].mean(),
            'yaw':   diff[:, 2].mean(),
        }

        w = self.axis_weights / self.axis_weights.sum() * 3.0
        weighted_diff = diff * w.unsqueeze(0)
        mean_loss = weighted_diff.mean()
        loss = mean_loss

        return loss, per_axis


class GeodesicRotationLoss(nn.Module):
    """SO(3) geodesic distance loss: arccos((tr(R_pred^T @ R_gt) - 1) / 2).

    Unlike quaternion distance, this operates directly on rotation matrices
    and provides well-behaved gradients even at sub-degree errors.
    """

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_rotation, gt_rotation):
        """
        Args:
            pred_rotation: (B, 3, 3)
            gt_rotation:   (B, 3, 3)
        Returns:
            loss: scalar (mean geodesic angle in radians)
        """
        R_diff = torch.bmm(pred_rotation.transpose(1, 2), gt_rotation)
        tr = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_angle = (tr - 1.0) / 2.0
        cos_angle = cos_angle.clamp(-1.0 + self.eps, 1.0 - self.eps)
        angle = torch.acos(cos_angle)
        return angle.mean()


class PC_reproj_loss(nn.Module):
    def __init__(self):
        super(PC_reproj_loss, self).__init__()
    
    def forward(self, pcs, gt_T_to_camera, pred_translation, pred_rotation, mask = None):
        """
        Args:
            pcs: (B, N, 3)
            gt_T_to_camera: (B, 4, 4)
            init_T_to_camera: (B, 4, 4)
            pred_translation: (B, 3), which is T_pred^{-1} * T_init
            pred_rotation: (B, 3, 3), which is T_pred^{-1} * T_init
            mask: (B, N), where N is the maximum number of points in a batch
        Returns:
            reproj_loss: ||T_gt^{-1} * T_pred^{-1} * T_init * pcs - pcs||_2
        """
        B, N, _ = pcs.shape
        loss = torch.tensor(0.0, device = pcs.device)
        for i in range(B):
            RT_gt = gt_T_to_camera[i]
            T_pred = torch.eye(4, device = pcs.device, dtype=pcs.dtype)
            T_pred[:3, :3] = pred_rotation[i]
            T_pred[:3, 3] = pred_translation[i]
            with torch.cuda.amp.autocast(enabled=False):
                RT_total = torch.matmul(
                    torch.linalg.inv(RT_gt.float()), T_pred.float()
                )
            pc = pcs[i]
            if mask is not None:
                pc = pc[mask[i] == 1]
            ones = torch.ones(pc.shape[0], 1, device=pc.device, dtype=pc.dtype)
            points_h = torch.cat([pc, ones], dim = 1) # (N, 4)
            points_transformed = torch.matmul(points_h, RT_total.t())[:, :3] # (N, 3)
            error = (points_transformed - pc).norm(dim = 1)
            loss += error.mean()
        
        return loss / B
            
class realworld_loss(nn.Module):
    def __init__(self, weight_translation = 1.0, weight_quat_norm = 0.5, weight_rotation = 0.5, weight_PCreproj = 0.5, 
                 weight_bev_reproj = 0.5, weight_feat_align = 1.0, l1 = False, rotation_only = False,
                 enable_axis_loss = False, weight_axis_rotation = 0.3,
                 axis_weights = (3.0, 1.5, 1.0),
                 use_geodesic_loss = False):
        super(realworld_loss, self).__init__()
        self.rotation_only = rotation_only
        self.enable_axis_loss = enable_axis_loss
        self.use_geodesic_loss = use_geodesic_loss
        if rotation_only:
            self.weight_translation = 0.0
            self.weight_rotation = weight_rotation * 2.0
            self.weight_PCreproj = weight_PCreproj * 2.0
            self.weight_quat_norm = weight_quat_norm
        else:
            self.weight_translation = weight_translation
            self.weight_rotation = weight_rotation
            self.weight_PCreproj = weight_PCreproj
            self.weight_quat_norm = weight_quat_norm
        self.weight_axis_rotation = weight_axis_rotation if enable_axis_loss else 0.0
        self.weight_bev_reproj = weight_bev_reproj
        self.weight_feat_align = weight_feat_align
        if not rotation_only:
            self.translation_loss = translation_loss(l1 = True)
            self.real_translation_loss = translation_loss(l1 = False)
        self.rotation_loss = rotation_loss()
        self.geodesic_loss = GeodesicRotationLoss()
        self.quat_norm_loss = quat_norm_loss()
        self.PC_reproj_loss = PC_reproj_loss()
        if enable_axis_loss:
            self.axis_rotation_loss = AdaptiveAxisRotationLoss(
                axis_weights=axis_weights,
            )
    
    def forward(self, pred_translation, pred_rotation, pcs, gt_T_to_camera, init_T_to_camera, mask = None):
        """
        Args:
            pcs: (B, N, 3)
            gt_T_to_camera: (B, 4, 4)
            init_T_to_camera: (B, 4, 4)
            pred_translation: (B, 3)
            pred_rotation: (B, 4)
            mask: (B, N), where N is the maximum number of points in a batch
        Expected:
            T_gt_expected = T_pred^{-1} * T_init
            T_pred = T_init * T_gt^{-1}
        Returns:
            total loss
        """
        T_pred = batch_tvector2mat(pred_translation)
        quat_norm_loss = self.quat_norm_loss(pred_rotation)
        R_pred = batch_quat2mat(pred_rotation)
        T_pred = torch.bmm(T_pred, R_pred) # (B, 4, 4)

        with torch.cuda.amp.autocast(enabled=False):
            T_gt_expected = torch.matmul(
                torch.linalg.inv(T_pred.float()), init_T_to_camera.float()
            )

        if self.rotation_only:
            # Rotation-only mode: SE(3) composition couples rotation into translation,
            # so override with init translation (== GT since no translation perturbation).
            T_gt_expected = T_gt_expected.clone()
            T_gt_expected[:, :3, 3] = init_T_to_camera[:, :3, 3]

        pred_translation = T_gt_expected[:, :3, 3]
        pred_rotation = T_gt_expected[:, :3, :3]
        gt_translation = gt_T_to_camera[:, :3, 3]
        gt_rotation = gt_T_to_camera[:, :3, :3]

        if not self.rotation_only:
            translation_loss = self.translation_loss(pred_translation, gt_translation)
            PC_reproj_loss = self.PC_reproj_loss(
                pcs, gt_T_to_camera, pred_translation, pred_rotation, mask)
        else:
            translation_loss = torch.tensor(0.0, device=pcs.device)
            # PC_reproj_loss: in rotation_only mode, use GT translation for both
            # pred and gt to eliminate translation coupling from the reprojection error.
            PC_reproj_loss = self.PC_reproj_loss(
                pcs, gt_T_to_camera, gt_translation, pred_rotation, mask)

        with torch.cuda.amp.autocast(enabled=False):
            quat_rotation_loss = self.rotation_loss(pred_rotation.float(), gt_rotation.float())
            if self.use_geodesic_loss:
                geo_loss = self.geodesic_loss(pred_rotation.float(), gt_rotation.float())
                rotation_loss = geo_loss
            else:
                rotation_loss = quat_rotation_loss

        loss = self.weight_translation * translation_loss \
                + self.weight_rotation * rotation_loss \
                + self.weight_PCreproj * PC_reproj_loss \
                + self.weight_quat_norm * quat_norm_loss

        axis_loss_val = torch.tensor(0.0, device=pcs.device)
        axis_details = {}
        if self.enable_axis_loss:
            with torch.cuda.amp.autocast(enabled=False):
                axis_loss_val, axis_details = self.axis_rotation_loss(
                    pred_rotation.float(), gt_rotation.float())
            loss = loss + self.weight_axis_rotation * axis_loss_val

        if not self.rotation_only:
            with torch.no_grad():
                real_trans_loss = self.real_translation_loss(pred_translation, gt_translation)
        else:
            real_trans_loss = torch.tensor(0.0, device=pcs.device)

        ret = {
            "total_loss" : loss,
            "translation_loss" : real_trans_loss, # l2, m
            "rotation_loss" : quat_rotation_loss / torch.pi * 180.0, # degree (always quat-based for display)
            "quat_norm_loss" : quat_norm_loss, 
            "PC_reproj_loss" : PC_reproj_loss,
        }
        if self.use_geodesic_loss:
            ret["geodesic_loss"] = geo_loss / torch.pi * 180.0
        if self.enable_axis_loss:
            ret["axis_rotation_loss"] = axis_loss_val / torch.pi * 180.0
            for k, v in axis_details.items():
                ret[f"axis_{k}_loss"] = v / torch.pi * 180.0
        return ret, T_gt_expected