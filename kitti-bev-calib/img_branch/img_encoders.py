import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel

class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(len(self.in_channels) - 1):
            l_conv = nn.Sequential(
                nn.Conv2d(self.in_channels[i] + (self.in_channels[i + 1] if i == len(self.in_channels) - 2 else self.out_channels), self.out_channels, 1),
                nn.BatchNorm2d(self.out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),   
            )
            f_conv = nn.Sequential(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(self.out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(f_conv)

    def forward(self, inputs):
        """
        Args:
            inputs: list of feature maps, from high to low resolution e.g. [(H, W), (H/2, W/2), (H/4, W/4)]
        Returns:
            outs: list of feature maps, from high to low resolution e.g. [(H, W), (H/2, W/2)]
        """
        laterals = inputs
        for i in range(len(inputs) - 2, -1, -1):
            x = F.interpolate(laterals[i + 1], laterals[i].shape[-2:], mode='bilinear', align_corners=False)
            laterals[i] = torch.cat([laterals[i], x], 1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])
        
        outs = [laterals[i] for i in range(len(laterals) - 1)]
        return outs
    
class SwinT_tiny_Encoder(nn.Module):
    def __init__(self, output_indices, featureShape, out_channels, FPN_in_channels, FPN_out_channels):
        super(SwinT_tiny_Encoder, self).__init__()
        self.model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.output_indices = output_indices
        self.FPN = FPN(FPN_in_channels, FPN_out_channels)
        _, self.fH, self.fW = featureShape
        self.out_channels = out_channels
        self._max_used_stage = max(output_indices)
        self._freeze_unused_stages(output_indices)
    
    def _freeze_unused_stages(self, output_indices):
        """Remove SwinT stages whose outputs are never used by FPN.
        
        reshaped_hidden_states indices map: [0]=patch_embed, [i]=stage(i-1).downsample
        FPN reads indices in output_indices. With output_indices=[1,2,3]:
          - Stage 3 (encoder.layers.3) output is never consumed by FPN
          - model.layernorm on Stage 3 output is also unused
        
        Simply freezing (requires_grad=False) is insufficient for DDP
        find_unused_parameters=False in PyTorch 2.0.1 — DDP still tracks
        frozen params and raises RuntimeError if they don't participate in
        forward. Solution: replace unused stages with nn.Identity() to
        fully remove their parameters from the module tree.
        """
        max_used_idx = max(output_indices)
        num_stages = len(self.model.encoder.layers)
        removed_count = 0
        for stage_idx in range(max_used_idx, num_stages):
            removed_count += sum(1 for _ in self.model.encoder.layers[stage_idx].parameters())
            self.model.encoder.layers[stage_idx] = nn.Identity()
        if hasattr(self.model, 'layernorm'):
            removed_count += sum(1 for _ in self.model.layernorm.parameters())
            self.model.layernorm = nn.Identity()
        if removed_count > 0:
            print(f"[SwinT] Removed {removed_count} unused params (replaced with Identity), "
                  f"encoder.layers[{max_used_idx}:{num_stages}] + layernorm")

    def forward(self, x):
        """
        Args:
            x: (B, N, C, H, W), N is the number of images at the same time
        Returns:
            out: (B, N, out_channels, fH, fW), feature maps
        """
        imgs = x
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)

        embedding_output, input_dimensions = self.model.embeddings(imgs)
        all_reshaped = []
        hidden_states = embedding_output

        batch_size, _, hidden_size = hidden_states.shape
        reshaped = hidden_states.view(batch_size, *input_dimensions, hidden_size)
        reshaped = reshaped.permute(0, 3, 1, 2)
        all_reshaped.append(reshaped)

        for i in range(self._max_used_stage):
            layer_outputs = self.model.encoder.layers[i](
                hidden_states, input_dimensions, None, False, False
            )
            hidden_states = layer_outputs[0]
            output_dimensions = layer_outputs[2]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            batch_size, _, hidden_size = hidden_states.shape
            reshaped = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped = reshaped.permute(0, 3, 1, 2)
            all_reshaped.append(reshaped)

        ret = [all_reshaped[i] for i in self.output_indices]
        out = self.FPN(ret)
        return out[0].view(B, N, self.out_channels, self.fH, self.fW)