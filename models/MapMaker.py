import torch
import torch.nn as nn
import torch.nn.functional as F


class MapMaker(nn.Module):
    def __init__(
        self,
        image_size: int,
        num_layers: int = 3,
        num_classes: int = 2,
        temperature: float = 0.07,
        use_refine: bool = True,       
        refine_groups: int = 1,        
    ):
        super().__init__()
        self.image_size = image_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.tau = temperature

        self.use_refine = False

        self.layer_logits = nn.Parameter(torch.zeros(num_layers), requires_grad=False)

    def forward(self, vision_adapter_features, propmt_adapter_features):

        B = vision_adapter_features[0].shape[0]

        txt = F.normalize(propmt_adapter_features, dim=0) 

        per_layer_maps = [] 
        for v in vision_adapter_features:
            
            v = F.normalize(v, dim=-1)
            B_, H_i, W_i, C = v.shape
            assert B_ == B, "Batch mismatch."

           
            logits = (v.view(B, H_i * W_i, C) @ txt) / self.tau  # [B, H_i*W_i, num_classes]
            logits = logits.view(B, H_i, W_i, self.num_classes).permute(0, 3, 1, 2).contiguous()

            logits = F.interpolate(
                logits,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=True
            )

            per_layer_maps.append(logits)

        stacked = torch.stack(per_layer_maps, dim=0)

        fused = stacked.mean(dim=0)  

        probs = torch.softmax(fused, dim=1)

        weights = torch.full(
            (self.num_layers,),
            fill_value=1.0 / max(1, self.num_layers),
            dtype=stacked.dtype,
            device=stacked.device
        )

        return probs, weights



