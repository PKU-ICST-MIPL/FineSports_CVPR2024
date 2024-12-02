# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
The code refers to https://github.com/facebookresearch/detr
"""
import torch
import torch.nn.functional as F
from torch import nn
from models.BLIP.models.blip_retrieval import blip_retrieval
from models.transformer.util import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, accuracy_sigmoid, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from models.BLIP.prompt_learner import PromptLearnerFuser
from models.backbone_builder import build_backbone
from models.detr.segmentation import (dice_loss, sigmoid_focal_loss)
from models.transformer.transformer import build_transformer
from models.transformer.transformer_layers import TransformerEncoderLayer, TransformerEncoder
from models.criterion import SetCriterion, PostProcess, SetCriterionAVA, PostProcessAVA, MLP
from einops import rearrange

class TextFuser(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        print('loading blip weights...')
        blip = blip_retrieval(pretrained=pretrained_path,
                               image_size=384, vit='large', 
                                vit_grad_ckpt=True, vit_ckpt_layer=10, 
                                queue_size=57600, negative_all_rank=False)
        self.tokenizer = blip.tokenizer
        self.text_encoder = blip.text_encoder
        print('load blip done')
        self.text_linear = nn.Linear(768, 2048)
        self.attn = nn.MultiheadAttention(embed_dim=2048, num_heads=8, batch_first=True)
        
        self.avgpool = torch.nn.AvgPool1d(35)
        
    def forward(self, video_feature, texts):
        text_embeds = []
        for text in texts:
            text_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt")
            text_output = self.text_encoder(text_input.input_ids.cuda(), attention_mask = text_input.attention_mask.cuda(), mode='text')
            text_embeds.append(text_output.last_hidden_state)
        text_embed = torch.cat(text_embeds, dim=0)
        text_embed = self.text_linear(text_embed)
        
        B, C, T, H, W = video_feature.shape
        video_fea_re = rearrange(video_feature, 'b c t h w->b (t h w) c')
        
        out, _ = self.attn(video_fea_re, text_embed, text_embed)
        
        output_fea = video_fea_re + out
        output = rearrange(output_fea, 'b (t h w) c->b c t h w', t=T,h=H,w=W)
        return output
        
        

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries,
                 hidden_dim, temporal_length, aux_loss=False, generate_lfb=False,
                 backbone_name='CSN-152', ds_rate=1, last_stride=True, dataset_mode='ava', text_fuser=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.temporal_length = temporal_length
        self.num_queries = num_queries
        self.transformer = transformer
        self.avg = nn.AvgPool3d(kernel_size=(temporal_length, 1, 1))
        self.dataset_mode = dataset_mode
        self.text_fuser = text_fuser

        if self.dataset_mode != 'ava':
            self.avg_s = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.query_embed = nn.Embedding(num_queries * temporal_length, hidden_dim)
        else:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
            
        if "SWIN" in backbone_name or 'I3D' in backbone_name:
            print("using swin or i3d")
            self.input_proj = nn.Conv3d(1024, hidden_dim, kernel_size=1)
            self.class_proj = nn.Conv3d(1024, hidden_dim, kernel_size=1)
        elif "SlowFast" in backbone_name:
            self.input_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)
            self.class_proj = nn.Conv3d(2048 + 512, hidden_dim, kernel_size=1)
        else:
            self.input_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)
            self.class_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)

        encoder_layer = TransformerEncoderLayer(hidden_dim, 8, 2048, 0.1, "relu", normalize_before=False)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=1, norm=None)
        self.cross_attn = nn.MultiheadAttention(256, num_heads=8, dropout=0.1)

        if self.dataset_mode == 'ava':
            self.class_embed_b = nn.Linear(hidden_dim, 3)
        else:
            if "SWIN" in backbone_name or 'I3D' in backbone_name:
                self.class_embed_b = nn.Linear(1024, 2)
            else:
                self.class_embed_b = nn.Linear(backbone.num_channels, 2)                
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        if self.dataset_mode == 'ava':
            self.class_fc = nn.Linear(hidden_dim, num_classes)
        else:
            self.class_fc = nn.Linear(hidden_dim, num_classes + 1)
        self.dropout = nn.Dropout(0.5)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.hidden_dim = hidden_dim
        self.is_swin = "SWIN" in backbone_name
        self.generate_lfb = generate_lfb
        self.last_stride = last_stride

    def freeze_params(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.query_embed.parameters():
            param.requires_grad = False
        for param in self.bbox_embed.parameters():
            param.requires_grad = False
        for param in self.input_proj.parameters():
            param.requires_grad = False
        for param in self.class_embed_b.parameters():
            param.requires_grad = False

    def forward(self, samples: NestedTensor, des, img_384, targets, stage='train'):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        """
        samples.tensors.shape: [BS, 3, 32, 224, 298]
        features.tensors.shape: [BS, 2048, 1, 14, 19]
        features.mask.shape: [BS, 1, 14, 19]
        pos.shape: [BS, 256, 4, 14, 19]
        xt.shape: [BS, 2048, 4, 14, 19]
        """
        labels = targets
        features, pos, xt = self.backbone(samples)
        src, mask = features[-1].decompose()
        if self.text_fuser is not None:
            src, label_pred, color_pred, number_pred = self.text_fuser(src, des, img_384, pos[-1], stage, labels)
        assert mask is not None

        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        if self.dataset_mode == 'ava':
            outputs_class_b = self.class_embed_b(hs)
        else:
            outputs_class_b = self.class_embed_b(self.avg_s(xt).squeeze(-1).squeeze(-1).squeeze(-1))
            outputs_class_b = outputs_class_b.unsqueeze(0).repeat(6, 1, 1)

        lay_n, bs, nb, dim = hs.shape

        src_c = self.class_proj(xt)

        hs_t_agg = hs.contiguous().view(lay_n, bs, 1, nb, dim)

        src_flatten = src_c.view(1, bs, self.hidden_dim, -1).repeat(lay_n, 1, 1, 1).view(lay_n * bs, self.hidden_dim, -1).permute(2, 0, 1).contiguous()
        if not self.is_swin:
            src_flatten, _ = self.encoder(src_flatten, orig_shape=src_c.shape)

        hs_query = hs_t_agg.view(lay_n * bs, nb, dim).permute(1, 0, 2).contiguous()
        q_class = self.cross_attn(hs_query, src_flatten, src_flatten)[0]
        q_class = q_class.permute(1, 0, 2).contiguous().view(lay_n, bs, nb, self.hidden_dim)

        outputs_class = self.class_fc(self.dropout(q_class))
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_logits_b': outputs_class_b[-1], 'label_pred': label_pred,
               'color_pred': color_pred, 'number_pred': number_pred}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_class_b)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_class_b):
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_logits_b': c}
                for a, b ,c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_b[:-1])]


def load_detr_weights(model, pretrain_dir, cfg):
    checkpoint = torch.load(pretrain_dir, map_location='cpu')
    model_dict = model.state_dict()

    pretrained_dict = {}
    for k, v in checkpoint['model'].items():
        if k.split('.')[1] == 'transformer':
            pretrained_dict.update({k: v})
        elif k.split('.')[1] == 'bbox_embed':
            pretrained_dict.update({k: v})
        elif k.split('.')[1] == 'query_embed':
            if not cfg.CONFIG.MODEL.SINGLE_FRAME:
                query_size = cfg.CONFIG.MODEL.QUERY_NUM * (cfg.CONFIG.MODEL.TEMP_LEN // cfg.CONFIG.MODEL.DS_RATE)
                pretrained_dict.update({k: v[:query_size].repeat(cfg.CONFIG.MODEL.DS_RATE, 1)})
            else:
                query_size = cfg.CONFIG.MODEL.QUERY_NUM
                pretrained_dict.update({k: v[:query_size]})
    if cfg.DDP_CONFIG.DISTRIBUTED:
        re_pretrained_dict = {k[7:]:v for k, v in pretrained_dict.items()}
    else:
        re_pretrained_dict = {k:v for k, v in pretrained_dict.items()}
    pretrained_dict_ = {k: v for k, v in re_pretrained_dict.items() if k in model_dict}
    unused_dict = {k: v for k, v in re_pretrained_dict.items() if not k in model_dict}

    if not (len(pretrained_dict_)==len(pretrained_dict)):
        raise ValueError('Not use pretrained models!!!')
        exit(-1)

    model_dict.update(pretrained_dict_)
    model.load_state_dict(model_dict)

    print('pretrain layers: {}, unused layers: {}'.format(len(pretrained_dict_), len(unused_dict)))
    print("load pretrain success")


def build_model(cfg):
    if cfg.CONFIG.DATA.DATASET_NAME == 'ava':
        from models.detr.matcher import build_matcher
    else:
        from models.detr.matcher_ucf import build_matcher
    num_classes = cfg.CONFIG.DATA.NUM_CLASSES
    print('num_classes', num_classes)

    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)

    if cfg.CONFIG.MODEL.USE_TEXT:
        text_fuser = PromptLearnerFuser(cfg.CONFIG.MODEL.PRETRAIN_BLIP)
    else:
        text_fuser = None

    model = DETR(backbone,
                 transformer,
                 num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                 num_queries=cfg.CONFIG.MODEL.QUERY_NUM,
                 aux_loss=cfg.CONFIG.TRAIN.AUX_LOSS,
                 hidden_dim=cfg.CONFIG.MODEL.D_MODEL,
                 temporal_length=cfg.CONFIG.MODEL.TEMP_LEN,
                 generate_lfb=cfg.CONFIG.MODEL.GENERATE_LFB,
                 backbone_name=cfg.CONFIG.MODEL.BACKBONE_NAME,
                 ds_rate=cfg.CONFIG.MODEL.DS_RATE,
                 last_stride=cfg.CONFIG.MODEL.LAST_STRIDE,
                 dataset_mode=cfg.CONFIG.DATA.DATASET_NAME,
                 text_fuser=text_fuser)
    load_detr_weights(model, cfg.CONFIG.MODEL.PRETRAIN_TRANSFORMER_DIR, cfg)
    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg.CONFIG.LOSS_COFS.DICE_COF, 'loss_bbox': cfg.CONFIG.LOSS_COFS.BBOX_COF}
    weight_dict['loss_giou'] = cfg.CONFIG.LOSS_COFS.GIOU_COF
    weight_dict['loss_ce_b'] = cfg.CONFIG.LOSS_COFS.BNY_COF


    if cfg.CONFIG.TRAIN.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.CONFIG.MODEL.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']

    if cfg.CONFIG.DATA.DATASET_NAME == 'ava':
        criterion = SetCriterionAVA(cfg.CONFIG.LOSS_COFS.WEIGHT,
                                    num_classes,
                                    num_queries=cfg.CONFIG.MODEL.QUERY_NUM,
                                    matcher=matcher, weight_dict=weight_dict,
                                    eos_coef=cfg.CONFIG.LOSS_COFS.EOS_COF,
                                    losses=losses,
                                    data_file=cfg.CONFIG.DATA.DATASET_NAME,
                                    evaluation=cfg.CONFIG.EVAL_ONLY)
    else:
        criterion = SetCriterion(cfg.CONFIG.LOSS_COFS.WEIGHT,
                        num_classes,
                        num_queries=cfg.CONFIG.MODEL.QUERY_NUM,
                        matcher=matcher, weight_dict=weight_dict,
                        eos_coef=cfg.CONFIG.LOSS_COFS.EOS_COF,
                        losses=losses,
                        data_file=cfg.CONFIG.DATA.DATASET_NAME,
                        evaluation=cfg.CONFIG.EVAL_ONLY)

    postprocessors = {'bbox': PostProcessAVA() if cfg.CONFIG.DATA.DATASET_NAME == 'ava' else PostProcess()}

    return model, criterion, postprocessors
