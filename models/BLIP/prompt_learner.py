from models.BLIP.models.blip_retrieval import blip_retrieval
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from models.transformer.position_encoding import build_position_encoding
from utils.misc import NestedTensor

label2class = { 0: 'Dribble Move',
                1: 'Drives Left',
                2: 'Drives Right',
                3: 'Drives Straight',
                4: 'Drive Baseline',
                5: 'Drive Middle',
                6: 'Post Up',
                7: 'Basket',
                8: 'To Basket',
                9: 'Free Throw',
                10: 'Jumper',
                11: 'Early Jumper',
                12: 'To Jumper',
                13: 'Takes Early Jump Shot',
                14: 'Dribble Jumper',
                15: 'No Dribble Jumper',
                16: 'Offensive Rebound',
                17: 'High P&R',
                18: 'Left P&R',
                19: 'Right P&R',
                20: 'Hand Off',
                21: 'Ball Delivered',
                22: "To Shooter's Left",
                23: "To Shooter's Right"
            }
color2color = {
    idx: v for idx, v in enumerate([
        "black",
        "white",
        "red",
        "blue",
        "yellow",
        "green"
    ])
}
number2number = {
    idx: v for idx, v in enumerate(["23",
        "30",
        "0",
        "35",
        "5",
        "11",
        "6",
        "8",
        "32",
        "34",
        "26",
        "22",
        "3",
        "13",
        "1",
        "2",
        "7",
        "9",
        "10",
        "17",
        "16",
        "55",
        "24",
        "43",
        "4",
        "81",
        "42",
        "41",
        "33",
        "21",
        "25",
        "18",
        "12",
        "36",
        "46",
        "15",
        "14",
        "37",
        "28",
        "44",
        "27",
        "45",
        "99"])
}
label2length = [4, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 4, 4, 5, 2, 4, 4, 4, 2, 2, 5, 5]
text_embdding_length = 5


class PromptLearnerFuser(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        model = blip_retrieval(pretrained=pretrained_path, image_size=384, vit='large', 
                             vit_grad_ckpt=True, vit_ckpt_layer=10, 
                             queue_size=57600, negative_all_rank=False, custimize=True)
        for param in model.parameters():
            param.requires_grad = False
            
        self.tokenizer = model.tokenizer
        self.embedding = model.text_encoder.embeddings.word_embeddings
        self.text_encoder = model.text_encoder
        self.image_encoder = model.visual_encoder
        
        self.text_linear = nn.Linear(768, 2048)
        self.input_linear = nn.Linear(2048,768)
        self.cls_net = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 24)
        )
        self.cls_color = nn.Sequential(
            nn.Linear(768, 256), #2048
            nn.ReLU(),
            nn.Linear(256, 6)
        )
        self.cls_number = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 43)
        )

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        self.positional_embed = build_position_encoding(2048)
        self.attn = nn.MultiheadAttention(embed_dim=2048, num_heads=8, batch_first=True)              
        
        # add SOS, EOS token
        tmp_text = 'hello'
        tmp_text_input = self.tokenizer(tmp_text, padding='max_length', truncation=True, max_length=35, return_tensors="pt")
        with torch.no_grad():
            tmp_text_output = self.embedding(tmp_text_input['input_ids'])
        self.register_buffer("token_prefix", tmp_text_output[:, 0, :])  # CLS TOKEN
        self.register_buffer("token_suffix", tmp_text_output[:, 2, :])  # SEP TOKEN
        
        # get color, number prompt
        import json
        with open('./colors_numbers.json', 'r') as f:
            color_number = json.load(f)
        
        for color in color_number['colors']:
            text = color
            text_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt")
            with torch.no_grad():
                text_output = self.embedding(text_input['input_ids'])
            color_embedding = text_output[:,1,:]
            self.register_buffer("token_color_"+color, color_embedding)
            
        for number in color_number['numbers']:
            text = number
            text_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt")
            with torch.no_grad():
                text_output = self.embedding(text_input['input_ids'])
            number_embedding = text_output[:,1,:]
            self.register_buffer("token_number_"+number, number_embedding)
            
        for label, class_ in label2class.items():
            text = class_
            length = label2length[label]
            text_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt")
            with torch.no_grad():
                text_output = self.embedding(text_input['input_ids'])
            cls_embedding = torch.cat((text_output[:,1:1+length,:], text_output[:,-1-(5-length):-1,:]), dim=1)
            self.register_buffer("token_cls_"+str(label), cls_embedding)
            
        # init cts_vector
        text = 'a player wearing a jersey number'
        self.ctx_length = len(text.split())
        text_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt")
        with torch.no_grad():
            text_output = self.embedding(text_input['input_ids'])
        ctx_vectors = text_output[:,1:1+self.ctx_length,:]
        self.ctx = nn.Parameter(ctx_vectors)
        self.color_idx = 4
        self.number_idx = 6
        
        # get position embedding
        position_ids = model.text_encoder.embeddings.position_ids[:, :self.ctx_length+2+2+text_embdding_length] # plus 2 for SOS, EOS
        self.position_embeddings = model.text_encoder.embeddings.position_embeddings(position_ids)
        self.attention_mask = torch.ones(1,self.ctx_length+2+2+text_embdding_length)
        
        # frozen
        self.attention_mask.requires_grad = False
        self.ctx.requires_grad = True
        
    def to_cuda(self, device):
        # Move model parameters to the device
        self.to(device)
        print(device)

        # Move registered buffers to the device
        self.token_prefix = self.token_prefix.to(device)
        self.token_suffix = self.token_suffix.to(device)

        # Move color and number tokens to the device
        for name, buffer in self.named_buffers():
            if name.startswith("token_color_") or name.startswith("token_number_") or name.startswith('token_cls_'):
                setattr(self, name, buffer.to(device))

        # Move context vectors and other parameters to the device
        self.ctx = self.ctx.to(device)

        # Move position embeddings and attention mask to the device
        self.position_embeddings = self.position_embeddings.to(device)
        self.attention_mask = self.attention_mask.to(device)

    
    def get_text_embeddings(self, text, label, stage, color_pred=None, number_pred=None):
        words = text.split()
        if stage == 'train':
            color, number = words[4], words[-1]
            color_embedding = getattr(self, "token_color_"+color)
            number_embedding = getattr(self, "token_number_"+number)
            cls_embedding = getattr(self, 'token_cls_'+str(label))
            prefix = self.token_prefix 
            suffix = self.token_suffix
        
            
            text_embedding = torch.cat([prefix.unsqueeze(0), self.ctx[:, :self.color_idx], 
                                        color_embedding.unsqueeze(0), self.ctx[:, self.color_idx:self.number_idx],
                                        number_embedding.unsqueeze(0), self.ctx[:, self.number_idx:],
                                        cls_embedding,
                                        suffix.unsqueeze(0)], dim=1)
        else:
            color_embedding = color_pred
            number_embedding = number_pred
            cls_embedding = getattr(self, 'token_cls_'+str(label))
            prefix = self.token_prefix 
            suffix = self.token_suffix
        
            
            text_embedding = torch.cat([prefix.unsqueeze(0), self.ctx[:, :self.color_idx], 
                                        color_embedding.unsqueeze(0).unsqueeze(0), self.ctx[:, self.color_idx:self.number_idx],
                                        number_embedding.unsqueeze(0).unsqueeze(0), self.ctx[:, self.number_idx:],
                                        cls_embedding,
                                        suffix.unsqueeze(0)], dim=1)
        
        return text_embedding + self.position_embeddings

    def forward(self, video_feature, texts, images_ls, pos, stage='train', label_gt=None):

        cls_fea = self.avgpool(video_feature).squeeze(2).squeeze(2).squeeze(2)  # [BS, 2048]
        cls_fea = self.input_linear(cls_fea)#  cls_fea [BS, 768]
        label_pred = self.cls_net(cls_fea)
        color_pred = cls_fea
        number_pred = cls_fea
        if stage == 'train':
            labels = label_gt
        else:
            labels = list(torch.argmax(label_pred, dim=1).detach().cpu().numpy())
        
        text_embeds = []
        for idx, (text, label) in enumerate(zip(texts, labels)):
            tmp_embeddings = self.get_text_embeddings(text, label, stage, color_pred[idx], number_pred[idx])
            text_output = self.text_encoder(inputs_embeds=tmp_embeddings, attention_mask = self.attention_mask, mode='text')
            text_embeds.append(text_output.last_hidden_state)
        text_embed = torch.cat(text_embeds, dim=0)
        text_embed = self.text_linear(text_embed)
        
        B, C, T, H, W = video_feature.shape

        video_fea_re = rearrange(video_feature, 'b c t h w->b (t h w) c')
        
        out, _ = self.attn(video_fea_re, text_embed, text_embed)
        
        output_fea = video_fea_re + out
        output = rearrange(output_fea, 'b (t h w) c->b c t h w', t=T,h=H,w=W)
        return output, label_pred, color_pred, number_pred