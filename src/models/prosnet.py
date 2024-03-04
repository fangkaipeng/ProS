import sys
import os
code_path = 'INSERT_CODE_PATH_HERE' # e.g. '/home/username/ProS' 
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "src"))
import torch
import torch.nn as nn
from clip import clip
from clip.model import CLIP, VisionTransformer
import math
import torch.nn as nn
from PIL import Image
from functools import reduce
from operator import mul
from utils import utils
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import copy
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
_tokenizer = _Tokenizer()

class TextEncoder(nn.Module):
    def __init__(self, clip_model:CLIP):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype) # 400, 77, 512
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 

        return x

class TextPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model:CLIP, device):
        super().__init__()
        n_cls = len(classnames) # 300
        n_ctx = cfg.tp_N_CTX  # number of context tokens 16
        dtype = clip_model.dtype # float32
        self.ctx_dim = clip_model.ln_final.weight.shape[0] # LayerNorm第0维，前一层的输出维度 512
        clip_imsize = clip_model.visual.input_resolution # 输入图片大小 # 224
        cfg_imsize = cfg.image_size # 设定的输入图片大小
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # random initialization
        # print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=dtype).to(device) # 16, 512
        nn.init.normal_(ctx_vectors, std=0.02)
        # prompt_prefix = " ".join(["X"] * n_ctx) # 生成n_ctx个 X，eg. X X X X X

        # print(f'Initial context: "{prompt_prefix}"') # 'X X X X X X X X X X X X X X X X'
        # print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized 16,512 

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] # 400，类名的长度
        # This is a photo of CLS from P domain.
        # an example of CLS from P domain.
        # a P photo of CLS.
        # P CLS
        # a photo of CLS from P domain.
        # prompts = ["a " + "X "*n_ctx +"photo of "+ name + '.' for name in classnames] # xxxxxxxxx classname .
        prompts = ["a photo of " + name + " from " + "X "*n_ctx +"domain." for name in classnames] # xxxxxxxxx classname .
        self.prefix_index = [length+5 for length in name_lens] # SOS a photo of classname from 
        print("Text Prompt Exampel:" + prompts[0])
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device) # 将prompt中的句子的每个单词转换成字典中的数字，长度固定为77，多的用0补齐, [300,77]
    
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # 300, 77, 512
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # token_prefix = [embedding[i, :prefix_index[i], :] for i in range(self.n_cls)]
        # token_suffix = [embedding[i, prefix_index[i]+n_ctx:, :] for i in range(self.n_cls)]
        self.register_buffer("origin_text_embedding",embedding)
        # for i in range(self.n_cls):
        #     self.register_buffer(f"token_prefix_{i}",embedding[i, :prefix_index[i], :])  # SOS
        #     self.register_buffer(f"token_suffix_{i}",embedding[i, prefix_index[i]+n_ctx:, :])  # CLS, EOS
       
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) # 400,16,512

        # prefix = self.token_prefix # 300,1,512
        # suffix = self.token_suffix # 300,60,512
        
        prompts = [torch.cat([self.origin_text_embedding[i,:self.prefix_index[i]],ctx[i],self.origin_text_embedding[i,self.prefix_index[i]+self.n_ctx:]],dim=0).view(1,-1,self.ctx_dim) for i in range(self.n_cls)]
        prompts = torch.cat(prompts, dim=0)
        # prompts = torch.cat(
        #     [
        #         prefix,  # (n_cls, 1, dim)
        #         ctx,     # (n_cls, n_ctx, dim)
        #         suffix,  # (n_cls, *, dim)
        #     ],
        #     dim=1,
        # )
        return prompts # 400,77,512


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        n_cls = len(classnames) # 400
        n_ctx = cfg.tp_N_CTX  # number of context tokens 16
        dtype = clip_model.dtype # float32
        ctx_dim = clip_model.ln_final.weight.shape[0] # LayerNorm第0维，前一层的输出维度 512
        clip_imsize = clip_model.visual.input_resolution # 输入图片大小 # 224
        cfg_imsize = cfg.image_size # 设定的输入图片大小
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # random initialization
        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).to(device) # 16, 512
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx) # 生成n_ctx个 X，eg. X X X X X

        print(f'Initial context: "{prompt_prefix}"') # 'X X X X X X X X X X X X X X X X'
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized 16,512 

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] # 400，类名的长度
        prompts = [prompt_prefix + " " + name + "." for name in classnames] # xxxxxxxxx classname .

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device) # 将prompt中的句子的每个单词转换成字典中的数字，长度固定为77，多的用0补齐, [400,77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # 400, 77, 512

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) # 400,16,512

        prefix = self.token_prefix # 400,1,512
        suffix = self.token_suffix # 400,60,512

        
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts # 400,77,512


class prosnet(nn.Module):
    def __init__(self, cfg, dict_clss:dict, dict_doms:dict, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dict_clss = dict_clss
       
        self.dict_doms = dict_doms
        self.dom_num_tokens = len(self.dict_doms)
        self.cls_num_tokens = len(self.dict_clss)
        clip:CLIP = self.load_clip()
        self.conv1 = clip.visual.conv1
        width = self.conv1.out_channels
        self.feature_template = clip.visual.class_embedding
        self.feature_proj = clip.visual.proj
        patch_size = self.conv1.kernel_size
        self.clip_positional_embedding = clip.visual.positional_embedding
        # self.generator = copy.deepcopy(clip.visual.transformer)
        self.generator = VisionTransformer(224, 32, 768, self.cfg.generator_layer, 12, 512)
        scale = width ** -0.5
        self.generator_proj = nn.Parameter(scale * torch.randn(width, 768)) # 768->768 作为LP来用
        self.ln_pre = clip.visual.ln_pre
        self.transformer = clip.visual.transformer
        self.ln_post = clip.visual.ln_post
        
        if self.cfg.tp_N_CTX != -1:
            if self.cfg.use_NTP == 1: # all text prompt tuning
                self.text_prompt_learner = PromptLearner(self.cfg, self.dict_clss.keys(),clip, device)
            else :
                self.text_prompt_learner = TextPromptLearner(self.cfg, self.dict_clss.keys(),clip, device)
            self.text_encoder = TextEncoder(clip)
            self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts
        else :
            self.text_encoder = clip.encode_text
        if self.cfg.DOM_PROJECT > -1:
            # only for prepend / add
            sp_dom_prompt_dim = self.cfg.DOM_PROJECT
            self.sp_dom_prompt_proj = nn.Linear(
                sp_dom_prompt_dim, width)
            nn.init.kaiming_normal_(
                self.sp_dom_prompt_proj.weight, a=0, mode='fan_out')
        else:
            sp_dom_prompt_dim = width
            self.sp_dom_prompt_proj = nn.Identity()  # 占位

        if self.cfg.CLS_PROJECT > -1:
            # only for prepend / add
            sp_cls_prompt_dim = self.cfg.CLS_PROJECT
            self.sp_cls_prompt_proj = nn.Linear(
                sp_cls_prompt_dim, width)
            nn.init.kaiming_normal_(
                self.sp_cls_prompt_proj.weight, a=0, mode='fan_out')
        else:
            sp_cls_prompt_dim = width
            self.sp_cls_prompt_proj = nn.Identity()  # 占位

        # definition of specific prompts 
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + sp_dom_prompt_dim))  # noqa
        self.specific_domain_prompts = nn.Parameter(torch.zeros(self.dom_num_tokens, sp_dom_prompt_dim)) # layer, num_token, prompt_dim
        nn.init.uniform_(self.specific_domain_prompts.data, -val, val)

        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + sp_cls_prompt_dim))  # noqa
        self.specific_class_prompts = nn.Parameter(torch.zeros(self.cls_num_tokens, sp_cls_prompt_dim)) # layer, num_token, prompt_dim
        nn.init.uniform_(self.specific_class_prompts.data, -val, val)
        
        # definition of Parameter about generated prompts 
        self.ge_cls_prompt_template = nn.Parameter(scale * torch.randn(self.cfg.GP_CLS_NUM_TOKENS, width))
        self.ge_dom_prompt_template = nn.Parameter(scale * torch.randn(self.cfg.GP_DOM_NUM_TOKENS, width))

    def incorporate_prompt(self, x, dom_index, cls_index, stage, dom_prompts=None, cls_prompts=None):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0] # batch size
        
        
        if stage == 1:
            """
            train specific prompts stage:
            x = [feature template, specific prompts of GT, image patch]
            """
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            x = torch.cat((
                (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1),
                self.sp_dom_prompt_proj(self.specific_domain_prompts[dom_index]).view(B, 1, -1),
                self.sp_cls_prompt_proj(self.specific_class_prompts[cls_index]).view(B, 1, -1), 
                x + self.clip_positional_embedding[1:]
        ), dim=1)
            
            
        elif stage == 2:
            """
            generate unseen prompts stage:
            x = [template prompt, specific prompts without GT, image patch]
            """
            x = self.generator.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            sp_dom_prompts = self.sp_dom_prompt_proj(self.specific_domain_prompts)
            sp_cls_prompts = self.sp_cls_prompt_proj(self.specific_class_prompts)
    
            cls_prompt_mask = torch.zeros(B, sp_cls_prompts.shape[0], sp_cls_prompts.shape[1]).type(torch.bool).to(self.device)
            dom_prompt_mask = torch.zeros(B, sp_dom_prompts.shape[0], sp_dom_prompts.shape[1]).type(torch.bool).to(self.device)
            cls_prompt_mask[range(B), cls_index, :] = 1
            dom_prompt_mask[range(B), dom_index, :] = 1
        
            sp_cls_prompts = sp_cls_prompts.expand(B,-1,-1).masked_fill(cls_prompt_mask, 0)
            sp_dom_prompts = sp_dom_prompts.expand(B,-1,-1).masked_fill(dom_prompt_mask, 0) 

            x = torch.cat((
                self.ge_dom_prompt_template.expand(B,-1,-1),
                self.ge_cls_prompt_template.expand(B,-1, -1),
                sp_dom_prompts,
                sp_cls_prompts,
                x + self.generator.positional_embedding[1:]
        ), dim=1)
            
        elif stage == 3:
            """
            use generated prompts get feature
            """
            # add template positional embedding
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            x = torch.cat((
            (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1),
            dom_prompts.view(B, self.cfg.GP_DOM_NUM_TOKENS, -1),
            cls_prompts.view(B, self.cfg.GP_CLS_NUM_TOKENS, -1), 
            x + self.clip_positional_embedding[1:]
        ), dim=1)
            
        elif stage == 4:
            """
            test time generated prompts: no need mask specific prompts
            """
            x = self.generator.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            x = torch.cat((
                self.ge_dom_prompt_template.expand(B,-1, -1),
                self.ge_cls_prompt_template.expand(B,-1, -1),
                self.sp_dom_prompt_proj(self.specific_domain_prompts).expand(B,-1,-1),
                self.sp_cls_prompt_proj(self.specific_class_prompts).expand(B,-1,-1),
                x + self.generator.positional_embedding[1:]
        ), dim=1)
        
        # elif stage == 5:
        #     """
        #     test time generated prompts: no need mask specific prompts
        #     """
        #     x = self.conv1(x)
        #     x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
        #     x = x.permute(0, 2, 1) # 65 49 768
        #     x = torch.cat((
        #         (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1),
        #         self.sp_dom_prompt_proj(self.specific_domain_prompts).expand(B,-1, -1),
        #         self.sp_cls_prompt_proj(self.specific_class_prompts).expand(B,-1, -1), 
        #         x + self.clip_positional_embedding[1:]
        # ), dim=1)
        return x
    
    def vit(self, x, out_token):
       
        if out_token == 1:
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:,out_token,:])
            x = x @ self.feature_proj
        else :
            x = self.generator.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.generator.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.generator.ln_post(x[:,:out_token,:])
            x = x @ self.generator_proj
        return x
    
    def image_encoder(self, image, dom_id, cls_id, stage):
        if stage == 1: # training for specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, stage)
            x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, 1)
        elif stage == 2: # input: template + specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, 2) # cat template + specific prompts + image patch
            # x = torch.nn.functional.dropout(x, 0.2)
            x = self.vit(x, self.cfg.GP_DOM_NUM_TOKENS+self.cfg.GP_CLS_NUM_TOKENS) # get generated prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, 3, x[:,:self.cfg.GP_DOM_NUM_TOKENS], x[:,self.cfg.GP_DOM_NUM_TOKENS:]) # cat CLS generated prompts + image patch
            # x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, 1)
        # elif stage == 3: # stage1 analyse
        #     x = self.incorporate_prompt(image, dom_id, cls_id, 5)
        #     # x = torch.nn.functional.dropout(x, self.cfg.dropout)
        #     x = self.vit(x, 1)
        elif stage == 4:
            x = self.incorporate_prompt(image, dom_id, cls_id, 4)
            x = self.vit(x, self.cfg.GP_DOM_NUM_TOKENS+self.cfg.GP_CLS_NUM_TOKENS)
            x = self.incorporate_prompt(image, dom_id, cls_id,3, x[:,:self.cfg.GP_DOM_NUM_TOKENS], x[:,self.cfg.GP_DOM_NUM_TOKENS:])
            x = self.vit(x, 1)
       
        return x

    def forward(self, image, domain_name, class_name, stage): # bt 3, 244, 244
        cls_id = utils.numeric_classes(class_name, self.dict_clss)
        dom_id = utils.numeric_classes(domain_name, self.dict_doms)
        # domain_name = list(dict.fromkeys(domain_name))
        # class_name = list(dict.fromkeys(class_name))
        
        image_features = self.image_encoder(image, dom_id, cls_id, stage) # batch, 512
        if self.cfg.tp_N_CTX != -1:
            text_prompts = self.text_prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(text_prompts, tokenized_prompts)
        else :
            text_template = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_name]).to(self.device)
            text_features = self.text_encoder(text_template)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        return image_features, text_features
    
    def load_clip(self):
        backbone_name = self.cfg.clip_backbone
        print(f"=======load CLIP:{backbone_name}=========")
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location=self.device).eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location=self.device)

        model = clip.build_model(state_dict or model.state_dict())
        return model.float().to(self.device) 

class c_spgnet(nn.Module):
    def __init__(self, cfg, dict_clss:dict, dict_doms:dict, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dict_doms = dict_doms
        self.dict_clss = dict_clss
        self.cls_num_tokens = len(self.dict_clss)
        clip:CLIP = self.load_clip()
        self.conv1 = clip.visual.conv1
        width = self.conv1.out_channels
        self.feature_template = clip.visual.class_embedding
        self.feature_proj = clip.visual.proj
        patch_size = self.conv1.kernel_size
        self.clip_positional_embedding = clip.visual.positional_embedding
        scale = width ** -0.5
        self.generator = VisionTransformer(224, 32, 768, self.cfg.generator_layer, 12, 512)
        self.generator_proj = nn.Parameter(scale * torch.randn(width, 768)) # 768->768 作为LP来用
        
        self.ln_pre = clip.visual.ln_pre
        self.transformer = clip.visual.transformer
        self.ln_post = clip.visual.ln_post
        
        if self.cfg.tp_N_CTX != -1:
            if self.cfg.use_NTP == 1: # all text prompt tuning
                self.text_prompt_learner = PromptLearner(self.cfg, self.dict_clss.keys(),clip, device)
            else :
                self.text_prompt_learner = TextPromptLearner(self.cfg, self.dict_clss.keys(),clip, device)
            self.text_encoder = TextEncoder(clip)
            self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts
        else :
            self.text_encoder = clip.encode_text
        

        if self.cfg.CLS_PROJECT > -1:
            # only for prepend / add
            sp_cls_prompt_dim = self.cfg.CLS_PROJECT
            self.sp_cls_prompt_proj = nn.Linear(
                sp_cls_prompt_dim, width)
            nn.init.kaiming_normal_(
                self.sp_cls_prompt_proj.weight, a=0, mode='fan_out')
        else:
            sp_cls_prompt_dim = width
            self.sp_cls_prompt_proj = nn.Identity()  # 占位

        # definition of specific prompts 

        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + sp_cls_prompt_dim))  # noqa
        self.specific_class_prompts = nn.Parameter(torch.zeros(self.cls_num_tokens, sp_cls_prompt_dim)) # layer, num_token, prompt_dim
        nn.init.uniform_(self.specific_class_prompts.data, -val, val)
        
        # definition of Parameter about generated prompts 
       
        self.ge_cls_prompt_template = nn.Parameter(scale * torch.randn(self.cfg.GP_CLS_NUM_TOKENS, width))
    
    def incorporate_prompt(self, x, dom_index, cls_index, stage, dom_prompts=None, cls_prompts=None):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0] # batch size
       
      
        if stage == 1:
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            """
            train specific prompts stage:
            x = [feature template, specific prompts of GT, image patch]
            """
            x = torch.cat((
                (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1),
                self.sp_cls_prompt_proj(self.specific_class_prompts[cls_index]).view(B, 1, -1), 
                x + self.clip_positional_embedding[1:]
        ), dim=1)
            
            
        elif stage == 2:
            """
            generate unseen prompts stage:
            x = [template prompt, specific prompts without GT, image patch]
            """
            
            x = self.generator.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            sp_cls_prompts = self.sp_cls_prompt_proj(self.specific_class_prompts)
    
            cls_prompt_mask = torch.zeros(B, sp_cls_prompts.shape[0], sp_cls_prompts.shape[1]).type(torch.bool).to(self.device)
            cls_prompt_mask[range(B), cls_index, :] = 1
            
            sp_cls_prompts = sp_cls_prompts.expand(B,-1,-1).masked_fill(cls_prompt_mask, 0)
            
            x = torch.cat((
                self.ge_cls_prompt_template.expand(B,-1, -1),
                sp_cls_prompts,
                x + self.generator.positional_embedding[1:]
        ), dim=1)
            
        elif stage == 3:
            """
            use generated prompts get feature
            """
            # add template positional embedding
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            x = torch.cat((
            (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1),
            cls_prompts.view(B, self.cfg.GP_CLS_NUM_TOKENS, -1), 
            x + self.clip_positional_embedding[1:]
        ), dim=1)
            
        elif stage == 4:
            """
            test time generated prompts: no need mask specific prompts
            """
            x = self.generator.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            x = torch.cat((
                self.ge_cls_prompt_template.expand(B,-1, -1),
                self.sp_cls_prompt_proj(self.specific_class_prompts).expand(B,-1,-1),
                x + self.generator.positional_embedding[1:]
        ), dim=1)
        return x
    
    def vit(self, x, out_token):
        
        if out_token == 1:
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:,out_token,:])
            x = x @ self.feature_proj
        else :
            x = self.generator.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.generator.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.generator.ln_post(x[:,:out_token,:])
            x = x @ self.generator_proj
        return x
    
    def image_encoder(self, image, dom_id, cls_id, stage):
        if stage == 1: # training for specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, stage)
            x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, 1)
        elif stage == 2: # input: template + specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, 2) # cat template + specific prompts + image patch
            # x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, self.cfg.GP_CLS_NUM_TOKENS+1) # get generated prompts
            x = self.incorporate_prompt(image, dom_id, cls_id,3, cls_prompts=x[:,1]) # cat CLS generated prompts + image patch
            # x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, 1)
        elif stage == 4:
            x = self.incorporate_prompt(image, dom_id, cls_id, 4)
            x = self.vit(x, self.cfg.GP_CLS_NUM_TOKENS+1)
            x = self.incorporate_prompt(image, dom_id, cls_id,3, cls_prompts=x[:,1])
            x = self.vit(x, 1)
       
        return x

    def forward(self, image, domain_name, class_name, stage): # bt 3, 244, 244
        cls_id = utils.numeric_classes(class_name, self.dict_clss)
        dom_id = utils.numeric_classes(domain_name, self.dict_doms)
        # domain_name = list(dict.fromkeys(domain_name))
        # class_name = list(dict.fromkeys(class_name))
        
        image_features = self.image_encoder(image, dom_id, cls_id, stage) # batch, 512
        if self.cfg.tp_N_CTX != -1:
            text_prompts = self.text_prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(text_prompts, tokenized_prompts)
        else :
            text_template = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_name]).to(self.device)
            text_features = self.text_encoder(text_template)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        return image_features, text_features
    
    def load_clip(self):
        backbone_name = self.cfg.clip_backbone
        print(f"=======load CLIP:{backbone_name}=========")
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location=self.device).eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location=self.device)

        model = clip.build_model(state_dict or model.state_dict())
        return model.float().to(self.device)   

class d_spgnet(nn.Module):

    def __init__(self, cfg, dict_clss:dict, dict_doms:dict, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dict_clss = dict_clss
       
        self.dict_doms = dict_doms
        self.dom_num_tokens = len(self.dict_doms)
        clip:CLIP = self.load_clip()
        self.conv1 = clip.visual.conv1
        width = self.conv1.out_channels
        self.feature_template = clip.visual.class_embedding
        self.feature_proj = clip.visual.proj
        patch_size = self.conv1.kernel_size
        self.clip_positional_embedding = clip.visual.positional_embedding
        scale = width ** -0.5
        self.generator = VisionTransformer(224, 32, 768, self.cfg.generator_layer, 12, 512)
        self.generator_proj = nn.Parameter(scale * torch.randn(width, 768)) # 768->768 作为LP来用
        self.ln_pre = clip.visual.ln_pre
        self.transformer = clip.visual.transformer
        self.ln_post = clip.visual.ln_post
        
        if self.cfg.tp_N_CTX != -1:
            if self.cfg.use_NTP == 1: # all text prompt tuning
                self.text_prompt_learner = PromptLearner(self.cfg, self.dict_clss.keys(),clip, device)
            else :
                self.text_prompt_learner = TextPromptLearner(self.cfg, self.dict_clss.keys(),clip, device)
            self.text_encoder = TextEncoder(clip)
            self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts
        else :
            self.text_encoder = clip.encode_text
        if self.cfg.DOM_PROJECT > -1:
            # only for prepend / add
            sp_dom_prompt_dim = self.cfg.DOM_PROJECT
            self.sp_dom_prompt_proj = nn.Linear(
                sp_dom_prompt_dim, width)
            nn.init.kaiming_normal_(
                self.sp_dom_prompt_proj.weight, a=0, mode='fan_out')
        else:
            sp_dom_prompt_dim = width
            self.sp_dom_prompt_proj = nn.Identity()  # 占位


        # definition of specific prompts 
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + sp_dom_prompt_dim))  # noqa
        self.specific_domain_prompts = nn.Parameter(torch.zeros(self.dom_num_tokens, sp_dom_prompt_dim)) # layer, num_token, prompt_dim
        nn.init.uniform_(self.specific_domain_prompts.data, -val, val)

        # definition of Parameter about generated prompts 
        self.ge_dom_prompt_template = nn.Parameter(scale * torch.randn(self.cfg.GP_DOM_NUM_TOKENS, width))

    def incorporate_prompt(self, x, dom_index, cls_index, stage, dom_prompts=None, cls_prompts=None):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0] # batch size
       
        if stage == 1:
            """
            train specific prompts stage:
            x = [feature template, specific prompts of GT, image patch]
            """
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            x = torch.cat((
                (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1),
                self.sp_dom_prompt_proj(self.specific_domain_prompts[dom_index]).view(B, 1, -1),
                x + self.clip_positional_embedding[1:]
        ), dim=1)
            
            
        elif stage == 2:
            """
            generate unseen prompts stage:
            x = [template prompt, specific prompts without GT, image patch]
            """
            x = self.generator.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            sp_dom_prompts = self.sp_dom_prompt_proj(self.specific_domain_prompts)
            dom_prompt_mask = torch.zeros(B, sp_dom_prompts.shape[0], sp_dom_prompts.shape[1]).type(torch.bool).to(self.device)
            dom_prompt_mask[range(B), dom_index, :] = 1
        
            sp_dom_prompts = sp_dom_prompts.expand(B,-1,-1).masked_fill(dom_prompt_mask, 0) 

            x = torch.cat((
                self.ge_dom_prompt_template.expand(B,-1,-1),
                sp_dom_prompts,
                x + self.generator.positional_embedding[1:]
        ), dim=1)
            
        elif stage == 3:
            """
            use generated prompts get feature
            """
            # add template positional embedding
        
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            x = torch.cat((
            (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1),
            dom_prompts.view(B, self.cfg.GP_DOM_NUM_TOKENS, -1),
            x + self.clip_positional_embedding[1:]
        ), dim=1)
            
        elif stage == 4:
            """
            test time generated prompts: no need mask specific prompts
            """
            x = self.generator.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            x = torch.cat((
                self.ge_dom_prompt_template.expand(B,-1, -1),
                self.sp_dom_prompt_proj(self.specific_domain_prompts).expand(B,-1,-1),
                x + self.generator.positional_embedding[1:]
        ), dim=1)
        return x
    
    def vit(self, x, out_token):
        
        if out_token == 1:
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:,out_token,:])
            x = x @ self.feature_proj
        else :
            x = self.generator.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.generator.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.generator.ln_post(x[:,:out_token,:])
            x = x @ self.generator_proj
        return x
    
    def image_encoder(self, image, dom_id, cls_id, stage):
        if stage == 1: # training for specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, stage)
            x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, 1)
        elif stage == 2: # input: template + specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, 2) # cat template + specific prompts + image patch
            # x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, self.cfg.GP_DOM_NUM_TOKENS+1) # get generated prompts
            x = self.incorporate_prompt(image, dom_id, cls_id,3, dom_prompts=x[:,1]) # cat CLS generated prompts + image patch
            # x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, 1)
        elif stage == 4:
            x = self.incorporate_prompt(image, dom_id, cls_id, 4)
            x = self.vit(x, self.cfg.GP_DOM_NUM_TOKENS+1)
            x = self.incorporate_prompt(image, dom_id, cls_id,3, dom_prompts=x[:,1])
            x = self.vit(x, 1)
       
        return x

    def forward(self, image, domain_name, class_name, stage): # bt 3, 244, 244
        cls_id = utils.numeric_classes(class_name, self.dict_clss)
        dom_id = utils.numeric_classes(domain_name, self.dict_doms)
        # domain_name = list(dict.fromkeys(domain_name))
        # class_name = list(dict.fromkeys(class_name))
        
        image_features = self.image_encoder(image, dom_id, cls_id, stage) # batch, 512
        if self.cfg.tp_N_CTX != -1:
            text_prompts = self.text_prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(text_prompts, tokenized_prompts)
        else :
            text_template = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_name]).to(self.device)
            text_features = self.text_encoder(text_template)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        return image_features, text_features
    
    def load_clip(self):
        backbone_name = self.cfg.clip_backbone
        print(f"=======load CLIP:{backbone_name}=========")
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location=self.device).eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location=self.device)

        model = clip.build_model(state_dict or model.state_dict())
        return model.float().to(self.device)
    
class no_c_d_spgnet(nn.Module):
    def __init__(self, cfg, dict_clss:dict, dict_doms:dict, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dict_doms = dict_doms
        self.dict_clss = dict_clss
        self.cls_num_tokens = len(self.dict_clss)
        clip:CLIP = self.load_clip()
        self.conv1 = clip.visual.conv1
        width = self.conv1.out_channels
        self.feature_template = clip.visual.class_embedding
        self.feature_proj = clip.visual.proj
        patch_size = self.conv1.kernel_size
        self.clip_positional_embedding = clip.visual.positional_embedding
        scale = width ** -0.5
        self.generator = VisionTransformer(224, 32, 768, self.cfg.generator_layer, 12, 512)
        self.generator_proj = nn.Parameter(scale * torch.randn(width, 768)) # 768->768 作为LP来用
        
        self.ln_pre = clip.visual.ln_pre
        self.transformer = clip.visual.transformer
        self.ln_post = clip.visual.ln_post
        
        if self.cfg.tp_N_CTX != -1:
            if self.cfg.use_NTP == 1: # all text prompt tuning
                self.text_prompt_learner = PromptLearner(self.cfg, self.dict_clss.keys(),clip, device)
            else :
                self.text_prompt_learner = TextPromptLearner(self.cfg, self.dict_clss.keys(),clip, device)
            self.text_encoder = TextEncoder(clip)
            self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts
        else :
            self.text_encoder = clip.encode_text
        
        # definition of Parameter about generated prompts 
       
        self.ge_cls_prompt_template = nn.Parameter(scale * torch.randn(self.cfg.GP_CLS_NUM_TOKENS, width))
    
    def incorporate_prompt(self, x, dom_index, cls_index, stage, dom_prompts=None, cls_prompts=None):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0] # batch size
       
            
        if stage == 2:
            """
            generate unseen prompts stage:
            x = [template prompt, specific prompts without GT, image patch]
            """
            
            x = self.generator.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768


            
            x = torch.cat((
                self.ge_cls_prompt_template.expand(B,-1, -1),
                x + self.generator.positional_embedding[1:]
        ), dim=1)
            
        elif stage == 3:
            """
            use generated prompts get feature
            """
            # add template positional embedding
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            x = torch.cat((
            (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1),
            cls_prompts.view(B, self.cfg.GP_CLS_NUM_TOKENS, -1), 
            x + self.clip_positional_embedding[1:]
        ), dim=1)
            
        elif stage == 4:
            """
            test time generated prompts: no need mask specific prompts
            """
            x = self.generator.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            x = torch.cat((
                self.ge_cls_prompt_template.expand(B,-1, -1),
                x + self.generator.positional_embedding[1:]
        ), dim=1)
        return x
    
    def vit(self, x, out_token):
        
        if out_token == 1:
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:,out_token,:])
            x = x @ self.feature_proj
        else :
            x = self.generator.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.generator.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.generator.ln_post(x[:,:out_token,:])
            x = x @ self.generator_proj
        return x
    
    def image_encoder(self, image, dom_id, cls_id, stage):
        if stage == 1: # training for specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, stage)
            x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, 1)
        elif stage == 2: # input: template + specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, 2) # cat template + specific prompts + image patch
            # x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, self.cfg.GP_CLS_NUM_TOKENS+1) # get generated prompts
            x = self.incorporate_prompt(image, dom_id, cls_id,3, cls_prompts=x[:,1]) # cat CLS generated prompts + image patch
            # x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, 1)
        elif stage == 4:
            x = self.incorporate_prompt(image, dom_id, cls_id, 4)
            x = self.vit(x, self.cfg.GP_CLS_NUM_TOKENS+1)
            x = self.incorporate_prompt(image, dom_id, cls_id,3, cls_prompts=x[:,1])
            x = self.vit(x, 1)
       
        return x

    def forward(self, image, domain_name, class_name, stage): # bt 3, 244, 244
        cls_id = utils.numeric_classes(class_name, self.dict_clss)
        dom_id = utils.numeric_classes(domain_name, self.dict_doms)
        # domain_name = list(dict.fromkeys(domain_name))
        # class_name = list(dict.fromkeys(class_name))
        
        image_features = self.image_encoder(image, dom_id, cls_id, stage) # batch, 512
        if self.cfg.tp_N_CTX != -1:
            text_prompts = self.text_prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(text_prompts, tokenized_prompts)
        else :
            text_template = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_name]).to(self.device)
            text_features = self.text_encoder(text_template)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        return image_features, text_features
    
    def load_clip(self):
        backbone_name = self.cfg.clip_backbone
        print(f"=======load CLIP:{backbone_name}=========")
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location=self.device).eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location=self.device)

        model = clip.build_model(state_dict or model.state_dict())
        return model.float().to(self.device)   

class no_mask_spgnet(nn.Module):
    def __init__(self, cfg, dict_clss:dict, dict_doms:dict, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dict_clss = dict_clss
       
        self.dict_doms = dict_doms
        self.dom_num_tokens = len(self.dict_doms)
        self.cls_num_tokens = len(self.dict_clss)
        clip:CLIP = self.load_clip()
        self.conv1 = clip.visual.conv1
        width = self.conv1.out_channels
        self.feature_template = clip.visual.class_embedding
        self.feature_proj = clip.visual.proj
        patch_size = self.conv1.kernel_size
        self.clip_positional_embedding = clip.visual.positional_embedding
        # self.generator = copy.deepcopy(clip.visual.transformer)
        self.generator = VisionTransformer(224, 32, 768, self.cfg.generator_layer, 12, 512)
        scale = width ** -0.5
        self.generator_proj = nn.Parameter(scale * torch.randn(width, 768)) # 768->768 作为LP来用
        self.ln_pre = clip.visual.ln_pre
        self.transformer = clip.visual.transformer
        self.ln_post = clip.visual.ln_post
        
        if self.cfg.tp_N_CTX != -1:
            if self.cfg.use_NTP == 1: # all text prompt tuning
                self.text_prompt_learner = PromptLearner(self.cfg, self.dict_clss.keys(),clip, device)
            else :
                self.text_prompt_learner = TextPromptLearner(self.cfg, self.dict_clss.keys(),clip, device)
            self.text_encoder = TextEncoder(clip)
            self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts
        else :
            self.text_encoder = clip.encode_text
        if self.cfg.DOM_PROJECT > -1:
            # only for prepend / add
            sp_dom_prompt_dim = self.cfg.DOM_PROJECT
            self.sp_dom_prompt_proj = nn.Linear(
                sp_dom_prompt_dim, width)
            nn.init.kaiming_normal_(
                self.sp_dom_prompt_proj.weight, a=0, mode='fan_out')
        else:
            sp_dom_prompt_dim = width
            self.sp_dom_prompt_proj = nn.Identity()  # 占位

        if self.cfg.CLS_PROJECT > -1:
            # only for prepend / add
            sp_cls_prompt_dim = self.cfg.CLS_PROJECT
            self.sp_cls_prompt_proj = nn.Linear(
                sp_cls_prompt_dim, width)
            nn.init.kaiming_normal_(
                self.sp_cls_prompt_proj.weight, a=0, mode='fan_out')
        else:
            sp_cls_prompt_dim = width
            self.sp_cls_prompt_proj = nn.Identity()  # 占位

        # definition of specific prompts 
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + sp_dom_prompt_dim))  # noqa
        self.specific_domain_prompts = nn.Parameter(torch.zeros(self.dom_num_tokens, sp_dom_prompt_dim)) # layer, num_token, prompt_dim
        nn.init.uniform_(self.specific_domain_prompts.data, -val, val)

        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + sp_cls_prompt_dim))  # noqa
        self.specific_class_prompts = nn.Parameter(torch.zeros(self.cls_num_tokens, sp_cls_prompt_dim)) # layer, num_token, prompt_dim
        nn.init.uniform_(self.specific_class_prompts.data, -val, val)
        
        # definition of Parameter about generated prompts 
        self.ge_cls_prompt_template = nn.Parameter(scale * torch.randn(self.cfg.GP_CLS_NUM_TOKENS, width))
        self.ge_dom_prompt_template = nn.Parameter(scale * torch.randn(self.cfg.GP_DOM_NUM_TOKENS, width))

    def incorporate_prompt(self, x, dom_index, cls_index, stage, dom_prompts=None, cls_prompts=None):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0] # batch size
        
        
        if stage == 1:
            """
            train specific prompts stage:
            x = [feature template, specific prompts of GT, image patch]
            """
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            x = torch.cat((
                (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1),
                self.sp_dom_prompt_proj(self.specific_domain_prompts).expand(B, -1, -1),
                self.sp_cls_prompt_proj(self.specific_class_prompts).expand(B, -1, -1), 
                x + self.clip_positional_embedding[1:]
        ), dim=1)
            
            
        elif stage == 2:
            """
            generate unseen prompts stage:
            x = [template prompt, specific prompts without GT, image patch]
            """
            x = self.generator.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            sp_dom_prompts = self.sp_dom_prompt_proj(self.specific_domain_prompts)
            sp_cls_prompts = self.sp_cls_prompt_proj(self.specific_class_prompts)
            
            sp_cls_prompts = sp_cls_prompts.expand(B,-1,-1)
            sp_dom_prompts = sp_dom_prompts.expand(B,-1,-1)

            x = torch.cat((
                self.ge_dom_prompt_template.expand(B,-1, -1),
                self.ge_cls_prompt_template.expand(B,-1, -1),
                sp_dom_prompts,
                sp_cls_prompts,
                x + self.generator.positional_embedding[1:]
        ), dim=1)
            
        elif stage == 3:
            """
            use generated prompts get feature
            """
            # add template positional embedding
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            x = torch.cat((
            (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1),
            dom_prompts.view(B, self.cfg.GP_DOM_NUM_TOKENS, -1),
            cls_prompts.view(B, self.cfg.GP_CLS_NUM_TOKENS, -1), 
            x + self.clip_positional_embedding[1:]
        ), dim=1)
            
        elif stage == 4:
            """
            test time generated prompts: no need mask specific prompts
            """
            x = self.generator.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) # 65 768 49
            x = x.permute(0, 2, 1) # 65 49 768
            x = torch.cat((
                self.ge_dom_prompt_template.expand(B,-1, -1),
                self.ge_cls_prompt_template.expand(B,-1, -1),
                self.sp_dom_prompt_proj(self.specific_domain_prompts).expand(B,-1,-1),
                self.sp_cls_prompt_proj(self.specific_class_prompts).expand(B,-1,-1),
                x + self.generator.positional_embedding[1:]
        ), dim=1)
        return x
    
    def vit(self, x, out_token):
       
        if out_token == 1:
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:,out_token,:])
            x = x @ self.feature_proj
        else :
            x = self.generator.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.generator.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.generator.ln_post(x[:,:out_token,:])
            x = x @ self.generator_proj
        return x
    
    def image_encoder(self, image, dom_id, cls_id, stage):
        if stage == 1: # training for specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, stage)
            x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, 1)
        elif stage == 2: # input: template + specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, 2) # cat template + specific prompts + image patch
            # x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, self.cfg.GP_DOM_NUM_TOKENS+self.cfg.GP_CLS_NUM_TOKENS) # get generated prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, 3, x[:,:self.cfg.GP_DOM_NUM_TOKENS], x[:,self.cfg.GP_DOM_NUM_TOKENS:]) # cat CLS generated prompts + image patch
            # x = torch.nn.functional.dropout(x, self.cfg.dropout)
            x = self.vit(x, 1)
        elif stage == 4:
            x = self.incorporate_prompt(image, dom_id, cls_id, 4)
            x = self.vit(x, self.cfg.GP_DOM_NUM_TOKENS+self.cfg.GP_CLS_NUM_TOKENS)
            x = self.incorporate_prompt(image, dom_id, cls_id,3, x[:,:self.cfg.GP_DOM_NUM_TOKENS], x[:,self.cfg.GP_DOM_NUM_TOKENS:])
            x = self.vit(x, 1)
       
        return x

    def forward(self, image, domain_name, class_name, stage): # bt 3, 244, 244
        cls_id = utils.numeric_classes(class_name, self.dict_clss)
        dom_id = utils.numeric_classes(domain_name, self.dict_doms)
        # domain_name = list(dict.fromkeys(domain_name))
        # class_name = list(dict.fromkeys(class_name))
        
        image_features = self.image_encoder(image, dom_id, cls_id, stage) # batch, 512
        if self.cfg.tp_N_CTX != -1:
            text_prompts = self.text_prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(text_prompts, tokenized_prompts)
        else :
            text_template = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_name]).to(self.device)
            text_features = self.text_encoder(text_template)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        return image_features, text_features
    
    def load_clip(self):
        backbone_name = self.cfg.clip_backbone
        print(f"=======load CLIP:{backbone_name}=========")
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location=self.device).eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location=self.device)

        model = clip.build_model(state_dict or model.state_dict())
        return model.float().to(self.device)
    