import pandas as pd
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset,Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM,default_data_collator
import os
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor,EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import random
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from pytorch_lightning.plugins import DDPPlugin
import torch.nn.functional as F
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
from torch.distributions import Normal

def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def em_algorithm_independent_dimensions_stable(samples, n_distributions, n_dimensions, max_iterations=500, tol=1e-4, min_variance=1e-20, device='cpu'):

    torch.manual_seed(2048)
    means = torch.rand(n_distributions, n_dimensions, device=device)  # Mean vectors
    variances = torch.rand(n_distributions, n_dimensions, device=device)  # Variance vectors

    log_likelihood_history = []  # Record log likelihood history
    means_history = []  # Record mean history

    samples = samples.to(device)

    for iteration in range(max_iterations):

        log_responsibilities = torch.zeros(samples.shape[0], n_distributions, device=device)
        for i in range(n_distributions):
            variances_with_min = torch.clamp(variances[i], min=min_variance)
            log_pdf = Normal(means[i], torch.sqrt(variances_with_min)).log_prob(samples)

            log_responsibilities[:, i] = log_pdf.sum(dim=1)



        min_log_responsibilities = torch.min(log_responsibilities, dim=1, keepdim=True)[0]
        max_log_responsibilities = torch.max(log_responsibilities, dim=1, keepdim=True)[0]
        log_responsibilities = (log_responsibilities - min_log_responsibilities) / (max_log_responsibilities - min_log_responsibilities)
        
  
        
        responsibilities = torch.nn.functional.softmax(log_responsibilities, dim=1)

        new_means = torch.zeros_like(means, device=device)
        new_variances = torch.zeros_like(variances, device=device)
        for i in range(n_distributions):
            weights = responsibilities[:, i].unsqueeze(1)
            total_weight = weights.sum()
            new_means[i] = torch.sum(samples * weights, dim=0) / total_weight
            new_variances[i] = torch.sum(weights * (samples - new_means[i]).pow(2), dim=0) / total_weight



        mean_change = torch.norm(new_means - means)
        print(mean_change)
        if mean_change < tol and iteration > 0:
            break

        means = new_means
        variances = new_variances
        means_history.append(means.detach().clone().cpu().numpy())

    return means, variances, log_likelihood_history, means_history


def get_sne(encoder_out):

    encoder_out_np = encoder_out.detach().cpu().numpy()

    tsne = TSNE(n_components=2, random_state=0)
    encoder_out_2d = tsne.fit_transform(encoder_out_np)


    plt.figure(figsize=(10, 6))
    plt.scatter(encoder_out_2d[:, 0], encoder_out_2d[:, 1])
    plt.title('t-SNE Visualization of encoder_out')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()




class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = focal_loss * self.alpha[targets]
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MSModel(nn.Module):
    def __init__(self,model,tokenizer,codebook_num=10,n=5,freeze=False,centrifugal=True,f_step=5000,init_method='mean',lang="zh"):
        super().__init__()
        self.model = model
        self.lang = lang

        self.centrifugal = centrifugal

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            print("wte:",parameter_count(self.model.transformer.wte))
            print("wpe:",parameter_count(self.model.transformer.wpe))
            for param in self.model.transformer.wte.parameters():
                param.requires_grad = True
            for param in self.model.transformer.wpe.parameters():
                param.requires_grad = True

        self.init_method = init_method
            
        self.encoder_length = 128
        self.decoder_length = 128

        self.tokenizer = tokenizer
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.ori_model_prepare_inputs_for_generation = self.model.prepare_inputs_for_generation

        self.codebook_num = codebook_num
        self.seq_len = n
        


        self.codebook = nn.Embedding(self.codebook_num, self.model.config.hidden_size)
        self.codebook.weight.data.uniform_(-1/self.codebook_num, 1/self.codebook_num)
        

        
        hidden_size = self.model.config.hidden_size
        codebook_num = self.codebook_num  
        dropout_rate = 0.1  
        n_layer = self.model.config.n_layer


        # Decoder multi-layer network
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size * n_layer),
        )
        
        self.seq_embedding = nn.Embedding(self.seq_len, self.model.config.hidden_size)


        # Predicter multi-layer network
        self.predicter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, codebook_num)
        )



        self.ori_prepare_model_inputs = self.model.prepare_inputs_for_generation

        self.past_keys_values = None
        
        self.generate_past_key_values = None

        self.codebook_initialized_count = 0
        
        self.f_step = f_step
        self.f_step_count = 0

    def pass_grad(self, x, y):
        return x + (y - x).detach()

    def forward(self, input_ids, attention_mask, labels=None, condition=None):

        if self.init_method == 'em' and self.codebook_initialized_count == 0:
            print(self.model.device)
            self.persona_vector = torch.load('./vector/persona_vector_'+self.lang)
            print(self.persona_vector.shape)
            device = self.model.device
            self.persona_vector = self.persona_vector.to(device)
            
            estimated_means, estimated_variances, log_likelihood_history, means_history = em_algorithm_independent_dimensions_stable(
                    self.persona_vector, self.codebook_num, self.persona_vector.shape[1], device=device
                )
            del estimated_variances, log_likelihood_history, means_history
            self.codebook.weight.data = estimated_means
            del self.persona_vector
            self.codebook_initialized_count += 1 

        hidden_states = self.model(output_hidden_states= True,**condition)

        # hidden_states = hidden_states.hidden_states[1:]

        # # [layer, B, T, D]
        # hidden_states_tensor = torch.stack(hidden_states)

        # # To merge layer and D, we first need to adjust the shape of the tensor
        # # We want the final shape to be [B, T, layer*D], so we can send this tensor into a fully connected layer
        # # First, we swap the dimensions of layer and batch size, then merge layer and D
        # encoder_in = hidden_states_tensor.permute(1, 2, 0, 3).reshape(hidden_states_tensor.shape[1], hidden_states_tensor.shape[2], hidden_states_tensor.shape[0]*hidden_states_tensor.shape[3])

        encoder_in = hidden_states.hidden_states[-1]
        not_neg_100 = condition["attention_mask"] != 1
        first_not_neg_100_positions = torch.argmax(not_neg_100.int(), dim=1)
        # Prevent indices from being -1, i.e., the entire row is -100, in which case we can choose 0 as the index or perform other processing
        # first_not_neg_100_positions = torch.clamp(first_not_neg_100_positions, min=0)
        # print(condition["attention_mask"])
        # print(first_not_neg_100_positions)
        # Select last_hidden_state based on the index calculated

        # [B*n, D]
        encoder_in = encoder_in[torch.arange(encoder_in.size(0)),first_not_neg_100_positions]
        
        

        encoder_out = encoder_in
        
        
        if self.init_method != "em":
        # Calculate how many codebook items still need to be initialized
            remaining = self.codebook_num - self.codebook_initialized_count
            if remaining > 0:
                # Calculate how many items the current batch can provide
                current_batch_size = encoder_out.size(0)
                # Calculate how many items can be initialized in this batch
                if self.init_method == "mean":
                    current_batch_size = 1
                elif self.init_method == "single":
                    current_batch_size = encoder_out.size(0)
                    
                init_size = min(remaining, current_batch_size)

                # Update codebook
                start_index = self.codebook_initialized_count
                end_index = start_index + init_size
                self.codebook.weight.data[start_index:end_index] = torch.mean(encoder_out[:init_size],dim=0).detach()

                # Update the number of initialized codebook items
                self.codebook_initialized_count += init_size
            
            


        
                
        
        distance = torch.norm(encoder_out.unsqueeze(1) - self.codebook.weight, dim=-1)
        index = torch.argmin(distance, dim=-1)
        
        # [batch*seqlen, 1]
        encoding_indices = index.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.codebook_num, device=encoder_out.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.codebook.weight).view(encoding_indices.shape[0], -1)


        # Contrastive learning
        try:
            positive_examples = quantized
            negative_examples_nums = 20
            mask = torch.zeros(negative_examples_nums, self.codebook.num_embeddings, device=encoder_out.device)

            unique_encoding_indices = torch.unique(encoding_indices.squeeze(1))
            all_indices = torch.arange(self.codebook.num_embeddings, device=encoder_out.device)

            expanded_unique = unique_encoding_indices.unsqueeze(0)
            expanded_all = all_indices.unsqueeze(1)

            comparison = expanded_all == expanded_unique

            # Any column with a True value indicates the presence in unique_encoding_indices
            # We invert the result, so True becomes False and vice versa
            # Then, we use any(dim=1) to check if any True values exist across columns for each row
            not_in_unique = ~(comparison.any(dim=1))

            # Use this mask to filter out indices that are not in unique_encoding_indices
            negative_indices = all_indices[not_in_unique]
            # Shuffle
            negative_indices = negative_indices[torch.randperm(negative_indices.size(0))]
            negative_indices = negative_indices[:negative_examples_nums]
            
            # print(unique_encoding_indices)
            # print(negative_indices)

            negative_indices = negative_indices[:negative_examples_nums]
            
            mask.scatter_(1, negative_indices.unsqueeze(1), 1)

            # print(mask)

            # negative_examples = torch.matmul(mask, self.codebook.weight).view(-1, self.model.config.hidden_size)

            negative_examples = self.codebook(negative_indices).view(-1, self.model.config.hidden_size)


            # Calculate the cosine similarity for positive and negative pairs
            tau = 0.07  # Temperature parameter
            cosine_sim_pos = F.cosine_similarity(encoder_out, positive_examples)
            cosine_sim_neg = F.cosine_similarity(encoder_out.unsqueeze(1), negative_examples.unsqueeze(0), dim=-1)

            # Contrastive loss calculation
            denominator_pos = torch.exp(cosine_sim_pos / tau)
            denominator_neg = torch.sum(torch.exp(cosine_sim_neg / tau), dim=1)
            
            contrastive_loss = -torch.log(denominator_pos / (denominator_pos + denominator_neg))
        except Exception as e:
            print(e)
            contrastive_loss = torch.tensor(0.0).to(input_ids.device)



        vae_loss = torch.tensor(0.0)
        mse_loss_func = torch.nn.MSELoss()
        vae_loss = vae_loss.to(input_ids.device)


        vae_loss += mse_loss_func(quantized, encoder_out.detach())
        
        if self.centrifugal:
            vae_loss += contrastive_loss.mean()
        
        
        decoder_in = self.pass_grad(quantized, encoder_out) #.view(-1, self.model.config.hidden_size * 2)
        # [B*seq_len, layer*D]
        decoder_out = self.decoder(decoder_in)
        # decoder_out = decoder_in

        # [B, seq_len, layer, h, D//h]
        decoder_out = decoder_out.view(-1,self.seq_len,self.model.config.n_layer,self.model.config.n_head,self.model.config.n_embd//self.model.config.n_head)

        # [layer, B, h, seq_len, D//h]
        decoder_out = decoder_out.permute(2, 0, 3, 1, 4)
        decoder_out = decoder_out.contiguous()

        # [layer,1, B, h, seq_len, D//h]
        decoder_out = decoder_out.unsqueeze(1)

        # [layer,2, B, h, seq_len, D//h]
        decoder_out = decoder_out.repeat_interleave(2,1)

        # layer * [2, B, h, seq_len, D//h]
        decoder_out = decoder_out.split(1,dim=0)
        decoder_out = [item.squeeze(0) for item in decoder_out]




        prefix_attention_mask = torch.ones(attention_mask.shape[0], self.seq_len).to(attention_mask.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels, past_key_values=decoder_out,output_hidden_states= True)
        model_loss = outputs.loss


        last_hidden_state = outputs.hidden_states[-1]
        index = index
        # Select the last index
        not_neg_100 = labels != -100
        first_not_neg_100_positions = torch.argmax(not_neg_100.int(), dim=1)
        # print(first_not_neg_100_positions)
        # Prevent indices from being -1, i.e., the entire row is -100, in which case we can choose 0 as the index or perform other processing
        first_not_neg_100_positions = torch.clamp(first_not_neg_100_positions, min=0)
        # Select last_hidden_state based on the index calculated
        last_hidden_state = last_hidden_state[torch.arange(last_hidden_state.size(0)), first_not_neg_100_positions]
        # last_hidden_state = last_hidden_state[:, -1, :]        
        
        # index B*n,h 
        # last_hidden_state B,h
        # pos_tensor B,n,h
        
        batch_size = last_hidden_state.size(0)
        hidden_size = last_hidden_state.size(1)
        
        seq_embed = self.seq_embedding(torch.arange(self.seq_len).to(last_hidden_state.device))
        # seq_n,h -> B*seq_n,h
        seq_embed = seq_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        
        last_hidden_state_expanded = last_hidden_state.unsqueeze(1).expand(-1, self.seq_len, -1)

        last_hidden_state_pos = last_hidden_state_expanded + seq_embed
        last_hidden_state_pos = last_hidden_state_pos.view(-1,hidden_size)
        
        # # Create position vectors, each vector is all 0, all 1, ..., all n-1
        # pos_tensor = torch.arange(self.seq_len).to(last_hidden_state.device)
        # pos_tensor = pos_tensor.unsqueeze(-1).expand(self.seq_len, hidden_size)
        # pos_tensor = pos_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
        # # Expand/repeat last_hidden_state from [B, h] to [B, n, h] to match pos_tensor
        # last_hidden_state_expanded = last_hidden_state.unsqueeze(1).expand(-1, self.seq_len, -1)
        # # Concatenate last_hidden_state and pos_tensor to form a [B, n, 2*h] tensor
        # concatenated = torch.cat((last_hidden_state_expanded, pos_tensor), dim=2)
        # concatenated = concatenated.view(-1,hidden_size*2)
        
        code_pred = self.predicter(last_hidden_state_pos)
        # code_pred = code_pred.view(-1, self.codebook_num)

        # softmax_loss_func = torch.nn.CrossEntropyLoss()
        # code_loss = softmax_loss_func(code_pred, index)

        print(index)


        focal_loss_func = FocalLoss(gamma=2)  # You can adjust the gamma value as needed
        code_loss = focal_loss_func(code_pred, index)
        
        return model_loss.mean(), vae_loss.mean(), code_loss.mean()
    
    def my_prepare_model_inputs(self,*args,**kwargs):
        model_kwargs = self.ori_prepare_model_inputs(*args, **kwargs)
        

        if model_kwargs["past_key_values"] is None:
            model_kwargs["attention_mask"],model_kwargs["past_key_values"] = self.my_get_prompt(model_kwargs["input_ids"],model_kwargs["attention_mask"])
        else:
            model_kwargs["attention_mask"],_ = self.my_get_prompt(model_kwargs["input_ids"],model_kwargs["attention_mask"])


        return model_kwargs


    def my_get_prompt(self,input_ids,attention_mask):
        if self.generate_past_key_values is None:

            # print(input_ids.shape,attention_mask.shape)
                
            outputs = self.model(input_ids, attention_mask=attention_mask,output_hidden_states= True)
            
            last_hidden_state = outputs.hidden_states[-1]

            if self.lang == "en":
                last_hidden_state = last_hidden_state[:, -2, :]
            else:
                last_hidden_state = last_hidden_state[:, -1, :]
            
            batch_size = last_hidden_state.size(0)
            hidden_size = last_hidden_state.size(1)            
            
            # # Create position vectors, each vector is all 0, all 1, ..., all n-1

            # # Expand/repeat last_hidden_state from [B, h] to [B, n, h] to match pos_tensor
            # last_hidden_state_expanded = last_hidden_state.unsqueeze(1).expand(-1, self.seq_len, -1)

            # # Concatenate last_hidden_state and pos_tensor to form a [B, n, 2*h] tensor
            # concatenated = torch.cat((last_hidden_state_expanded, pos_tensor), dim=2)
            # concatenated = concatenated.view(-1,hidden_size*2)
            
            seq_embed = self.seq_embedding(torch.arange(self.seq_len).to(last_hidden_state.device))
            # seq_n,h -> B*seq_n,h
            seq_embed = seq_embed.unsqueeze(0).repeat(batch_size, 1, 1)
            
            last_hidden_state_expanded = last_hidden_state.unsqueeze(1).expand(-1, self.seq_len, -1)

            last_hidden_state_pos = last_hidden_state_expanded + seq_embed
            last_hidden_state_pos = last_hidden_state_pos.view(-1,hidden_size)
        
            score = self.predicter(last_hidden_state_pos)

            generate_index = score.view(-1, self.codebook_num).argmax(dim=-1)
            
            # print(generate_index)
            
            generate_past_key_values = self.codebook.weight.data[generate_index]


            # [B*seq_len, layer*D]
            generate_past_key_values = self.decoder(generate_past_key_values)

            # [B, seq_len, layer, h, D//h]
            generate_past_key_values = generate_past_key_values.view(-1,self.seq_len,self.model.config.n_layer,self.model.config.n_head,self.model.config.n_embd//self.model.config.n_head)

            # [layer, B, h, seq_len, D//h]
            generate_past_key_values = generate_past_key_values.permute(2, 0, 3, 1, 4)
            generate_past_key_values = generate_past_key_values.contiguous()

            # [layer,1, B, h, seq_len, D//h]
            generate_past_key_values = generate_past_key_values.unsqueeze(1)

            # [layer,2, B, h, seq_len, D//h]
            generate_past_key_values = generate_past_key_values.repeat_interleave(2,1)

            # 12 * [2, B, h, seq_len, D//h]
            generate_past_key_values = generate_past_key_values.split(1,dim=0)
            generate_past_key_values = [item.squeeze(0) for item in generate_past_key_values]

            self.generate_past_key_values = generate_past_key_values
        
        # Mask needs to be expanded each time
        prefix_attention_mask = torch.ones(attention_mask.shape[0], self.seq_len).to(attention_mask.device)
        # print(attention_mask.shape)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        return attention_mask,self.generate_past_key_values
        

    def generate(self,*args,**kwargs):
        # print(attention_mask.shape)
        # print(input_ids.shape)
        self.generate_past_key_values = None
        self.model.prepare_inputs_for_generation = self.my_prepare_model_inputs

        return self.model.generate(*args,**kwargs)
    
    
class MSModel_EM(nn.Module):
    def __init__(self,model,tokenizer,codebook_num=10,n=5,freeze=False,centrifugal=True,f_step=2000):
        super().__init__()
        self.model = model
        
        self.model.eval()

  
        self.persona_vector = None


    def forward(self, input_ids, attention_mask, labels=None, condition=None):

        hidden_states = self.model(output_hidden_states= True,**condition)


        encoder_in = hidden_states.hidden_states[-1]
        not_neg_100 = condition["attention_mask"] != 1
        first_not_neg_100_positions = torch.argmax(not_neg_100.int(), dim=1)


        # [B*n, D]
        encoder_in = encoder_in[torch.arange(encoder_in.size(0)),first_not_neg_100_positions]

        
        if self.persona_vector is None:
            self.persona_vector = encoder_in
        else:

            self.persona_vector = torch.cat((self.persona_vector, encoder_in.detach()), dim=0)
            self.persona_vector = torch.unique(self.persona_vector,dim=0)
            
        return self.persona_vector