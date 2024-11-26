import os
# os.environ["CUDA_VISIBLE_DEVIES"]="0,1,2,3"
import pandas as pd
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset,Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM,default_data_collator
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
from models import MSModel,MSModel_EM
from peft import LoraConfig, TaskType, PromptEncoderConfig, PrefixTuningConfig, PromptTuningConfig, get_peft_model
import json
# from pytorch_lightning.plugins import DeepSpeedPlugin

random.seed(2025)


def calculate_total_steps(train_dataset_size, batch_size, num_epochs, num_gpus):

    # 每个epoch中全局批次的数量
    global_batches_per_epoch = (train_dataset_size + batch_size - 1) // batch_size
    # 每个epoch中每个GPU处理的批次数量
    batches_per_epoch_per_gpu = (global_batches_per_epoch + num_gpus - 1) // num_gpus
    # 计算总步数
    total_steps_per_gpu = batches_per_epoch_per_gpu * num_epochs
    return total_steps_per_gpu

def multi_task_loss(losses, epsilon=1e-8):

    weighted_losses = [li / (li.detach() + epsilon) for li in losses]
    total_loss = sum(weighted_losses)
    return total_loss


class MyData(pl.LightningDataModule):
    def __init__(self,batch_size=16,tokenizer=None,max_condition_length = 128,max_length = 256, lang="zh", data_path="",role_aware=True, seq_len=5):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_condition_length = max_condition_length
        self.max_length = max_length
        self.lang = lang
        self.data_path = data_path
        self.role_aware = role_aware
        self.seq_len = seq_len
        self.user_name = "<P1>"
        self.bot_name = "<P2>"

    
    def prepare_data(self):
        print("prepare_data!!!!!!!!!!")
        role_aware = self.role_aware
        self.data_train = []
        with open(self.data_path + "/train_datas_" + self.lang + ".json", "r" ,encoding="utf-8") as f:
            datas = json.load(f)
            random.shuffle(datas)
            count = 0
            select_list = random.sample(range(len(datas)),len(datas)//100)
            for data in datas:
                
                
                persona, context, response = data["persona"], data["context"], data["response"]
                
                if self.lang == "en":
                    for c in range(len(context)):
                        if c%2==0:
                            context[c] = self.user_name + context[c]
                        else:
                            context[c] = self.bot_name + context[c]

                    if len(context)%2 == 0:
                        response = self.user_name + response
                    else:
                        response = self.bot_name + response

                    # context = "".join(context)
                    context = self.tokenizer.sep_token.join(context)

                if self.lang == "zh":
                    context = self.tokenizer.sep_token.join(context)
                
                if context == "":
                    context = " "
                    


                context = self.tokenizer.sep_token.join(persona) + self.tokenizer.sep_token + context
                
                self.data_train.append([context, response, persona])
                count+=1 
        
        random.shuffle(self.data_train)
        
        self.data_valid = []
        self.data_test = []
        
        with open(self.data_path + "/val_datas_" + self.lang + ".json", "r" ,encoding="utf-8") as f:
            datas = json.load(f)
            for data in datas:
                persona, context, response = data["persona"], data["context"], data["response"]
                

                if self.lang == "en":
                    for c in range(len(context)):
                        if c%2==0:
                            context[c] = self.user_name + context[c]
                        else:
                            context[c] = self.bot_name + context[c]

                    if len(context)%2 == 0:
                        response = self.user_name + response
                    else:
                        response = self.bot_name + response

                    context = self.tokenizer.sep_token.join(context)

                if self.lang == "zh":
                    context = self.tokenizer.sep_token.join(context)
                
                if context == "":
                    context = " "
                    

                context = self.tokenizer.sep_token.join(persona) + self.tokenizer.sep_token + context
                
                self.data_valid.append([context, response, persona])

        with open(self.data_path + "/test_datas_" + self.lang + ".json", "r" ,encoding="utf-8") as f:
            datas = json.load(f)
            for data in datas:
                persona, context, response = data["persona"], data["context"], data["response"]

                if self.lang == "en":
                    for c in range(len(context)):
                        if c%2==0:
                            context[c] = self.user_name + context[c]
                        else:
                            context[c] = self.bot_name + context[c]

                    if len(context)%2 == 0:
                        response = self.user_name + response
                    else:
                        response = self.bot_name + response



                if len(context) == 0:
                    context = [" "]
                

                self.data_test.append([context, response, persona])       
            
        random.shuffle(self.data_valid)
        print(self.data_train[123],self.data_valid[123],self.data_test[123])        


    def setup(self, stage=None):
        pass
        
    def data_collator(self,samples):

        tokenizer = self.tokenizer
        max_length_condition= self.max_condition_length
        max_length = self.max_length

        def process_func(examples):
            batch_size = len(examples["context"])
            
            
            # print(batch_size,examples["instruction"])
            inputs = []
            for i in range(batch_size):
                if not examples["context"][i]:
                    inputs = inputs + [" "]
                else:
                    inputs = inputs + [examples["context"][i]]
            # print(inputs)
            targets = [str(x) for x in examples["response"]]


            model_inputs = self.tokenizer(inputs,add_special_tokens=False)
            labels = self.tokenizer(targets,add_special_tokens=False)

            for i in range(batch_size):
                if self.lang == "zh":
                    sample_input_ids = [tokenizer.cls_token_id] + model_inputs["input_ids"][i] + [tokenizer.sep_token_id]
                else:
                    sample_input_ids = [tokenizer.cls_token_id] + model_inputs["input_ids"][i] + [tokenizer.sep_token_id]

                label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
                # print(i, sample_input_ids, label_input_ids)

                model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
                
                # labels["input_ids"][i] = sample_input_ids + label_input_ids
                labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
                
                model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
  
            for i in range(batch_size):
                sample_input_ids = model_inputs["input_ids"][i]
                label_input_ids = labels["input_ids"][i]

                model_inputs["input_ids"][i] = sample_input_ids + [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
                )
                model_inputs["attention_mask"][i] =  model_inputs["attention_mask"][i] + [0] * (max_length - len(sample_input_ids))
                labels["input_ids"][i] = label_input_ids + [-100] * (max_length - len(sample_input_ids))

                model_inputs["input_ids"][i] = model_inputs["input_ids"][i][:max_length]
                model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i][:max_length]
                labels["input_ids"][i] = labels["input_ids"][i][:max_length]

            # decode
            decode_ex = self.tokenizer.decode(model_inputs["input_ids"][0],skip_special_tokens=False)
            print(decode_ex)

            model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"])
            model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"])
            model_inputs["labels"] = torch.tensor(labels["input_ids"])

            batch = dict()
            batch["input_ids"] = model_inputs["input_ids"]
            batch["attention_mask"] = model_inputs["attention_mask"]
            batch["labels"] = model_inputs["labels"]


            return batch
        
        examples = dict()
        examples["context"] = [x[0] for x in samples]
        examples["response"] = [x[1] for x in samples]
        examples["persona"] = [x[2] for x in samples]

        batch_data = process_func(examples)

        return batch_data
    
    def data_collator_test(self,samples):
        examples = dict()
        examples["context"] = [x[0] for x in samples]
        examples["response"] = [x[1] for x in samples]
        examples["persona"] = [x[2] for x in samples]

        return examples

    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, collate_fn=self.data_collator, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_valid, collate_fn=self.data_collator, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, collate_fn=self.data_collator_test, batch_size=self.batch_size)


class MyLightningModel(pl.LightningModule):
    def __init__(self, model, tokenizer,t_total,lr,model_file,total_steps_hand,lang="zh"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer 
        self.t_total = t_total
        self.lr = lr
        self.model_file = model_file
        self.lang = lang

        # self.automatic_optimization = False
        
        self.total_steps_hand = total_steps_hand
        self.val_step_outputs = []

    def training_step(self, batch, batch_idx):

        
        outputs = self.model(**batch)
        loss = outputs.loss

        self.log('train_loss',loss.item(),on_epoch=True,on_step = True,prog_bar=True,logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('val_loss', loss.item(),on_epoch=True,on_step = True,prog_bar=True,logger=True)
        
        self.val_step_outputs.append(loss.item())

    def on_validation_epoch_end(self):
        avg_loss = torch.tensor(self.val_step_outputs).mean()
        self.log('avg_val_loss', avg_loss,on_epoch=True,prog_bar=True,logger=True)
        self.val_step_outputs.clear()
        
    def test_step(self, batch, batch_idx):
        # generat
        self.model.eval()
        
        temp_dict = {'persona':[],'context':[],'pred':[],'target':[]}
        
        print(batch)
        
        for b in range(len(batch["context"])):
            
            
            if len(batch["context"][b]) == 0:
                batch["context"][b] = [" "]
            persona_context = batch["context"][b]
            gold_response = batch["response"][b]


            while len(self.tokenizer.encode(self.tokenizer.cls_token + self.tokenizer.sep_token.join(persona_context) + self.tokenizer.eos_token + "<P2>",add_special_tokens=False)) > 512:
                persona_context = persona_context[1:] 


            if self.lang == "en":
                if "<P1>" in persona_context[-1]:
                    persona_context = self.tokenizer.cls_token + self.tokenizer.sep_token.join(persona_context) + self.tokenizer.eos_token + "<P2>"
                else:
                    persona_context = self.tokenizer.cls_token + self.tokenizer.sep_token.join(persona_context) + self.tokenizer.eos_token + "<P1>"
            else:
                persona_context = self.tokenizer.cls_token + self.tokenizer.sep_token.join(persona_context) + self.tokenizer.eos_token

            
            inputs = self.tokenizer(
                persona_context,
                return_tensors="pt",
                truncation = True,
                max_length = 512,
                add_special_tokens = False,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            
            print("eos_token",self.tokenizer.eos_token_id,self.tokenizer.eos_token)
            
            outputs = model.model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],max_new_tokens=128,eos_token_id = self.tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,top_k=50,top_p=0.95,
            )
            
            generated_response = self.tokenizer.decode(outputs.detach().cpu().numpy()[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            print(self.tokenizer.decode(inputs["input_ids"].detach().cpu().numpy()[0], skip_special_tokens=False))
            print(self.tokenizer.decode(outputs.detach().cpu().numpy()[0], skip_special_tokens=False))
            
            if self.lang == "zh":
                generated_response = generated_response.replace(" ", "")
            else:
                generated_response = generated_response.replace("Ġ", " ")
                
            print(generated_response)
            

            batch["context"][b] = ''.join(batch["context"][b]).replace(self.tokenizer.sep_token,"").replace("<P2>","").replace("<P1>","")
            gold_response = gold_response.replace("<P2>","").replace("<P1>","")

            if self.lang == "zh":
                temp_dict['persona'].append("。".join(batch["persona"][b])+"。")
            else:
                temp_dict['persona'].append(".".join(batch["persona"][b])+".")

            temp_dict['context'].append(batch["context"][b])
            temp_dict['pred'].append(generated_response)
            temp_dict['target'].append(gold_response)
        
        return temp_dict
    
    def test_epoch_end(self, outputs):
        temp_dict = {'persona':[],'context':[],'pred':[],'target':[]}
        for output in outputs:
            temp_dict['persona'] += output['persona']
            temp_dict['context'] += output['context']
            temp_dict['pred'] += output['pred']
            temp_dict['target'] += output['target']
        

        pred_list = []
        for i in range(len(temp_dict['persona'])):
            pred_list.append(temp_dict['persona'][i]+ '\n' + temp_dict['context'][i] + '\n'+  temp_dict['pred'][i]+ '\n'+ temp_dict['target'][i])
        with open('./generation/' + self.model_file + '-pred_result.txt','w',encoding='utf-8') as f:
            f.write('\n\n'.join(pred_list))
            
        pred_list = []
        for i in range(len(temp_dict['persona'])):
            pred_list.append({"persona":temp_dict['persona'][i], "context":temp_dict['context'][i], "pred":temp_dict['pred'][i],"target":temp_dict['target'][i]})
        with open('./generation/' + self.model_file + '-pred_result.json','w',encoding='utf-8') as f:
            json.dump(pred_list,f,ensure_ascii=False)
        
            
            



    def configure_optimizers(self):
        print("self.trainer.estimated_stepping_batches:",self.trainer.estimated_stepping_batches)
        self.warm_up_steps = self.trainer.estimated_stepping_batches // 20
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.001)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, self.warm_up_steps, self.trainer.estimated_stepping_batches
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        
        
        return [optimizer], [scheduler]


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='GPT model training parameters')
    

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--gpus', nargs='+', type=int, default=[2])
    

    parser.add_argument('--max_condition_length', type=int, default=128)
    parser.add_argument('--max_input_length', type=int, default=256)
    parser.add_argument('--max_output_length', type=int, default=256)
    parser.add_argument('--model_file', type=str, default='gpt_0424_peft')
    
 
    parser.add_argument('--lang', type=str, default='en',choices=['zh','en'])
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--role_aware', type=bool, default=True)
    

    parser.add_argument('--initial_method', type=str, default='em')
    parser.add_argument('--centrifugal', type=bool, default=True)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--codebook_num', type=int, default=100)
    parser.add_argument('--em_init', type=bool, default=True)
    parser.add_argument('--freeze', type=bool, default=False)
    parser.add_argument('--peft', type=str, default=None,choices=[None,'lora','p-tuning','prefix-tuning','prompt-tuning'])
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    max_condition_length = args.max_condition_length
    max_input_length = args.max_input_length
    max_output_length = args.max_output_length
    model_file = args.model_file
    lang = args.lang
    gpus = args.gpus
    data_path = args.data_path
    role_aware = args.role_aware
    initial_method = args.initial_method
    centrifugal = args.centrifugal
    seq_len = args.seq_len
    codebook_num = args.codebook_num
    em_init = args.em_init
    freeze = args.freeze
    peft = args.peft
    lr = args.lr
    
    model_file = model_file + "_" + "_".join([lang,"peft="+str(peft),"role_aware="+str(role_aware),initial_method,"seq_len="+str(seq_len),"codebook_num="+str(codebook_num),"centrifugal="+str(centrifugal)])



    if lang == "zh":
        model = AutoModelForCausalLM.from_pretrained("./models/gpt2-chinese-cluecorpussmall")
        tokenizer = AutoTokenizer.from_pretrained("./models/gpt2-chinese-cluecorpussmall")
        tokenizer.eos_token = tokenizer.sep_token 
    else:
        model = AutoModelForCausalLM.from_pretrained("./models/gpt2")
        tokenizer = AutoTokenizer.from_pretrained("./models/gpt2")        
        tokenizer.sep_token = tokenizer.eos_token
        tokenizer.cls_token = tokenizer.eos_token
        tokenizer.pad_token = tokenizer.eos_token
        special_tokens = {'additional_special_tokens': ["<P1>", "<P2>"]}
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model.tie_weights()

    print(tokenizer.eos_token,tokenizer.eos_token_id)
    
    
    dm = MyData(batch_size=batch_size,tokenizer=tokenizer,max_length=max_input_length+max_output_length,lang=lang,data_path=data_path,role_aware=role_aware,seq_len=seq_len)

    if em_init:
        dm.prepare_data()
        model_init = MSModel_EM(model,tokenizer,codebook_num=codebook_num,n=seq_len,centrifugal=centrifugal)
        model_init.to("cuda:0")

        from tqdm import tqdm
        for batch in tqdm(dm.train_dataloader()):
            batch = {k: v.to("cuda:0") for k, v in batch.items()}
            persona_vector = model_init.forward(**batch)
            torch.save(persona_vector,"./vector/persona_vector_"+lang)
 
        model_init.persona_vector = torch.unique(model_init.persona_vector, dim=0)
        torch.save(model_init.persona_vector,"./vector/persona_vector_"+lang)

        del model_init
    


    train_dataset_size = 50000
    num_gpus = 1  
    total_steps_per_gpu = calculate_total_steps(train_dataset_size, batch_size, num_epochs, num_gpus)
    print(f"Steps/GPU: {total_steps_per_gpu}")

    # peft 
    if peft == "lora":
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    elif peft == "p-tuning":
        peft_config = config = PromptEncoderConfig(
                peft_type="P_TUNING",
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=5,
                token_dim=768,
                num_transformer_submodules=1,
                num_attention_heads=12,
                num_layers=12,
                encoder_reparameterization_type="MLP",
                encoder_hidden_size=768,
            )
    elif peft == "prefix-tuning":
        peft_config = PrefixTuningConfig(
            peft_type="PREFIX_TUNING",
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=5,
            token_dim=768,
            num_transformer_submodules=1,
            num_attention_heads=12,
            num_layers=12,
            encoder_hidden_size=768,
        )
    elif peft == "prompt-tuning":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=5,
        )

    if peft:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()





    model = MyLightningModel(model,tokenizer,t_total=1,lr=lr,model_file=model_file,total_steps_hand=total_steps_per_gpu,lang=lang)

    import time
    now_time = time.strftime("%m%d%H", time.localtime())
    now_time = str(now_time)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        save_top_k=1,
        monitor="avg_val_loss",
        mode="min",
        save_weights_only = True,
        filename= model_file + '-{epoch:02d}-{step}-{avg_val_loss:.2f}'+ "_" + now_time,
        save_last=True
    )

    learning_rate_callback = LearningRateMonitor()

    early_stopping = EarlyStopping(
        monitor='avg_val_loss', 
        min_delta=0.0, 
        patience=3, 
        mode='min', 
        strict=True
    )

    logger = TensorBoardLogger('log', name=model_file)


    trainer = pl.Trainer(
        default_root_dir='./checkpoints',
        max_epochs=num_epochs,
        devices=gpus,
        val_check_interval=0.5,
        callbacks=[learning_rate_callback, checkpoint_callback,early_stopping],
        logger = logger,
        accelerator = 'gpu',
    )  

    trainer.fit(model, datamodule=dm)

    
    best_model_path = checkpoint_callback.best_model_path
    
    
    best_model_state = torch.load(best_model_path)["state_dict"]
    model.load_state_dict(best_model_state, strict=False)
    
    print(best_model_path)
    
    trainer.test(model,datamodule=dm)

   


