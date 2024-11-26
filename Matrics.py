from rouge import Rouge
from tqdm import tqdm 
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
from nltk import word_tokenize, sent_tokenize 
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from nltk import bigrams, FreqDist
import torch
from Consis_Model import predict
from transformers import AutoTokenizer
import json
import random
import math
import argparse

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

def get_tokens_list(strs,tokenizer):

    encoded_input_ids = tokenizer.encode(strs, return_tensors="pt")

    input_ids_list = encoded_input_ids.squeeze().tolist()

    decoded_output_list = [tokenizer.decode([input_id], skip_special_tokens=True) for input_id in input_ids_list]


    decoded_output_list = list(filter(None, decoded_output_list))

    # print("Decoded output list:", decoded_output_list)
    
    return decoded_output_list

class NLP_Gen_Matrics:
    def __init__(self,condition_list=['test'],model_name='test_model',model_path_en="./models/consis_model_EN.ckpt",model_path_zh="./models/consis_model_ZH.ckpt"):
        super(NLP_Gen_Matrics, self).__init__()
        self.judge_value_list = dict()
        self.condition_list = condition_list
        self.model_name = model_name
        self.model_path_en = model_path_en
        self.model_path_zh = model_path_zh
    
    def get_judge_data(self,result_path):
        # 格式: 'I[CSE]am[CSE]you\nyou[CSE]am[CSE]I\n\n'
        with open(result_path,'r',encoding='utf-8') as f:
            data = json.load(f)
        persona_list = []
        context_list = []
        pred_list = []
        target_list = []
        for i in range(len(data)):
            temp = data[i]

            
            if len(temp)==4:

                persona_list.append(temp["persona"])
                context_list.append(temp["context"])
                pred_list.append(temp["pred"])
                target_list.append(temp["target"])
                
        
        print(pred_list[0],target_list[0])
    
        return self.get_result(persona_list,context_list,pred_list,target_list)
        
        
    def get_result(self,persona_list,context_list,pred_list,target_list):
        
        if is_contains_chinese(target_list[0]):
            model_name = "./models/bert-base-chinese"
        else:
            model_name = "./models/bert-base-uncased"
        
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        pred_list_tokens = []
        target_list_tokens = []
        for pred in pred_list:
            pred_list_tokens.append(get_tokens_list(pred,tokenizer))
        for target in target_list:
            target_list_tokens.append(get_tokens_list(target,tokenizer))
            
        self.get_distinct(pred_list_tokens,target_list_tokens)
        self.get_coherence(pred_list_tokens,target_list_tokens)
        self.get_pc(persona_list,pred_list)
        
        print(self.judge_value_list)
        
        self.get_consistency(persona_list,context_list,pred_list,target_list)

        
        file_name = '_'.join(self.condition_list) + '_' + self.model_name

        # 保存结果
        with open('./result/' + file_name + '-metrics.txt','w',encoding='utf-8') as f:
            f.write('condition_list:  '+ self.model_name + str(self.condition_list) + '\n')
            for key,value in self.judge_value_list.items():
                f.write(key + ' : ' + str(value) + '\n')
        
        return self.judge_value_list
            

    def get_distinct(self, pred,target):
        
        
        corpus = pred
        unigrams = []
        bigrams = []
        for n,rep in enumerate(corpus):
            temp = rep
            unigrams += temp
            for i in range(len(temp)-1):
                bigrams.append(temp[i] + ' ' + temp[i+1])
        self.judge_value_list['c-distinct-1'] = len(set(unigrams)) * 1.0 / len(unigrams)
        self.judge_value_list['c-distinct-2'] = len(set(bigrams)) * 1.0 / len(bigrams)


        pred_list = pred[:]
        target_list = pred[:]
        
        random.shuffle(target_list)

        num = 0
        bleu_score_all_1 = 0
        bleu_score_all_2 = 0
        bleu_score_all_3 = 0
        bleu_score_all_4 = 0
        for pred,target in zip(pred_list,target_list):
            if pred == target:
                continue
            bleu_score_1 = sentence_bleu([target], pred,weights=(1, 0, 0, 0))
            bleu_score_all_1 += bleu_score_1
            bleu_score_2 = sentence_bleu([target], pred,weights=(0.5, 0.5, 0, 0))
            bleu_score_all_2 += bleu_score_2
            bleu_score_3 = sentence_bleu([target], pred,weights=(0.33, 0.33, 0.33, 0))
            bleu_score_all_3 += bleu_score_3
            bleu_score_4 = sentence_bleu([target], pred,weights=(0.25, 0.25, 0.25, 0.25))
            bleu_score_all_4 += bleu_score_4
            num+=1
        self.judge_value_list['sbleu-1'] = bleu_score_all_1/num
        self.judge_value_list['sbleu-2'] = bleu_score_all_2/num
        self.judge_value_list['sbleu-3'] = bleu_score_all_3/num
        self.judge_value_list['sbleu-4'] = bleu_score_all_4/num
        


    def get_pc(self,personas_list, pred_list):

        all_personas_list = []
        for personas in personas_list:
            if "。" in personas:
                all_personas_list += personas.split('。')
            else:
                all_personas_list += personas.split('.')

        if is_contains_chinese(all_personas_list[0]):
            model_name = "./models/bert-base-chinese"
        else:
            model_name = "./models/bert-base-uncased"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        all_personas_list_tokens = []
        for persona in all_personas_list:
            all_personas_list_tokens.append(get_tokens_list(persona,tokenizer))

        idf_dict = dict()
        for persona in all_personas_list_tokens:
            for token in persona:
                if token in idf_dict:
                    idf_dict[token] += 1
                else:
                    idf_dict[token] = 1
        
        idf_dict["。"] = len(all_personas_list)
        idf_dict["."] = len(all_personas_list)
        for key in idf_dict.keys():
            idf_dict[key] = 1 / (1 + math.log(1+idf_dict[key]))

        p_cover_value_all = []
        for i in range(len(pred_list)):
            if "。" in personas_list[i]:
                personas = personas_list[i].split('。')
                personas = [i+"。" for i in personas if i != '']
            else:
                personas = personas_list[i].split('.')
                personas = [i+"." for i in personas if i != '']
            pred = pred_list[i]
            pred_tokens = get_tokens_list(pred,tokenizer)
            p_cover = []
            for persona in personas:
                perspona_tokens = get_tokens_list(persona,tokenizer)
                # 共现词
                cover_word_list = [token for token in perspona_tokens if token in pred_tokens]
                f = 0
                for token in cover_word_list:
                    f += idf_dict[token]
                if len(cover_word_list) == 0:
                    p_cover.append(0)
                else:
                    f = f / len(cover_word_list)
                    p_cover.append(f)
            # 选择最大的
            if len(p_cover) == 0:
                p_cover_value_all.append(0)
            else:
                p_cover_value_all.append(max(p_cover))
        
        self.judge_value_list['idf'] = sum(p_cover_value_all) / len(p_cover_value_all)




    
    def get_consistency(self,persona_list,context_list,pred_list,target_list):

        if is_contains_chinese(target_list[0]):
            lang = "zh"
        else:
            lang = "en"

        batch_data = []
        for i in range(len(pred_list)):
            print([persona_list[i],context_list[i],pred_list[i]])
            batch_data.append([persona_list[i],context_list[i],pred_list[i]])
        y_hat = predict(batch_data, lang=lang,model_path_en=self.model_path_en,model_path_zh=self.model_path_zh)

        self.judge_value_list['consistency-coherence'] = y_hat.count(2) / len(y_hat)
        self.judge_value_list['consistency'] = (y_hat.count(1) + y_hat.count(2)) / len(y_hat)
    
    

    def get_coherence(self,pred_list,target_list):
        
        # bleu
        num = 0
        bleu_score_all_1 = 0
        bleu_score_all_2 = 0
        bleu_score_all_3 = 0
        bleu_score_all_4 = 0
        for pred,target in zip(pred_list,target_list):
            bleu_score_1 = sentence_bleu([target], pred,weights=(1, 0, 0, 0))
            bleu_score_all_1 += bleu_score_1
            bleu_score_2 = sentence_bleu([target], pred,weights=(0.5, 0.5, 0, 0))
            bleu_score_all_2 += bleu_score_2
            bleu_score_3 = sentence_bleu([target], pred,weights=(0.33, 0.33, 0.33, 0))
            bleu_score_all_3 += bleu_score_3
            bleu_score_4 = sentence_bleu([target], pred,weights=(0.25, 0.25, 0.25, 0.25))
            bleu_score_all_4 += bleu_score_4
            num+=1
        self.judge_value_list['bleu-1'] = bleu_score_all_1/num
        self.judge_value_list['bleu-2'] = bleu_score_all_2/num
        self.judge_value_list['bleu-3'] = bleu_score_all_3/num
        self.judge_value_list['bleu-4'] = bleu_score_all_4/num        
        
        
        # rouge
        rouge = Rouge()
        rouge_list = [[],[],[]]
        for pred,target in zip(pred_list,target_list):
            if len(target) <= 1:
                continue 
            if len(pred) <= 1:
                pred.append('<UNK>')
            if len(target) <= 1:
                target.append('<UNK>')
            rouge_score = rouge.get_scores(" ".join(pred), " ".join(target))
            rouge_list[0].append(rouge_score[0]['rouge-1']['r'])
            rouge_list[1].append(rouge_score[0]['rouge-2']['r'])
            rouge_list[2].append(rouge_score[0]['rouge-l']['r'])
        self.judge_value_list['rouge-1'] = sum(rouge_list[0]) / len(rouge_list[0])
        self.judge_value_list['rouge-2'] = sum(rouge_list[1]) / len(rouge_list[1])
        self.judge_value_list['rouge-l'] = sum(rouge_list[2]) / len(rouge_list[2])

    

        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='fine-tuning')
    parser.add_argument('--condition_list', type=str, default='2024-xx-xx,fine-tuning')
    parser.add_argument('--result_path', type=str, default='./generation/xxx.json')
    parser.add_argument('--model_path_en', type=str, default='./models/consis_model_EN.ckpt')
    parser.add_argument('--model_path_zh', type=str, default='./models/consis_model_ZH.ckpt')
    args = parser.parse_args()
    
    condition_list = args.condition_list.split(',')
    judger = NLP_Gen_Matrics(model_name=args.model_name,condition_list=condition_list,model_path_en=args.model_path_en,model_path_zh=args.model_path_zh)
    result = judger.get_judge_data(args.result_path)

    for key in result.keys():
        print(key,result[key])
        
