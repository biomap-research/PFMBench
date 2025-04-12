import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data.protein_dataset import dynamic_pad
from src.data.esm.sdk.api import LogitsConfig
import os
import pickle
from tqdm import tqdm
import sys; sys.path.append('/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom')

MODEL_ZOOM_PATH = '/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom'

class PretrainModelInterface(nn.Module):
    """
    Interface for pre-trained models.
    """

    def __init__(self, pretrain_model_name, batch_size = 64, max_length = 1022, device = 'cuda', sequence_only=False):
        super(PretrainModelInterface, self).__init__()
        self.pretrain_model_name = pretrain_model_name
        self.sequence_only = sequence_only
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.setup_model()
        
    def setup_model(self):
        """
        Setup the pre-trained model based on the specified name.
        """
        if self.pretrain_model_name == 'esm2_650m':
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            self.pretrain_model = AutoModelForMaskedLM.from_pretrained(f"{MODEL_ZOOM_PATH}/esm2_650m").to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_ZOOM_PATH}/esm2_650m")
        elif self.pretrain_model_name == 'esm3_1.4b':
            from model_zoom.esm.models.esm3 import ESM3
            self.pretrain_model = ESM3.from_pretrained("esm3-sm-open-v1", ).to(self.device)
        elif self.pretrain_model_name == 'esmc_600m':
            from model_zoom.esm.models.esmc import ESMC
            self.pretrain_model = ESMC.from_pretrained("esmc_600m").to(self.device)
        elif self.pretrain_model_name == 'procyon':
            # ProCyon uses ESM2-3B and GearNet as the protein sequence encoder and the protein structure 
            # encoder, respectively.
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            from model_zoom.procyon.model.model_unified import UnifiedProCyon
            os.environ["HOME_DIR"] = MODEL_ZOOM_PATH
            os.environ["DATA_DIR"] = "/nfs_beijing/wanghao/2025-onesystem/vllm/ProCyon-Instruct"
            os.environ["LLAMA3_PATH"] = "/nfs_beijing/wanghao/2025-onesystem/vllm/Meta-Llama-3-8B"
            procyon_ckpt = '/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/procyon/model_weights/ProCyon-Full'
            self.esm_pretrain_model = AutoModelForMaskedLM.from_pretrained(f"{MODEL_ZOOM_PATH}/esm2_3b").to(self.device)
            self.esm_pretrain_model.eval()
            self.esm_tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_ZOOM_PATH}/esm2_3b")

            # Since the env problem of torchdrug, we load the structure embeddings instead of using GearNet to perfrom the online inference
            with open(f"{MODEL_ZOOM_PATH}/../datasets/gearnet_features/gearnet_features.pkl", "rb") as file:
                self.structure_embeddings = pickle.load(file)

            # procyon initialization
            self.pretrain_model, _ = UnifiedProCyon.from_pretrained(
                pretrained_weights_dir=procyon_ckpt, 
                checkpoint_dir=procyon_ckpt
            )
            self.pretrain_model = self.pretrain_model.to(self.device)
        elif self.pretrain_model_name == 'prollama':
            from transformers import LlamaForCausalLM, LlamaTokenizer
            llama_path = "/nfs_beijing/kubeflow-user/wanghao/workspace/ai4sci/protein_benchmark_project/data/ProLLaMA"
            self.pretrain_model = LlamaForCausalLM.from_pretrained(
                llama_path,
                # torch_dtype=torch.float16,
                # low_cpu_mem_usage=True,
                # device_map='auto',
                quantization_config=None
            ).to(self.device)
            self.tokenizer = LlamaTokenizer.from_pretrained(llama_path)
        elif self.pretrain_model_name == 'progen2':
            from model_zoom.progen2.modeling_progen import ProGenForCausalLM
            from tokenizers import Tokenizer
            def create_tokenizer_custom(file):
                with open(file, 'r') as f:
                    return Tokenizer.from_str(f.read())
                
            self.pretrain_model = ProGenForCausalLM.from_pretrained(f'{MODEL_ZOOM_PATH}/progen2').to(self.device)
            self.tokenizer = create_tokenizer_custom(file=f'{MODEL_ZOOM_PATH}/progen2/tokenizer.json')
        elif self.pretrain_model_name == 'prostt5':
            from transformers import T5Tokenizer, T5EncoderModel
            import mini3di
            self.tokenizer = T5Tokenizer.from_pretrained(f'{MODEL_ZOOM_PATH}/ProstT5', do_lower_case=False)
            self.pretrain_model = T5EncoderModel.from_pretrained(f"{MODEL_ZOOM_PATH}/ProstT5").to(self.device)
            self.encoder_3di = mini3di.Encoder()
        elif self.pretrain_model_name == 'protgpt2':
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_ZOOM_PATH}/ProtGPT2")
            self.pretrain_model = AutoModelForCausalLM.from_pretrained(f"{MODEL_ZOOM_PATH}/ProtGPT2").to(self.device)
        elif self.pretrain_model_name == 'protrek':
            from model_zoom.ProTrek.model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel
            import mini3di
            self.encoder_3di = mini3di.Encoder()
            config = {
                "protein_config": f"{MODEL_ZOOM_PATH}/ProTrek/ProTrek_650M_UniRef50/esm2_t33_650M_UR50D",
                "text_config": f"{MODEL_ZOOM_PATH}/ProTrek/ProTrek_650M_UniRef50/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                "structure_config": f"{MODEL_ZOOM_PATH}/ProTrek/ProTrek_650M_UniRef50/foldseek_t30_150M",
                "load_protein_pretrained": False,
                "load_text_pretrained": False,
                "from_checkpoint": f"{MODEL_ZOOM_PATH}/ProTrek/ProTrek_650M_UniRef50/ProTrek_650M_UniRef50.pt"
            }
            self.pretrain_model = ProTrekTrimodalModel(**config).eval().to(self.device)
        elif self.pretrain_model_name == 'saport':
            from transformers import EsmTokenizer, EsmForMaskedLM
            import mini3di
            self.encoder_3di = mini3di.Encoder()
            self.tokenizer = EsmTokenizer.from_pretrained(f'{MODEL_ZOOM_PATH}/SaPort/ckpt')
            self.pretrain_model = EsmForMaskedLM.from_pretrained(f'{MODEL_ZOOM_PATH}/SaPort/ckpt').to(self.device)
        else:
            raise ValueError(f"Unknown pretrain model name: {self.pretrain_model_name}")
            
    def construct_batch(self, data, batch_size, task_name=None):
        """
        Constructs batches of data.

        Args:
            data (list): List of data samples.
            batch_size (int): Size of each batch.

        Yields:
            dict: A batch of data.
        """
        for i in range(0, len(data), batch_size):
            if self.pretrain_model_name == 'esm2_650m':
                name_batch, X_batch, S_batch, mask_batch, label_batch = [], [], [], [], []
                
                max_length_batch = max([len(sample['seq']) for sample in data[i:i + batch_size]])+2
                    
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    label = sample['label']
                    X = sample['X']
                    attention_mask = torch.zeros(max_length_batch)
                    
                    seq_token = torch.tensor(self.tokenizer.encode(seq))
                    attention_mask[:len(seq_token)] = 1
                    seq_token = self.pad_data(seq_token, dim=0, max_length=max_length_batch)
                    
                    X = dynamic_pad(X, [1, 1], dim=0, pad_value=0)
                    X = self.pad_data(X, dim=0, max_length=max_length_batch)
                    
                    
                    S_batch.append(seq_token)
                    X_batch.append(X)
                    mask_batch.append(attention_mask)
                    label_batch.append(torch.tensor(label))
                    name_batch.append(sample['name'])
        
                yield {
                    'name': name_batch,
                    'seq': torch.stack(S_batch).to(self.device),
                    'X': torch.stack(X_batch).to(self.device),
                    'attention_mask': torch.stack(mask_batch).to(self.device)==1,
                    'label': torch.stack(label_batch).to(self.device),
                }
            
            if self.pretrain_model_name == 'esm3_1.4b':
                from model_zoom.esm.utils.misc import stack_variable_length_tensors
                from model_zoom.esm.utils.sampling import _BatchedESMProteinTensor
                from model_zoom.esm.utils import encoding
                model = self.pretrain_model
                name_batch, sequence_list, coordinates_list, label_batch = [], [], [], []
                seq_tokenizer = model.tokenizers.sequence
                struct_tokenizer = model.tokenizers.structure
                pad = model.tokenizers.sequence.pad_token_id
                for sample in data[i:i + batch_size]:
                    sequence_list.append(sample['seq'])
                    coordinates_list.append(sample['X'])
                    name_batch.append(sample['name'])
                    label_batch.append(torch.tensor(sample['label']))
                
                sequence_tokens = stack_variable_length_tensors(
                    [
                        encoding.tokenize_sequence(x, seq_tokenizer, add_special_tokens=True)
                        for x in sequence_list
                    ],
                    constant_value=pad,
                ).to(self.device)
                
                structure_tokens_batch = []
                coordinates_batch = []
                mask_batch = []
                for coordinate in coordinates_list:
                    mask = torch.zeros(self.max_length)
                    coordinates, plddt, structure_token = encoding.tokenize_structure(coordinate, model.get_structure_encoder(), struct_tokenizer, add_special_tokens=True)
                    mask[:coordinates.shape[0]] = 1
                    structure_tokens_batch.append(structure_token)
                    coordinates_batch.append(coordinates)
                    mask_batch.append(mask)
                
                structure_tokens_batch = stack_variable_length_tensors(
                            structure_tokens_batch,
                            constant_value=pad,
                        ).to(self.device)
                                
                coordinates_batch = stack_variable_length_tensors(
                            coordinates_batch,
                            constant_value=pad,
                        ).to(self.device)
                
                if self.sequence_only:
                    protein_tensor = _BatchedESMProteinTensor(sequence=sequence_tokens, coordinates=coordinates_batch).to(self.device)
                else:
                    protein_tensor = _BatchedESMProteinTensor(sequence=sequence_tokens,
                                                          structure=structure_tokens_batch, coordinates=coordinates_batch).to(self.device)

                yield {
                    'name': name_batch,
                    'protein_tensor': protein_tensor,
                    'label': torch.stack(label_batch).to(self.device),
                    'attention_mask': torch.stack(mask_batch).to(self.device)==1,
                }
                    
            
            if self.pretrain_model_name == 'esmc_600m':
                from model_zoom.esm.utils.sampling import _BatchedESMProteinTensor
                name_batch, seq_batch, mask_batch, label_batch = [], [], [], []
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    label = sample['label']
                    attention_mask = torch.zeros(self.max_length)
                    attention_mask[:len(seq)] = 1
                    
                    seq_batch.append(seq)
                    mask_batch.append(attention_mask)
                    label_batch.append(torch.tensor(label))
                    name_batch.append(sample['name'])

                sequence_tokens = self.pretrain_model._tokenize(seq_batch)
                protein_tensor = _BatchedESMProteinTensor(sequence=sequence_tokens).to(self.device)
                
                yield {
                    'name': name_batch,
                    'protein_tensor': protein_tensor,
                    'attention_mask': torch.stack(mask_batch).to(self.device)==1,
                    'label': torch.stack(label_batch).to(self.device),
                }
            
            if self.pretrain_model_name == 'procyon':
                name_batch, X_batch, S_batch, label_batch = [], [], [], []
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    label = sample['label']
                    uid = sample['unique_id']
                    
                    seq_token = torch.tensor(self.esm_tokenizer.encode(seq))
                    seq_token = self.esm_tokenizer([seq], return_tensors="pt", padding=True, truncation=True)
                    seq_embedding = self.esm_pretrain_model.esm(
                        seq_token['input_ids'].to(self.device),
                        attention_mask=seq_token['attention_mask'].to(self.device),
                        return_dict=True,
                    ).last_hidden_state.squeeze(0).mean(0).flatten()
                    
                    X = torch.tensor(
                        self.structure_embeddings[task_name][uid]["graph_feature"]
                    ).flatten()

                    
                    S_batch.append(seq_embedding)
                    X_batch.append(X)
                    label_batch.append(torch.tensor(label))
                    name_batch.append(sample['name'])
        
                yield {
                    'name': name_batch,
                    'seq': torch.stack(S_batch).to(self.device),
                    'X': torch.stack(X_batch).unsqueeze(1).to(self.device),
                    'label': torch.stack(label_batch).to(self.device),
                }
            
            if self.pretrain_model_name == 'prollama':
                max_length_batch = 0
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    seq = f"[Determine superfamily] Seq=<{seq}>"
                    seq_token = torch.tensor(self.tokenizer.encode(seq))
                    max_length_batch = max(max_length_batch, len(seq_token))
                    
                name_batch, X_batch, S_batch, mask_batch, label_batch = [], [], [], [], []
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    seq = f"[Determine superfamily] Seq=<{seq}>"
                    label = sample['label']
                    X = sample['X']

                    attention_mask = torch.zeros(self.max_length)
                    seq_token = torch.tensor(self.tokenizer.encode(seq))
                    attention_mask[:len(seq_token)] = 1
                    seq_token = self.pad_data(seq_token, dim=0, max_length=max_length_batch)
                    
                    X = dynamic_pad(X, [1, 1], dim=0, pad_value=0)
                    X = self.pad_data(X, dim=0, max_length=max_length_batch)
 
                    S_batch.append(seq_token)
                    X_batch.append(X)
                    mask_batch.append(attention_mask)
                    label_batch.append(torch.tensor(label))
                    name_batch.append(sample['name'])
        
                yield {
                    'name': name_batch,
                    'seq': torch.stack(S_batch).to(self.device),
                    'X': torch.stack(X_batch).to(self.device),
                    'attention_mask': torch.stack(mask_batch).to(self.device)==1,
                    'label': torch.stack(label_batch).to(self.device),
                }
            
            if self.pretrain_model_name == 'progen2': # 没有BOS, EOS, 需要限制长度在1024以内
                max_length_batch = max([len(sample['seq']) for sample in data[i:i + batch_size]])
                name_batch, S_batch, mask_batch, label_batch = [], [], [], []
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    label = sample['label']
                    attention_mask = torch.zeros(self.max_length)
                    
                    seq_token = torch.tensor(self.tokenizer.encode(seq).ids)
                    attention_mask[:len(seq_token)] = 1
                    seq_token = self.pad_data(seq_token, dim=0, max_length=max_length_batch)
                    attention_mask = self.pad_data(attention_mask, dim=0, max_length=max_length_batch)
                    
                    S_batch.append(seq_token[:1024])
                    mask_batch.append(attention_mask[:1024])
                    label_batch.append(torch.tensor(label))
                    name_batch.append(sample['name'])
        
                yield {
                    'name': name_batch,
                    'seq': torch.stack(S_batch).to(self.device),
                    'attention_mask': torch.stack(mask_batch).to(self.device)==1,
                    'label': torch.stack(label_batch).to(self.device),
                }
            
            if self.pretrain_model_name == 'prostt5':
                max_length_batch = max([len(sample['seq']) for sample in data[i:i + batch_size]])+2
                import re
                name_batch, struct_batch, S_batch, mask_batch, label_batch = [], [], [], [], []
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    X = sample['X']
                    N,  CA, C,  CB,  O = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
                    label = sample['label']
                    attention_mask = torch.zeros(2, self.max_length)
                    
                    states = self.encoder_3di.encode_atoms(ca = CA.numpy(), cb = CB.numpy(), n = N.numpy(), c = C.numpy())
                    struct_sequence = self.encoder_3di.build_sequence(states).lower()
                    
                    if self.sequence_only:
                        sequence_examples = [seq, seq]
                        sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
                        sequence_examples = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s # this expects 3Di sequences to be already lower-case
                        for s in sequence_examples
                        ]
                    else:
                        sequence_examples = [seq, struct_sequence]
                        sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
                        sequence_examples = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s # this expects 3Di sequences to be already lower-case
                        for s in sequence_examples
                        ]
                        
                    seq_token = self.tokenizer.batch_encode_plus(sequence_examples,
                                  add_special_tokens=True,
                                  padding="longest",
                                  return_tensors='pt').to(self.device)
                    
                    attention_mask[:, :seq_token.input_ids.shape[1]] = 1
                    seq_token = self.pad_data(seq_token.input_ids, dim=1, max_length=max_length_batch)
                    attention_mask = self.pad_data(attention_mask, dim=1, max_length=max_length_batch)
                    
                    
                    
                    S_batch.append(seq_token)
                    mask_batch.append(attention_mask)
                    label_batch.append(torch.tensor(label))
                    name_batch.append(sample['name'])
        
                yield {
                    'name': name_batch,
                    'seq': torch.cat(S_batch, dim=0).to(self.device),
                    'attention_mask': torch.cat(mask_batch, dim=0).to(self.device)==1,
                    'label': torch.stack(label_batch).to(self.device),
                }
            
            if self.pretrain_model_name == 'protgpt2': # 没有BOS, EOS, 并且长度与氨基酸数量不一致
                max_length_batch = 0
                for sample in data[i:i + batch_size]:
                    seq_token = torch.tensor(self.tokenizer.encode(sample['seq']))
                    max_length_batch = max(max_length_batch, len(seq_token))
                    
                name_batch, S_batch, mask_batch, label_batch = [], [], [], []
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    label = sample['label']
                    attention_mask = torch.zeros(self.max_length)
                    
                    seq_token = torch.tensor(self.tokenizer.encode(seq))
                    attention_mask[:len(seq_token)] = 1
                    seq_token = self.pad_data(seq_token, dim=0, max_length=max_length_batch)
                    attention_mask = self.pad_data(attention_mask, dim=0, max_length=max_length_batch)
                    
                    S_batch.append(seq_token)
                    mask_batch.append(attention_mask)
                    label_batch.append(torch.tensor(label))
                    name_batch.append(sample['name'])
        
                yield {
                    'name': name_batch,
                    'seq': torch.stack(S_batch).to(self.device),
                    'attention_mask': torch.stack(mask_batch).to(self.device)==1,
                    'label': torch.stack(label_batch).to(self.device),
                }
            
            if self.pretrain_model_name == 'protrek':
                max_length_batch = max([len(sample['seq']) for sample in data[i:i + batch_size]])+2
                name_batch, struct_batch, seq_batch, label_batch = [], [], [], []
                attention_mask = torch.zeros(len(data[i:i + batch_size]), max_length_batch)
                for idx, sample in enumerate(data[i:i + batch_size]):
                    seq = sample['seq']
                    X = sample['X']
                    N,  CA, C,  CB,  O = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
                    label = sample['label']
                    
                    states = self.encoder_3di.encode_atoms(ca = CA.numpy(), cb = CB.numpy(), n = N.numpy(), c = C.numpy())
                    struct_sequence = self.encoder_3di.build_sequence(states).lower()
                    if self.sequence_only:
                        struct_sequence = ''.join(['#' for one in struct_sequence])
                    
                    attention_mask[idx, :len(seq)+2] = 1
                    struct_batch.append(struct_sequence)
                    seq_batch.append(seq)
                
                    label_batch.append(torch.tensor(label))
                    name_batch.append(sample['name'])

                yield {
                    'name': name_batch,
                    'seq_batch': seq_batch,
                    'struct_batch': struct_batch,
                    'attention_mask': attention_mask==1,
                    'label': torch.stack(label_batch).to(self.device),
                }
            
            if self.pretrain_model_name == 'saport':
                max_length_batch = max([len(sample['seq']) for sample in data[i:i + batch_size]])+2
                name_batch, struct_batch, S_batch, mask_batch, label_batch = [], [], [], [], []
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    X = sample['X']
                    N,  CA, C,  CB,  O = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
                    label = sample['label']
                    attention_mask = torch.zeros(self.max_length)
                    
                    states = self.encoder_3di.encode_atoms(ca = CA.numpy(), cb = CB.numpy(), n = N.numpy(), c = C.numpy())
                    struct_sequence = self.encoder_3di.build_sequence(states).lower()
                    if self.sequence_only:
                        struct_sequence = ''.join(['#' for one in struct_sequence])
                    merged_seq = ''.join(a + b.lower() for a, b in zip(seq, struct_sequence))
                    
                    seq_token = self.tokenizer(merged_seq, return_tensors="pt").input_ids[0]
                    
                    attention_mask[:seq_token.shape[0]] = 1
                    seq_token = self.pad_data(seq_token, dim=0, max_length=max_length_batch)
                    attention_mask = self.pad_data(attention_mask, dim=0, max_length=max_length_batch)
                    
                    
                    
                    S_batch.append(seq_token)
                    mask_batch.append(attention_mask)
                    label_batch.append(torch.tensor(label))
                    name_batch.append(sample['name'])
        
                yield {
                    'name': name_batch,
                    'seq': torch.stack(S_batch, dim=0).to(self.device),
                    'attention_mask': torch.stack(mask_batch, dim=0).to(self.device)==1,
                    'label': torch.stack(label_batch).to(self.device),
                }
    
    def forward(self, x) -> torch.Tensor:
        names, labels = x['name'], x['label']
        if self.pretrain_model_name == 'esm2_650m':
            # Forward pass through the pre-trained model
            outputs = self.pretrain_model.esm(
                        x['seq'],
                        attention_mask=x['attention_mask'],
                        return_dict=True,
                    )
            embeddings = outputs.last_hidden_state
            ends = x['attention_mask'].sum(dim=-1)
            starts = torch.ones_like(ends)
            attention_mask = x['attention_mask']
            
        if self.pretrain_model_name == 'esm3_1.4b':
            output = self.pretrain_model.logits(
                x['protein_tensor'],
                LogitsConfig(
                    sequence=True,
                    structure=True,
                    secondary_structure=True,
                    sasa=True,
                    function=True,
                    residue_annotations=True,
                    return_embeddings=True,
                ),
            )
            embeddings = output.embeddings
            ends = x['attention_mask'].sum(dim=-1)
            starts = torch.ones_like(ends)
            attention_mask = x['attention_mask']
            # embeddings = F.pad(embeddings, (0, 0, 0, self.max_length-embeddings.shape[1], 0, 0), value=0)
        
        if self.pretrain_model_name == 'esmc_600m':
            logits_output = self.pretrain_model.logits(
            x['protein_tensor'], LogitsConfig(sequence=True, return_embeddings=True)
            )
            attention_mask = x['attention_mask']
            embeddings = logits_output.embeddings
            ends = x['attention_mask'].sum(dim=-1)
            starts = torch.ones_like(ends)
        
        if self.pretrain_model_name == 'procyon': # 长度和氨基酸数量不同
            seq_embeddings = self.pretrain_model.token_projectors["aaseq"](
                x["seq"]
            )
            struct_embeddings = self.pretrain_model.token_projectors["prot_structure"](
                x["X"]
            )
            instructions = [
                "Describe the following protein with features: <|protein|> <|struct|>"
            ] * seq_embeddings.shape[0]
            
            input_ids, attn_masks = self.pretrain_model._prepare_text_inputs_and_tokenize(instructions, [[]] * seq_embeddings.shape[0], no_pad=True)
            input_ids, attn_masks = input_ids.to(self.device), attn_masks.to(self.device)
            
            if self.sequence_only:
                input_embeds, ret_output_indices = self.pretrain_model._prepare_input_embeddings(
                    input_ids, 
                    protein_soft_tokens=seq_embeddings
                )
            else:
                input_embeds, ret_output_indices = self.pretrain_model._prepare_input_embeddings(
                    input_ids, 
                    protein_soft_tokens=seq_embeddings,
                    protein_struct_tokens=struct_embeddings
                )
            x["attention_mask"] = ~(input_ids == self.pretrain_model.tokenizer.pad_token_id)
            outputs = self.pretrain_model.text_encoder(
                input_embeds = input_embeds,
                attn_masks = attn_masks,
            )
            embeddings = outputs.hidden_states[-1] # shape(b, 2048, 4096)
            attention_mask = x['attention_mask']
            ends = attention_mask.sum(dim=-1)
            starts = torch.zeros_like(ends)

        if self.pretrain_model_name == 'prollama': # 长度和氨基酸数量不同
            out = self.pretrain_model(
                input_ids = x["seq"],
                attention_mask = x['attention_mask'],
                output_hidden_states=True
            )
            embeddings = out.hidden_states[-1].float()
            attention_mask = x['attention_mask']
            ends = attention_mask.sum(dim=-1)
            starts = torch.zeros_like(ends)
        
        if self.pretrain_model_name == 'progen2':
            transformer_outputs = self.pretrain_model.transformer(x['seq'], return_dict=True)
            embeddings = transformer_outputs[0]
            attention_mask = x['attention_mask']
            ends = attention_mask.sum(dim=-1)+1
            starts = torch.zeros_like(ends)

        if self.pretrain_model_name == 'prostt5':
            embedding_repr = self.pretrain_model(
                x['seq'], 
                attention_mask=x['attention_mask']
                )
            embeddings = embedding_repr.last_hidden_state
            embeddings = embeddings.reshape(embeddings.shape[0]//2, 2, embeddings.shape[1], embeddings.shape[2])
            embeddings = torch.cat([embeddings[:,0], embeddings[:,1]], dim=-1)
            attention_mask = x['attention_mask'][::2]
            ends = attention_mask.sum(dim=-1)
            starts = torch.ones_like(ends)
        
        if self.pretrain_model_name == 'protgpt2':
            outputs = self.pretrain_model.transformer(x['seq'])
            embeddings = outputs.last_hidden_state
            attention_mask = x['attention_mask']
            ends = attention_mask.sum(dim=-1)
            starts = torch.ones_like(ends)
    
        if self.pretrain_model_name == 'protrek':
            seq_embedding = self.pretrain_model.get_protein_repr(x['seq_batch'])
            struc_embedding = self.pretrain_model.get_structure_repr(x['struct_batch'])
            embeddings = torch.cat([seq_embedding, struc_embedding], dim=-1)
            attention_mask = x['attention_mask']
            ends = attention_mask.sum(dim=-1)
            starts = torch.ones_like(ends)
        
        if self.pretrain_model_name == 'saport':
            outputs = self.pretrain_model.esm(
                        x['seq'],
                        attention_mask=x['attention_mask'],
                        return_dict=True,
                    )
            embeddings = outputs.last_hidden_state
            attention_mask = x['attention_mask']
            ends = attention_mask.sum(dim=-1)
            starts = torch.ones_like(ends)
        
        # 这个embedding 是[B,L,D]的形式, L是氨基酸数量, 但对于语言模型老师, L这个维度可能不刚好等于氨基酸数量，无法用于氨基酸级别的预测任务。另外，有些方法会加上[EOS, BOS]，有些又不需要，最好是在最后的embedding上统一去除[EOS, BOS]
        return self.post_process(names, labels, embeddings, attention_mask, starts, ends)
    
    def post_process(self, names, labels, embeddings, attention_mask, starts, ends):
        results = []
        for i, end in enumerate(ends):
            start = starts[i]
            results.append( 
                            {'name': names[i],
                            'embedding': self.pad_data(embeddings[i,start:end], dim=0, max_length=self.max_length).cpu(),
                            'attention_mask': self.pad_data(attention_mask[i,start:end], dim=0, max_length=self.max_length).cpu(),
                            'label': labels[i].cpu()}
            )
        return results

    
    def pad_data(self, data, dim=0, pad_value=0, max_length=1022):
        if data.shape[dim] < max_length:
            data = dynamic_pad(data, [0, max_length-data.shape[dim]], dim=dim, pad_value=pad_value)
        else:
            # start = torch.randint(0, data.shape[0]-self.max_length+1, (1,)).item()
            start = 0
            data = data[start:start+max_length]
        return data
    
    def inference_datasets(self, data, task_name=None):
        self.pretrain_model.eval()
        with torch.no_grad():
            proccessed_data = []
            for batch in tqdm(self.construct_batch(data, self.batch_size, task_name), desc='Extracting embeddings'):
                # print('batch size:', len(batch['name']))
                results = self.forward(batch)
                proccessed_data.extend(results)
                # for i in range(len(batch['name'])):
                #     proccessed_data.append({
                #         'name': batch['name'][i],
                #         'attention_mask': result['attention_mask'][i].cpu(),
                #         'label': batch['label'][i].cpu(),
                #         'embedding': result['embeddings'][i].cpu()
                #     })
        return proccessed_data
            


