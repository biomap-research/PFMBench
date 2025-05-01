import torch.nn as nn
import torch
import os
import numpy as np
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from src.data.protein_dataset import dynamic_pad
from transformers import AutoTokenizer, AutoModelForMaskedLM, LlamaForCausalLM, LlamaTokenizer, T5Tokenizer, T5EncoderModel, AutoModelForCausalLM, AutoModel
from src.data.esm.sdk.api import LogitsConfig
from model_zoom.procyon.model.model_unified import UnifiedProCyon
from model_zoom.progen2.modeling_progen import ProGenForCausalLM
from tokenizers import Tokenizer
from model_zoom.GearNet.data.protein import Protein
from model_zoom.GearNet.data.transform import ProteinView
from model_zoom.GearNet.data.transform import Compose
from model_zoom.GearNet.data.geo_graph import GraphConstruction
from model_zoom.GearNet.data.function import AlphaCarbonNode, SpatialEdge, KNNEdge, SequentialEdge
from model_zoom.GearNet.gearnet import GeometryAwareRelationalGraphNeuralNetwork
from model_zoom.esm.utils.sampling import _BatchedESMProteinTensor
from model_zoom.ProTrek.model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel
from vplm import TransformerForMaskedLM, TransformerConfig
from vplm import VPLMTokenizer
import mini3di
MODEL_ZOOM_PATH = '/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom'

class BaseProteinModel(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()
        self.device = device
    
    def construct_batch(self, data, batch_size, task_name=None):
        raise NotImplementedError
    
    def forward(self, batch):
        raise NotImplementedError

class UtilsModel:
    def __init__(self):
        super().__init__()

    def post_process_cpu(self, batch, embeddings, attention_masks, start, ends, task_type='binary_classification'):

        # sparse return
        results = []
        for i, end in enumerate(ends):
            end = int(end.item())
            embedding = embeddings[i][start:end].cpu()
            name = batch['name'][i]
            attention_mask = attention_masks[i][start:end].cpu()
            label = torch.tensor(batch['label'][i])
            
            results.append({'name': name,
                            'embedding': embedding,
                            'attention_mask': attention_mask.bool(),
                            'label': label}  )
        return results

    def pad_data(self, data, dim=0, pad_value=0, max_length=1022):
        if data.shape[dim] < max_length:
            data = self.dynamic_pad(data, [0, max_length-data.shape[dim]], dim=dim, pad_value=pad_value)
        else:
            # start = torch.randint(0, data.shape[0]-self.max_length+1, (1,)).item()
            # start = torch.randint(0, data.shape[0]-self.max_length+1, (1,)).item()
            start = 0
            end = start + max_length
            # 构建切片列表，对其他维度用 slice(None)，目标维度用 slice(start, end)
            slices = [slice(None)] * data.ndim
            slices[dim] = slice(start, end)
            data = data[tuple(slices)]  # 正确应用多维切片
        return data

    def dynamic_pad(self, tensor, pad_size, dim=0, pad_value=0):
        # 获取原始形状
        shape = list(tensor.shape)
        num_dims = len(shape)
        
        # 生成 padding 参数
        pad = [0] * (2 * num_dims)
        prev_pad_size, post_pad_size = pad_size
        pad_index = 2 * (num_dims - dim - 1)
        pad[pad_index] = prev_pad_size  # 前面 padding
        pad[pad_index + 1] = post_pad_size  # 后面 padding

        # 应用 padding
        padded_tensor = F.pad(tensor, pad, mode="constant", value=pad_value)
        return padded_tensor
    
class ESM2Model(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022, **kwargs):
        super().__init__(device)
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.pretrain_model = AutoModelForMaskedLM.from_pretrained(f"{MODEL_ZOOM_PATH}/esm2_650m").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_ZOOM_PATH}/esm2_650m")
        self.max_length = max_length

    def construct_batch(self, batch):
        MAXLEN = self.max_length
        max_length_batch = min(max([len(sample['seq']) for sample in batch]) + 2, self.max_length + 2) # +2 for <s> and </s>
        result = {
            'name': [],
            'seq': [],
            'attention_mask': [],
            'label': []
        }
        for sample in batch:
            seq_token = torch.tensor(self.tokenizer.encode(sample['seq']))[:MAXLEN]
            attention_mask = torch.zeros(max_length_batch)
            attention_mask[:len(seq_token)] = 1
            seq_token = self.pad_data(seq_token, dim=0, max_length=max_length_batch)
            result['name'].append(sample['name'])
            result['seq'].append(seq_token)
            result['attention_mask'].append(attention_mask)
            result['label'].append(sample['label'])

        result['seq'] = torch.stack(result['seq'], dim=0).to(self.device)
        result['attention_mask'] = torch.stack(result['attention_mask'], dim=0).to(self.device)
            
        return result

    def forward(self, batch, post_process=True, task_type='binary_classification', return_prob=False):
        attention_mask = batch['attention_mask']
        outputs = self.pretrain_model.esm(
                        batch['seq'],
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
        if return_prob:
            logits = self.pretrain_model.lm_head(outputs.last_hidden_state)
            probs = F.softmax(logits, dim=-1)
            return probs
        
        embeddings = outputs.last_hidden_state
        ends = attention_mask.sum(dim=-1)-1
        start = 1
        if post_process:
            result = self.post_process_cpu(batch, embeddings, attention_mask, start, ends, task_type=task_type)
        else:
            result = embeddings
        return result

class SmilesModel(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022, **kwargs):
        super().__init__(device)


    def construct_batch(self, batch):
        result = {'smiles': []}
        for sample in batch:
            mol = Chem.MolFromSmiles(sample['smiles'])
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                smiles = torch.tensor([int(ele) for ele in list(fp.ToBitString())]).float()
            else:
                smiles = torch.tensor([0]*2048).float()
            result['smiles'].append(smiles)
        return result
    
    def forward(self, batch, post_process=True, task_type='binary_classification'):
        return batch
        
class ESM3Model(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022, sequence_only=False):
        super().__init__(device)
        from model_zoom.esm.models.esm3 import ESM3
        self.model = ESM3.from_pretrained("esm3-sm-open-v1").to(self.device)
        self.sequence_only = sequence_only
        self.max_length = max_length

    def construct_batch(self, batch):
        from model_zoom.esm.utils import encoding
        from model_zoom.esm.utils.misc import stack_variable_length_tensors
        addBOS = 1
        pad_id = self.model.tokenizers.sequence.pad_token_id
        max_len = min(max([len(s['seq']) for s in batch]) + 2*addBOS, self.max_length + 2*addBOS)
        names, prot_tensors, labels, masks = [], [], [], []
        sequence_list, coordinates_list, structure_tokens_batch = [], [], []
        for sample in batch:
            seq, coords = sample['seq'], sample['X']
            seq_tokenizer = self.model.tokenizers.sequence
            struct_tokenizer = self.model.tokenizers.structure
            # tokenize
            seq_tok = encoding.tokenize_sequence(seq, seq_tokenizer, add_special_tokens=True)
            coords_tok, _plddt, struct_tok = encoding.tokenize_structure(
                            np.array(coords), 
                            self.model.get_structure_encoder(), 
                            struct_tokenizer, 
                            add_special_tokens=True
                        )
            coords_tok, struct_tok = torch.tensor(coords_tok), torch.tensor(struct_tok)
            mask = torch.zeros(max_len)
            mask[:seq_tok.shape[0]] = 1
            seq_tok = self.pad_data(seq_tok, dim=0, pad_value=pad_id, max_length=max_len)
            struct_tok = self.pad_data(struct_tok, dim=0, pad_value=pad_id, max_length=max_len)
            coords_tok = dynamic_pad(coords_tok, [addBOS, addBOS], dim=0, pad_value=0) # 坐标需要和seq一样加上BOS, EOS
            coords_tok = self.pad_data(coords_tok, dim=0, max_length=max_len)
            
            sequence_list.append(seq_tok)
            coordinates_list.append(coords_tok)
            structure_tokens_batch.append(struct_tok)
            names.append(sample['name'])
            masks.append(mask)
            labels.append(sample['label'])
            
        sequence_tokens = stack_variable_length_tensors(
                    sequence_list,
                    constant_value=pad_id,
                ).to(self.device)
                
        structure_tokens_batch = stack_variable_length_tensors(
            structure_tokens_batch,
            constant_value=pad_id,
        ).to(self.device)
                        
        coordinates_batch = stack_variable_length_tensors(
            coordinates_list,
            constant_value=pad_id,
        ).to(self.device)
        protein_tensor = _BatchedESMProteinTensor(sequence=sequence_tokens,
                                                          structure=structure_tokens_batch, coordinates=coordinates_batch).to(self.device)
        return {
            'name': names,
            'protein_tensor': protein_tensor,
            'attention_mask': torch.stack([m.bool() for m in masks]).to(self.device),
            'label': labels
        }

    def forward(self, batch, post_process=True, task_type='binary_classification'):
        tens, mask = batch['protein_tensor'], batch['attention_mask']
        out = self.model.logits(
            tens, LogitsConfig(
                sequence=True, structure=True, secondary_structure=True,
                sasa=True, function=True, residue_annotations=True, return_embeddings=True
            )
        )
        embeddings = out.embeddings
        ends = mask.sum(dim=-1) - 1
        start = 1
        if post_process:
            return self.post_process_cpu(batch, embeddings, mask, start, ends, task_type)
        return embeddings
    
class ESMC600MModel(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022):
        super().__init__(device)
        from model_zoom.esm.models.esmc import ESMC
        self.model = ESMC.from_pretrained("esmc_600m").to(self.device)
        self.max_length = max_length
    def construct_batch(self, batch):
        from model_zoom.esm.utils.misc import stack_variable_length_tensors
        addBOS = 1
        pad_id = self.model.tokenizer.pad_token_id
        max_len = min(max([len(s['seq']) for s in batch]) + 2*addBOS, self.max_length+2*addBOS)
        names, prots, masks, labels = [], [], [], []
        token_ids_list = []
        for sample in batch:
            seq = sample['seq']
            token_ids = self.model._tokenize([seq]).flatten()
            mask = torch.zeros(max_len)
            mask[:len(token_ids)] = 1
            token_ids = self.pad_data(token_ids, dim=0, pad_value=pad_id, max_length=max_len)
            token_ids_list.append(token_ids)
            names.append(sample['name'])
            masks.append(mask)
            labels.append(sample['label'])
        sequence_tokens = stack_variable_length_tensors(
                    token_ids_list,
                    constant_value=self.model.tokenizer.pad_token_id,
                )
        protein_tensor = _BatchedESMProteinTensor(sequence=sequence_tokens).to(self.device)
        return {
            'name': names,
            'protein_tensor': protein_tensor,
            'attention_mask': torch.stack([m.bool() for m in masks]).to(self.device),
            'label': labels
        }
    def forward(self, batch, post_process=True, task_type='binary_classification'):
        tens, mask = batch['protein_tensor'], batch['attention_mask']
        outputs = self.model.logits(tens, LogitsConfig(sequence=True, return_embeddings=True))
        embeddings = outputs.embeddings
        ends = mask.sum(dim=-1) - 1
        start = 1
        if post_process:
            return self.post_process_cpu(batch, embeddings, mask, start, ends, task_type)
        return embeddings

# ProCyon
class ProCyonModel(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022, sequence_only=False):
        super().__init__(device)
        protein_view_transform = ProteinView(view='residue')
        self.transform = Compose([protein_view_transform])
        self.graph_construction_model = GraphConstruction(
                                            node_layers=[AlphaCarbonNode()], 
                                            edge_layers=[SpatialEdge(radius=10.0, min_distance=5),
                                            KNNEdge(k=10, min_distance=5),
                                            SequentialEdge(max_distance=2)],
                                            edge_feature="gearnet"
                                        )
        self.gearnet_edge = GeometryAwareRelationalGraphNeuralNetwork(input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512], 
                                                                        num_relation=7, edge_input_dim=59, num_angle_bin=8,
                                                                        batch_norm=True, concat_hidden=True, short_cut=True, readout="sum"
        )
        ckpt = torch.load("/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/GearNet/mc_gearnet_edge.pth")
        self.gearnet_edge.load_state_dict(ckpt)
        self.gearnet_edge = self.gearnet_edge.to(self.device)
        self.gearnet_edge.eval()
        os.environ["HOME_DIR"] = MODEL_ZOOM_PATH
        os.environ["DATA_DIR"] = "/nfs_beijing/wanghao/2025-onesystem/vllm/ProCyon-Instruct"
        os.environ["LLAMA3_PATH"] = "/nfs_beijing/wanghao/2025-onesystem/vllm/Meta-Llama-3-8B"
        procyon_ckpt = '/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/procyon/model_weights/ProCyon-Full'
        self.esm_pretrain_model = AutoModelForMaskedLM.from_pretrained(f"{MODEL_ZOOM_PATH}/esm2_3b").to(self.device)
        self.esm_pretrain_model.eval()
        self.esm_tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_ZOOM_PATH}/esm2_3b")
        # procyon initialization
        self.model, _ = UnifiedProCyon.from_pretrained(
            pretrained_weights_dir=procyon_ckpt, 
            checkpoint_dir=procyon_ckpt
        )
        self.model = self.model.to(self.device)

        self.max_length = max_length
        self.sequence_only = sequence_only

    def construct_batch(self, batch):
        names, seqs, structs, labels = [], [], [], []
        for sample in batch:
            try:
                seqs_list = sample['seq'] if isinstance(sample['seq'], list) else [sample['seq']]
                pdbs = sample['pdb_path'] if isinstance(sample['pdb_path'], list) else [sample['pdb_path']]
                seq_embs, struct_embs = [], []
                for s, p in zip(seqs_list, pdbs):
                    toks = self.esm_tokenizer([s], return_tensors='pt', padding=True, max_length=self.max_length, truncation=True)
                    out = self.esm_pretrain_model.esm(toks.input_ids.to(self.device), attention_mask=toks.attention_mask.to(self.device), return_dict=True)
                    seq_embs.append(out.last_hidden_state.squeeze(0).mean(0))
                    prot = Protein.from_pdb(p, bond_feature="length", residue_feature="symbol")
                    prot = self.transform({"graph": prot})["graph"]
                    packed = Protein.pack([prot])
                    protein = self.graph_construction_model(packed).to(self.device)
                    with torch.no_grad():
                        gea = self.gearnet_edge(protein, protein.node_feature.float())
                    struct_embs.append(gea["graph_feature"].flatten())
                names.append(sample['name'])
                seqs.append(torch.cat(seq_embs, dim=-1))
                structs.append(torch.cat(struct_embs, dim=-1))
                labels.append(sample['label'])
            except Exception as e:
                print(f"Error processing sample {sample['name']}: {e}")
                continue
        return {
            'name': names,
            'seq': torch.stack(seqs).to(self.device),
            'X': torch.stack(structs).unsqueeze(1).to(self.device),
            'label': labels
        }

    def forward(self, batch, post_process=True, task_type='binary_classification'):
        seq_emb, struct_emb = batch['seq'], batch['X']
        aaseq = self.model.token_projectors['aaseq'](seq_emb)
        struct_proj = self.model.token_projectors['prot_structure'](struct_emb)
        B = aaseq.shape[0]
        instr = ["Describe the following protein with functions: <|protein|> <|struct|>"] * B
        input_ids, attn = self.model._prepare_text_inputs_and_tokenize(instr, [[]]*B, no_pad=True)
        input_ids, attn = input_ids.to(self.device), attn.to(self.device)
        if self.sequence_only:
            embeds, _ = self.model._prepare_input_embeddings(input_ids, protein_soft_tokens=aaseq)
        else:
            embeds, _ = self.model._prepare_input_embeddings(input_ids, protein_soft_tokens=aaseq, protein_struct_tokens=struct_proj)
        mask = ~(input_ids == self.model.tokenizer.pad_token_id)
        out = self.model.text_encoder(input_embeds=embeds, attn_masks=attn)
        h = out.hidden_states[-1]
        ends = mask.sum(dim=-1)
        start = 0
        if post_process:
            return self.post_process_cpu(batch, h, mask, start, ends, task_type)
        return h

# GearNet
class GearNetModel(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022):
        super().__init__(device)
        pv = ProteinView(view='residue')
        self.transform = Compose([pv])
        self.graph_construction = GraphConstruction(
            node_layers=[AlphaCarbonNode()], edge_layers=[SpatialEdge(radius=10.0, min_distance=5), KNNEdge(k=10, min_distance=5), SequentialEdge(max_distance=2)], edge_feature="gearnet"
        )
        self.gearnet = GeometryAwareRelationalGraphNeuralNetwork(
            input_dim=21, hidden_dims=[512]*6, num_relation=7, edge_input_dim=59, num_angle_bin=8,
            batch_norm=True, concat_hidden=True, short_cut=True, readout="sum"
        )
        ckpt = torch.load(f"{MODEL_ZOOM_PATH}/GearNet/mc_gearnet_edge.pth")
        self.gearnet.load_state_dict(ckpt)
        self.gearnet = self.gearnet.to(self.device).eval()
        self.max_length = max_length

    def construct_batch(self, batch):
        names, embeddings, attention_masks, labels = [], [], [], []
        for sample in batch:
            try:
                pdbs = sample['pdb_path'] if isinstance(sample['pdb_path'], list) else [sample['pdb_path']]
                prots = []
                for p in pdbs:
                    pr = Protein.from_pdb(p, bond_feature="length", residue_feature="symbol")
                    prots.append(self.transform({"graph": pr})["graph"])

                pack = Protein.pack(prots)
                max_res = pack.num_residues.max().item() 
                gc = self.graph_construction(pack.to(self.device))
                node = self.gearnet(gc.to(self.device), gc.node_feature.float().to(self.device))["node_feature"]
                splits = torch.cumsum(F.pad(pack.num_residues, (1,0)), dim=0)
                attention_mask = torch.zeros(len(splits)-1, max_res).to(self.device)
                embeddings_temp = []
                for i in range(len(splits)-1):
                    start, end = splits[i], splits[i+1]
                    embedding = node[start:end]
                    attention_mask[i, :embedding.shape[0]] = 1
                    embedding = self.pad_data(embedding, dim=0, max_length=max_res)
                    embeddings_temp.append(embedding)
                embeddings_temp = torch.stack(embeddings_temp)
                embeddings.append(embeddings_temp)
                attention_masks.append(attention_mask)
                labels.append(sample['label'])
                names.append(sample['name'])
            except Exception as e:
                print(f"Error processing sample {sample['name']}: {e}")
                continue
        
        max_len = max([one.shape[1] for one in embeddings])
        
        
        embeddings = torch.stack([F.pad(one[0], (0,0,0, max_len-one.shape[1])) for one in embeddings], dim=0)
        attention_masks = torch.stack([F.pad(one[0], (0, max_len-one.shape[1])) for one in attention_masks], dim=0)
        return {
            'name': names,
            'X': embeddings,
            'attention_mask': attention_masks,
            'label': labels
        }

    def forward(self, batch, post_process=True, task_type='binary_classification'):
        emb = batch['X']; mask = batch['attention_mask']
        ends = mask.sum(dim=-1)
        start = 0
        if post_process:
            return self.post_process_cpu(batch, emb, mask, start, ends, task_type)
        return emb

# ProLLAMA
class ProLLAMAModel(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022):
        super().__init__(device)
        llama_path = f"{MODEL_ZOOM_PATH}/ProLLaMA"
        self.model = LlamaForCausalLM.from_pretrained(llama_path).to(self.device)
        self.tokenizer = LlamaTokenizer.from_pretrained(llama_path)
        self.max_length = max_length

    def construct_batch(self, batch):
        max_len = min(
            max(len(s) for sample in batch for s in (sample['seq'] if isinstance(sample['seq'], list) else [sample['seq']])) + 2,
            self.max_length + 2
        )
        names, seqs, masks, labels = [], [], [], []
        for sample in batch:
            seqs_list = sample['seq'] if isinstance(sample['seq'], list) else [sample['seq']]
            tok_ids, m = [], []
            for s in seqs_list:
                s2 = f"[Determine superfamily] Seq=<{s}>"
                tid = torch.tensor(self.tokenizer.encode(s2))
                mask = torch.zeros(max_len, dtype=torch.bool); mask[:len(tid)] = True
                tid = self.pad_data(tid, dim=0, max_length=max_len)
                tok_ids.append(tid); m.append(mask)
            names.append(sample['name']); seqs.append(torch.hstack(tok_ids)); masks.append(torch.hstack(m)); labels.append(sample['label'])
        return {
            'name': names,
            'seq': torch.stack(seqs).to(self.device),
            'attention_mask': torch.stack(masks).to(self.device),
            'label': labels
        }

    def forward(self, batch, post_process=True, task_type='binary_classification'):
        seq, mask = batch['seq'], batch['attention_mask']
        out = self.model(input_ids=seq, attention_mask=mask, output_hidden_states=True)
        emb = out.hidden_states[-1].float()
        ends = mask.sum(dim=-1) ; start = 0
        if post_process:
            return self.post_process_cpu(batch, emb, mask, start, ends, task_type)
        return emb

# ProST
class ProSTModel(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022):
        super().__init__(device)
        self.prost = AutoModel.from_pretrained(
                f"{MODEL_ZOOM_PATH}/protst", 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16
            ).protein_model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_ZOOM_PATH}/esm1b_650m")
        self.max_length = max_length

    def construct_batch(self, batch):
        names, seqs, masks, labels = [], [], [], []
        for sample in batch:
            seq = sample['seq'][:self.max_length]
            tid = torch.tensor(self.tokenizer.encode(seq))
            mask = torch.zeros(self.max_length, dtype=torch.bool); mask[:len(tid)] = True
            tid = self.pad_data(tid, dim=0, max_length=self.max_length)
            names.append(sample['name']); seqs.append(tid); masks.append(mask); labels.append(sample['label'])
        return {
            'name': names,
            'seq': torch.stack(seqs).to(self.device),
            'attention_mask': torch.stack(masks).to(self.device),
            'label': labels
        }

    def forward(self, batch, post_process=True, task_type='binary_classification'):
        out = self.prost(input_ids=batch['seq'], attention_mask=batch['attention_mask'], return_dict=True)
        emb = out.residue_feature
        ends = batch['attention_mask'].sum(dim=-1) - 1; start = 1
        if post_process:
            return self.post_process_cpu(batch, emb, batch['attention_mask'], start, ends, task_type)
        return emb

# ProGen2
class ProGen2Model(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022):
        super().__init__(device)
        self.model = ProGenForCausalLM.from_pretrained(f"{MODEL_ZOOM_PATH}/progen2").to(self.device)
        def create_tokenizer_custom(file):
            with open(file, 'r') as f:
                return Tokenizer.from_str(f.read())
        self.tokenizer = create_tokenizer_custom(file=f"{MODEL_ZOOM_PATH}/progen2/tokenizer.json")
        self.max_length = max_length

    def construct_batch(self, batch):
        max_len = max(len(self.tokenizer.encode(s).ids) for sample in batch for s in ([sample['seq']] if not isinstance(sample['seq'], list) else sample['seq']))
        names, seqs, masks, labels = [], [], [], []
        for sample in batch:
            seqs_list = sample['seq'] if isinstance(sample['seq'], list) else [sample['seq']]
            tids, m = [], []
            for s in seqs_list:
                tok = torch.tensor(self.tokenizer.encode(s).ids)
                mask = torch.zeros(max_len, dtype=torch.bool); mask[:len(tok)] = True
                tok = self.pad_data(tok, dim=0, max_length=max_len)
                mask = self.pad_data(mask, dim=0, max_length=max_len)
                tids.append(tok); m.append(mask)
            stacked = torch.hstack(tids)[:self.max_length]; mstack = torch.hstack(m)[:self.max_length]
            names.append(sample['name']); seqs.append(stacked); masks.append(mstack); labels.append(sample['label'])
        return {
            'name': names,
            'seq': torch.stack(seqs).to(self.device),
            'attention_mask': torch.stack(masks).to(self.device),
            'label': labels
        }

    def forward(self, batch, post_process=True, task_type='binary_classification'):
        out = self.model.transformer(batch['seq'], return_dict=True)
        emb = out.last_hidden_state
        ends = batch['attention_mask'].sum(dim=-1) - 1; start = 0
        if post_process:
            return self.post_process_cpu(batch, emb, batch['attention_mask'], start, ends, task_type)
        return emb

# ProstT5
class ProstT5Model(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022, sequence_only=False):
        super().__init__(device)
        self.tokenizer = T5Tokenizer.from_pretrained(f"{MODEL_ZOOM_PATH}/ProstT5", do_lower_case=False, legacy=False)
        self.model = T5EncoderModel.from_pretrained(f"{MODEL_ZOOM_PATH}/ProstT5").to(self.device)
        self.encoder_3di = mini3di.Encoder()
        self.max_length = max_length
        self.sequence_only = sequence_only

    def construct_batch(self, batch):
        import re
        max_length_batch = min(
            max([len(sample['seq']) for sample in batch]) + 2,
            self.max_length + 2
        )
        names, seqs, masks, labels = [], [], [], []
        seq_tokens, attention_masks = [], []
        for sample in batch:
            seq = sample['seq']; X = sample['X']
            N, CA, C, CB = X[:,0], X[:,1], X[:,2], X[:,3]
            attention_mask = torch.zeros(2, max_length_batch, device=self.device)
            states = self.encoder_3di.encode_atoms(ca=CA.numpy(), cb=CB.numpy(), n=N.numpy(), c=C.numpy())
            struct_seq = self.encoder_3di.build_sequence(states).lower()
            if self.sequence_only:
                sequence_examples = [seq, seq]
                sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
                sequence_examples = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s # this expects 3Di sequences to be already lower-case
                    for s in sequence_examples
                ]
            else:
                sequence_examples = [seq, struct_seq]
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
            seq_tokens.append(seq_token)
            attention_masks.append(attention_mask)
            names.append(sample['name'])
            labels.append(sample['label'])
        return {
            'name': names,
            'seq': torch.cat(seq_tokens, dim=0),
            'attention_mask': torch.cat(attention_masks, dim=0),
            'label': labels
        }

    def forward(self, batch, post_process=True, task_type='binary_classification'):
        seq, attention_mask = batch['seq'], batch['attention_mask']
        embedding_repr = self.model(
                        seq,
                        attention_mask=attention_mask
                )
        last = embedding_repr.last_hidden_state
        # embeddings = embeddings.reshape(embeddings.shape[0]//2, 2, embeddings.shape[1], embeddings.shape[2])
        # embeddings = torch.cat([embeddings[:,0], embeddings[:,1]], dim=-1)
                
        # reshape back [B,2,L,H] -> concat axes
        B2, L, H = last.size()
        b = len(batch['name'])
        last = last.view(b, 2, L, H)
        emb = torch.cat([last[:,0], last[:,1]], dim=-1)
        mask = batch['attention_mask'][::2]  # use every two rows
        ends = mask.sum(dim=-1) - 1; start = 1
        if post_process:
            return self.post_process_cpu(batch, emb, mask, start, ends, task_type)
        return emb

# ProtGPT2 -gzy
class ProtGPT2Model(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022):
        super().__init__(device)
        self.tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_ZOOM_PATH}/ProtGPT2")
        self.model = AutoModelForCausalLM.from_pretrained(f"{MODEL_ZOOM_PATH}/ProtGPT2").to(self.device)
        self.max_length = max_length

    def construct_batch(self, batch):
        max_len = max(len(self.tokenizer.encode(s)) for sample in batch for s in ([sample['seq']] if not isinstance(sample['seq'], list) else sample['seq']))
        names, seqs, masks, labels = [], [], [], []
        for sample in batch:
            seqs_list = sample['seq'] if isinstance(sample['seq'], list) else [sample['seq']]
            tids, m = [], []
            for s in seqs_list:
                tok = torch.tensor(self.tokenizer.encode(s))
                mask = torch.zeros(max_len, dtype=torch.bool); mask[:len(tok)] = True
                tok = self.pad_data(tok, dim=0, max_length=max_len)
                mask = self.pad_data(mask, dim=0, max_length=max_len)
                tids.append(tok); m.append(mask)
            names.append(sample['name']); seqs.append(torch.hstack(tids)); masks.append(torch.hstack(m)); labels.append(sample['label'])
        return {
            'name': names,
            'seq': torch.stack(seqs).to(self.device),
            'attention_mask': torch.stack(masks).to(self.device),
            'label': labels
        }

    def forward(self, batch, post_process=True, task_type='binary_classification'):
        out = self.model(input_ids=batch['seq'], attention_mask=batch['attention_mask'], output_hidden_states=True)
        emb = out.hidden_states[-1]
        ends = batch['attention_mask'].sum(dim=-1); start = 0
        if post_process:
            return self.post_process_cpu(batch, emb, batch['attention_mask'], start, ends, task_type)
        return emb

# ProTrek
class ProTrekModel(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022):
        super().__init__(device)
        config = {
            "protein_config": f"{MODEL_ZOOM_PATH}/ProTrek/ProTrek_650M_UniRef50/esm2_t33_650M_UR50D",
            "text_config": f"{MODEL_ZOOM_PATH}/ProTrek/ProTrek_650M_UniRef50/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "structure_config": f"{MODEL_ZOOM_PATH}/ProTrek/ProTrek_650M_UniRef50/foldseek_t30_150M",
            "load_protein_pretrained": False,
            "load_text_pretrained": False,
            "from_checkpoint": f"{MODEL_ZOOM_PATH}/ProTrek/ProTrek_650M_UniRef50/ProTrek_650M_UniRef50.pt"
        }
        self.model = ProTrekTrimodalModel(**config).eval().to(self.device)
        self.encoder_3di = mini3di.Encoder()
        self.max_length = max_length

    def construct_batch(self, batch):
        names, seqs, structs, masks, labels = [], [], [], [], []
        max_length_batch = min(
            max([len(sample['seq']) for sample in batch]) + 2,
            self.max_length + 2
        )
        for sample in batch:
            seq = sample['seq']; X = sample['X']
            if X is None: continue
            N, CA, C, CB = X[:,0], X[:,1], X[:,2], X[:,3]
            states = self.encoder_3di.encode_atoms(ca=CA.numpy(), cb=CB.numpy(), n=N.numpy(), c=C.numpy())
            struct_seq = self.encoder_3di.build_sequence(states).lower()
            # merged sequence
            mask = torch.zeros(max_length_batch, dtype=torch.bool)
            mask[:len(seq)+2] = True
            names.append(sample['name'])
            seqs.append(seq)
            structs.append(struct_seq)
            masks.append(mask)
            labels.append(sample['label'])
        return {
            'name': names,
            'seq': seqs,
            'struct': structs,
            'attention_mask': torch.stack(masks).to(self.device),
            'label': labels
        }

    def forward(self, batch, post_process=True, task_type='binary_classification'):
        # get representations
        prot = self.model.get_protein_repr(batch['seq'])
        struct = self.model.get_structure_repr(batch['struct']) 
        emb = torch.cat([prot, struct], dim=-1)
        mask = batch['attention_mask']
        ends = mask.sum(dim=-1) - 1; start = 0
        if post_process:
            return self.post_process_cpu(batch, emb, mask, start, ends, task_type)
        return emb

# SaPort - seq + struct
class SaPortModel(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022, sequence_only=False):
        super().__init__(device)
        from transformers import EsmTokenizer, EsmForMaskedLM
        self.encoder_3di = mini3di.Encoder()
        self.tokenizer = EsmTokenizer.from_pretrained(f'{MODEL_ZOOM_PATH}/SaPort/ckpt')
        self.model = EsmForMaskedLM.from_pretrained(f'{MODEL_ZOOM_PATH}/SaPort/ckpt').to(self.device)
        self.max_length = max_length
        self.sequence_only = sequence_only

    def construct_batch(self, batch):
        names, seqs, masks, labels = [], [], [], []
        max_len = min(
            max([len(s['seq']) for s in batch]) + 2,
            self.max_length + 2
        )
        for sample in batch:
            seq, X = sample['seq'], sample['X']
            N, CA, C, CB = X[:,0], X[:,1], X[:,2], X[:,3]
            states = self.encoder_3di.encode_atoms(ca=CA.numpy(), cb=CB.numpy(), n=N.numpy(), c=C.numpy())
            struct_seq = self.encoder_3di.build_sequence(states).lower()
            merged = ''.join(a + b.lower() for a, b in zip(seq, struct_seq))
            tid = torch.tensor(self.tokenizer(merged, return_tensors='pt').input_ids[0])
            mask = torch.zeros(max_len, dtype=torch.bool)
            mask[:len(tid)] = True
            tid = self.pad_data(tid, dim=0, max_length=max_len)
            mask = self.pad_data(mask, dim=0, max_length=max_len)

            names.append(sample['name'])
            seqs.append(tid)
            masks.append(mask)
            labels.append(torch.tensor(sample['label']))
        return {
            'name': names,
            'seq': torch.stack(seqs).to(self.device),
            'attention_mask': torch.stack(masks).to(self.device),
            'label': labels
        }

    def forward(self, batch, post_process=True, task_type='binary_classification'):
        seq, mask = batch['seq'], batch['attention_mask']
        out = self.model.esm(input_ids=seq, attention_mask=mask, return_dict=True)
        emb = out.last_hidden_state
        start = 0
        ends = mask.sum(dim=-1) - 1
        if post_process:
            return self.post_process_cpu(batch, emb, mask, start, ends, task_type)
        return emb


# VenusPLM: seq only
class VenusPLMModel(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022):
        super().__init__(device)
        config = TransformerConfig.from_pretrained(MODEL_ZOOM_PATH + '/venusplm', attn_impl="sdpa")
        self.model = TransformerForMaskedLM.from_pretrained(MODEL_ZOOM_PATH + '/venusplm', config=config).to(self.device)
        self.tokenizer = VPLMTokenizer.from_pretrained(MODEL_ZOOM_PATH + '/venusplm')
        self.max_length = max_length

    def construct_batch(self, batch):
        names, seqs, masks, labels = [], [], [], []
        max_len = min(
            max([len(self.tokenizer.encode(s['seq'])) for s in batch]) + 2,
            self.max_length + 2
        )
        for sample in batch:
            seq = sample['seq']
            seq_tokens = torch.tensor(self.tokenizer.encode(seq))[:self.max_length]
            attention_mask = torch.zeros(max_len, dtype=torch.bool)
            attention_mask[:len(seq_tokens)] = True
            seq_tokens = self.pad_data(seq_tokens, dim=0, max_length=max_len)
            seq_tokens = self.pad_data(seq_tokens, dim=0, max_length=max_len)

            names.append(sample['name'])
            seqs.append(seq_tokens)
            masks.append(attention_mask)
            labels.append(torch.tensor(sample['label']))

        return {
            'name': names,
            'seq': torch.stack(seqs).to(self.device),
            'attention_mask': torch.stack(masks).to(self.device),
            'label': labels
        }

    def forward(self, batch, post_process=True, task_type='binary_classification'):
        out = self.model(input_ids=batch['seq'], attention_mask=batch['attention_mask'], output_hidden_states=True)
        emb = out.hidden_states[-1]
        start = 0
        ends = batch['attention_mask'].sum(dim=-1) - 1
        if post_process:
            return self.post_process_cpu(batch, emb, batch['attention_mask'], start, ends, task_type)
        return emb
    

 # ProSST-2048 seq + struct    
class ProSST2048Model(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022):
        super().__init__(device)
        from model_zoom.ProSST.prosst.structure.quantizer import PdbQuantizer
        weight_path = f"{MODEL_ZOOM_PATH}/ProSST/prosst_2048_weight"
        self.tokenizer = AutoTokenizer.from_pretrained(weight_path, trust_remote_code=True)
        self.quantizer = PdbQuantizer(structure_vocab_size=2048)
        self.model = AutoModelForMaskedLM.from_pretrained(weight_path, trust_remote_code=True).to(self.device)
        self.max_length = max_length

    def construct_batch(self, batch):
        max_len = min(max([len(s['seq']) for s in batch]) + 2, self.max_length + 2)
        names, seqs, xs, masks, labels = [], [], [], [], []
        for sample in batch:
            seq = sample['seq']
            pdb_path = sample['pdb_path']
            pdb_name = os.path.basename(pdb_path)
            seq_tokens = torch.tensor(self.tokenizer.encode(seq))[:self.max_length]
            struct = self.quantizer(pdb_path, return_residue_seq=False)['2048'][pdb_name]["struct"]
            struct_seq = [i + 3 for i in struct]
            struct_seq = [1] + struct_seq + [2]
            struct_tokens = torch.tensor(struct_seq)[:self.max_length]
            attention_mask = torch.zeros(max_len, dtype=torch.bool)
            attention_mask[:len(seq_tokens)] = True
            seq_tokens = self.pad_data(seq_tokens, dim=0, max_length=max_len)
            struct_tokens = self.pad_data(struct_tokens, dim=0, max_length=max_len)
            names.append(sample['name'])
            seqs.append(seq_tokens)
            xs.append(struct_tokens)
            masks.append(attention_mask)
            labels.append(torch.tensor(sample['label']))
        return {
            'name': names,
            'seq': torch.stack(seqs).to(self.device),
            'X': torch.stack(xs).to(self.device),
            'attention_mask': torch.stack(masks).to(self.device),
            'label': labels
        }

    def forward(self, batch, post_process=True, task_type='binary_classification'):
        outputs = self.model(
            input_ids=batch['seq'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=True,
            ss_input_ids=batch['X']
        )
        embeddings = outputs.hidden_states[-1]
        ends = batch['attention_mask'].sum(dim=-1) - 1
        start = 1
        if post_process:
            return self.post_process_cpu(batch, embeddings, batch['attention_mask'], start, ends, task_type)
        return embeddings


# ProtTrans https://github.com/agemagician/ProtTrans
class ProtT5(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022):
        super().__init__(device)
        from transformers import T5Tokenizer, T5EncoderModel
        weight_path = f"{MODEL_ZOOM_PATH}/ProtT5"
        self.tokenizer = T5Tokenizer.from_pretrained(weight_path, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(weight_path).to(device)
        self.max_length = max_length

    def construct_batch(self, batch):
        max_len = min(max([len(s['seq']) for s in batch]) + 2, self.max_length + 2)
        # max_len = min(max([len(s['seq']) for s in batch]), self.max_length) + 2
        names, seqs, masks, labels = [], [], [], []
        for sample in batch:
            seq = " ".join(list(sample['seq'][:1022]))
            seq_tokens = torch.tensor(self.tokenizer.encode(seq, add_special_tokens=False))[:self.max_length]
            attention_mask = torch.zeros(max_len, dtype=torch.bool)
            attention_mask[:len(seq_tokens)] = True
            seq_tokens = self.pad_data(seq_tokens, dim=0, max_length=max_len)
            names.append(sample['name'])
            seqs.append(seq_tokens)
            masks.append(attention_mask)
            labels.append(torch.tensor(sample['label']))
        return {
            'name': names,
            'seq': torch.stack(seqs).to(self.device),
            'attention_mask': torch.stack(masks).to(self.device),
            'label': labels
        }

    def forward(self, batch, post_process=True, task_type='binary_classification'):
        embedding_repr = self.model(
            input_ids=batch['seq'],
            attention_mask=batch['attention_mask'],
        )
        embeddings = embedding_repr.last_hidden_state
        ends = batch['attention_mask'].sum(dim=-1) - 1
        start = 1
        if post_process:
            return self.post_process_cpu(batch, embeddings, batch['attention_mask'], start, ends, task_type)
        return embeddings

class DPLMModel(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022, **kwargs):
        super().__init__(device)
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.pretrain_model = AutoModelForMaskedLM.from_pretrained(f"{MODEL_ZOOM_PATH}/esm2_650m").to(self.device)
        params = torch.load(f'{MODEL_ZOOM_PATH}/dplm_650m/pytorch_model.bin')
        self.pretrain_model.load_state_dict(params, strict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_ZOOM_PATH}/esm2_650m")
        self.max_length = max_length

    def construct_batch(self, batch):
        MAXLEN = self.max_length
        max_length_batch = min(
            max([len(sample['seq']) for sample in batch]) + 2,  # +2 for <s> and </s>
            self.max_length + 2
        )
        result = {
            'name': [],
            'seq': [],
            'attention_mask': [],
            'label': []
        }
        for sample in batch:
            seq_token = torch.tensor(self.tokenizer.encode(sample['seq']))[:MAXLEN]
            attention_mask = torch.zeros(max_length_batch)
            attention_mask[:len(seq_token)] = 1
            seq_token = self.pad_data(seq_token, dim=0, max_length=max_length_batch)
            result['name'].append(sample['name'])
            result['seq'].append(seq_token)
            result['attention_mask'].append(attention_mask)
            result['label'].append(sample['label'])

        result['seq'] = torch.stack(result['seq'], dim=0).to(self.device)
        result['attention_mask'] = torch.stack(result['attention_mask'], dim=0).to(self.device)
            
        return result

    def forward(self, batch, post_process=True, task_type='binary_classification', return_prob=False):
        attention_mask = batch['attention_mask']
        outputs = self.pretrain_model.esm(
                        batch['seq'],
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
        if return_prob:
            logits = self.pretrain_model.lm_head(outputs.last_hidden_state)
            probs = F.softmax(logits, dim=-1)
            return probs
        
        embeddings = outputs.last_hidden_state
        ends = attention_mask.sum(dim=-1)-1
        start = 1
        if post_process:
            result = self.post_process_cpu(batch, embeddings, attention_mask, start, ends, task_type=task_type)
        else:
            result = embeddings
        return result

class OntoProteinModel(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022, **kwargs):
        super().__init__(device)
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_ZOOM_PATH}/OntoProtein")
        self.model = AutoModelForMaskedLM.from_pretrained(f"{MODEL_ZOOM_PATH}/OntoProtein").to(self.device)
        self.max_length = max_length

    def construct_batch(self, batch):
        import re
        MAXLEN = self.max_length
        max_length_batch = min(
            max([len(sample['seq']) for sample in batch]) + 2, # +2 for <s> and </s>
            MAXLEN + 2
        )
        result = {
            'name': [],
            'seq': [],
            'attention_mask': [],
            'token_type_ids': [],
            'label': []
        }
        for sample in batch:
            sequence_Example = ' '.join(sample['seq'])
            sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
            encoded_input = self.tokenizer(sequence_Example, return_tensors='pt')
            
            input_ids = self.pad_data(encoded_input['input_ids'][0], dim=0, max_length=max_length_batch)
            attention_mask = self.pad_data(encoded_input['attention_mask'][0], dim=0, max_length=max_length_batch)
            token_type_ids = self.pad_data(encoded_input['token_type_ids'][0], dim=0, max_length=max_length_batch)
            
            
            result['name'].append(sample['name'])
            result['seq'].append(input_ids)
            result['attention_mask'].append(attention_mask)
            result['token_type_ids'].append(token_type_ids)
            result['label'].append(sample['label'])

        result['seq'] = torch.stack(result['seq'], dim=0).to(self.device)
        result['attention_mask'] = torch.stack(result['attention_mask'], dim=0).to(self.device)
        result['token_type_ids'] = torch.stack(result['token_type_ids'], dim=0).to(self.device)
            
        return result

    def forward(self, batch, post_process=True, task_type='binary_classification', return_prob=False):
        output = self.model.bert(
            input_ids=batch['seq'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
        )
        if return_prob:
            logits = self.model.cls(output.last_hidden_state)
            probs = F.softmax(logits, dim=-1)
            return probs
        
        attention_mask = batch['attention_mask']    
        embeddings = output.last_hidden_state
        ends = attention_mask.sum(dim=-1)-1
        start = 1
        if post_process:
            result = self.post_process_cpu(batch, embeddings, attention_mask, start, ends, task_type=task_type)
        else:
            result = embeddings
        return result


class ANKHBase(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022):
        super().__init__(device)
        from transformers import AutoTokenizer, T5EncoderModel
        weight_path = f"{MODEL_ZOOM_PATH}/ankh_base"
        self.tokenizer = AutoTokenizer.from_pretrained(weight_path)
        self.model = T5EncoderModel.from_pretrained(weight_path).to(device)
        self.max_length = max_length

    def construct_batch(self, batch):
        max_len = min(max([len(s['seq']) for s in batch]) + 2, self.max_length + 2)
        names, seqs, masks, labels = [], [], [], []
        for sample in batch:
            seq = sample['seq'][:1022]
            seq_tokens = torch.tensor(self.tokenizer.encode(seq, add_special_tokens=False))[:self.max_length]
            attention_mask = torch.zeros(max_len, dtype=torch.bool)
            attention_mask[:len(seq_tokens)] = True
            seq_tokens = self.pad_data(seq_tokens, dim=0, max_length=max_len)
            names.append(sample['name'])
            seqs.append(seq_tokens)
            masks.append(attention_mask)
            labels.append(torch.tensor(sample['label']))

        return {
            'name': names,
            'seq': torch.stack(seqs).to(self.device),
            'attention_mask': torch.stack(masks).to(self.device),
            'label': labels
        }

    def forward(self, batch, post_process=True, task_type='binary_classification'):
        embedding_repr = self.model(
            input_ids=batch['seq'],
            attention_mask=batch['attention_mask'],
        )
        embeddings = embedding_repr.last_hidden_state
        ends = batch['attention_mask'].sum(dim=-1) - 1
        start = 1
        if post_process:
            return self.post_process_cpu(batch, embeddings, batch['attention_mask'], start, ends, task_type)
        return embeddings

class PGLMModel(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022, **kwargs):
        super().__init__(device)
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.tokenizer  = AutoTokenizer.from_pretrained(f"{MODEL_ZOOM_PATH}/proteinglm-1b-mlm", trust_remote_code=True, use_fast=True)
        self.model = AutoModelForMaskedLM.from_pretrained(f"{MODEL_ZOOM_PATH}/proteinglm-1b-mlm",  trust_remote_code=True).to(self.device)
        self.max_length = max_length

    def construct_batch(self, batch):
        MAXLEN = self.max_length
        max_length_batch = min(
            max([len(sample['seq']) for sample in batch]) + 2, # +2 for <s> and </s>
            self.max_length + 2
        )
        result = {
            'name': [],
            'seq': [],
            'attention_mask': [],
            'label': []
        }
        for sample in batch:
            output = self.tokenizer(sample['seq'], add_special_tokens=True, return_tensors='pt')
            seq_token = output['input_ids'][0]
            attention_mask = torch.zeros(max_length_batch)
            attention_mask[:len(seq_token)] = 1
            seq_token = self.pad_data(seq_token, dim=0, max_length=max_length_batch)
            result['name'].append(sample['name'])
            result['seq'].append(seq_token)
            result['attention_mask'].append(attention_mask)
            result['label'].append(sample['label'])

        result['seq'] = torch.stack(result['seq'], dim=0).to(self.device)
        result['attention_mask'] = torch.stack(result['attention_mask'], dim=0).to(self.device)
            
        return result

    def forward(self, batch, post_process=True, task_type='binary_classification', return_prob=False):
        attention_mask = batch['attention_mask']
        outputs = self.model(
                        batch['seq'],
                        attention_mask=batch['attention_mask'],
                        output_hidden_states=True, return_last_hidden_state=True
                    )
        if return_prob:
            logits = self.pretrain_model.lm_head(outputs.last_hidden_state)
            probs = F.softmax(logits, dim=-1)
            return probs
        
        embeddings = outputs.hidden_states.permute(1,0,2)
        ends = attention_mask.sum(dim=-1)-1
        start = 1
        if post_process:
            result = self.post_process_cpu(batch, embeddings, attention_mask, start, ends, task_type=task_type)
        else:
            result = embeddings
        return result

# TODO: 
# ProtTrans https://github.com/agemagician/ProtTrans --> OK
# xTrimoPGLM https://github.com/ONERAI/xTrimoPGLM
# Ankh https://github.com/agemagician/Ankh --> OK
# ProteinBERT https://github.com/nadavbra/protein_bert --> keras模型, 不必整合
# IgLM https://github.com/Graylab/IgLM --> 小模型, 不必整合
# OntoProtein https://github.com/zjunlp/OntoProtein --> OK
# ABGNN https://github.com/KyGao/ABGNN --> 不必整合
# DPLM  https://github.com/bytedance/dplm --> OK
