import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data.protein_dataset import dynamic_pad
from src.data.esm.sdk.api import LogitsConfig
import os
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import sys; sys.path.append('/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom')
from model_zoom.esm.utils.sampling import _BatchedESMProteinTensor
MODEL_ZOOM_PATH = '/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom'
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class PretrainModelInterface(nn.Module):
    """
    setup_model: 注册预训练模型
    construct_batch: 构建模型的输入batch, 需要进行padding。如果序列加上EOS, BOS, 结构也需要在这里进行padding
    forward: 调用不同模型提取蛋白质氨基酸级别的embedding, 在后处理阶段去除EOS, BOS embedding，只返回氨基酸embedding
    """

    def __init__(self, pretrain_model_name, batch_size = 64, max_length = 1022, device = 'cuda', sequence_only=False, task_type=None):
        super(PretrainModelInterface, self).__init__()
        self.pretrain_model_name = pretrain_model_name
        self.sequence_only = sequence_only
        self.batch_size = batch_size
        self.max_length = max_length
        self.task_type = task_type
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
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            from model_zoom.procyon.model.model_unified import UnifiedProCyon
            from model_zoom.GearNet.data.protein import Protein
            from model_zoom.GearNet.data.transform import ProteinView
            from model_zoom.GearNet.data.transform import Compose
            from model_zoom.GearNet.data.geo_graph import GraphConstruction
            from model_zoom.GearNet.data.function import AlphaCarbonNode, SpatialEdge, KNNEdge, SequentialEdge
            from model_zoom.GearNet.gearnet import GeometryAwareRelationalGraphNeuralNetwork
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
            self.pretrain_model, _ = UnifiedProCyon.from_pretrained(
                pretrained_weights_dir=procyon_ckpt, 
                checkpoint_dir=procyon_ckpt
            )
            self.pretrain_model = self.pretrain_model.to(self.device)
        elif self.pretrain_model_name == 'gearnet':
            from model_zoom.GearNet.data.transform import ProteinView
            from model_zoom.GearNet.data.transform import Compose
            from model_zoom.GearNet.data.geo_graph import GraphConstruction
            from model_zoom.GearNet.data.function import AlphaCarbonNode, SpatialEdge, KNNEdge, SequentialEdge
            from model_zoom.GearNet.gearnet import GeometryAwareRelationalGraphNeuralNetwork
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
            self.pretrain_model = self.gearnet_edge.to(self.device)
            self.pretrain_model.eval()
        elif self.pretrain_model_name == 'prollama':
            from transformers import LlamaForCausalLM, LlamaTokenizer
            llama_path = "/nfs_beijing/kubeflow-user/wanghao/workspace/ai4sci/protein_benchmark_project/data/ProLLaMA"
            self.pretrain_model = LlamaForCausalLM.from_pretrained(
                llama_path,
                quantization_config=None
            ).to(self.device)
            self.tokenizer = LlamaTokenizer.from_pretrained(llama_path)
        elif self.pretrain_model_name == 'prost':
            from transformers import AutoModel, AutoTokenizer, AutoConfig
            prost_weights = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/protst"
            prost_tokenizer_weights = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/esm1b_650m"
            protst_model = AutoModel.from_pretrained(
                prost_weights, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16
            )
            self.pretrain_model = protst_model.protein_model.to(self.device)
            self.tokenizer =  AutoTokenizer.from_pretrained(prost_tokenizer_weights)
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
        elif self.pretrain_model_name == 'venusplm':
            from vplm import TransformerForMaskedLM, TransformerConfig
            from vplm import VPLMTokenizer
            venusplm_weight_path = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/venusplm"
            config = TransformerConfig.from_pretrained(venusplm_weight_path, attn_impl="sdpa") # or "flash_attn" if you have installed flash-attn
            self.pretrain_model = TransformerForMaskedLM.from_pretrained(venusplm_weight_path, config=config).to(self.device)
            self.pretrain_model.eval()
            self.tokenizer = VPLMTokenizer.from_pretrained(venusplm_weight_path)
        elif self.pretrain_model_name == 'prosst2048':
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            from model_zoom.ProSST.prosst.structure.quantizer import PdbQuantizer
            prosst_2048_weight_path = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/ProSST/prosst_2048_weight"

            self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(prosst_2048_weight_path, trust_remote_code=True)
            self.ss_processor = PdbQuantizer(structure_vocab_size=2048)
            self.pretrain_model = AutoModelForMaskedLM.from_pretrained(prosst_2048_weight_path, trust_remote_code=True).to(self.device)
            self.pretrain_model.eval()
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
        MAXLEN = 1022
        for i in range(0, len(data), batch_size):
            if self.pretrain_model_name == 'esm2_650m':
                add_BOS = 1
                name_batch, label_batch, smiles_batch = [], [], []
                if "pair" in self.task_type:
                    current_batch = [sample["seq"] for sample in data[i:i + batch_size]]
                    max_length_batch = [max(len(s) for s in column)+2 for column in zip(*current_batch)]
                else:
                    max_length_batch = [max([len(sample['seq']) for sample in data[i:i + batch_size]]) + 2]
                X_batch, S_batch, mask_batch, t_lengths = [[] for _ in range(len(max_length_batch))], \
                                                          [[] for _ in range(len(max_length_batch))], \
                                                          [[] for _ in range(len(max_length_batch))], \
                                                          [[] for _ in range(len(max_length_batch))] 
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    label = sample['label']
                    smiles = sample['smiles'] if 'smiles' in sample else None
                    if smiles:
                        mol = Chem.MolFromSmiles(smiles)
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                        smiles = torch.tensor([int(ele) for ele in list(fp.ToBitString())]).float()
                    if not isinstance(seq, list):
                        seq = [seq]

                    for j, _seq in enumerate(seq):
                        _seq = _seq[:MAXLEN]
                        t_lengths[j].append(len(_seq))
                        attention_mask = torch.zeros(max_length_batch[j])
                        seq_token = torch.tensor(self.tokenizer.encode(_seq))
                        attention_mask[:len(seq_token)] = 1
                        seq_token = self.pad_data(seq_token, dim=0, max_length=max_length_batch[j])
                        # _X = dynamic_pad(_X, [add_BOS, add_BOS], dim=0, pad_value=0) # 坐标需要和seq一样加上BOS, EOS
                        # _X = self.pad_data(_X, dim=0, max_length=max_length_batch[j])

                        S_batch[j].append(seq_token)
                        # X_batch[j].append(_X)
                        mask_batch[j].append(attention_mask)
                    
                    if task_name == 'contact_map':
                        label = F.pad(label, [add_BOS, self.max_length-label.shape[0]-add_BOS, add_BOS, self.max_length-label.shape[0]-add_BOS])
                    else:
                        label = torch.tensor(label)

                    label_batch.append(label)
                    name_batch.append(sample['name'])
                    if smiles is not None:
                        smiles_batch.append(smiles)

                S_batch = [torch.stack(ele).to(self.device) for ele in S_batch]
                # X_batch = [torch.stack(ele).to(self.device) for ele in X_batch]
                mask_batch = [torch.stack(ele).to(self.device)==1 for ele in mask_batch]
                t_lengths = [torch.tensor(ele).to(self.device) for ele in t_lengths]
                smiles_batch = None if len(smiles_batch) == 0 else torch.stack(smiles_batch)
                yield {
                    'name': name_batch,
                    'seq': S_batch,
                    # 'X': X_batch,
                    't_lengths': t_lengths,
                    'smiles': smiles_batch,
                    'attention_mask': mask_batch,
                    'label': torch.stack(label_batch),
                }
            
            if self.pretrain_model_name == 'esm3_1.4b':
                addBOS = 1
                from model_zoom.esm.utils.misc import stack_variable_length_tensors
                from model_zoom.esm.utils.sampling import _BatchedESMProteinTensor
                from model_zoom.esm.utils import encoding
                model = self.pretrain_model
                name_batch, sequence_list, coordinates_list, label_batch, structure_tokens_batch, mask_batch = [], [], [], [], [], []
                seq_tokenizer = model.tokenizers.sequence
                struct_tokenizer = model.tokenizers.structure
                pad = model.tokenizers.sequence.pad_token_id
                if "pair" in self.task_type:
                    max_length_batch = max(
                        [max(len(seq) for seq in sample['seq']) for sample in data[i:i + batch_size]]
                    ) + addBOS*2
                else:
                    max_length_batch = max([len(sample['seq']) for sample in data[i:i + batch_size]])+addBOS*2
                for sample in data[i:i + batch_size]:
                    name, label, seq, X = sample['name'], sample['label'], sample['seq'], sample['X']
                    sequence_tokens, structure_tokens, coordinates, masks = [], [], [], []
                    for _seq, _X in zip(seq, X):
                        _seq, _X = _seq[:MAXLEN], _X[:MAXLEN]
                        _seq_token = encoding.tokenize_sequence(_seq, seq_tokenizer, add_special_tokens=True)
                        _coordinates, _plddt, _structure_token = encoding.tokenize_structure(
                            _X, 
                            model.get_structure_encoder(), 
                            struct_tokenizer, 
                            add_special_tokens=True
                        )
                        mask = torch.zeros(max_length_batch)
                        mask[:_coordinates.shape[0]] = 1

                        # pad
                        _seq_token = self.pad_data(_seq_token, dim=0, pad_value=pad, max_length=max_length_batch)
                        _structure_token = self.pad_data(_structure_token, dim=0, pad_value=pad, max_length=max_length_batch)
                        _coordinates = dynamic_pad(_coordinates, [addBOS, addBOS], dim=0, pad_value=0) # 坐标需要和seq一样加上BOS, EOS
                        _coordinates = self.pad_data(_coordinates, dim=0, max_length=max_length_batch)
                        sequence_tokens.append(_seq_token)
                        structure_tokens.append(_structure_token)
                        coordinates.append(_coordinates)
                        masks.append(mask)
                    
                    sequence_tokens = torch.hstack(sequence_tokens)
                    structure_tokens = torch.hstack(structure_tokens)
                    coordinates = torch.vstack(coordinates)
                    masks = torch.hstack(masks)

                    sequence_list.append(sequence_tokens)
                    coordinates_list.append(coordinates)
                    structure_tokens_batch.append(structure_tokens)
                    mask_batch.append(masks)
                    name_batch.append(sample['name'])
                    label = sample['label']
                    if task_name == 'contact_map':
                        label = F.pad(label, [add_BOS, self.max_length-label.shape[0]-add_BOS, add_BOS, self.max_length-label.shape[0]-add_BOS])
                    else:
                        label = torch.tensor(label)
                        
                    label_batch.append(label)
                
                sequence_tokens = stack_variable_length_tensors(
                    sequence_list,
                    constant_value=pad,
                ).to(self.device)
                
                structure_tokens_batch = stack_variable_length_tensors(
                    structure_tokens_batch,
                    constant_value=pad,
                ).to(self.device)
                                
                coordinates_batch = stack_variable_length_tensors(
                    coordinates_list,
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
                    'attention_mask': torch.stack(mask_batch).to(self.device)==1
                }
                    
            
            if self.pretrain_model_name == 'esmc_600m':
                addBOS = 1
                from model_zoom.esm.utils.sampling import _BatchedESMProteinTensor
                from model_zoom.esm.utils.misc import stack_variable_length_tensors
                name_batch, seq_batch, mask_batch, label_batch = [], [], [], []
                if "pair" in self.task_type:
                    max_length_batch = max(
                        [max(len(seq) for seq in sample['seq']) for sample in data[i:i + batch_size]]
                    ) + addBOS*2
                else:
                    max_length_batch = max([len(sample['seq']) for sample in data[i:i + batch_size]])+addBOS*2
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    label = sample['label']
                    if not isinstance(seq, list):
                        seq = [seq]
                    seq_tokens, attention_masks = [], []
                    for _seq in seq:
                        attention_mask = torch.zeros(max_length_batch)
                        attention_mask[:len(_seq)] = 1
                        _seq = _seq[:max_length_batch]
                        _seq_token = self.pretrain_model._tokenize([_seq]).flatten()
                        pad = self.pretrain_model.tokenizer.pad_token_id
                        _seq_token = self.pad_data(_seq_token, dim=0, pad_value=pad, max_length=max_length_batch)
                        seq_tokens.append(_seq_token)
                        attention_masks.append(attention_mask)
                    seq_tokens = torch.hstack(seq_tokens)
                    attention_masks = torch.hstack(attention_masks)

                    seq_batch.append(seq_tokens)
                    mask_batch.append(attention_masks)
                    if task_name == 'contact_map':
                        label = F.pad(label, [add_BOS, self.max_length-label.shape[0]-add_BOS, add_BOS, self.max_length-label.shape[0]-add_BOS])
                    else:
                        label = torch.tensor(label)
                    label_batch.append(label)
                    name_batch.append(sample['name'])

                sequence_tokens = stack_variable_length_tensors(
                    seq_batch,
                    constant_value=self.pretrain_model.tokenizer.pad_token_id,
                )
                # sequence_tokens = self.pretrain_model._tokenize(seq_batch)
                protein_tensor = _BatchedESMProteinTensor(sequence=sequence_tokens).to(self.device)
                
                yield {
                    'name': name_batch,
                    'protein_tensor': protein_tensor,
                    'attention_mask': torch.stack(mask_batch).to(self.device)==1,
                    'label': torch.stack(label_batch).to(self.device),
                }
            
            if self.pretrain_model_name == 'procyon':
                add_BOS = 0
                from model_zoom.GearNet.data.protein import Protein
                name_batch, X_batch, S_batch, label_batch = [], [], [], []
                if "pair" in self.task_type:
                    max_length_batch = max(
                        [max(len(seq) for seq in sample['seq']) for sample in data[i:i + batch_size]]
                    ) + 2
                else:
                    max_length_batch = max([len(sample['seq']) for sample in data[i:i + batch_size]])+2

                for sample in data[i:i + batch_size]:
                    try:
                        seq = sample['seq']
                        label = sample['label']
                        pdb_path = sample['pdb_path']
                        if not isinstance(seq, list) and not isinstance(pdb_path, list):
                            seq, pdb_path = [seq], [pdb_path]
                        seq_embeddings, struct_embeddings = [], []
                        for _seq, _pdb_path in zip(seq, pdb_path):
                            # seq_token = torch.tensor(self.esm_tokenizer.encode(_seq))
                            seq_token = self.esm_tokenizer(
                                [_seq], 
                                return_tensors="pt",
                                padding=True,
                                max_length=max_length_batch,
                                truncation=True
                            )
                            seq_embedding = self.esm_pretrain_model.esm(
                                seq_token['input_ids'].to(self.device),
                                attention_mask=seq_token['attention_mask'].to(self.device),
                                return_dict=True,
                            ).last_hidden_state.squeeze(0).mean(0).flatten()
                            seq_embeddings.append(seq_embedding)

                            # strcuture
                            protein = Protein.from_pdb(_pdb_path, bond_feature="length", residue_feature="symbol")
                            protein = self.transform({"graph": protein})["graph"]
                            _protein = Protein.pack([protein])
                            protein_ = self.graph_construction_model(_protein).to(self.device)
                            with torch.no_grad():
                                out = self.gearnet_edge(protein_, protein_.node_feature.float())
                            struct_embeddings.append(out["graph_feature"].flatten())

                        seq_embeddings = torch.cat(seq_embeddings, dim=-1)
                        struct_embeddings = torch.cat(struct_embeddings, dim=-1)
                        
                        if task_name == 'contact_map':
                            label = F.pad(label, [add_BOS, self.max_length-label.shape[0]-add_BOS, add_BOS, self.max_length-label.shape[0]-add_BOS])
                        else:
                            label = torch.tensor(label)

                        S_batch.append(seq_embeddings)
                        X_batch.append(struct_embeddings)
                        label_batch.append(label)
                        name_batch.append(sample['name'])
                    except:
                        print(f"Error processing sample {sample['name']}")
                        continue
        
                yield {
                    'name': name_batch,
                    'seq': torch.stack(S_batch).to(self.device),
                    'X': torch.stack(X_batch).unsqueeze(1).to(self.device),
                    'label': torch.stack(label_batch).to(self.device),
                }
            
            if self.pretrain_model_name == 'gearnet':
                add_BOS = 0
                from model_zoom.GearNet.data.protein import Protein
                name_batch, X_batch, S_batch, label_batch = [], [], [], []
                # proteins = []
                proteins_pair_1, proteins_pair_2 = [], []
                for sample in data[i:i + batch_size]:
                    try:
                        label = sample['label']
                        pdb_path = sample['pdb_path']
                        if not isinstance(pdb_path, list):
                            pdb_path = [pdb_path]
                        temp_proteins = []
                        for j, _pdb_path in enumerate(pdb_path):
                            protein = Protein.from_pdb(_pdb_path, bond_feature="length", residue_feature="symbol")
                            protein = self.transform({"graph": protein})["graph"]
                            temp_proteins.append(protein)
                            # sub_proteins.append(protein)
                        proteins_pair_1.append(temp_proteins[0])
                        proteins_pair_2.append(temp_proteins[1])
                        if task_name == 'contact_map':
                            label = F.pad(label, [add_BOS, self.max_length-label.shape[0]-add_BOS, add_BOS, self.max_length-label.shape[0]-add_BOS])
                        else:
                            label = torch.tensor(label)
                        label_batch.append(label)
                        name_batch.append(sample['name'])
                    except:
                        print(f"Error processing sample {sample['name']}")
                        continue
                proteins_pairs = [proteins_pair_1, proteins_pair_2]
                embeddings, attention_masks = [], []
                if len(proteins_pair_2) == 0:
                    max_length_batch = Protein.pack(proteins_pair_1).num_residues.max()
                else:
                    max_length_batch = max([Protein.pack(proteins).num_residues.max() for proteins in proteins_pairs])
                for proteins in proteins_pairs:
                    X_batch = []
                    if len(proteins_pairs) == 0:
                        continue
                    _protein = Protein.pack(proteins)
                    protein_ = self.graph_construction_model(_protein).to(self.device)
                    with torch.no_grad():
                        out = self.gearnet_edge(protein_, protein_.node_feature.float())
                    node_feature = out["node_feature"]
                    # max_length_batch = _protein.num_residues.max()
                    split = torch.cumsum(F.pad(_protein.num_residues, (1,0)), dim=0)
                    attention_mask = torch.zeros(len(split)-1, max_length_batch).to(self.device)
                    for i in range(len(split)-1):
                        start, end = split[i], split[i+1]
                        embedding = node_feature[start:end]
                        attention_mask[i, :embedding.shape[0]] = 1
                        embedding = self.pad_data(embedding, dim=0, max_length=max_length_batch)
                        X_batch.append(embedding)
                    X_batch = torch.stack(X_batch)
                    embeddings.append(X_batch)
                    attention_masks.append(attention_mask)
                embeddings = torch.cat(embeddings, dim=-1)
                if "pair" in self.task_type:
                    attention_mask = (attention_masks[0].int() + attention_masks[1].int() > 0)
                else:
                    attention_mask = attention_masks[0]
                
                yield {
                    'name': name_batch,
                    'attention_mask': attention_mask,
                    'X': embeddings.to(self.device),
                    'label': torch.stack(label_batch).to(self.device),
                }
            
            if self.pretrain_model_name == 'prollama':
                name_batch, S_batch, mask_batch, label_batch = [], [], [], []
                if "pair" in self.task_type:
                    max_length_batch = max(
                        [max(len(seq) for seq in sample['seq']) for sample in data[i:i + batch_size]]
                    ) + 2
                else:
                    max_length_batch = max([len(sample['seq']) for sample in data[i:i + batch_size]])+2

                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    label = sample['label']
                    if not isinstance(seq, list):
                        seq = [seq]
                    seq = [f"[Determine superfamily] Seq=<{_seq}>" for _seq in seq]
                    seq_tokens, attention_masks = [], []
                    for _seq in seq:
                        attention_mask = torch.zeros(max_length_batch)
                        seq_token = torch.tensor(self.tokenizer.encode(_seq))
                        attention_mask[:len(seq_token)] = 1
                        seq_token = self.pad_data(seq_token, dim=0, max_length=max_length_batch)
                        seq_tokens.append(seq_token)
                        attention_masks.append(attention_mask)
                    seq_tokens = torch.hstack(seq_tokens)
                    attention_masks = torch.hstack(attention_masks)
                    S_batch.append(seq_tokens)
                    mask_batch.append(attention_masks)
                    if task_name == 'contact_map':
                        label = F.pad(label, [add_BOS, self.max_length-label.shape[0]-add_BOS, add_BOS, self.max_length-label.shape[0]-add_BOS])
                    else:
                        label = torch.tensor(label)
                    label_batch.append(label)
                    name_batch.append(sample['name'])
        
                yield {
                    'name': name_batch,
                    'seq': torch.stack(S_batch).to(self.device),
                    'attention_mask': torch.stack(mask_batch).to(self.device)==1,
                    'label': torch.stack(label_batch).to(self.device),
                }
            
            if self.pretrain_model_name == "venusplm":
                name_batch, X_batch, S_batch, mask_batch, label_batch = [], [], [], [], []
                if "pair" in self.task_type:
                    max_length_batch = max(
                        [max(len(seq) for seq in sample['seq']) for sample in data[i:i + batch_size]]
                    ) + 2
                else:
                    max_length_batch = max([len(sample['seq']) for sample in data[i:i + batch_size]]) + 2
                    
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    label = sample['label']
                    if not isinstance(seq, list):
                        seq = [seq]
                    seq_tokens, attention_masks = [], []
                    for _seq in seq:
                        attention_mask = torch.zeros(max_length_batch)
                        seq_token = torch.tensor(self.tokenizer.encode(_seq))
                        attention_mask[:len(seq_token)] = 1
                        seq_token = self.pad_data(seq_token, dim=0, max_length=max_length_batch)
                        seq_tokens.append(seq_token)
                        attention_masks.append(attention_mask)
                    seq_tokens = torch.hstack(seq_tokens)
                    attention_masks = torch.hstack(attention_masks)
                    S_batch.append(seq_tokens)
                    mask_batch.append(attention_masks)
                    if task_name == 'contact_map':
                        label = F.pad(label, [add_BOS, self.max_length-label.shape[0]-add_BOS, add_BOS, self.max_length-label.shape[0]-add_BOS])
                    else:
                        label = torch.tensor(label)
                    label_batch.append(label)
                    name_batch.append(sample['name'])
        
                yield {
                    'name': name_batch,
                    'seq': torch.stack(S_batch).to(self.device),
                    'attention_mask': torch.stack(mask_batch).to(self.device)==1,
                    'label': torch.stack(label_batch).to(self.device),
                }
            
            if self.pretrain_model_name == "prosst2048":
                def tokenize_structure_sequence(structure_sequence):
                    shift_structure_sequence = [i + 3 for i in structure_sequence]
                    shift_structure_sequence = [1, *shift_structure_sequence, 2]
                    return torch.tensor(
                        # [
                            shift_structure_sequence,
                        # ],
                        dtype=torch.long,
                    )

                name_batch, X_batch, S_batch, mask_batch, label_batch = [], [], [], [], []
                
                max_length_batch = max([len(sample['seq']) for sample in data[i:i + batch_size]])+2
                    
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    label = sample['label']
                    X = sample['pdb_path']
                    attention_mask = torch.zeros(max_length_batch)
                    
                    seq_token = torch.tensor(self.tokenizer.encode(seq))
                    attention_mask[:len(seq_token)] = 1
                    seq_token = self.pad_data(seq_token, dim=0, max_length=max_length_batch)
                    
                    X = self.ss_processor(X, return_residue_seq=False)['2048']['ranked_unrelax_0.pdb']["struct"]
                    X = tokenize_structure_sequence(X)
                    X = self.pad_data(X, dim=0, max_length=max_length_batch)
                    
                    S_batch.append(seq_token)
                    X_batch.append(X)
                    mask_batch.append(attention_mask)
                    if task_name == 'contact_map':
                        label = F.pad(label, [add_BOS, self.max_length-label.shape[0]-add_BOS, add_BOS, self.max_length-label.shape[0]-add_BOS])
                    else:
                        label = torch.tensor(label)
                    label_batch.append(label)
                    name_batch.append(sample['name'])
        
                yield {
                    'name': name_batch,
                    'seq': torch.stack(S_batch).to(self.device),
                    'X': torch.stack(X_batch).to(self.device),
                    'attention_mask': torch.stack(mask_batch).to(self.device)==1,
                    'label': torch.stack(label_batch).to(self.device),
                }

            if self.pretrain_model_name == "prost":
                name_batch, X_batch, S_batch, mask_batch, label_batch = [], [], [], [], []
                
                if "pair" in self.task_type:
                    max_length_batch = max(
                        [max(len(seq) for seq in sample['seq']) for sample in data[i:i + batch_size]]
                    ) + 2
                else:
                    max_length_batch = max([len(sample['seq']) for sample in data[i:i + batch_size]]) + 2
                max_length_batch = self.max_length
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    label = sample['label']
                    if not isinstance(seq, list):
                        seq = [seq]
                    seq_tokens, attention_masks = [], []
                    for _seq in seq:
                        _seq = _seq[:1022]
                        attention_mask = torch.zeros(max_length_batch)
                        seq_token = torch.tensor(self.tokenizer.encode(_seq))
                        attention_mask[:len(seq_token)] = 1
                        seq_token = self.pad_data(seq_token, dim=0, max_length=max_length_batch)
                        seq_tokens.append(seq_token)
                        attention_masks.append(attention_mask)
                    seq_tokens = torch.hstack(seq_tokens)
                    attention_masks = torch.hstack(attention_masks)
                    S_batch.append(seq_tokens)
                    mask_batch.append(attention_masks)
                    if task_name == 'contact_map':
                        label = F.pad(label, [add_BOS, self.max_length-label.shape[0]-add_BOS, add_BOS, self.max_length-label.shape[0]-add_BOS])
                    else:
                        label = torch.tensor(label)
                    label_batch.append(label)
                    name_batch.append(sample['name'])
                yield {
                    'name': name_batch,
                    'seq': torch.stack(S_batch).to(self.device),
                    'attention_mask': torch.stack(mask_batch).to(self.device)==1,
                    'label': torch.stack(label_batch).to(self.device),
                }
            
            if self.pretrain_model_name == 'progen2': # 没有BOS, EOS, 需要限制长度在1024以内
                if "pair" in self.task_type:
                    max_length_batch = max(
                        [max(len(seq) for seq in sample['seq']) for sample in data[i:i + batch_size]]
                    ) 
                else:
                    max_length_batch = max([len(sample['seq']) for sample in data[i:i + batch_size]])
                name_batch, S_batch, mask_batch, label_batch = [], [], [], []

                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    label = sample['label']
                    if not isinstance(seq, list):
                        seq = [seq]
                    seq_tokens, attention_masks = [], []
                    for _seq in seq:
                        attention_mask = torch.zeros(self.max_length)
                        seq_token = torch.tensor(self.tokenizer.encode(_seq).ids)
                        attention_mask[:len(seq_token)] = 1
                        seq_token = self.pad_data(seq_token, dim=0, max_length=max_length_batch)
                        attention_mask = self.pad_data(attention_mask, dim=0, max_length=max_length_batch)
                        seq_tokens.append(seq_token)
                        attention_masks.append(attention_mask)
                    seq_tokens = torch.hstack(seq_tokens)
                    attention_masks = torch.hstack(attention_masks)
                    
                    S_batch.append(seq_tokens[:1024])
                    mask_batch.append(attention_masks[:1024])
                    if task_name == 'contact_map':
                        label = F.pad(label, [add_BOS, self.max_length-label.shape[0]-add_BOS, add_BOS, self.max_length-label.shape[0]-add_BOS])
                    else:
                        label = torch.tensor(label)
                    label_batch.append(label)
                    name_batch.append(sample['name'])
        
                yield {
                    'name': name_batch,
                    'seq': torch.stack(S_batch).to(self.device),
                    'attention_mask': torch.stack(mask_batch).to(self.device)==1,
                    'label': torch.stack(label_batch).to(self.device),
                }
            
            if self.pretrain_model_name == 'prostt5':
                import re
                if "pair" in self.task_type:
                    max_length_batch = max(
                        [max(len(seq) for seq in sample['seq']) for sample in data[i:i + batch_size]]
                    ) + 2
                else:
                    max_length_batch = max([len(sample['seq']) for sample in data[i:i + batch_size]]) + 2
                
                name_batch, struct_batch, S_batch, mask_batch, label_batch = [], [], [], [], []
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    X = sample['X']
                    label = sample['label']
                    if not isinstance(seq, list) and not isinstance(X, list):
                        seq, X = [seq], [X]
                    seq_tokens, attention_masks = [], []
                    for _seq, _X in zip(seq, X):
                        N, CA, C, CB, O = _X[:, 0], _X[:, 1], _X[:, 2], _X[:, 3], _X[:, 4]
                        attention_mask = torch.zeros(2, max_length_batch)
                        states = self.encoder_3di.encode_atoms(ca = CA.numpy(), cb = CB.numpy(), n = N.numpy(), c = C.numpy())
                        struct_sequence = self.encoder_3di.build_sequence(states).lower()
                        if self.sequence_only:
                            sequence_examples = [_seq, _seq]
                            sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
                            sequence_examples = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s # this expects 3Di sequences to be already lower-case
                                for s in sequence_examples
                            ]
                        else:
                            sequence_examples = [_seq, struct_sequence]
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
                    seq_tokens = torch.hstack(seq_tokens)
                    attention_masks = torch.hstack(attention_masks)
                    
                    S_batch.append(seq_tokens)
                    mask_batch.append(attention_masks)
                    if task_name == 'contact_map':
                        label = F.pad(label, [add_BOS, self.max_length-label.shape[0]-add_BOS, add_BOS, self.max_length-label.shape[0]-add_BOS])
                    else:
                        label = torch.tensor(label)
                    label_batch.append(label)
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
                    if "pair" in self.task_type:
                        submax = max([len(torch.tensor(self.tokenizer.encode(_seq))) for _seq in sample['seq']])
                        max_length_batch = max(max_length_batch, submax)
                    else:
                        seq_token = torch.tensor(self.tokenizer.encode(sample['seq']))
                        max_length_batch = max(max_length_batch, len(seq_token))
                    
                name_batch, S_batch, mask_batch, label_batch = [], [], [], []
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    label = sample['label']
                    if not isinstance(seq, list):
                        seq = [seq]
                    seq_tokens, attention_masks = [], []
                    for _seq in seq:
                        attention_mask = torch.zeros(self.max_length)
                        seq_token = torch.tensor(self.tokenizer.encode(_seq))
                        attention_mask[:len(seq_token)] = 1
                        seq_token = self.pad_data(seq_token, dim=0, max_length=max_length_batch)
                        attention_mask = self.pad_data(attention_mask, dim=0, max_length=max_length_batch)
                        seq_tokens.append(seq_token)
                        attention_masks.append(attention_mask)
                    seq_tokens = torch.hstack(seq_tokens)
                    attention_masks = torch.hstack(attention_masks)
                    
                    S_batch.append(seq_tokens)
                    mask_batch.append(attention_masks)
                    if task_name == 'contact_map':
                        label = F.pad(label, [add_BOS, self.max_length-label.shape[0]-add_BOS, add_BOS, self.max_length-label.shape[0]-add_BOS])
                    else:
                        label = torch.tensor(label)
                    label_batch.append(label)
                    name_batch.append(sample['name'])
        
                yield {
                    'name': name_batch,
                    'seq': torch.stack(S_batch).to(self.device),
                    'attention_mask': torch.stack(mask_batch).to(self.device)==1,
                    'label': torch.stack(label_batch).to(self.device),
                }

            if self.pretrain_model_name == 'protrek':
                if "pair" in self.task_type:
                    max_length_batch = max(
                        [max(len(seq) for seq in sample['seq']) for sample in data[i:i + batch_size]]
                    ) + 2
                else:
                    max_length_batch = max([len(sample['seq']) for sample in data[i:i + batch_size]]) + 2
                name_batch, struct_batch, seq_batch, label_batch, mask_batch = [], [], [], [], []
                # attention_mask = torch.zeros(len(data[i:i + batch_size]), max_length_batch)
                for idx, sample in enumerate(data[i:i + batch_size]):
                    seq = sample['seq']
                    X = sample['X']
                    label = sample['label']
                    if not isinstance(seq, list) and not isinstance(X, list):
                        seq, X = [seq], [X]
                    seq_tokens, attention_masks, struct_tokens = [], [], []
                    for _seq, _X in zip(seq, X):
                        N, CA, C, CB, O = _X[:, 0], _X[:, 1], _X[:, 2], _X[:, 3], _X[:, 4]
                        states = self.encoder_3di.encode_atoms(ca = CA.numpy(), cb = CB.numpy(), n = N.numpy(), c = C.numpy())
                        struct_sequence = self.encoder_3di.build_sequence(states).lower()
                        if self.sequence_only:
                            struct_sequence = ''.join(['#' for one in struct_sequence])
                        attention_mask = torch.zeros(max_length_batch)
                        attention_mask[:len(_seq)+2] = 1
                        attention_masks.append(attention_mask)
                        seq_tokens.append(_seq)
                        struct_tokens.append(struct_sequence)
                    attention_masks = torch.hstack(attention_masks)
                    struct_batch.append(struct_tokens)
                    seq_batch.append(seq_tokens)
                    if task_name == 'contact_map':
                        label = F.pad(label, [add_BOS, self.max_length-label.shape[0]-add_BOS, add_BOS, self.max_length-label.shape[0]-add_BOS])
                    else:
                        label = torch.tensor(label)
                    label_batch.append(label)
                    name_batch.append(sample['name'])
                    mask_batch.append(attention_masks)

                yield {
                    'name': name_batch,
                    'seq_batch': seq_batch,
                    'struct_batch': struct_batch,
                    'attention_mask': torch.stack(mask_batch).to(self.device)==1,
                    'label': torch.stack(label_batch).to(self.device)
                }
            
            if self.pretrain_model_name == 'saport':
                if "pair" in self.task_type:
                    max_length_batch = max(
                        [max(len(seq) for seq in sample['seq']) for sample in data[i:i + batch_size]]
                    ) + 2
                else:
                    max_length_batch = max([len(sample['seq']) for sample in data[i:i + batch_size]]) + 2
                name_batch, struct_batch, S_batch, mask_batch, label_batch = [], [], [], [], []
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    X = sample['X']
                    label = sample['label']
                    if not isinstance(seq, list) and not isinstance(X, list):
                        seq, X = [seq], [X]
                    seq_tokens, attention_masks, struct_tokens = [], [], []
                    for _seq, _X in zip(seq, X):
                        N,  CA, C,  CB,  O = _X[:, 0], _X[:, 1], _X[:, 2], _X[:, 3], _X[:, 4]
                        states = self.encoder_3di.encode_atoms(ca = CA.numpy(), cb = CB.numpy(), n = N.numpy(), c = C.numpy())
                        struct_sequence = self.encoder_3di.build_sequence(states).lower()
                        attention_mask = torch.zeros(self.max_length)
                    
                        if self.sequence_only:
                            struct_sequence = ''.join(['#' for one in struct_sequence])
                        merged_seq = ''.join(a + b.lower() for a, b in zip(_seq, struct_sequence))
                        seq_token = self.tokenizer(merged_seq, return_tensors="pt").input_ids[0]
                    
                        attention_mask[:seq_token.shape[0]] = 1
                        seq_token = self.pad_data(seq_token, dim=0, max_length=max_length_batch)
                        attention_mask = self.pad_data(attention_mask, dim=0, max_length=max_length_batch)
                        seq_tokens.append(seq_token)
                        attention_masks.append(attention_mask)
                    seq_tokens = torch.hstack(seq_tokens)
                    attention_masks = torch.hstack(attention_masks)
                    S_batch.append(seq_tokens)
                    mask_batch.append(attention_masks)
                    if task_name == 'contact_map':
                        label = F.pad(label, [add_BOS, self.max_length-label.shape[0]-add_BOS, add_BOS, self.max_length-label.shape[0]-add_BOS])
                    else:
                        label = torch.tensor(label)
                    label_batch.append(label)
                    name_batch.append(sample['name'])
        
                yield {
                    'name': name_batch,
                    'seq': torch.stack(S_batch, dim=0).to(self.device),
                    'attention_mask': torch.stack(mask_batch, dim=0).to(self.device)==1,
                    'label': torch.stack(label_batch).to(self.device),
                }
        
    
    def forward(self, x):
        names, labels = x['name'], x['label']
        if self.pretrain_model_name == 'esm2_650m':
            # Forward pass through the pre-trained model
            seq, attention_mask, t_lengths, smiles = x['seq'], x['attention_mask'], x['t_lengths'], x['smiles']
            embeddings, attention_masks = [], []
            for i, (_seq, _attention_mask, _t_length) in enumerate(zip(seq, attention_mask, t_lengths)):
                outputs = self.pretrain_model.esm(
                            _seq,
                            attention_mask=_attention_mask,
                            return_dict=True,
                        )
                _embeddings = outputs.last_hidden_state
                _t_length = torch.where(_t_length==_t_length.max(), _t_length-1, _t_length)
                if i != len(seq) - 1:
                    _attention_mask.scatter_(1, _t_length.unsqueeze(1), True)
                embeddings.append(_embeddings)
                attention_masks.append(_attention_mask)
            embeddings = torch.cat(embeddings, dim=1)
            attention_mask = torch.cat(attention_masks, dim=-1)

            if "pair" in self.task_type:
                ends = torch.tensor([attention_mask.shape[1]]*attention_mask.shape[0])-1
                starts = torch.ones_like(ends)
            else:
                ends = attention_mask.sum(dim=-1)-1
                starts = torch.ones_like(ends)
            
        if self.pretrain_model_name == 'esm3_1.4b': # WORKING
            protein_tensor, attention_mask = x['protein_tensor'], x['attention_mask']
            embeddings = []
            if "pair" in self.task_type:
                split_idx = protein_tensor.sequence.shape[1] // 2
                protein_tensor = [
                    _BatchedESMProteinTensor(
                        sequence=protein_tensor.sequence[:, :split_idx],
                        structure=protein_tensor.structure[:, :split_idx],
                        coordinates=protein_tensor.coordinates[:, :split_idx],
                    ),
                    _BatchedESMProteinTensor(
                        sequence=protein_tensor.sequence[:, split_idx:],
                        structure=protein_tensor.structure[:, split_idx:],
                        coordinates=protein_tensor.coordinates[:, split_idx:],
                    )
                ]
                attention_mask = [attention_mask[:, :split_idx], attention_mask[:, split_idx:]]
            else:
                protein_tensor, attention_mask = [protein_tensor], [attention_mask]

            for _protein_tensor in protein_tensor:
                output = self.pretrain_model.logits(
                    _protein_tensor,
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
                embeddings.append(output.embeddings)
            embeddings = torch.cat(embeddings, dim=-1)

            if "pair" in self.task_type:
                attention_mask = (attention_mask[0].int() + attention_mask[1].int() > 0)
                ends = attention_mask.sum(dim=-1)-1
                starts = torch.ones_like(ends)
                attention_mask = attention_mask
            else:
                ends = attention_mask[0].sum(dim=-1)-1
                starts = torch.ones_like(ends)
                attention_mask = attention_mask[0]
            # embeddings = F.pad(embeddings, (0, 0, 0, self.max_length-embeddings.shape[1], 0, 0), value=0)
        
        if self.pretrain_model_name == 'esmc_600m':
            protein_tensor, attention_mask = x['protein_tensor'], x['attention_mask']
            embeddings = []
            if "pair" in self.task_type:
                split_idx = protein_tensor.sequence.shape[1] // 2
                protein_tensor = [
                    _BatchedESMProteinTensor(
                        sequence=protein_tensor.sequence[:, :split_idx]
                    ),
                    _BatchedESMProteinTensor(
                        sequence=protein_tensor.sequence[:, split_idx:]
                    )
                ]
                attention_mask = [attention_mask[:, :split_idx], attention_mask[:, split_idx:]]
            else:
                protein_tensor, attention_mask = [protein_tensor], [attention_mask]
            
            for _protein_tensor in protein_tensor:
                logits_output = self.pretrain_model.logits(
                    _protein_tensor, 
                    LogitsConfig(sequence=True, return_embeddings=True)
                )
                embeddings.append(logits_output.embeddings)
            embeddings = torch.cat(embeddings, dim=-1)

            if "pair" in self.task_type:
                attention_mask = (attention_mask[0].int() + attention_mask[1].int() > 0)
                ends = attention_mask.sum(dim=-1)-1
                starts = torch.ones_like(ends)
                attention_mask = attention_mask
            else:
                ends = attention_mask[0].sum(dim=-1)-1
                starts = torch.ones_like(ends)
                attention_mask = attention_mask[0]
        
        if self.pretrain_model_name == 'procyon': # 长度和氨基酸数量不同
            seq_embedding, struct_embedding = x['seq'], x['X']
            embeddings, attention_masks = [], []
            if "pair" in self.task_type:
                seq_split_idx = seq_embedding.shape[-1] // 2
                struct_split_idx = struct_embedding.shape[-1] // 2
                seq_embedding = [
                    seq_embedding[:, :seq_split_idx], seq_embedding[:, seq_split_idx:]
                ]
                struct_embedding = [
                    struct_embedding[:, :, :struct_split_idx], struct_embedding[:, :, struct_split_idx:]
                ]
            else:
                seq_embedding, struct_embedding = [seq_embedding], [struct_embedding]

            for _seq_embedding, _strcture_embedding in zip(seq_embedding, struct_embedding):
                seq_embeddings = self.pretrain_model.token_projectors["aaseq"](
                    _seq_embedding
                )
                struct_embeddings = self.pretrain_model.token_projectors["prot_structure"](
                    _strcture_embedding
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
                attention_mask = ~(input_ids == self.pretrain_model.tokenizer.pad_token_id)
                outputs = self.pretrain_model.text_encoder(
                    input_embeds = input_embeds,
                    attn_masks = attn_masks,
                )
                embeddings.append(outputs.hidden_states[-1]) # shape(b, 2048, 4096)
                attention_masks.append(attention_mask)
            embeddings = torch.cat(embeddings, dim=-1)
            if "pair" in self.task_type:
                attention_mask = (attention_masks[0].int() + attention_masks[1].int() > 0)
                ends = attention_mask.sum(dim=-1)-1
                starts = torch.ones_like(ends)
            else:
                ends = attention_masks[0].sum(dim=-1)-1
                starts = torch.ones_like(ends)
                attention_mask = attention_masks[0]
        
        if self.pretrain_model_name == 'gearnet': 
            embeddings = x['X']
            attention_mask = x['attention_mask']
            ends = attention_mask.sum(dim=-1)
            starts = torch.zeros_like(ends)

        if self.pretrain_model_name == 'prollama': # 长度和氨基酸数量不同
            seq, attention_mask = x['seq'], x['attention_mask']
            if "pair" in self.task_type:
                split_idx = seq.shape[1] // 2
                seq = [
                    seq[:, :split_idx], seq[:, split_idx:]
                ]
                attention_mask = [attention_mask[:, :split_idx], attention_mask[:, split_idx:]]
            else:
                seq, attention_mask = [seq], [attention_mask]
            embeddings = []
            for _seq, _attention_mask in zip(seq, attention_mask):
                out = self.pretrain_model(
                    input_ids = _seq,
                    attention_mask = _attention_mask,
                    output_hidden_states=True
                )
                embeddings.append(out.hidden_states[-1].float())
            embeddings = torch.cat(embeddings, dim=-1)
            if "pair" in self.task_type:
                attention_mask = (attention_mask[0].int() + attention_mask[1].int() > 0)
                ends = attention_mask.sum(dim=-1)-1
                starts = torch.ones_like(ends)
            else:
                ends = attention_mask[0].sum(dim=-1)-1
                starts = torch.ones_like(ends)
                attention_mask = attention_mask[0]

        if self.pretrain_model_name == 'venusplm':
            seq, attention_mask = x['seq'], x['attention_mask']
            if "pair" in self.task_type:
                split_idx = seq.shape[1] // 2
                seq = [
                    seq[:, :split_idx], seq[:, split_idx:]
                ]
                attention_mask = [attention_mask[:, :split_idx], attention_mask[:, split_idx:]]
            else:
                seq, attention_mask = [seq], [attention_mask]
            embeddings = []
            for _seq, _attention_mask in zip(seq, attention_mask):
                outputs = self.pretrain_model(
                            input_ids=_seq,
                            attention_mask=_attention_mask,
                            output_hidden_states=True
                        )
                embeddings.append(outputs.hidden_states[-1])
            embeddings = torch.cat(embeddings, dim=-1)
            if "pair" in self.task_type:
                attention_mask = (attention_mask[0].int() + attention_mask[1].int() > 0)
                ends = attention_mask.sum(dim=-1)-1
                starts = torch.ones_like(ends)
            else:
                ends = attention_mask[0].sum(dim=-1)-1
                starts = torch.ones_like(ends)
                attention_mask = attention_mask[0]

        if self.pretrain_model_name == 'prost': 
            seq, attention_mask = x['seq'], x['attention_mask']
            if "pair" in self.task_type:
                split_idx = seq.shape[1] // 2
                seq = [
                    seq[:, :split_idx], seq[:, split_idx:]
                ]
                attention_mask = [attention_mask[:, :split_idx], attention_mask[:, split_idx:]]
            else:
                seq, attention_mask = [seq], [attention_mask]

            embeddings = []
            for _seq, _attention_mask in zip(seq, attention_mask):
                outputs = self.pretrain_model(
                            input_ids=_seq,
                            attention_mask=_attention_mask,
                            return_dict=True
                        )
                embeddings.append(outputs.residue_feature)
            embeddings = torch.cat(embeddings, dim=-1)
            if "pair" in self.task_type:
                attention_mask = (attention_mask[0].int() + attention_mask[1].int() > 0)
                ends = attention_mask.sum(dim=-1)-1
                starts = torch.ones_like(ends)
            else:
                ends = attention_mask[0].sum(dim=-1)-1
                starts = torch.ones_like(ends)
                attention_mask = attention_mask[0]
        
        if self.pretrain_model_name == 'progen2':
            seq, attention_mask = x['seq'], x['attention_mask']
            if "pair" in self.task_type:
                split_idx = seq.shape[1] // 2
                seq = [
                    seq[:, :split_idx], seq[:, split_idx:]
                ]
                attention_mask = [attention_mask[:, :split_idx], attention_mask[:, split_idx:]]
            else:
                seq, attention_mask = [seq], [attention_mask]
            
            embeddings = []
            for _seq, _attention_mask in zip(seq, attention_mask):
                outputs = self.pretrain_model.transformer(
                            _seq,
                            return_dict=True
                        )
                embeddings.append(outputs[0])
            embeddings = torch.cat(embeddings, dim=-1)
            if "pair" in self.task_type:
                attention_mask = (attention_mask[0].int() + attention_mask[1].int() > 0)
                ends = attention_mask.sum(dim=-1)-1
                starts = torch.ones_like(ends)
            else:
                ends = attention_mask[0].sum(dim=-1)-1
                starts = torch.ones_like(ends)
                attention_mask = attention_mask[0]

        if self.pretrain_model_name == 'prostt5':
            seq, attention_mask = x['seq'], x['attention_mask']
            if "pair" in self.task_type:
                split_idx = seq.shape[1] // 2
                seq = [
                    seq[:, :split_idx], seq[:, split_idx:]
                ]
                attention_mask = [attention_mask[:, :split_idx], attention_mask[:, split_idx:]]
            else:
                seq, attention_mask = [seq], [attention_mask]
            embeddings = []
            for _seq, _attention_mask in zip(seq, attention_mask):
                embedding_repr = self.pretrain_model(
                        _seq,
                        attention_mask=_attention_mask
                )
                _embeddings = embedding_repr.last_hidden_state
                _embeddings = _embeddings.reshape(_embeddings.shape[0]//2, 2, _embeddings.shape[1], _embeddings.shape[2])
                _embeddings = torch.cat([_embeddings[:,0], _embeddings[:,1]], dim=-1)
                embeddings.append(_embeddings)
            embeddings = torch.cat(embeddings, dim=-1)

            if "pair" in self.task_type:
                attention_mask = (attention_mask[0].int() + attention_mask[1].int() > 0)
                attention_mask = attention_mask[::2]
                ends = attention_mask.sum(dim=-1)-1
                starts = torch.ones_like(ends)
            else:
                attention_mask = attention_mask[0][::2]
                ends = attention_mask.sum(dim=-1)-1
                starts = torch.ones_like(ends)
        
        if self.pretrain_model_name == 'protgpt2':
            seq, attention_mask = x['seq'], x['attention_mask']
            if "pair" in self.task_type:
                split_idx = seq.shape[1] // 2
                seq = [
                    seq[:, :split_idx], seq[:, split_idx:]
                ]
                attention_mask = [attention_mask[:, :split_idx], attention_mask[:, split_idx:]]
            else:
                seq, attention_mask = [seq], [attention_mask]
            
            embeddings = []
            for _seq, _attention_mask in zip(seq, attention_mask):
                outputs = self.pretrain_model.transformer(_seq)
                embeddings.append(outputs.last_hidden_state)
            embeddings = torch.cat(embeddings, dim=-1)
            if "pair" in self.task_type:
                attention_mask = (attention_mask[0].int() + attention_mask[1].int() > 0)
                ends = attention_mask.sum(dim=-1)-1
                starts = torch.ones_like(ends)
            else:
                ends = attention_mask[0].sum(dim=-1)-1
                starts = torch.ones_like(ends)
                attention_mask = attention_mask[0]
    
        if self.pretrain_model_name == 'protrek':
            seq, attention_mask, struct = x['seq_batch'], x['attention_mask'], x['struct_batch']
            if "pair" in self.task_type:
                split_idx = attention_mask.shape[1] // 2
                seq = [
                    [ele[0] for ele in seq], [ele[1] for ele in seq]
                ]
                struct = [
                    [ele[0] for ele in struct], [ele[1] for ele in struct]
                ]
                attention_mask = [attention_mask[:, :split_idx], attention_mask[:, split_idx:]]
            else:
                seq, attention_mask, struct = [[ele[0] for ele in seq]], [attention_mask], [[ele[0] for ele in struct]]

            embeddings = []
            for _seq, _attention_mask, _struct in zip(seq, attention_mask, struct):
                seq_embedding = self.pretrain_model.get_protein_repr(_seq)
                struc_embedding = self.pretrain_model.get_structure_repr(_struct)
                _embeddings = torch.cat([seq_embedding, struc_embedding], dim=-1)
                if "pair" in self.task_type:
                    _embeddings = self.pad_data(_embeddings, dim=1, max_length=split_idx)
                embeddings.append(_embeddings)
            embeddings = torch.cat(embeddings, dim=-1)

            if "pair" in self.task_type:
                attention_mask = (attention_mask[0].int() + attention_mask[1].int() > 0)
                ends = attention_mask.sum(dim=-1)-1
                starts = torch.ones_like(ends)
            else:
                ends = attention_mask[0].sum(dim=-1)-1
                starts = torch.ones_like(ends)
                attention_mask = attention_mask[0]

        if self.pretrain_model_name == 'prosst2048':
            # Forward pass through the pre-trained model
            outputs = self.pretrain_model(
                        input_ids=x['seq'],
                        attention_mask=x['attention_mask'],
                        output_hidden_states=True,
                        ss_input_ids=x['X']
                    )
            embeddings = outputs.hidden_states[-1]
            ends = x['attention_mask'].sum(dim=-1)-1
            starts = torch.ones_like(ends)
            attention_mask = x['attention_mask']

        if self.pretrain_model_name == 'saport':
            seq, attention_mask = x['seq'], x['attention_mask']
            if "pair" in self.task_type:
                split_idx = seq.shape[1] // 2
                seq = [
                    seq[:, :split_idx], seq[:, split_idx:]
                ]
                attention_mask = [attention_mask[:, :split_idx], attention_mask[:, split_idx:]]
            else:
                seq, attention_mask = [seq], [attention_mask]
            embeddings = []
            for _seq, _attention_mask in zip(seq, attention_mask):
                output = self.pretrain_model.esm(
                        _seq,
                        attention_mask=_attention_mask,
                        return_dict=True,
                )
                embeddings.append(output.last_hidden_state)
            embeddings = torch.cat(embeddings, dim=-1)

            if "pair" in self.task_type:
                attention_mask = (attention_mask[0].int() + attention_mask[1].int() > 0)
                attention_mask = attention_mask[::2]
                ends = attention_mask.sum(dim=-1)-1
                starts = torch.ones_like(ends)
            else:
                attention_mask = attention_mask[0][::2]
                ends = attention_mask.sum(dim=-1)-1
                starts = torch.ones_like(ends)

        # 这个embedding 是[B,L,D]的形式, L是氨基酸数量, 但对于语言模型老师, L这个维度可能不刚好等于氨基酸数量，无法用于氨基酸级别的预测任务。另外，有些方法会加上[EOS, BOS]，有些又不需要，最好是在最后的embedding上统一去除[EOS, BOS]
        return self.post_process(names, labels, embeddings, attention_mask, starts, ends, smiles)
    
    def post_process(self, names, labels, embeddings, attention_mask, starts, ends, smiles)->list:
        results = []
        for i, end in enumerate(ends):
            start = starts[i]
            label = labels[i].cpu()
            if self.task_type == 'contact':
                label = labels[i,start:end, start:end].cpu()
            
            results.append( 
                {
                    'name': names[i],
                    'embedding': embeddings[i,start:end].cpu(),
                    'attention_mask': attention_mask[i,start:end].cpu(),
                    'label': label,
                    'smiles': smiles[i] if smiles is not None else None
                }
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
            for i, batch in enumerate(tqdm(self.construct_batch(data, self.batch_size, task_name), desc='Extracting embeddings')):
                # print('batch size:', len(batch['name']))
                try:
                    results = self.forward(batch)
                    proccessed_data.extend(results)
                except:
                    print(f"Error processing batch {i}")
        return proccessed_data
            


