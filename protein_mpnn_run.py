#!/usr/bin/env python3

import argparse
import os
import json
import numpy as np
import torch
import random
import copy
import time
import subprocess
import glob

from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB, parse_fasta
from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN

def main(args):
    print("Starting MPNN")
    if args.seed:
        seed = args.seed
    else:
        seed = int(np.random.randint(0, high=999, size=1, dtype=int)[0])

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    hidden_dim = 128
    num_layers = 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_folder_path = args.path_to_model_weights.rstrip('/') + '/'
    checkpoint_path = model_folder_path + f'{args.model_name}.pt'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ProteinMPNN(
        ca_only=args.ca_only,
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=args.backbone_noise,
        k_neighbors=checkpoint['num_edges']
    )
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model Loaded")
    model.eval()

    # Hook for embeddings
    embedding_storage = {}
    def get_embeddings_hook(module, input, output):
        #print(output)
        embedding_storage['embeddings'] = output[0].detach().cpu()

    model.encoder_layers[0].register_forward_hook(get_embeddings_hook)
    
    print("Embeddings Hook")

    # Load data
    #pdb_files = glob.glob(os.path.join(args.pdb_path, "*.pdb"))
    pdb_files = [args.pdb_path]
    dataset_valid = []

    for pdb_file in pdb_files:
        pdb_dict_list = parse_PDB(pdb_file, ca_only=args.ca_only)
        dataset_valid.extend(pdb_dict_list)
    print("PDB's Loaded")

    folder_for_outputs = args.out_folder.rstrip('/') + '/'
    os.makedirs(folder_for_outputs, exist_ok=True)
    os.makedirs(folder_for_outputs + 'embeddings', exist_ok=True)

    NUM_BATCHES = args.num_seq_per_target // args.batch_size

    with torch.no_grad():
        for ix, protein in enumerate(dataset_valid):
            batch_clones = [copy.deepcopy(protein) for _ in range(args.batch_size)]
            print("Clones Created")

            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, None)
        
            print("Featurize Completed")

            name_ = batch_clones[0]['name']
            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)

            # Save embedding
            if 'embeddings' in embedding_storage:
                print("Saving embedding!")
                embedding_array = embedding_storage['embeddings'].numpy()
                save_path = os.path.join(folder_for_outputs, 'embeddings', f"{name_}_embedding.npy")
                np.save(save_path, embedding_array)


argparser = argparse.ArgumentParser()
argparser.add_argument("--pdb_path", type=str, default="", help="Path to a folder of PDB files")
argparser.add_argument("--jsonl_path", type=str, default="", help="Path to JSONL if not using PDB")
argparser.add_argument("--out_folder", type=str, required=True, help="Path to save outputs")
argparser.add_argument("--path_to_model_weights", type=str, required=True, help="Path to model weights folder")
argparser.add_argument("--model_name", type=str, default="v_48_020", help="Model filename without .pt")
argparser.add_argument("--num_seq_per_target", type=int, default=1)
argparser.add_argument("--batch_size", type=int, default=1)
argparser.add_argument("--ca_only", action='store_true')
argparser.add_argument("--backbone_noise", type=float, default=0.0)
argparser.add_argument("--seed", type=int, default=None)

args = argparser.parse_args()
main(args)