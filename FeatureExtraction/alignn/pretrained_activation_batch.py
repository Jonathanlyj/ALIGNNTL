import requests
import os
import zipfile
from tqdm import tqdm
# from alignn.models.alignn import ALIGNN, ALIGNNConfig
from .models.alignn import ALIGNN, ALIGNNConfig
import tempfile
import torch
import re
import numpy as np
import pandas as pd
import argparse
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
import gc
import h5py
import dgl
import time
import io

io_time = 0
infer_time = 0
preprocess_time = 0
model_load_time = 0
end_to_end_time = 0
transfer_time = 0


all_models = {
    "jv_formation_energy_peratom_alignn": [
        "https://figshare.com/ndownloader/files/31458679",
        1,
    ],
    "jv_optb88vdw_total_energy_alignn": [
        "https://figshare.com/ndownloader/files/31459642",
        1,
    ],
    "jv_optb88vdw_bandgap_alignn": [
        "https://figshare.com/ndownloader/files/31459636",
        1,
    ],
    "jv_mbj_bandgap_alignn": [
        "https://figshare.com/ndownloader/files/31458694",
        1,
    ],
    "jv_spillage_alignn": [
        "https://figshare.com/ndownloader/files/31458736",
        1,
    ],
    "jv_slme_alignn": ["https://figshare.com/ndownloader/files/31458727", 1],
    "jv_bulk_modulus_kv_alignn": [
        "https://figshare.com/ndownloader/files/31458649",
        1,
    ],
    "jv_shear_modulus_gv_alignn": [
        "https://figshare.com/ndownloader/files/31458724",
        1,
    ],
    "jv_n-Seebeck_alignn": [
        "https://figshare.com/ndownloader/files/31458718",
        1,
    ],
    "jv_n-powerfact_alignn": [
        "https://figshare.com/ndownloader/files/31458712",
        1,
    ],
    "jv_magmom_oszicar_alignn": [
        "https://figshare.com/ndownloader/files/31458685",
        1,
    ],
    "jv_kpoint_length_unit_alignn": [
        "https://figshare.com/ndownloader/files/31458682",
        1,
    ],
    "jv_avg_elec_mass_alignn": [
        "https://figshare.com/ndownloader/files/31458643",
        1,
    ],
    "jv_avg_hole_mass_alignn": [
        "https://figshare.com/ndownloader/files/31458646",
        1,
    ],
    "jv_epsx_alignn": ["https://figshare.com/ndownloader/files/31458667", 1],
    "jv_mepsx_alignn": ["https://figshare.com/ndownloader/files/31458703", 1],
    "jv_max_efg_alignn": [
        "https://figshare.com/ndownloader/files/31458691",
        1,
    ],
    "jv_ehull_alignn": ["https://figshare.com/ndownloader/files/31458658", 1],
    "jv_dfpt_piezo_max_dielectric_alignn": [
        "https://figshare.com/ndownloader/files/31458652",
        1,
    ],
    "jv_dfpt_piezo_max_dij_alignn": [
        "https://figshare.com/ndownloader/files/31458655",
        1,
    ],
    "jv_exfoliation_energy_alignn": [
        "https://figshare.com/ndownloader/files/31458676",
        1,
    ],
    "mp_e_form_alignnn": [
        "https://figshare.com/ndownloader/files/31458811",
        1,
    ],
    "mp_gappbe_alignnn": [
        "https://figshare.com/ndownloader/files/31458814",
        1,
    ],
    "qm9_U0_alignn": ["https://figshare.com/ndownloader/files/31459054", 1],
    "qm9_U_alignn": ["https://figshare.com/ndownloader/files/31459051", 1],
    "qm9_alpha_alignn": ["https://figshare.com/ndownloader/files/31459027", 1],
    "qm9_gap_alignn": ["https://figshare.com/ndownloader/files/31459036", 1],
    "qm9_G_alignn": ["https://figshare.com/ndownloader/files/31459033", 1],
    "qm9_HOMO_alignn": ["https://figshare.com/ndownloader/files/31459042", 1],
    "qm9_LUMO_alignn": ["https://figshare.com/ndownloader/files/31459045", 1],
    "qm9_ZPVE_alignn": ["https://figshare.com/ndownloader/files/31459057", 1],
    "hmof_co2_absp_alignnn": [
        "https://figshare.com/ndownloader/files/31459198",
        5,
    ],
    "hmof_max_co2_adsp_alignnn": [
        "https://figshare.com/ndownloader/files/31459207",
        1,
    ],
    "hmof_surface_area_m2g_alignnn": [
        "https://figshare.com/ndownloader/files/31459222",
        1,
    ],
    "hmof_surface_area_m2cm3_alignnn": [
        "https://figshare.com/ndownloader/files/31459219",
        1,
    ],
    "hmof_pld_alignnn": ["https://figshare.com/ndownloader/files/31459216", 1],
    "hmof_lcd_alignnn": ["https://figshare.com/ndownloader/files/31459201", 1],
    "hmof_void_fraction_alignnn": [
        "https://figshare.com/ndownloader/files/31459228",
        1,
    ],
    "mof_dband": ['',
        1
    ]
}

def load_alignn_model(model_name, device):
    global model_load_time
    model_load_start = time.time()
    url, output_features = all_models[model_name]
    zfile = f"{model_name}.zip"
    path = os.path.join(os.path.dirname(__file__), zfile)

    if not os.path.isfile(path):
        response = requests.get(url, stream=True)
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    zp = zipfile.ZipFile(path)
    checkpoint_file = [i for i in zp.namelist() if "checkpoint_" in i and i.endswith("pt")][0]
    data = zp.read(checkpoint_file)
    buffer = io.BytesIO(data)
    

    model = ALIGNN(ALIGNNConfig(name="alignn", output_features=output_features))
    # num_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {num_params}")
    # model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    # print(f"Model size: {model_size_mb:.2f} MB")
    # _, filename = tempfile.mkstemp()
    # with open(filename, "wb") as f:
    #     f.write(data)
    # Get file size in MB
    # file_size_mb = os.path.getsize(filename) / (1024 ** 2)
    # print(f"Temp checkpoint file size: {file_size_mb:.2f} MB")

    # model.load_state_dict(torch.load(filename, map_location=device)["model"])
    # os.remove(filename)
    model.load_state_dict(torch.load(buffer, map_location=device)["model"])
    model_load_time = time.time() - model_load_start
    model.to(device)
    model.eval()
    
    
    return model


def atoms_from_file(input_file_path, file_format):
    if file_format == "poscar":
        return Atoms.from_poscar(input_file_path)
    elif file_format == "cif":
        return Atoms.from_cif(input_file_path)
    elif file_format == "xyz":
        return Atoms.from_xyz(input_file_path, box_size=500)
    elif file_format == "pdb":
        return Atoms.from_pdb(input_file_path, max_lat=500)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def get_batch_embeddings(model, atoms_list, cutoff):
    global infer_time
    global transfer_time   
    global preprocess_time  
    g_list, lg_list = [], []
    # print(len(atoms_list))
    preprocess_start = time.time()
    for atoms in atoms_list:
        g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=float(cutoff)) #source code: https://jarvis-tools.readthedocs.io/en/master/_modules/jarvis/core/graphs.html#Graph.atom_dgl_multigraph
        g_list.append(g)
        lg_list.append(lg)
    preprocess_end = time.time()
    preprocess_time += preprocess_end - preprocess_start
    transfer_start = time.time()
    bg = dgl.batch(g_list).to(device)
    blg = dgl.batch(lg_list).to(device)
    transfer_time += time.time() - transfer_start
    torch.cuda.synchronize()
    infer_start = time.time()
    with torch.no_grad():
        output, act_list_x, act_list_y, act_list_z = model([bg, blg], return_per_graph=True)
    
    torch.cuda.synchronize()
    infer_time += time.time() - infer_start

    embeddings = []
    for i in range(len(atoms_list)):
        emb_x = np.concatenate([a[i].detach().cpu().numpy().flatten() for a in act_list_x], axis=0)
        emb_y = np.concatenate([a[i].detach().cpu().numpy().flatten() for a in act_list_y], axis=0)
        emb_z = np.concatenate([a[i].detach().cpu().numpy().flatten() for a in act_list_z], axis=0)
        embeddings.append(np.concatenate([emb_x, emb_y, emb_z]))
    # Clean up GPU memory
    del output
    del bg, blg, g_list, lg_list
    del act_list_x, act_list_y, act_list_z
    torch.cuda.empty_cache()
    gc.collect()
    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) and obj.is_cuda:
    #             print("Leaked tensor", type(obj), obj.size())
    #     except:
    #         pass
    return embeddings


def main():
    global end_to_end_time
    global model_load_time
    global preprocess_time
    global device
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="jv_formation_energy_peratom_alignn")
    parser.add_argument("--file_format", default="poscar")
    parser.add_argument("--file_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--cutoff", type=float, default=8)
    parser.add_argument("--source_file")
    parser.add_argument("--h5_filename", default="embeddings.h5")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--subset", type=int, default=100)
    args = parser.parse_args()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    end_to_end_start = time.time()
    model = load_alignn_model(args.model_name, device)

    input_files = os.listdir(args.file_path)
    if args.source_file:
        src_df = pd.read_csv(args.source_file)
        src_filelist = set(src_df['sample'].values)
        input_files = [f for f in input_files if f in src_filelist]

    all_embeddings = []
    sample_ids = []
    batch_atoms, batch_ids = [], []
    # 0.6/14 s here
    
    for input_file in tqdm(input_files[:args.subset], desc="Batch embedding extraction"):
 
        id_str = os.path.splitext(input_file)[0].lstrip("POSCAR-")
        input_file_path = os.path.join(args.file_path, input_file)


        preprocess_start = time.time()
        atoms = atoms_from_file(input_file_path, args.file_format)
        
        batch_atoms.append(atoms)
        batch_ids.append(id_str)
        preprocess_time += time.time() - preprocess_start
        
        if len(batch_atoms) == args.batch_size:

            batch_embs = get_batch_embeddings(model, batch_atoms, args.cutoff)

            all_embeddings.extend(batch_embs)
            sample_ids.extend(batch_ids)
            batch_atoms, batch_ids = [], []
        
    if batch_atoms:
        batch_embs = get_batch_embeddings(model, batch_atoms, args.cutoff)
        all_embeddings.extend(batch_embs)
        sample_ids.extend(batch_ids)
    # end_to_end_end = time.time()
    all_embeddings = np.stack(all_embeddings, axis=0)
    sample_ids = np.array(sample_ids, dtype='S')
    
    os.makedirs(args.output_path, exist_ok=True)
    h5_path = os.path.join(args.output_path, args.h5_filename)
    io_start = time.time()
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("embeddings", data=all_embeddings)
        f.create_dataset("sample_ids", data=sample_ids)
    io_time = time.time() - io_start
    end_to_end_end = time.time() 
    end_to_end_time = end_to_end_end - end_to_end_start
    other_time = end_to_end_time - (model_load_time + io_time + preprocess_time + infer_time + transfer_time)
    print(f"End_to_end time: {end_to_end_time:.4f} seconds")
    print(f"Model load time: {model_load_time:.4f} seconds")
    print(f"IO time: {io_time:.4f} seconds")
    print(f"Preprocess time: {preprocess_time:.4f} seconds")
    print(f"Inference time: {infer_time:.4f} seconds")
    print(f"Transfer time: {transfer_time:.4f} seconds")
    print(f"Other time: {other_time:.4f} seconds")

    print(end_to_end_time)
    print(model_load_time)
    print(io_time)
    print(preprocess_time)
    print(infer_time)
    print(transfer_time)
    print(other_time)
    print(f"Saved {len(sample_ids)} batch embeddings to {h5_path}")


if __name__ == "__main__":
    main()
