#!/usr/bin/env python

"""Module to download and load pre-trained ALIGNN models."""
import requests
import os
import zipfile
from tqdm import tqdm
from alignn.models.alignn import ALIGNN, ALIGNNConfig
import tempfile
import torch
import sys
import re
import numpy as np
import pandas as pd
from jarvis.db.jsonutils import loadjson
import argparse
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
import gc

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

# Name of the model, figshare link, number of outputs
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

    model = ALIGNN(ALIGNNConfig(name="alignn", output_features=output_features))
    _, filename = tempfile.mkstemp(dir="/data/yll6162/tmp")
    with open(filename, "wb") as f:
        f.write(data)
    model.load_state_dict(torch.load(filename, map_location=device)["model"])
    os.remove(filename)

    model.to(device)
    model.eval()
    return model

def get_prediction(model, atoms, cutoff, output_path, input_file_path):
    g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=float(cutoff))
    g, lg = g.to(device), lg.to(device)

    with torch.no_grad():
        out_data, act_list_x, act_list_y, act_list_z = model([g, lg])

    act_list_x = act_list_x[-9:]
    act_list_y = act_list_y[-9:]
    act_list_z = act_list_z[-5:]

    base_name = os.path.basename(input_file_path)
    struct_file = re.sub(r'\.(vasp|cif)$', '', base_name)

    def save_activation(act_list, suffix):
        arrs = [a.detach().cpu().numpy() if not isinstance(a, np.ndarray) else a for a in act_list]
        np_arr = np.concatenate(arrs, axis=0)
        pd.DataFrame(np_arr).to_csv(f"{output_path}/{struct_file}_{suffix}.csv", index=False)
        for a in arrs:
            del a
        del np_arr

    save_activation(act_list_x, 'x')
    save_activation(act_list_y, 'y')
    save_activation(act_list_z, 'z')

    out_data = out_data.detach().cpu().numpy().flatten().tolist()

    del g, lg, act_list_x, act_list_y, act_list_z
    gc.collect()
    torch.cuda.empty_cache()

    return out_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="jv_formation_energy_peratom_alignn")
    parser.add_argument("--file_format", default="poscar")
    parser.add_argument("--file_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--cutoff", type=float, default=8)
    parser.add_argument("--source_file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_alignn_model(args.model_name, device)

    input_files = os.listdir(args.file_path)
    if args.source_file:
        src_df = pd.read_csv(args.source_file)
        src_filelist = set(src_df['sample'].values)
        input_files = [f for f in input_files if f in src_filelist]

    for input_file in tqdm(input_files, desc="extract alignn embeddings", unit='item'):
        id_str = os.path.splitext(input_file)[0].lstrip("POSCAR-")
        if any(id_str in f for f in os.listdir(args.output_path)):
            print(f"file {id_str} exists, skipping")
            continue

        input_file_path = os.path.join(args.file_path, input_file)

        if args.file_format == "poscar":
            atoms = Atoms.from_poscar(input_file_path)
        elif args.file_format == "cif":
            atoms = Atoms.from_cif(input_file_path)
        elif args.file_format == "xyz":
            atoms = Atoms.from_xyz(input_file_path, box_size=500)
        elif args.file_format == "pdb":
            atoms = Atoms.from_pdb(input_file_path, max_lat=500)
        else:
            raise ValueError(f"Unsupported file format: {args.file_format}")

        out_data = get_prediction(model, atoms, args.cutoff, args.output_path, input_file_path)
        print("Predicted:", args.model_name, input_file, out_data)
        # except Exception as e:
        #     print(f"Failed on {input_file}: {e}")


if __name__ == "__main__":
    main()