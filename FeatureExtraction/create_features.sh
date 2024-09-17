# for filename in /scratch/yll6162/MOF-oxo/MOFs_oms/*.cif; do
#     python alignn/pretrained_activation.py --model_name mof_dband --file_format cif --file_path "$filename" --output_path "../examples/mof_dband_embed"
# done


# for filename in /scratch/yll6162/atomgpt/structure/*.vasp; do
#     python alignn/pretrained_activation.py --model_name mp_e_form_alignnn --file_format poscar --file_path "$filename" --output_path "/data/yll6162/alignntl_dft_3d/jid"
# done


# python alignn/pretrained_activation_all.py --model_name mof_dband --file_format cif \--file_path "/scratch/yll6162/MOF-oxo/MOFs_oms/" --output_path "/data/yll6162/mof/mof_dband_embed/" 
#  python alignn/pretrained_activation_all.py --model_name mp_e_form_alignnn --file_format cif \--file_path "/scratch/yll6162/MOF-oxo/MOFs_oms/" --output_path "../examples/mof_form_e_embed" \

#  python alignn/pretrained_activation.py --model_name mof_dband --file_format cif --file_path "/scratch/yll6162/MOF-oxo/MOFs_oms/" \
#  --output_path "../examples/mof_dband_embed" --source_file "/scratch/yll6162/MOF-oxo/MOFs_oms/id_prop_oxo.csv"


# python alignn/pretrained_activation.py --model_name mp_e_form_alignnn --file_format poscar \
# --file_path "/scratch/yll6162/atomgpt/structure" --output_path "/scratch/yll6162/ALIGNNTL/FeatureExtraction"

# python alignn/pretrained_activation_all.py --model_name mp_e_form_alignnn --file_format poscar \
# --file_path "/scratch/yll6162/atomgpt/structure" --output_path "/data/yll6162/alignntl_dft_3d/jid"



# find /scratch/yll6162/atomgpt/structure/ -maxdepth 1 -type f -name "*.vasp" | tail -n +51500 | while IFS= read -r filename; do
#     python alignn/pretrained_activation.py --model_name mp_e_form_alignnn --file_format poscar --file_path "$filename" --output_path "/data/yll6162/alignntl_dft_3d/jid"

# done


# find /scratch/yll6162/MOF-oxo/MOFs_oms/ -maxdepth 1 -type f -name "*.cif" | while IFS= read -r filename; do
#     python alignn/pretrained_activation.py --model_name mp_e_form_alignnn --file_format cif --file_path "$filename" --output_path "/data/yll6162/mof/mof_form_e_embed"
# done

#  python alignn/pretrained_activation.py --model_name mp_e_form_alignnn --file_format poscar --file_path "/scratch/yll6162/atomgpt/structure/POSCAR-JVASP-801.vasp" --output_path "/data/yll6162/alignntl_dft_3d/jid"

# echo "Processed $counter files."


# find /scratch/yll6162/MOF-oxo/MOFs_oms_q1 -maxdepth 1 -type f -name "*.cif" | while IFS= read -r filename; do
#     python alignn/pretrained_activation.py --model_name jv_optb88vdw_bandgap_alignn --file_format cif --file_path "$filename" --output_path "/data/yll6162/mof/mof_opt_bandgap"
# done

# find /scratch/yll6162/MOF-oxo/MOFs_oms -maxdepth 1 -type f -name "*.cif" | while IFS= read -r filename; do
#     python alignn/pretrained_activation.py --model_name jv_mbj_bandgap_alignn --file_format cif --file_path "$filename" --output_path "/data/yll6162/mof/mof_bandgap_embed"
# done

find /scratch/yll6162/MOF-oxo/MOFs_oms_q3_NN_other -maxdepth 1 -type f -name "*.cif" | while IFS= read -r filename; do
    python alignn/pretrained_activation.py --model_name mof_dband --file_format cif --file_path "$filename" --output_path "/data/yll6162/mof/mof_dband_embed"
done