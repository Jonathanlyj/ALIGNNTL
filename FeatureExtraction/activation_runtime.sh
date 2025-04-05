
for filename in /scratch/yll6162/atomgpt/structure/*.vasp; do
    # Extract the number from the filename using pattern matching
    basename=$(basename "$filename")
    number=$(echo "$basename" | grep -oP '\d+(?=\.vasp)')

    # Proceed only if the number is less than 10000
    if [ "$number" -lt 10000 ]; then
        python alignn/pretrained_activation.py \
            --model_name mp_e_form_alignnn \
            --file_format poscar \
            --file_path "$filename" \
            --output_path "/data/yll6162/activation_runtime"
    fi
done