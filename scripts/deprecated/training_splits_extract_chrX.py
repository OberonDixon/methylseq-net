input_file = "../configs/training-splits/borzoi-human/sequences_human.bed"
output_file = "../configs/training-splits/borzoi-human/chrX_sequences.bed"

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        fields = line.strip().split('\t')
        
        # Check if this is a chrX line
        if fields[0] == 'chrX':
            # Extract the fold number from the last column (e.g., "fold0" -> 0)
            fold_info = fields[-1]
            fold_num = int(fold_info.replace('fold', ''))
            
            # Determine new fold label based on even/odd
            if fold_num % 2 == 0:
                fields[-1] = 'chrX_fold0'
            else:
                fields[-1] = 'chrX_fold1'
            
            # Write the modified line
            outfile.write('\t'.join(fields) + '\n')

print(f"chrX lines extracted and saved to {output_file}")