#!/bin/bash
INPUT_DIR="/global/scratch/projects/vector_streetslab/oberon/datasets/vollger_mendelian/UDN318336/fiber_bams/"

# Define output file
OUTPUT_BAM="${INPUT_DIR}/PS0019X.merged.phased.bam"

# Merge BAMs (samtools merge automatically sorts by coordinate)
samtools merge -@ 8 "${OUTPUT_BAM}" "${INPUT_DIR}PS00190.phased.bam" "${INPUT_DIR}PS00191.phased.bam" "${INPUT_DIR}PS00192.phased.bam"

# "PS00190.phased" "PS00191.phased" "PS00192.phased" "PS00288.phased"
for filedesc in "PS0019X.merged.phased"; do
    samtools index "${INPUT_DIR}${filedesc}.bam"

    # Extract reads to canonical chromosomes
    samtools view -@ 8 -b "${INPUT_DIR}${filedesc}.bam" $(echo chr{1..22} chrX chrY chrM) > ${INPUT_DIR}${filedesc}.canonical.tmp.bam
    
    # Reheader to only include canonical chromosomes in @SQ lines
    samtools view -H ${INPUT_DIR}${filedesc}.canonical.tmp.bam | \
        awk '$1 == "@SQ" && $2 ~ /^SN:chr([0-9]+|X|Y|M)$/ {print; next} $1 != "@SQ" {print}' \
        > ${INPUT_DIR}${filedesc}.header.sam
    
    samtools reheader ${INPUT_DIR}${filedesc}.header.sam ${INPUT_DIR}${filedesc}.canonical.tmp.bam > ${INPUT_DIR}${filedesc}.canonical.bam
    
    rm ${INPUT_DIR}${filedesc}.canonical.tmp.bam ${INPUT_DIR}${filedesc}.header.sam
    samtools index "${INPUT_DIR}${filedesc}.canonical.bam"
    
    aligned_bam_to_cpg_scores --bam "${INPUT_DIR}${filedesc}.canonical.bam" \
        --output-prefix "${INPUT_DIR}${filedesc}" --threads 8 \
        --modsites-mode reference --ref /clusterfs/nilah/oberon/genomes/hg38.fa
done
