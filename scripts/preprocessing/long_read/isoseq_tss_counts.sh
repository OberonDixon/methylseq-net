#!/bin/bash

folder="/global/scratch/projects/vector_streetslab/oberon/datasets/vollger_mendelian/rna_bams/"

isoseq collapse \
  --min-aln-coverage 0.85 \
  ${folder}GM12878.kinnex.bam \
  ${folder}GM12878.kinnex.collapsed.no5exon.gff

echo "unphased metrics for min-aln-coverage 0.85"
awk '{sum+=$3} END {print "Total reads:", sum}' ${folder}GM12878.kinnex.collapsed.no5exon.abundance.txt
wc -l ${folder}GM12878.kinnex.collapsed.no5exon.abundance.txt
cut -f3 ${folder}GM12878.kinnex.collapsed.no5exon.abundance.txt | sort -n | uniq -c | tail -20

isoseq collapse \
  --min-aln-coverage 0.85 \
  ${folder}GM12878.kinnex.HP1.bam \
  ${folder}GM12878.kinnex.HP1.collapsed.no5exon.gff

echo "HP1 metrics for min-aln-coverage 0.85"
awk '{sum+=$3} END {print "Total reads:", sum}' ${folder}GM12878.kinnex.HP1.collapsed.no5exon.abundance.txt
wc -l ${folder}GM12878.kinnex.HP1.collapsed.no5exon.abundance.txt
cut -f3 ${folder}GM12878.kinnex.HP1.collapsed.no5exon.abundance.txt | sort -n | uniq -c | tail -20

isoseq collapse \
  --min-aln-coverage 0.85 \
  ${folder}GM12878.kinnex.HP2.bam \
  ${folder}GM12878.kinnex.HP2.collapsed.no5exon.gff

echo "HP2 metrics for min-aln-coverage 0.85"
awk '{sum+=$3} END {print "Total reads:", sum}' ${folder}GM12878.kinnex.HP2.collapsed.no5exon.abundance.txt
wc -l ${folder}GM12878.kinnex.HP2.collapsed.no5exon.abundance.txt
cut -f3 ${folder}GM12878.kinnex.HP2.collapsed.no5exon.abundance.txt | sort -n | uniq -c | tail -20