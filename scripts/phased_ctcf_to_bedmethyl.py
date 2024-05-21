from dimelo import parse_bam

ref_genome = '/clusterfs/nilah/oberon/jupyter/chm13.draft_v1.0.fasta'
hp1_ctcf = '/clusterfs/nilah/oberon/datasets/deep_ctcf/phased/ALLCTCF_guppy_winnowmap_merge_chrX_NanoMethPhase_HP1.retagged.bam'
hp2_ctcf = '/clusterfs/nilah/oberon/datasets/deep_ctcf/phased/ALLCTCF_guppy_winnowmap_merge_chrX_NanoMethPhase_HP2.retagged.bam'
hp1_pileup, _ = parse_bam.pileup(
    input_file = hp1_ctcf,
    output_name = 'chrX_hp1_processed',
    ref_genome = ref_genome,
    motifs = ['A,0','CG,0'],
    thresh = 190,
    regions='chrX:0-200000000',
    cores=32,
)
hp2_pileup, _ = parse_bam.pileup(
    input_file = hp2_ctcf,
    output_name = 'chrX_hp2_processed',
    ref_genome = ref_genome,
    motifs = ['A,0','CG,0'],
    thresh = 190,
    regions='chrX:0-200000000',
    cores=32,
)

print(str(hp1_pileup))
print(str(hp2_pileup))