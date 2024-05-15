from dimelo import parse_bam

genome_path = '/clusterfs/nilah/oberon/jupyter/chm13.draft_v1.0.fasta'
megalodon_bam_filepath = '/clusterfs/nilah/oberon/datasets/deep_ctcf/deep_ctcf_mod_mappings_merge.retagged.sorted.bam'
cpg_bam_filepath = '/clusterfs/nilah/oberon/datasets/deep_ctcf/deep_ctcf_CpG_mod_mappings.retagged.sorted.bam'

# whole genome pileup for all context model
mA_pileup, _ = parse_bam.pileup(
    input_file = megalodon_bam_filepath,
    output_name = 'allcontext_bam_whole_genome',
    ref_genome = genome_path,
    motifs = ['A,0','CG,0'],
    thresh = 190,
)

# whole genome pileup for cpg model
# cpg_pileup, _ = parse_bam.pileup(
#     input_file = cpg_bam_filepath,
#     output_name = 'cpg_bam_whole_genome',
#     ref_genome = genome_path,
#     motifs = ['A,0','CG,0'],
#     thresh = 190,
# )