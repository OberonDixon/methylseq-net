from dimelo import parse_bam

ref_genome = '/clusterfs/nilah/oberon/jupyter/chm13.draft_v1.0.fasta'
# hp1_ctcf = '/clusterfs/nilah/oberon/datasets/deep_ctcf/phased/ALLCTCF_guppy_winnowmap_merge_chrX_NanoMethPhase_HP1.retagged.bam'
# hp2_ctcf = '/clusterfs/nilah/oberon/datasets/deep_ctcf/phased/ALLCTCF_guppy_winnowmap_merge_chrX_NanoMethPhase_HP2.retagged.bam'
# hp1_pileup, _ = parse_bam.pileup(
#     input_file = hp1_ctcf,
#     output_name = 'chrX_hp1_processed',
#     ref_genome = ref_genome,
#     motifs = ['A,0','CG,0'],
#     thresh = 190,
#     regions='chrX:0-200000000',
#     cores=32,
# )
# hp2_pileup, _ = parse_bam.pileup(
#     input_file = hp2_ctcf,
#     output_name = 'chrX_hp2_processed',
#     ref_genome = ref_genome,
#     motifs = ['A,0','CG,0'],
#     thresh = 190,
#     regions='chrX:0-200000000',
#     cores=32,
# )

# print(str(hp1_pileup))
# print(str(hp2_pileup))

files_dict = {'hp1_cpg':'/clusterfs/nilah/oberon/datasets/deep_ctcf/phased/megalodon/deep_ctcf_CpG_mod_mappings_HP1_chrX.retagged.sorted.bam',
'hp2_cpg':'/clusterfs/nilah/oberon/datasets/deep_ctcf/phased/megalodon/deep_ctcf_CpG_mod_mappings_HP2_chrX.retagged.sorted.bam',
'hp1_ma':'/clusterfs/nilah/oberon/datasets/deep_ctcf/phased/megalodon/deep_ctcf_mod_mappings_merge_HP1_chrX.retagged.sorted.bam',
'hp2_ma':'/clusterfs/nilah/oberon/datasets/deep_ctcf/phased/megalodon/deep_ctcf_mod_mappings_merge_HP2_chrX.retagged.sorted.bam'}
for output_name,input_file in files_dict.items():
    pileup,_ = parse_bam.pileup(
        input_file = input_file,
        output_name = output_name,
        ref_genome = ref_genome,
        motifs = ['A,0','CG,0'],
        thresh = 190,
        regions='chrX:0-200000000',
        cores=32,
    )