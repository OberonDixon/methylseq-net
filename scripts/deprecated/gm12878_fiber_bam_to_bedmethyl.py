from dimelo import parse_bam

genome_path = '/clusterfs/nilah/oberon/genomes/hg38.fa'
hp1_filepath = '/global/scratch/projects/vector_streetslab/oberon/datasets/vollger_mendelian/fiberseq_bams/GM12878_WGS.hap1.retagged.bam'
hp2_filepath = '/global/scratch/projects/vector_streetslab/oberon/datasets/vollger_mendelian/fiberseq_bams/GM12878_WGS.hap2.retagged.bam'

mA_pileup, _ = parse_bam.pileup(
    input_file = hp1_filepath,
    output_name = 'GM12878_hp1_whole_genome_CpG',
    ref_genome = genome_path,
    motifs = ['CG,0'],
    thresh = 128,
)

cpg_pileup, _ = parse_bam.pileup(
    input_file = hp2_filepath,
    output_name = 'GM12878_hp2_whole_genome_CpG',
    ref_genome = genome_path,
    motifs = ['CG,0'],
    thresh = 128,
)