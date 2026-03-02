import pysam
from tqdm import tqdm

hp1_ctcf = '/clusterfs/nilah/oberon/datasets/deep_ctcf/phased/ALLCTCF_guppy_winnowmap_merge_chrX_NanoMethPhase_HP1.retagged.bam'
hp2_ctcf = '/clusterfs/nilah/oberon/datasets/deep_ctcf/phased/ALLCTCF_guppy_winnowmap_merge_chrX_NanoMethPhase_HP2.retagged.bam'

hp1_bam = pysam.AlignmentFile(hp1_ctcf)
hp2_bam = pysam.AlignmentFile(hp2_ctcf)
hp1_txt = '/clusterfs/nilah/oberon/datasets/deep_ctcf/phased/qnames/hp1.txt'
hp2_txt = '/clusterfs/nilah/oberon/datasets/deep_ctcf/phased/qnames/hp2.txt'

with open(hp1_txt,'w') as f:
    for read in tqdm(hp1_bam.fetch('chrX',0,200000000)):
        f.write(read.qname + '\n')
with open(hp2_txt,'w') as f:
    for read in tqdm(hp2_bam.fetch('chrX',0,200000000)):
        f.write(read.qname + '\n')