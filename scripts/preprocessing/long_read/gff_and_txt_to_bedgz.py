"""
Extract TSS positions with counts from IsoSeq collapse output
"""
import subprocess
import sys

def parse_gff(gff_path):
    """Extract TSS positions from GFF, return dict of {transcript_id: (chrom, pos, strand)}"""
    tss_dict = {}
    
    with open(gff_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'transcript':
                continue
            
            chrom = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            attributes = fields[8]
            
            # Extract transcript_id
            for attr in attributes.split(';'):
                if 'transcript_id' in attr:
                    transcript_id = attr.split('"')[1]
                    break
            else:
                continue
            
            # TSS depends on strand
            if strand == '+':
                tss_pos = start - 1  # Convert to 0-based
            else:
                tss_pos = end - 1
            
            tss_dict[transcript_id] = (chrom, tss_pos, strand)
    
    return tss_dict

def parse_abundance(abundance_path):
    """Parse abundance file, return dict of {transcript_id: count}"""
    count_dict = {}
    
    with open(abundance_path) as f:
        for line in f:
            if line.startswith('#') or line.startswith('pbid'):
                continue
            
            fields = line.strip().split('\t')
            transcript_id = fields[0]
            count = int(fields[1])
            count_dict[transcript_id] = count
    
    return count_dict

def main():
    folder = "/global/scratch/projects/vector_streetslab/oberon/datasets/vollger_mendelian/rna_bams/"
    for gff_path, abundance_path, output_path in [
        (f"{folder}GM12878.kinnex.collapsed.no5exon.gff",f"{folder}GM12878.kinnex.collapsed.no5exon.abundance.txt",f"{folder}GM12878.kinnex.no5exon.tss.counts.bed.gz"),
        (f"{folder}GM12878.kinnex.HP1.collapsed.no5exon.gff",f"{folder}GM12878.kinnex.HP1.collapsed.no5exon.abundance.txt",f"{folder}GM12878.kinnex.HP1.no5exon.tss.counts.bed.gz"),
        (f"{folder}GM12878.kinnex.HP2.collapsed.no5exon.gff",f"{folder}GM12878.kinnex.HP2.collapsed.no5exon.abundance.txt",f"{folder}GM12878.kinnex.HP2.no5exon.tss.counts.bed.gz"),
    ]:
        print(f"Processing {output_path}...", file=sys.stderr)
        print("Parsing GFF...", file=sys.stderr)
        tss_dict = parse_gff(gff_path)
        print(f"Found {len(tss_dict)} transcripts in GFF", file=sys.stderr)
        
        print("Parsing abundance...", file=sys.stderr)
        count_dict = parse_abundance(abundance_path)
        print(f"Found {len(count_dict)} transcripts in abundance file", file=sys.stderr)
        
        # Join and write output through bgzip
        print("Writing output...", file=sys.stderr)
        matched = 0
        
        # Sort by chrom, position and pipe to bgzip
        bgzip_proc = subprocess.Popen(['bgzip', '-c'], 
                                    stdin=subprocess.PIPE, 
                                    stdout=open(output_path, 'wb'),
                                    text=True)
        
        # Collect all entries for sorting
        entries = []
        for transcript_id, (chrom, tss_pos, strand) in tss_dict.items():
            if transcript_id in count_dict:
                count = count_dict[transcript_id]
                entries.append((chrom, tss_pos, transcript_id, count, strand))
                matched += 1
        
        # Sort by chromosome and position
        entries.sort(key=lambda x: (x[0], x[1]))
        
        # Write sorted entries with count in column 4 (0-indexed)
        for chrom, tss_pos, transcript_id, count, strand in entries:
            bgzip_proc.stdin.write(f"{chrom}\t{tss_pos}\t{tss_pos+1}\t{transcript_id}\t{count}\t{strand}\n")
        
        bgzip_proc.stdin.close()
        bgzip_proc.wait()
        
        print(f"Wrote {matched} TSS sites to {output_path}", file=sys.stderr)
        
        # Index with tabix
        print("Indexing with tabix...", file=sys.stderr)
        subprocess.run(['tabix', '-p', 'bed', output_path], check=True)
        print("Done!", file=sys.stderr)

if __name__ == '__main__':
    main()