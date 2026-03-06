from pathlib import Path

import pysam

def point_in_intervals(intervals, chrom, pos):
    """Check if a position falls within any interval for the given chrom."""
    return any(s <= pos < e for s, e in intervals.get(chrom, []))

def fetch_tss_from_gtf(
    tabix_path: str | Path,
    fetch_params,
    subset=set(),
    intervals=None,
    require_canonical=True,
    transcript_types={'protein_coding',},
  ):
    """
    Fetch TSS information for a specific chromosome from a tabix-indexed GTF file.
    """
    gtf = pysam.TabixFile(str(tabix_path))
    tss_list = []
    for record in gtf.fetch(*fetch_params):
        fields = record.split("\t")
        if fields[2] == "transcript":
            chrom = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            attributes = fields[8]

            # Parse attributes
            attr_dict = {}
            for attr in attributes.strip().split(';'):
                attr = attr.strip()
                if attr:
                    parts = attr.split(' ', 1)
                    if len(parts) == 2:
                        key = parts[0]
                        val = parts[1].strip('"')
                        attr_dict[key] = val
            
            # TSS is start for + strand, end for - strand
            tss = start if strand == '+' else end
            threeprime = end if strand == '+' else start

            tss_dict = {
                'chrom': chrom,
                'tss': tss,
                'threeprime': threeprime,
                'strand': strand,
                'gene_id': attr_dict.get('gene_id', ''),
                'gene_name': attr_dict.get('gene_name', '').upper(),
                'transcript_id': attr_dict.get('transcript_id', ''),
                'transcript_type': attr_dict.get('transcript_type', ''),
                'canonical': 'Ensembl_canonical' in attributes
            }

            if (
                    (len(subset)==0 or (tss_dict['gene_name'] in subset))
                    and (tss_dict['canonical'] or not require_canonical)
                    and (tss_dict['transcript_type'] in transcript_types)
                    and (intervals is None or point_in_intervals(intervals, chrom, tss))
                ):
                tss_list.append(tss_dict)
    return tss_list