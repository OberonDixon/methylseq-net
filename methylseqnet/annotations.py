from pathlib import Path

import pysam

def load_bed_intervals(bed_file, descriptions=None, fetch_params=()):
    """Load a BED file into a dict of {chrom: [(start, end), ...]}."""
    intervals = {}
    with open(bed_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if descriptions is not None and parts[3].upper() not in [desc.upper() for desc in descriptions]:
                continue
            intervals.setdefault(parts[0], []).append((int(parts[1]), int(parts[2])))
    return intervals

def load_annotated_bed_rows(bed_file, descriptions=None,fetch_params=()):
    # if fetch_params is provided, only load rows that overlap the specified region
    rows = []
    if Path(bed_file).suffix == '.gz':
        for line in pysam.TabixFile(str(bed_file)).fetch(*fetch_params):
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if descriptions is not None and parts[3].upper() not in [desc.upper() for desc in descriptions]:
                continue
            rows.append(parts)
    else:
        with open(bed_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if descriptions is not None and parts[3].upper() not in [desc.upper() for desc in descriptions]:
                    continue
                if fetch_params:
                    chrom, start, end = fetch_params
                    if parts[0] != chrom or int(parts[2]) <= start or int(parts[1]) >= end:
                        continue
                rows.append(parts)
    return rows

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