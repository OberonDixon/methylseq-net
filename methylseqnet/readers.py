from pathlib import Path
import warnings
import os

import numpy as np

def load_sequence(
    file_path: str | Path | dict[str, str | Path],
    contig: str,
    start: int | None = None,
    end: int | None = None,
    file_type: str | None = None,
) -> str:
    if (isinstance(file_path, str) or isinstance(file_path, Path)):
        try:
            if (Path(file_path).suffix in [".fa", ".fasta", ".fna"] or file_type == "fasta"):
                return _load_fasta_sequence(file_path, contig, start, end)
            else:
                raise NotImplementedError(f"File type for {file_path} not supported.")
        except Exception as e:
            raise RuntimeError(f"Error in sequence loading for {contig}:{start}-{end} from {file_path}.") from e
    else:
        if ("fasta" in file_path and "vcf" in file_path) or file_type == "vcf_fasta":
            raise NotImplementedError("VCF + FASTA variant loading not implemented yet.")
        elif ("fasta" in file_path and "chain" in file_path) or file_type == "chain_fasta":
            raise NotImplementedError("Chain + FASTA assembly liftover loading not implemented yet.")
        raise NotImplementedError(f"File type for {file_path} not supported.")

def load_track(
    file_path: str | Path | dict[str, str | Path],
    contig: str,
    start: int | None = None,
    end: int | None = None,
    file_type: str | None = None,
    negative_to_value: float | None = None,
    nan_to_zero: bool = False,
    bin_size: int = 1,
    **kwargs,
) -> np.ndarray:
    if (isinstance(file_path, str) or isinstance(file_path, Path)):
        if (Path(file_path).suffix in [".bw",".bigwig"] or file_type == "bigwig"):
            return _load_bigwig_track(file_path, contig, start, end, negative_to_value, nan_to_zero, bin_size)
        elif (Path(file_path).suffix in [".bam"] or file_type == "bam_coverage"):
            return _load_bam_coverage_track(file_path, contig, start, end, negative_to_value, nan_to_zero, bin_size)
        elif file_type == "bedcounts":
            return _load_bedcounts_track(file_path, contig, start, end, negative_to_value, nan_to_zero, bin_size)
        elif file_type == "bedmethyl":
            return _load_bedmethyl_track(file_path, contig, start, end, negative_to_value, nan_to_zero, bin_size, **kwargs)
        elif Path(file_path).name.endswith(".bed.gz"):
            return _load_bedcounts_track(file_path, contig, start, end, negative_to_value, nan_to_zero, bin_size)
        else:
            raise NotImplementedError(f"File type for {file_path} not supported.")
    else:
        raise NotImplementedError(f"File type for {file_path} not supported.")

def load_masked_track(
    file_path: str | Path | dict[str, str | Path],
    contig: str,
    start: int | None = None,
    end: int | None = None,
    file_type: str | None = None,
    negative_to_value: float | None = None,
    nan_to_zero: bool = False,
    bin_size: int = 1,
    **kwargs,
) -> np.ndarray:
    track = load_track(
        file_path=file_path, 
        contig=contig,
        start=start,
        end=end,
        file_type=file_type,
        negative_to_value=None,
        nan_to_zero=False,
        bin_size=bin_size,
        **kwargs,
    )
    valid_mask = ~np.isnan(track)
    if negative_to_value is not None:
        track[track < 0] = negative_to_value
    if nan_to_zero:
        track[np.isnan(track)] = 0
    return track, valid_mask

def _load_fasta_sequence(
    file_path: str | Path,
    contig: str,
    start: int | None = None,
    end: int | None = None,
) -> str:
    import pysam
    fasta = pysam.FastaFile(file_path)
    try:
        if start is not None and end is not None:
            start_pad = 0 - min(start,0)
            chrom_length = fasta.get_reference_length(contig)
            end_pad = max(end,chrom_length) - chrom_length
            sequence = start_pad*"N" + fasta.fetch(contig,max(start,0),min(end,chrom_length)) + end_pad*"N"
        else:
            sequence = fasta.fetch(contig)
    except Exception as e:
        raise RuntimeError(f"Error in fasta loading for {contig}:{start}-{end} from {file_path}.") from e
    finally:
        fasta.close()
    return sequence

def _load_bigwig_track(
    file_path: str | Path,
    contig: str,
    start: int | None,
    end: int | None,
    negative_to_value: float | None,
    nan_to_zero: bool,
    bin_size: int,
) -> np.ndarray:
    import pyBigWig
    bw = pyBigWig.open(str(file_path))
    contigs = bw.chroms()
    if start is not None and end is not None:
        start_pad = max(-start,0)
        end_pad = max(end,contigs[contig]) - contigs[contig]
    else:
        start = 0
        end = contigs[contig]
        start_pad = 0
        end_pad = 0
    try:
        track = np.array(
            start_pad*[np.nan]
            + bw.values(contig, max(start,0), min(end,contigs[contig]))
            + end_pad*[np.nan]
            )
    except Exception as e:
        raise RuntimeError(f"Error in bigwig loading for {contig}:{start}-{end} from {file_path}.") from e
    finally:
        bw.close()
    if negative_to_value is not None:
        track[track < 0] = negative_to_value
    if nan_to_zero:
        track[np.isnan(track)] = 0
    if bin_size > 1:
        return track.reshape(-1, bin_size).mean(axis=1)
    else:
        return track

def _load_bedcounts_track(
    file_path: str | Path,
    contig: str,
    start: int | None,
    end: int | None,
    negative_to_value: float | None,
    nan_to_zero: bool,
    bin_size: int,
) -> np.ndarray:
    import pysam
    if start is None or end is None:
        raise ValueError("Start and end must be provided for bedcounts track loading.")
    counts_vector = np.zeros(end-start)
    start_pad = 0 - min(start,0)
    tabix = pysam.TabixFile(str(file_path))
    try:
        for row in tabix.fetch(contig,max(start,0),end):
            tabix_fields = row.split("\t")
            genomic_coord = int(tabix_fields[1])
            counts = int(tabix_fields[4])
            counts_vector[genomic_coord-start]+=counts
    except Exception as e:
        raise RuntimeError(f"Error in bedcounts loading for {contig}:{start}-{end} from {file_path}.") from e
    finally:
        tabix.close()
    if negative_to_value is not None:
        # we should never hit this branch: no valid counts bed file should have negative counts
        counts_vector[counts_vector < 0] = negative_to_value
    if bin_size > 1:
        return counts_vector.reshape(-1, bin_size).mean(axis=1)
    else:
        return counts_vector

def _load_bedmethyl_track(
    file_path: str | Path,
    contig: str,
    start: int | None,
    end: int | None,
    negative_to_value: float | None,
    nan_to_zero: bool,
    bin_size: int,
    motif: str = "CG,0",
) -> np.ndarray:
    from dimelo import load_processed
    if start is None or end is None:
        raise ValueError("Start and end must be provided for bedmethyl track loading.")
    start_pad = 0 - min(start,0)
    try:
        modified_base_counts,valid_base_counts = load_processed.pileup_vectors_from_bedmethyl(
            bedmethyl_file=file_path,
            motif=motif,
            regions=f'{contig}:{max(start,0)}-{end}',
            cores=1,
            quiet=True,
        )
    except Exception as e:
        raise RuntimeError(f"Error in bedmethyl loading for {contig}:{start}-{end} from {file_path}.") from e
    modified_base_counts_binned = modified_base_counts.reshape(-1,bin_size).sum(axis=1)
    valid_base_counts_binned = valid_base_counts.reshape(-1,bin_size).sum(axis=1)

    nans_everywhere = np.full_like(
        modified_base_counts_binned, np.nan, dtype=float
    )
    mod_ratios = np.array(
        start_pad*[np.nan] + list(
            np.divide(
                modified_base_counts_binned,
                valid_base_counts_binned,
                out=nans_everywhere,
                where=valid_base_counts_binned != 0,
            )
        )
    )
    # Mod ratios are always [0,1] or NaN, so negative_to_value is not applicable
    if nan_to_zero:
        mod_ratios[np.isnan(mod_ratios)] = 0
    return mod_ratios

def _load_bam_coverage_track(
    file_path: str | Path,
    contig: str,
    start: int | None,
    end: int | None,
    negative_to_value: float | None,
    nan_to_zero: bool,
    bin_size: int,   
) -> np.ndarray:
    import pysam
    bam = pysam.AlignmentFile(file_path, "rb")
    contig_length = bam.get_reference_length(contig)
    if start is None:
        start = 0
    if end is None:
        end = contig_length
    start_pad = max(-start,0)
    end_pad = max(end,contig_length) - contig_length
    try:
        coverage_vector = np.array(
            start_pad*[0]
            + [sum(x) for x in zip(*(bam.count_coverage(contig,max(start,0),min(end,contig_length))))]
            + end_pad*[0]
        )
    except Exception as e:
        raise RuntimeError(f"Error in coverage loading for {contig}:{start}-{end} from {file_path}. contig length {contig_length}") from e
    finally:
        bam.close()
    # Coverage is always >=0, so negative_to_value and nan_to_zero are not applicable. No .bam file that pysam can open could give such values.
    if bin_size > 1:
        coverage_vector = coverage_vector.reshape(-1, bin_size).mean(axis=1)
    return coverage_vector

