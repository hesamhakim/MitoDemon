"""
VCF Processing Module for Single-Cell Mitochondrial Variant Analysis
Author: [Your Name]
Date: 2025
Description: Processes VAULT-generated VCF files containing single-mitochondrion variant profiles
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import vcf
import warnings
from pathlib import Path

class VCFProcessor:
    """Process VCF files from VAULT pipeline for mitochondrial variant analysis"""
    
    def __init__(self, vcf_path: str, verbose: bool = True):
        """
        Initialize VCF processor
        
        Parameters:
        -----------
        vcf_path : str
            Path to the VCF file
        verbose : bool
            Print processing information
        """
        self.vcf_path = Path(vcf_path)
        self.verbose = verbose
        self.variants = pd.DataFrame()
        self.umi_groups = {}
        
        if not self.vcf_path.exists():
            raise FileNotFoundError(f"VCF file not found: {vcf_path}")
    
    def read_vcf(self) -> pd.DataFrame:
        """
        Read VCF file and extract variant information
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing variant information
        """
        if self.verbose:
            print(f"Reading VCF file: {self.vcf_path}")
        
        vcf_reader = vcf.Reader(open(str(self.vcf_path), 'r'))
        
        variants_list = []
        for record in vcf_reader:
            # Extract UMI information from ID field
            umi_info = record.ID.split('_') if record.ID else ['0', 'UNKNOWN', 'UNKNOWN']
            read_count = int(umi_info[0]) if umi_info[0].isdigit() else 0
            umi_seq = '_'.join(umi_info[1:]) if len(umi_info) > 1 else 'UNKNOWN'
            
            # Extract variant information
            alt_str = str(record.ALT[0]) if record.ALT else ''
            ref_len = len(record.REF)
            alt_len = len(alt_str)
            is_snp = (ref_len == 1 and alt_len == 1)
            is_indel = not is_snp and (record.ALT is not None)
            
            indel_length = abs(alt_len - ref_len) if is_indel else 0
            indel_type = 'INS' if alt_len > ref_len else 'DEL' if alt_len < ref_len else 'NONE'
            
            variant_dict = {
                'chrom': record.CHROM,
                'pos': record.POS,
                'ref': record.REF,
                'alt': alt_str,
                'qual': record.QUAL,
                'filter': ','.join(record.FILTER) if record.FILTER else 'PASS',
                'umi_id': record.ID,
                'read_count': read_count,
                'umi_seq': umi_seq,
                'is_snp': is_snp,
                'is_indel': is_indel,
                'indel_length': indel_length,
                'indel_type': indel_type
            }
            
            # Extract INFO fields
            for key, value in record.INFO.items():
                variant_dict[f'info_{key}'] = value[0] if isinstance(value, list) else value
            
            # Calculate VAF if DP4 is available
            if 'DP4' in record.INFO:
                dp4 = record.INFO['DP4']
                ref_count = dp4[0] + dp4[1]
                alt_count = dp4[2] + dp4[3]
                total = ref_count + alt_count
                variant_dict['vaf'] = alt_count / total if total > 0 else 0
                variant_dict['total_depth'] = total
                variant_dict['alt_depth'] = alt_count
            else:
                # Try to get depth from DP field
                variant_dict['total_depth'] = record.INFO.get('DP', 0)
                variant_dict['vaf'] = 0.5  # Default if no VAF info available
                variant_dict['alt_depth'] = variant_dict['total_depth'] // 2
            
            variants_list.append(variant_dict)
        
        self.variants = pd.DataFrame(variants_list)
        
        if self.verbose:
            print(f"Loaded {len(self.variants)} variants from VCF file")
            print(f"Unique UMIs: {self.variants['umi_id'].nunique()}")
        
        return self.variants
    
    def apply_filters(self, 
                      filter_pass: bool = True,
                      include_indels: bool = True,
                      min_vaf: float = 0.15,
                      min_depth: int = 5,
                      min_alt_reads: int = 1,
                      max_missing_rate: float = 0.95) -> pd.DataFrame:
        """
        Apply quality filters to variants with relaxed thresholds and INDEL inclusion
        
        Parameters:
        -----------
        filter_pass : bool
            Keep only PASS filter variants
        include_indels : bool
            Include INDELs (new: default True, replaces snps_only)
        min_vaf : float
            Minimum variant allele frequency (relaxed)
        min_depth : int
            Minimum total read depth (relaxed)
        min_alt_reads : int
            Minimum reads supporting alternate allele (relaxed)
        max_missing_rate : float
            Maximum missing rate across UMIs (new: retain variants in >=5% UMIs)
        
        Returns:
        --------
        pd.DataFrame
            Filtered variants
        """
        if self.variants.empty:
            raise ValueError("No variants loaded. Run read_vcf() first.")
        
        filtered = self.variants.copy()
        initial_count = len(filtered)
        
        # Apply filters
        if filter_pass:
            filtered = filtered[filtered['filter'] == 'PASS']
            if self.verbose:
                print(f"After PASS filter: {len(filtered)} variants ({initial_count - len(filtered)} removed)")
        
        if not include_indels:
            before = len(filtered)
            filtered = filtered[filtered['is_snp'] == True]
            if self.verbose:
                print(f"After SNPs only: {len(filtered)} variants ({before - len(filtered)} indels removed)")
        else:
            if self.verbose:
                print(f"INDELs included (encoded as continuous features)")
        
        # Relaxed quality filters
        before = len(filtered)
        filtered = filtered[filtered['vaf'] >= min_vaf]
        if self.verbose:
            print(f"After VAF >= {min_vaf}: {len(filtered)} variants ({before - len(filtered)} removed)")
        
        before = len(filtered)
        filtered = filtered[filtered['total_depth'] >= min_depth]
        if self.verbose:
            print(f"After depth >= {min_depth}: {len(filtered)} variants ({before - len(filtered)} removed)")
        
        before = len(filtered)
        filtered = filtered[filtered['alt_depth'] >= min_alt_reads]
        if self.verbose:
            print(f"After alt reads >= {min_alt_reads}: {len(filtered)} variants ({before - len(filtered)} removed)")
        
        # Frequency-based filter (revised: relax to ensure more UMIs retained)
        if max_missing_rate < 1.0:
            before = len(filtered)
            variant_counts = filtered.groupby(['pos', 'ref', 'alt']).size()
            total_umis = self.variants['umi_id'].nunique()  # Revised: use original total UMIs before heavy filtering to relax threshold
            min_occurrences = max(1, int(total_umis * (1 - max_missing_rate) * 0.1))  # Revised: multiply by 0.1 to further relax (e.g., 0.5% instead of 5%)
            frequent_variants = variant_counts[variant_counts >= min_occurrences].index
            
            variant_keys = set(frequent_variants)
            filtered = filtered[filtered.apply(lambda x: (x['pos'], x['ref'], x['alt']) in variant_keys, axis=1)]
            
            if self.verbose:
                print(f"After frequency filter (min {min_occurrences} occurrences): {len(filtered)} variants ({before - len(filtered)} removed)")
        
        self.filtered_variants = filtered
        
        if self.verbose:
            print(f"\nFinal: {len(filtered)} variants passed all filters ({len(filtered)/initial_count*100:.1f}%)")
            print(f"Unique UMIs remaining: {filtered['umi_id'].nunique()}")
        
        return filtered
    
    def get_variant_matrix(self, n_positions: int = 16299, continuous: bool = True, include_indels: bool = True) -> np.ndarray:  # Revised: mouse mtDNA length 16299
        """
        Create variant matrix (continuous VAF or binary) for all mitochondrial positions
        
        Parameters:
        -----------
        n_positions : int
            Number of mitochondrial genome positions (revised: 16299 for mouse)
        continuous : bool
            Use continuous VAF values instead of binary (new: default True)
        include_indels : bool
            Include INDEL encodings in matrix (new)
        
        Returns:
        --------
        np.ndarray
            Variant matrix of shape (n_features, n_umis) where n_features >= n_positions (expanded for INDELs)
        """
        if not hasattr(self, 'filtered_variants'):
            raise ValueError("No filtered variants. Run apply_filters() first.")
        
        # Get unique UMIs
        unique_umis = self.filtered_variants['umi_id'].unique()
        n_umis = len(unique_umis)
        
        # For INDELs, we'll create additional features: presence (binary or VAF) + normalized length + type encoding
        if include_indels:
            indel_variants = self.filtered_variants[self.filtered_variants['is_indel']]
            n_indel_features = len(indel_variants) * 3  # presence, length, type
            if self.verbose:
                print(f"Encoding {len(indel_variants)} INDELs as 3 features each (presence, length, type)")
        else:
            n_indel_features = 0
        
        # Total features = positions (for SNVs) + indel features
        total_features = n_positions + n_indel_features
        variant_matrix = np.zeros((total_features, n_umis), dtype=float if continuous else int)
        
        # Map UMIs to column indices
        umi_to_idx = {umi: idx for idx, umi in enumerate(unique_umis)}
        
        # Fill SNV part (first n_positions rows)
        snv_variants = self.filtered_variants[self.filtered_variants['is_snp']]
        for _, var in snv_variants.iterrows():
            pos_idx = int(var['pos']) - 1
            if 0 <= pos_idx < n_positions:
                umi_idx = umi_to_idx[var['umi_id']]
                value = var['vaf'] if continuous else 1
                variant_matrix[pos_idx, umi_idx] = value
        
        # Fill INDEL part (rows n_positions onwards)
        if include_indels and n_indel_features > 0:
            indel_feature_idx = n_positions
            indel_groups = indel_variants.groupby(['pos', 'ref', 'alt', 'indel_type'])
            
            for _, group in indel_groups:
                for _, var in group.iterrows():
                    umi_idx = umi_to_idx.get(var['umi_id'], -1)
                    if umi_idx == -1:
                        continue
                    
                    # Feature 1: Presence (VAF or binary)
                    presence = var['vaf'] if continuous else 1
                    variant_matrix[indel_feature_idx, umi_idx] = presence
                    
                    # Feature 2: Normalized length (length / max_length in data)
                    max_length = indel_variants['indel_length'].max()
                    norm_length = var['indel_length'] / max_length if max_length > 0 else 0
                    variant_matrix[indel_feature_idx + 1, umi_idx] = norm_length
                    
                    # Feature 3: Type encoding (INS=1, DEL=0, other=0.5) - positive to avoid issues
                    type_code = 1 if var['indel_type'] == 'INS' else 0 if var['indel_type'] == 'DEL' else 0.5
                    variant_matrix[indel_feature_idx + 2, umi_idx] = type_code
                
                indel_feature_idx += 3
        
        if self.verbose:
            print(f"Created variant matrix: {variant_matrix.shape}")
            print(f"Total variants/features: {np.sum(variant_matrix > 0)}")
            print(f"Mean value per UMI: {variant_matrix.mean(axis=0).mean():.2f}")
            print(f"Sparsity: {1 - np.sum(variant_matrix > 0) / variant_matrix.size:.6f}")
        
        return variant_matrix
