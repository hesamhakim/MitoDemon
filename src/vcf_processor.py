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
        self.variants = []
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
            variant_dict = {
                'chrom': record.CHROM,
                'pos': record.POS,
                'ref': record.REF,
                'alt': str(record.ALT[0]) if record.ALT else '',
                'qual': record.QUAL,
                'filter': ','.join(record.FILTER) if record.FILTER else 'PASS',
                'umi_id': record.ID,
                'read_count': read_count,
                'umi_seq': umi_seq,
                'is_snp': len(record.REF) == 1 and len(str(record.ALT[0])) == 1 if record.ALT else False,
                'is_indel': not (len(record.REF) == 1 and len(str(record.ALT[0])) == 1) if record.ALT else False
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
                      snps_only: bool = True,
                      min_vaf: float = 0.25,
                      min_depth: int = 10,
                      min_alt_reads: int = 2) -> pd.DataFrame:
        """
        Apply quality filters to variants
        
        Parameters:
        -----------
        filter_pass : bool
            Keep only PASS filter variants
        snps_only : bool
            Keep only SNPs (exclude indels)
        min_vaf : float
            Minimum variant allele frequency
        min_depth : int
            Minimum total read depth
        min_alt_reads : int
            Minimum reads supporting alternate allele
        
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
        
        if snps_only:
            before = len(filtered)
            filtered = filtered[filtered['is_snp'] == True]
            if self.verbose:
                print(f"After SNPs only: {len(filtered)} variants ({before - len(filtered)} indels removed)")
        
        # VAF filter
        before = len(filtered)
        filtered = filtered[filtered['vaf'] >= min_vaf]
        if self.verbose:
            print(f"After VAF >= {min_vaf}: {len(filtered)} variants ({before - len(filtered)} removed)")
        
        # Depth filter
        before = len(filtered)
        filtered = filtered[filtered['total_depth'] >= min_depth]
        if self.verbose:
            print(f"After depth >= {min_depth}: {len(filtered)} variants ({before - len(filtered)} removed)")
        
        # Alt reads filter
        before = len(filtered)
        filtered = filtered[filtered['alt_depth'] >= min_alt_reads]
        if self.verbose:
            print(f"After alt reads >= {min_alt_reads}: {len(filtered)} variants ({before - len(filtered)} removed)")
        
        self.filtered_variants = filtered
        
        if self.verbose:
            print(f"\nFinal: {len(filtered)} variants passed all filters ({len(filtered)/initial_count*100:.1f}%)")
            print(f"Unique UMIs remaining: {filtered['umi_id'].nunique()}")
        
        return filtered
    
    def get_variant_matrix(self, n_positions: int = 16569) -> np.ndarray:
        """
        Create binary variant matrix for all mitochondrial positions
        
        Parameters:
        -----------
        n_positions : int
            Number of mitochondrial genome positions (default: 16569)
        
        Returns:
        --------
        np.ndarray
            Binary matrix of shape (n_positions, n_umis)
        """
        if not hasattr(self, 'filtered_variants'):
            raise ValueError("No filtered variants. Run apply_filters() first.")
        
        # Get unique UMIs
        unique_umis = self.filtered_variants['umi_id'].unique()
        n_umis = len(unique_umis)
        
        # Create binary matrix
        variant_matrix = np.zeros((n_positions, n_umis), dtype=int)
        
        # Fill matrix
        for umi_idx, umi in enumerate(unique_umis):
            umi_variants = self.filtered_variants[self.filtered_variants['umi_id'] == umi]
            for _, var in umi_variants.iterrows():
                # Adjust for 0-based indexing
                pos_idx = int(var['pos']) - 1
                if 0 <= pos_idx < n_positions:
                    variant_matrix[pos_idx, umi_idx] = 1
        
        if self.verbose:
            print(f"Created variant matrix: {variant_matrix.shape}")
            print(f"Total variants: {variant_matrix.sum()}")
            print(f"Mean variants per UMI: {variant_matrix.sum(axis=0).mean():.2f}")
        
        return variant_matrix
