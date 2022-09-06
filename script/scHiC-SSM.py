#!/usr/bin/env python

import sys
import os
import glob
import gc
import argparse
import pickle
from tqdm import tqdm
import scanpy as sc
import numpy as np
import pandas as pd
import anndata
import scvi
from joblib import Parallel, delayed
from time import time

print(scvi.__version__)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#helper function which returns n_samples indices per label in labels_key
def subsample_dataset(train_data, labels_key, n_samples):
    sample_idx = []
    labels, counts = np.unique(train_data.obs[labels_key], return_counts=True)
    for i, label in enumerate(labels):
        label_locs = np.where(train_data.obs[labels_key] == label)[0]
        if counts[i] < n_samples:
            sample_idx.append(label_locs)
        else:
            label_subset = np.random.choice(label_locs, n_samples, replace=False)
            sample_idx.append(label_subset)
    sample_idx = np.concatenate(sample_idx)
    return sample_idx

#n_obs. Number of observations. n_vars. Number of variables/features. e.g.AnnData object with n_obs × n_vars = 40 × 249
def normalize(bandM, cellInfo, chromSelect, bandDist, nLatent = 10, batchFlag = False, gpuFlag = False):
    cellSelect = [i for i, val in enumerate(bandM.sum(axis = 1)>0) if val]#cellSelect是[0,...,399]的列表
    if len(cellSelect) == 0:
        normCount = None
        latentDF = pd.DataFrame(np.zeros((len(bandM), nLatent)), index = range(len(bandM)))
    else:
        bandDepth = bandM[cellSelect,].sum(axis = 1).mean()
        adata = sc.AnnData(bandM)
        adata.obs[['cell_type']] = cellInfo['cell_type'].values
        if(batchFlag is True):
            adata.obs['batch'] = cellInfo['batch'].values

        ###option 1 for assigning cell labels
        # batch_1_idx = np.where(adata.obs['batch'] == '181218_21yr')[0]
        # batch_2_idx = np.where(adata.obs['batch'] == '190305_21yr')[0]
        # batch_3_idx = np.where(adata.obs['batch'] == '190305_29yr')[0]
        # batch_4_idx = np.where(adata.obs['batch'] == '190315_21yr')[0]
        # batch_5_idx = np.where(adata.obs['batch'] == '190315_29yr')[0]

        # labelled_batch_idx = np.concatenate((batch_1_idx, batch_2_idx, batch_4_idx)) #train on these batches
        # unlabelled_batch_idx = np.concatenate((batch_3_idx, batch_5_idx)) #make predictions on these batches

        # labelled_adata = adata[labelled_batch_idx].copy()
        # unlabelled_adata = adata[unlabelled_batch_idx].copy()
        # #del unlabelled_adata.uns["_scvi"]
        # unlabelled_data = unlabelled_adata.obs['cell_type']

        # sample_idx = subsample_dataset(labelled_adata, 'cell_type', 100)
        # labelled_adata.obs['labels_for_scanvi'] = 'unknown'
        # labelled_adata.obs['labels_for_scanvi'].iloc[sample_idx] = labelled_adata.obs['cell_type'][sample_idx]

        # unlabelled_adata.obs['labels_for_scanvi'] = 'unknown'
        # scandata = labelled_adata.concatenate(unlabelled_adata, batch_key="_batch")

        ###option 2 for assigning cell labels
        sample_idx = subsample_dataset(adata, 'cell_type',100)
        adata.obs['labels_for_scanvi'] = 'unknown'
        adata.obs['labels_for_scanvi'].iloc[sample_idx] = adata.obs['cell_type'][sample_idx]
        print(adata.obs[['labels_for_scanvi']].value_counts())
        sc.pp.filter_cells(adata, min_counts=1)

        if(batchFlag is True):
            scvi.model.SCVI.setup_anndata(adata, batch_key="batch", labels_key = 'labels_for_scanvi')
        else:
            scvi.model.SCVI.setup_anndata(adata, labels_key = 'labels_for_scanvi')
        scvi_model = scvi.model.SCVI(adata, n_latent = nLatent, n_layers=2)
        scvi_model.train(use_gpu = gpuFlag)
        scanvi_model = scvi.model.SCANVI.from_scvi_model(scvi_model, 'unknown')    
        scanvi_model.train(50, use_gpu = gpuFlag)

        ##save model
        #scanvi_model.save("scHiC-SSM_Lee2019_model_scvi_scanvi_version_10latent_50semi_100seeds_10band", overwrite=True)
        if(batchFlag is True):
            imputeTmp = np.zeros((len(cellSelect), bandM.shape[1]))
            for batchName in list(set(cellInfo['batch'].values)):
                imputeTmp = imputeTmp + scanvi_model.get_normalized_expression(library_size = bandDepth, transform_batch = batchName)
            imputeM = imputeTmp/len(list(set(cellInfo['batch'].values)))
        else:
            imputeM = scanvi_model.get_normalized_expression(library_size = bandDepth)
        normCount = get_locuspair(imputeM, chromSelect, bandDist)
        latent = scanvi_model.get_latent_representation(adata)
        latentDF = pd.DataFrame(latent, index = cellSelect)
        latentDF = latentDF.reindex([i for i in range(len(bandM))]).fillna(0)
    return(latentDF, normCount)

##Note: code from 3DVI; Github:https://github.com/yezhengSTAT/scVI-3D; Citation:Normalization and De-noising of Single-cell Hi-C Data with BandNorm and 3DVI.
def create_band_mat(x: np.array, count: np.array, diag: int, maxChromosomeSize: int) -> np.array:
    bandMat = np.zeros(maxChromosomeSize - diag)
    bandMat[x] = count
    return bandMat

class Process(object):
    def __init__(self, resolution, chromSize=None):
        self._RESOLUTION = resolution
        self._chromSize = chromSize
        self.df = None
        self._lastchrom = None
        self._chormdf = None

    def rescale(self, chrA, x, y, counts, resolution = None):
        if resolution:
            self._RESOLUTION = resolution
        xR = x // self._RESOLUTION
        yR = y // self._RESOLUTION
        self.df = pd.DataFrame({'chrA': chrA,
                        'x': xR,
                        'y': yR,
                        'counts': counts})
        self.df.loc[:,'diag'] = abs(yR - xR)
        return True
    
    def band(self, chrom, diag, maxBand):
        if self.df is None:
            raise "Run process.rescale(chrA, binA, binY, counts, resolution) first."
        if self._lastchrom is None or (self._lastchrom != chrom):
            self._lastchrom = chrom
            self._chormdf = self.df[self.df.chrA == chrom]            
        dat =  self._chormdf[self._chormdf.diag == diag]
        mat = create_band_mat(dat.x.values, dat.counts.values, diag, maxBand)
        return mat
    
    def band_all(self, chromSize, used_chroms = 'whole', used_diags = [i for i in range(1, 11)]):
        if self.df is None:
            raise "Run process.rescale(chrA, binA, binY, counts, resolution) first"
        if chromSize:
            self._chromSize = chromSize
        chrom = 'chrA'
        diag_s = 'diag'
        cell_band = {}
        for chromosome, chromosome_data in self.df.groupby(chrom):#以染色体字段分组
            if (used_chroms != 'whole' and chromosome not in used_chroms) or chromosome not in self._chromSize:
                continue
            bandSize = self._chromSize[chromosome] // self._RESOLUTION + 1
            chromosome_band = {}
            for diag, chromosome_diag in chromosome_data.groupby(diag_s):
                if used_diags != 'whole' and diag not in used_diags:
                    continue
                x = chromosome_diag.x.values
                count = chromosome_diag.counts.values
                chromosome_band[diag] = create_band_mat(x, count, diag, bandSize)
            cell_band[chromosome] = chromosome_band
        return cell_band
    
def read_file(file):
    df = pd.read_csv(file, sep = "\t", header = None, names = ['chrA', 'binA', 'chrB', 'binB', 'counts'])
    df.loc[:,'cell'] = file
    return df

def read_file_chrom(file, used_chroms):
    dfTmp = pd.read_csv(file, sep = "\t", header = None, names = ['chrA', 'binA', 'chrB', 'binB', 'counts'])
    dfTmp.loc[:,'cell'] = file    
    if used_chroms == 'whole':
        df = dfTmp
    else:
        df = dfTmp[dfTmp.chrA.isin(used_chroms)]
    return df

def read_files(file_list, used_chroms = 'whole', cores = 8):
    df_list = Parallel(n_jobs=cores)(delayed(read_file_chrom)(file, used_chroms) for file in file_list)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df
    
def process_cell(cell_raw: pd.DataFrame, binSizeDict: dict, 
                 resolution: int, cell: str,
                 used_chroms,
                 used_diags):
    process = Process(resolution)
    process.rescale(cell_raw.chrA, cell_raw.binA, cell_raw.binB, cell_raw.counts)    
    cell_band = process.band_all(binSizeDict, used_chroms=used_chroms, used_diags=used_diags)
    return (cell_band, cell)

def get_locuspair(imputeM, chromSelect, bandDist):
    xvars = ['chrA','binA', 'chrB', 'binB', 'count', 'cellID']
    tmp = imputeM.transpose().copy()
    tmp.index.name = 'binA'
    normCount = pd.melt(tmp.reset_index(), id_vars = ['binA'], var_name='cellID', value_name='count')
    normCount.loc[:,'binA'] = normCount.binA.astype(int)
    normCount.loc[:,'binB'] = normCount.binA + bandDist
    normCount.loc[:,'chrA'] = chromSelect
    normCount.loc[:,'chrB'] = chromSelect
    normCount = normCount[xvars]
    return normCount

def get_args():
    '''Get arguments'''
    parser = argparse.ArgumentParser(description = '------------Usage Start------------',
                                     epilog = '------------Usage End------------')
    parser.add_argument('-b', '--bandMax', help = 'Maximum genomic distance to be processed, e.g. 10. Use "whole" to include all the band matrix for each chromosome. Default is "whole".', default = 'whole')
    parser.add_argument('-c', '--chromList', help = 'List of chromosome to be processed separate by comma, e.g. "chr1,chr2,chrX". Use "whole" to include all chromosomes in the cell summary file (args.cellSummary). Default is "whole".', default = 'whole')
    parser.add_argument('-r', '--resolution', help = 'Resolution of scHi-C data, e.g., 1000000.', default = None)
    parser.add_argument('-i', '--inPath', help = 'Path to the folder where input scHi-C data are saved.', default = None)
    parser.add_argument('-o', '--outdir', help = 'Path to output directory.', default = None)
    parser.add_argument('-cs', '--cellSummary', help = '(Optional) Cell summary file with columns names to be "name" for scHi-C data file name including extension, "batch" for batch factor, "cell_type" for cluster or cell type label (tab separated file).', default = None)
    parser.add_argument('-g', '--genome', help = 'Path to genome size file (tab separated file).', default = None)
    parser.add_argument('-br', '--batchRemoval', help = 'Indicator to remove batch or not. Default is False.', action='store_true')
    parser.add_argument('-n', '--nLatent', help = 'Dimension of latent space. Default is 100.', default = 100)
    parser.add_argument('-gpu', '--gpuFlag', help = '(Optional) Use GPU or not. Default is False.', action='store_true')
    parser.add_argument('-p', '--parallelCPU', help = '(Optional) Number of CPUs to be used for parallel running. Default is 1 and no parallel computing is used.', default = 1)
    parser.add_argument('-v', '--verbose', help = '(Optional) Verbose. Default is False.', action='store_true')

    args = parser.parse_args()
    if args.bandMax != "whole":
        if int(args.bandMax) <= 0:
            print("Maximum distance as positive integer for band matrix need to be specified.")
            parser.print_help()
            sys.exit()
    if args.resolution is None:
        print("Please provide the resolution of the data.")
        parser.print_help()
        sys.exit()
    if args.inPath is None:
        print("Path to the input scHi-C data need to be specified.")
        parser.print_help()
        sys.exit()
    if args.outdir is None:
        print("Path to output directory need to be specified.")
        parser.print_help()
        sys.exit()
    else:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
    if args.batchRemoval is True:
        if not os.path.exists(args.cellSummary):
            print("Cell summary file with batch information does not exist.")
            parser.print_help()
            sys.exit()
        cellInfo = pd.read_csv(args.cellSummary, sep = "\t", header = 0).sort_values(by = 'name')
        if 'batch' not in cellInfo.columns:
            print("There is no column in cell summary file called 'batch'.")
            parser.print_help()
            sys.exit()
    if not os.path.exists(args.genome):
        print("The genome size file does not exist.")
        parser.print_help()
        sys.exit()
    if int(args.nLatent) <= 0:
        print("The number of dimension of latent space need to be set as nonnegative integer.")
        parser.print_help()
        sys.exit()
    if int(args.parallelCPU) == -1:
        print("All the CPUs will be used.")
    if int(args.parallelCPU) == -2:
        print("All the CPUs but one will be used.")
    if int(args.parallelCPU) < -2:
        print("Please provide number of CPUs for parallel computing.")
        parser.print_help()
        sys.exit()
    return args

if __name__ == "__main__":
    from time import time
    t0 = time()
    args = get_args()
    if args.verbose:
        print('Maximum genomic distance:', args.bandMax)
        print('Chromosomes to be processed:', args.chromList)
        print('Resolution:', args.resolution)
        print('Path to the input scHi-C data:', args.inPath)
        print('Path to output directory:', args.outdir)
        print('Cell summary file:', args.cellSummary)
        print('Genome size file:', args.genome)
        print('Remove batch effect or not:', args.batchRemoval)
        print('Dimension of latent space:', args.nLatent)
        print('Use GPU or not:', args.gpuFlag)
        print('Number of CPUs for parallel computing:', args.parallelCPU)

    outdir = args.outdir
    ## number of bin per chromosome
    print("Caculate total number of bin per chromosome.")
    binSize = pd.read_csv(args.genome, sep = "\t", header = None)
    binSizeDict = {}
    N = binSize.shape[0]
    for i in range(N):
        chrome = binSize.iloc[i,0]
        size = binSize.iloc[i,1]
        binSizeDict[chrome] = size
    ## cell info file
    print("Prepare cell summary file.")
    if args.cellSummary is not None:
        cellInfo = pd.read_csv(args.cellSummary, sep = "\t", header = 0).sort_values(by = 'name')
    else:
        cellName = {'name': os.listdir(args.inPath)}
        cellInfo = pd.DataFrame(cellName).sort_values(by = 'name')
    cellInfo.index = range(cellInfo.shape[0])
    ## read in scHi-C data
    print("Read in scHi-C data.")
    files = list(args.inPath + '/' + cellInfo.name)
    files.sort()
    ## read all the files and sort by cell file name
    resolution = int(args.resolution)
    coreN = int(args.parallelCPU)
    if args.bandMax == "whole":
        used_diags = "whole"
    else:
        used_diags = [i for i in range(1, int(args.bandMax) + 1)]
    if args.chromList == "whole":
        used_chroms = "whole"
    else:
        used_chroms = args.chromList.split(',')
    raws = read_files(files, used_chroms, coreN)
    print("Convert interactions into band matrix.")
    raw_cells = Parallel(n_jobs=coreN)(delayed(process_cell)(cell_df, binSizeDict, resolution, cell, used_chroms, used_diags) for cell, cell_df in tqdm(raws.groupby('cell')))
    raw_cells.sort(key=lambda x: x[1]) ##x[1] is 'cell', used for sorting
    cells = [cell for _, cell in raw_cells]
    raw_cells = [raw_cell for raw_cell, _ in raw_cells]
    if not os.path.exists(outdir + '/pickle'):
        os.mkdir(outdir + '/pickle')
    with open(outdir + '/pickle/raw_cells', 'wb') as f:
        pickle.dump(raw_cells, f)
    # del raws
    # gc.collect()
    print("Concat cells into cell x locus-pair matrix.")
    band_chrom_diag = {}
    for chrom, chromSize in binSizeDict.items():
        if used_chroms != "whole" and chrom not in used_chroms:
            continue
        chromSize = chromSize // resolution + 1
        chrom_diag = {}
        for band in range(1, chromSize - 4):
            if used_diags != "whole" and band not in used_diags:
                continue
            mat = []
            for fi in range(len(files)):
                if band not in raw_cells[fi][chrom]:
                    tmp = np.zeros(chromSize - band)
                else:
                    tmp = raw_cells[fi][chrom][band]
                mat.append(tmp)
            chrom_diag[band] = np.vstack(mat)
        band_chrom_diag[chrom] = chrom_diag
    
    with open(outdir + '/pickle/band_chrom_diag', 'wb') as f:
        pickle.dump(band_chrom_diag, f)
    
    # del raw_cells
    # gc.collect()
    '''for chromSelect, band_diags in band_chrom_diag.items():
        for bandDist, bandM in band_diags.items():
            print(chromSelect)'''
    bandMiter = [[bandM, chromSelect, bandDist] for chromSelect, band_diags in band_chrom_diag.items() for bandDist, bandM in band_diags.items()]
    #print(bandMiter)
    nLatent = int(args.nLatent) #int(args.nLatent)
    batchFlag = args.batchRemoval
    gpuFlag = args.gpuFlag
    if coreN == 1:
        latentCombinedNormCounts = [normalize(bandM, cellInfo, chromSelect, bandDist, nLatent, batchFlag, gpuFlag) for bandM, chromSelect, bandDist in bandMiter]
    else:
        latentCombinedNormCounts = Parallel(n_jobs=coreN,backend='multiprocessing')(delayed(normalize)(bandM, cellInfo, chromSelect, bandDist, nLatent, batchFlag, gpuFlag) for bandM, chromSelect, bandDist in bandMiter)
        #print(res)
    with open(outdir + '/pickle/scHiC-SSM_Lee2019_scvi_scanvi_version_latent10_100seeds_50semi_10Band', 'wb') as f:
        pickle.dump(latentCombinedNormCounts, f)
    print("Writing out latent embeddings.")
    if not os.path.exists(outdir + '/normalization'):
        os.mkdir(outdir + '/normalization')
    runningTimeFile = open('scHiC-SSM_running_time_Lee2019_scvi_scanvi_version_10latent_100seeds_50semi_10Band.txt','w')
    runningTimeFile.write('Total time: %d seconds.' % int(time() - t0))