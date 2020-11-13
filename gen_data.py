import pandas as pd

def gen_SMILES():
    data = pd.read_csv('./CHEMBL/chembl_27_chemreps.txt', sep='\t', header=0)
    SMILES = data['canonical_smiles'].dropna()
    # show statistics
    print('total instances:', SMILES.size)
    lengths = SMILES.apply(len)
    print(lengths.min(), lengths.max(), lengths.mean(), lengths.median())
    SMILES = SMILES[lengths <= 50]
    SMILES = SMILES[lengths >= 40]
    print('length between [40, 50]:', SMILES.size)
    SMILES = SMILES.str.ljust(51, 'Y')
    for idx, irow in SMILES.items():
        SMILES[idx] = 'X' + irow
    # save to file
    SMILES.to_csv('./CHEMBL/chembl_27_smiles.txt', header=None, index=False)


if __name__ == '__main__':
    
    gen_SMILES()
