import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
rdkit.RDLogger.logger.setLevel(4, 4)

def cano_smiles(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
        if tmp is None:
            return tmp, None
        tmp = Chem.RemoveHs(tmp)
        if tmp is None:
            return tmp, None
        [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    except:
        return None, None
    return tmp, Chem.MolToSmiles(tmp)

