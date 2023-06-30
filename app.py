#Uses streamlit 1.3.1
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFreeSASA
from rdkit.Chem import Draw
import SessionState

def filter_valid_smiles(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Check validity of SMILES in each row
    valid_smiles = []
    for smiles in df["Ligand SMILES"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)

    # Filter the DataFrame to include only rows with valid SMILES
    filtered_df = df[df["Ligand SMILES"].isin(valid_smiles)].reset_index(drop=True)

    return filtered_df

def calculate_sasa(mol):
    try:
        hmol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(hmol)
        radii = rdFreeSASA.classifyAtoms(hmol)
        sasa = rdFreeSASA.CalcSASA(hmol, radii)
    except:
        sasa = float('nan')
    return sasa

def calculate_qed(mol):
    qed = Descriptors.qed(mol)
    return qed

def generate(smiles, verbose=False):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)

    properties = []
    for mol in moldata:
        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol)

        desc_MolLogP = round(Descriptors.MolLogP(mol), 3)
        desc_MolWt = round(Descriptors.MolWt(mol), 3)
        desc_NumRotatableBonds = round(Descriptors.NumRotatableBonds(mol), 3)
        desc_HBONDAcceptors = round(Chem.Lipinski.NumHDonors(mol), 3)
        desc_HBONDDonors = round(Chem.Lipinski.NumHAcceptors(mol), 3)
        desc_TPSA = round(Descriptors.TPSA(mol), 3)
        desc_HeavyAtoms = round(Descriptors.HeavyAtomCount(mol), 3)
        desc_NumAromaticRings = round(Descriptors.NumAromaticRings(mol), 3)
        desc_QED = round(calculate_qed(mol), 3)
        desc_SASA = round(calculate_sasa(mol), 3)

        properties.append([desc_MolLogP, desc_MolWt, desc_NumRotatableBonds,
                           desc_HBONDAcceptors, desc_HBONDDonors, desc_TPSA,
                           desc_HeavyAtoms, desc_NumAromaticRings, desc_SASA, desc_QED])

    columnNames = ["MolLogP", "MolWt", "NumRotatableBonds", "HBondDonors", "HBondAcceptors",
                   "TPSA", "HeavyAtoms", "NumAromaticRings", "SASA", "QED"]

    descriptors = pd.DataFrame(data=properties, columns=columnNames)

    return descriptors

def generate_single(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None, {}

        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol)

        desc_MolLogP = round(Descriptors.MolLogP(mol), 3)
        desc_MolWt = round(Descriptors.MolWt(mol), 3)
        desc_NumRotatableBonds = round(Descriptors.NumRotatableBonds(mol), 3)
        desc_HBONDAcceptors = round(Chem.Lipinski.NumHDonors(mol), 3)
        desc_HBONDDonors = round(Chem.Lipinski.NumHAcceptors(mol), 3)
        desc_TPSA = round(Descriptors.TPSA(mol), 3)
        desc_HeavyAtoms = round(Descriptors.HeavyAtomCount(mol), 3)
        desc_NumAromaticRings = round(Descriptors.NumAromaticRings(mol), 3)
        desc_QED = round(calculate_qed(mol), 3)
        desc_SASA = round(calculate_sasa(mol), 3)

        properties = {
            "MolLogP": desc_MolLogP,
            "MolWt": desc_MolWt,
            "NumRotatableBonds": desc_NumRotatableBonds,
            "HBondDonors": desc_HBONDDonors,
            "HBondAcceptors": desc_HBONDAcceptors,
            "TPSA": desc_TPSA,
            "HeavyAtoms": desc_HeavyAtoms,
            "NumAromaticRings": desc_NumAromaticRings,
            "SASA": desc_SASA,
            "QED": desc_QED
        }

        return mol, properties
    except:
        return None, {}

def ret_final_df(df):
    df_orig = df.copy()  # Make a copy of the original DataFrame
    smiles = df_orig['Ligand SMILES']
    df = generate(smiles)
    df_sub = df_orig[['FDA drugnames', 'Ligand SMILES']]
    merged_df = pd.concat([df_sub, df], axis=1)

    # Create RDKit molecules from the SMILES
    molecules = [Chem.MolFromSmiles(i) for i in smiles]

    # Read the SDF file and extract the structures as images
    struc_images = [Draw.MolToImage(mol) for mol in molecules]

    # Create a DataFrame with structure images
    struc_df = pd.DataFrame({'Structures': struc_images})

    # Merge the structure images DataFrame with the properties DataFrame
    merged_df_final = pd.concat([merged_df, struc_df], axis=1)

    return merged_df_final

def main():
    st.title("Molecular Properties App")

    # Input field for molecule
    molecule_input = st.text_input("Enter a SMILES string:")

    if molecule_input:
        mol, properties = generate_single(molecule_input)

        if mol is None:
            st.write("Invalid molecule.")

        else:
            # Display molecule properties
            st.subheader("Molecule Properties")
            st.write("MolLogP:", properties["MolLogP"])
            st.write("MolWt:", properties["MolWt"])
            st.write("NumRotatableBonds:", properties["NumRotatableBonds"])
            st.write("HBondDonors:", properties["HBondDonors"])
            st.write("HBondAcceptors:", properties["HBondAcceptors"])
            st.write("TPSA:", properties["TPSA"])
            st.write("HeavyAtoms:", properties["HeavyAtoms"])
            st.write("NumAromaticRings:", properties["NumAromaticRings"])
            st.write("SASA:", properties["SASA"])
            st.write("QED:", properties["QED"])

            # Display 2D structure
            st.subheader("2D Structure")
            st.image(Draw.MolToImage(mol), use_column_width=False, width=300)

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    st.write("Results will keep only the valid smiles strings from the uploaded dataframe")
    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df=filter_valid_smiles(uploaded_file)
        N = 10
        session_state = SessionState.get(page_number = 0)
        last_page = len(df) // N
        prev, _ ,next = st.columns([1, 10, 1])
        if next.button("Next"):
            if session_state.page_number + 1 > last_page:
                session_state.page_number = 0
            else:
                session_state.page_number += 1
        if prev.button("Previous"):
            if session_state.page_number - 1 < 0:
                session_state.page_number = last_page
            else:
                session_state.page_number -= 1
        start = session_state.page_number * N 
        end = (1 + session_state.page_number) * N

       # Slice the DataFrame based on the current page
        sliced_df = df[start:end].reset_index(drop=True)

        # Display the sliced DataFrame
        st.subheader("Original DataFrame Head")
        st.dataframe(sliced_df)

        # Process the DataFrame
        final_df = ret_final_df(sliced_df)

        # Display the structures with properties for the current page
        for index, row in final_df.iterrows():
            st.subheader(f"FDA Drug Name: {row['FDA drugnames']}")
            st.image(row['Structures'], use_column_width=False,width=300)
            st.write("MolLogP:", row['MolLogP'])
            st.write("MolWt:", row['MolWt'])
            st.write("NumRotatableBonds:", row['NumRotatableBonds'])
            st.write("HBondDonors:", row['HBondDonors'])
            st.write("HBondAcceptors:", row['HBondAcceptors'])
            st.write("TPSA:", row['TPSA'])
            st.write("HeavyAtoms:", row['HeavyAtoms'])
            st.write("NumAromaticRings:", row['NumAromaticRings'])
            st.write("SASA:", row['SASA'])
            st.write("QED:", row['QED'])
            st.write("---")

if __name__ == "__main__":
    main()

