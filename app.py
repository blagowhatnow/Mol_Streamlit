#Uses streamlit 1.3.1
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFreeSASA
from rdkit.Chem import PandasTools, Draw
import base64
from io import BytesIO
from PIL import Image
import SessionState



def image_base64(im):
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

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

        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_HBONDAcceptors = Chem.Lipinski.NumHDonors(mol)
        desc_HBONDDonors = Chem.Lipinski.NumHAcceptors(mol)
        desc_TPSA = Descriptors.TPSA(mol)
        desc_HeavyAtoms = Descriptors.HeavyAtomCount(mol)
        desc_NumAromaticRings = Descriptors.NumAromaticRings(mol)
        desc_QED = calculate_qed(mol)
        desc_SASA = calculate_sasa(mol)

        properties.append([desc_MolLogP, desc_MolWt, desc_NumRotatableBonds,
                           desc_HBONDAcceptors, desc_HBONDDonors, desc_TPSA,
                           desc_HeavyAtoms, desc_NumAromaticRings, desc_SASA, desc_QED])

    columnNames = ["MolLogP", "MolWt", "NumRotatableBonds", "HBondDonors", "HBondAcceptors",
                   "TPSA", "HeavyAtoms", "NumAromaticRings", "SASA", "QED"]

    descriptors = pd.DataFrame(data=properties, columns=columnNames)

    return descriptors


def ret_final_df(df):
    df_orig = df.copy()  # Make a copy of the original DataFrame
    smiles = df_orig['Ligand SMILES']
    df = generate(smiles)
    df_sub = df_orig[['FDA drugnames', 'Ligand SMILES']]
    merged_df = pd.concat([df_sub, df], axis=1)

    # Create RDKit molecules from the SMILES
    molecules = [Chem.MolFromSmiles(i) for i in smiles]

    # Write molecules to an SDF file
    writer = Chem.SDWriter('molecules.sdf')
    for mol in molecules:
        writer.write(mol)
    writer.close()

    # Read the SDF file and extract the structures as images
    suppl = Chem.SDMolSupplier('molecules.sdf')
    struc_images = [Draw.MolToImage(mol) for mol in suppl]

    # Create a DataFrame with structure images
    struc_df = pd.DataFrame({'Structures': struc_images})

    # Merge the structure images DataFrame with the properties DataFrame
    merged_df_final = pd.concat([merged_df, struc_df], axis=1)

    return merged_df_final

def main():
    st.title("Molecular Properties App - Knowdis")

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

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

