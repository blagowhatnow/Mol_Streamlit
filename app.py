#Made wth streamlit 1.3.1. Uses the files etoxpred_best_model.joblib, sascore.py, fpscores.pkl.gz from eToxPred. Can be retrained on new data.
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFreeSASA
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import SessionState

import argparse

import pandas as pd

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem

import numpy as np

from sascore import SAscore
from joblib import load

rdBase.DisableLog('rdApp.error')

def load_data(smiles_list):
    mols = [Chem.MolFromSmiles(x) for x in smiles_list]
    X = []
    cnt = 0
    for mol in mols:
        mol = Chem.AddHs(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_string = fp.ToBitString()
        tmpX = np.array(list(fp_string),dtype=float)
        X.append(tmpX)
        cnt += 1
    X = np.array(X)
    return X, smiles_list

def predict(smiles_list, model_file):
    df = pd.DataFrame(columns=['smiles', 'Tox-score', 'SAscore'])
    # laod the data
    X, smiles_list = load_data(smiles_list)
    # load the saved model and make predictions
    clf = load(model_file)
    reg = SAscore()
    for i in range(X.shape[0]):
        tox_score = clf.predict_proba(X[i,:].reshape((1,1024)))[:,1]
        sa_score = reg(smiles_list[i])
        df.at[i, 'smiles'] = smiles_list[i]
        df.at[i, 'Tox-score'] = tox_score[0]
        df.at[i, 'SAscore'] = sa_score
    return df

st.set_page_config(page_title="Drug Property")

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

#No SASA
def generate_copy(smiles, verbose=False):
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

        properties.append([desc_MolLogP, desc_MolWt, desc_NumRotatableBonds,
                           desc_HBONDAcceptors, desc_HBONDDonors, desc_TPSA,
                           desc_HeavyAtoms, desc_NumAromaticRings, desc_QED])

    columnNames = ["MolLogP", "MolWt", "NumRotatableBonds", "HBondDonors", "HBondAcceptors",
                   "TPSA", "HeavyAtoms", "NumAromaticRings", "QED"]

    descriptors = pd.DataFrame(data=properties, columns=columnNames)

    return descriptors

def generate_single(smiles, model_file):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles_list=[smiles]
        df_toxi=predict(smiles_list,model_file)

        if mol is None:
            return None, {}

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
            "QED": desc_QED,
            "Toxicity" : df_toxi['Tox-score'][0],
            "SA Score" : df_toxi['SAscore'][0]
        }

        return mol, properties
    except:
        return None, {}

def ret_final_df(df, model_file):
    df_orig = df.copy()  # Make a copy of the original DataFrame
    smiles = df_orig['Ligand SMILES']
    df = generate(smiles)
    df_tox=predict(smiles, model_file)
    df_sub_0=df_tox[['Tox-score', 'SAscore']]
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
    merged_df_fin=pd.concat([df_sub_0, merged_df_final], axis=1)
    return merged_df_fin

def main():
    model_file='etoxpred_best_model.joblib'
    st.title("Molecular Properties App")

    # Input field for molecule
    molecule_input = st.text_input("Enter a SMILES string:")

    if molecule_input:
        mol, properties = generate_single(molecule_input, model_file)

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
            st.write("Toxicity:", properties["Toxicity"])
            st.write("SA Score:", properties["SA Score"])

            # Display 2D structure
            st.subheader("2D Structure")
            st.image(Draw.MolToImage(mol), use_column_width=False, width=300)
     # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv", key="csv_uploader")
    st.write("Results will keep only the valid SMILES strings from the uploaded dataframe. Default dataframe will be shown if no file is uploaded.")                   
    if uploaded_file is not None:
      try:
        # Read the CSV file into a DataFrame
          df=filter_valid_smiles(uploaded_file)
          N = 10
          molecules_per_row = 3 
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

          if session_state.page_number==0:
                df_graphs=generate_copy(df['Ligand SMILES'])
                non_empty_figures = []
                properties = ["MolLogP", "MolWt", "NumRotatableBonds", "HBondDonors", "HBondAcceptors",
                      "TPSA", "HeavyAtoms", "NumAromaticRings", "QED"]

                fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(30, 30))
                axes = axes.flatten()

                for i, prop in enumerate(properties):
                   ax = axes[i]
                   _, _, patches = ax.hist(df_graphs[prop], bins=13, alpha=0.7)
                   ax.hist(df_graphs[prop], bins=13, alpha=0.7)
                   ax.set_title(prop, fontsize=25)
                   ax.set_xlabel("Value",fontsize=22 )
                   ax.set_ylabel("Frequency", fontsize=22)
                   ax.tick_params(axis='x', labelsize=18)
                   ax.tick_params(axis='y', labelsize=18)
               # Check if the figure is empty
                   if any(patches):
                      non_empty_figures.append(ax)
# Check if there are non-empty figures
                if non_empty_figures:
    # Display the non-empty figures
                    fig.subplots_adjust(hspace=0.5)
                    st.pyplot(fig)
                else:
                    pass
       # Slice the DataFrame based on the current page
          sliced_df = df[start:end].reset_index(drop=True)

        # Display the sliced DataFrame
          st.subheader("Original DataFrame Head")
          st.dataframe(sliced_df)


        # Process the DataFrame
          final_df = ret_final_df(sliced_df, model_file)

        # Display the structures with properties for the current page
          for i in range(0, len(final_df), molecules_per_row):
              col1, col2, col3 = st.columns(3)

              for j, (_, row) in enumerate(final_df.iloc[i:i + molecules_per_row].iterrows()):
                  col = col1 if j % 3 == 0 else col2 if j % 3 == 1 else col3
                  with col:
                      st.markdown(f"<h3 style='font-size: 13px;'>FDA Drug Name: {row['FDA drugnames']}</h3>", unsafe_allow_html=True)
                      st.image(row['Structures'], use_column_width=False, width=210)
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
                      st.write("Toxicity:", row['Tox-score'])
                      st.write("SA Score:", row['SAscore'])
                      st.write("---") 
      except:
        st.write("Dataframe uploaded is not in the proper format")     
    else:
        df=pd.read_csv("FDA_Human_2022-11-14_2.csv")
        N = 10
        molecules_per_row = 3 
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
        if session_state.page_number==0:
            df_graphs=generate_copy(df['Ligand SMILES'])
            non_empty_figures = []
            properties = ["MolLogP", "MolWt", "NumRotatableBonds", "HBondDonors", "HBondAcceptors",
                      "TPSA", "HeavyAtoms", "NumAromaticRings", "QED"]

            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(30, 30))
            axes = axes.flatten()

            for i, prop in enumerate(properties):
               ax = axes[i]
               _, _, patches = ax.hist(df_graphs[prop], bins=13, alpha=0.7)
               ax.hist(df_graphs[prop], bins=13, alpha=0.7)
               ax.set_title(prop, fontsize=25)
               ax.set_xlabel("Value",fontsize=22 )
               ax.set_ylabel("Frequency", fontsize=22)
               ax.tick_params(axis='x', labelsize=18)
               ax.tick_params(axis='y', labelsize=18)
               # Check if the figure is empty
               if any(patches):
                  non_empty_figures.append(ax)
# Check if there are non-empty figures
            if non_empty_figures:
    # Display the non-empty figures
               fig.subplots_adjust(hspace=0.5)
               st.pyplot(fig)
            else:
                pass
   
       # Slice the DataFrame based on the current page
        sliced_df = df[start:end].reset_index(drop=True)

        # Display the sliced DataFrame
        st.subheader("Original DataFrame Head")
        st.dataframe(sliced_df)

        # Process the DataFrame
        final_df = ret_final_df(sliced_df, model_file)

        # Display the structures with properties for the current page
        # Display the structures with properties for the current page
        for i in range(0, len(final_df), molecules_per_row):
            col1, col2, col3 = st.columns(3)

            for j, (_, row) in enumerate(final_df.iloc[i:i + molecules_per_row].iterrows()):
                col = col1 if j % 3 == 0 else col2 if j % 3 == 1 else col3
                with col:
                    st.markdown(f"<h3 style='font-size: 13px;'>FDA Drug Name: {row['FDA drugnames']}</h3>", unsafe_allow_html=True)
                    st.image(row['Structures'], use_column_width=False, width=210)
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
                    st.write("Toxicity:", row['Tox-score'])
                    st.write("SA Score:", row['SAscore'])
                    st.write("---")
        


if __name__ == "__main__":
    main()
