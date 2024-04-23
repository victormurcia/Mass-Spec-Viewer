# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 19:19:25 2024

@author: vmurc
"""
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw,AllChem
import py3Dmol
import matplotlib.pyplot as plt
import plotly.express as px
from stmol import showmol

def load_data():
    #Load dataframe
    return pd.read_parquet("filtered_df.parquet.gzip")

def display_molecular_structure(mol_content):
    m = Chem.MolFromMolBlock(mol_content)
    if m:
        img = Draw.MolToImage(m)
        return img
    else:
        return None

def display_mass_spectrum(df, row_index):
    spectrum_data = df.loc[row_index, 'parsed_spectrum']
    compound_name = df['name'][row_index]
    
    fig, ax = plt.subplots(figsize=(10, 5.8))
    wavelengths, intensities = zip(*spectrum_data)
    ax.vlines(wavelengths, 0, intensities, color='b', linewidth=2)
    ax.set_xlabel('m/z')
    ax.set_ylabel('Intensity')
    ax.set_title(f'Mass Spectrum for {compound_name}')
    ax.grid(True)
    
    return fig

def create_classification_df(classification_series):
    # Create an empty DataFrame to store expanded classification data
    classification_df = pd.DataFrame()

    # Loop through each item in the series and convert each dictionary into a DataFrame row
    for item in classification_series:
        # Convert the dictionary into a DataFrame and append to the main DataFrame
        item_df = pd.DataFrame([item])  # Make sure to pass [item] to treat it as a single row
        classification_df = pd.concat([classification_df, item_df], ignore_index=True)
    
    # Initialize parent column
    classification_df['parent'] = ""
    
    # Variable to keep track of the current parent for alternative parents
    last_non_alternative_parent = ""

    # Set parent for each item
    for i in range(len(classification_df)):
        if 'alternative parent' not in classification_df.loc[i, 'name']:
            if i > 0:
                classification_df.loc[i, 'parent'] = classification_df.loc[i - 1, 'value']
            last_non_alternative_parent = classification_df.loc[i, 'value']
        else:
            classification_df.loc[i, 'parent'] = last_non_alternative_parent

    return classification_df


def create_sunburst(classification_df, compound_name):
    # Generate the figure using Plotly Express
    fig = px.sunburst(
        classification_df,
        names='value',       # Column containing labels for the leaves
        parents='parent',   # Column containing labels of parent nodes
        values=[1] * len(classification_df),  # Optional: sizes of the sectors, assumed uniform here
        title=f'Concept Hierarchy for {compound_name}',
        width=800, height=800,
        color='value',  # Define which column to use for coloring
        color_discrete_sequence=px.colors.qualitative.Antique  # Define the color palette
    )

    # Update layout for aesthetics
    fig.update_layout(
        title_font_size=25,
        font=dict(
            family="Arial, sans-serif",
            size=22,
            color="black"
        )
    )
    return fig

def smiles_to_xyz(smiles):
    # Create a molecule from a SMILES string
    molecule = Chem.MolFromSmiles(smiles)
    molecule = Chem.AddHs(molecule)  # Add hydrogens

    # Generate 3D coordinates
    AllChem.EmbedMolecule(molecule, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(molecule)

    # Get the number of atoms, and the coordinates
    num_atoms = molecule.GetNumAtoms()
    atoms = [molecule.GetAtomWithIdx(i).GetSymbol() for i in range(num_atoms)]
    conf = molecule.GetConformer()

    # Construct XYZ content
    xyz_content = f"{num_atoms}\n"
    xyz_content += "SMILES molecule converted to XYZ format\n"
    for atom, symbol in enumerate(atoms):
        pos = conf.GetAtomPosition(atom)
        xyz_content += f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n"

    return xyz_content

st.set_page_config(layout="wide")

def main():
    st.title('Molecular Data Viewer')
    st.markdown("""
        This application allows users to visualize different aspects of molecular data.
        Use the 'Select Row Index' to choose different compounds from the loaded dataset. 
        The app displays:
        - The **Molecular Structure** of the selected compound.
        - The **Mass Spectra**, providing insights into the compound's composition.
        - A **Sunburst Plot**, showing the hierarchical classification of the compound.
        """)
    
    
    # Load data
    merged_df = load_data()  # Assume load_data is defined to load your dataset
    row_index = st.number_input('Select Row Index', min_value=1, max_value=len(merged_df)-1, value=1, step=1)
    compound_name = merged_df['name'][row_index]
    smiles = merged_df['SMILES'][row_index]  # Correct to use dynamic row index
    xyz_data = smiles_to_xyz(smiles)  # Assume smiles_to_xyz is defined to convert SMILES to XYZ

    # Create two major columns
    col1,col2,col3= st.columns([1,1.5,2])

    # Within the left column, create two rows for the 3D rendering and the static image
    with col1:
        st.write(f"Molecular Structure for {compound_name}:")
        # Set up and display the 3D molecule viewer
        xyzview = py3Dmol.view(width=350, height=350)
        xyzview.addModel(xyz_data, 'xyz')
        xyzview.setStyle({'stick': {}})
        xyzview.zoomTo()
        showmol(xyzview, height=350, width=350)

        # Display static image
        mol_content = merged_df['molFile'][row_index]
        img = display_molecular_structure(mol_content)  # Assume this function returns an image object
        if img:
            st.image(img, width=350)

    # Within the right column, display mass spectra and sunburst plot
    with col2:
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        
        
        st.write("Mass Spectra:")
        fig = display_mass_spectrum(merged_df, row_index)  # Assume this function returns a matplotlib figure
        st.pyplot(fig)
        
    with col3:
        st.write("Sunburst Plot:")
        classification_series = merged_df.at[row_index, 'classification']
        classification_df = create_classification_df(classification_series)  # Assume this function is defined
        sunburst_fig = create_sunburst(classification_df, compound_name)  # Assume this function returns a plotly figure
        st.plotly_chart(sunburst_fig)

if __name__ == "__main__":
    main()
