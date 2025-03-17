import streamlit as st
import pandas as pd
import difflib

st.set_page_config(
    page_title="Enhanced vs Original Data Dashboard",
    layout="wide"
)


# Load datasets: Enhanced and Original data remain unchanged
@st.cache_data
def load_data():
    # Load enhanced data from CSV
    enhanced_df = pd.read_csv("data/Enhanced_Results.csv")
    # Load original data from Excel
    original_df = pd.read_excel("data/eComMasterDataResultsUpwork.xlsx", sheet_name='eComMasterDataResults', engine='openpyxl')

    # Ensure SKU columns are strings
    enhanced_df['SKU'] = enhanced_df['SKU'].astype(str)
    original_df['SKU'] = original_df['SKU'].astype(str)

    # Optional: Convert entire DataFrames to string for display to avoid pyarrow issues
    enhanced_df = enhanced_df.astype(str)
    original_df = original_df.astype(str)

    return enhanced_df, original_df


enhanced_df, original_df = load_data()


# Load the new CSV file with SKU and image links
@st.cache_data
def load_images():
    images_df = pd.read_csv("data/image_links.csv")
    images_df['SKU'] = images_df['SKU'].astype(str)
    return images_df


images_df = load_images()

st.title("🚀 Enhanced Data vs Original Data Explorer")

# Ensure only SKUs present in both datasets are selectable
common_skus = enhanced_df['SKU'][enhanced_df['SKU'].isin(original_df['SKU'])].tolist()
selected_sku = st.selectbox("Select a Product (SKU)", common_skus)

# Safely select rows using try/except in case of errors
try:
    enhanced_selected = enhanced_df.loc[enhanced_df['SKU'] == selected_sku].iloc[0]
    original_selected = original_df.loc[original_df['SKU'] == selected_sku].iloc[0]
except IndexError:
    st.error("Selected SKU not found in both datasets.")
    st.stop()

# Display side-by-side data comparison
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.subheader("📝 Original Data")
    original_data_df = pd.DataFrame([original_selected.dropna().to_dict()]).T
    st.dataframe(original_data_df, height=400)

with col2:
    st.subheader("🖼️ Product Image")
    # Retrieve image URLs for the selected SKU from the new images_df
    images_row = images_df[images_df['SKU'] == selected_sku]
    if images_row.empty:
        st.warning("No image available.")
    else:
        image_urls = []
        for col in ['Cloudinary_1', 'Cloudinary_2', 'Cloudinary_3', 'Cloudinary_4']:
            if col in images_row.columns:
                url = images_row.iloc[0][col]
                if pd.notna(url) and str(url).strip() != '' and str(url).lower() != 'nan':
                    image_urls.append(url)
        if image_urls:
            st.image(image_urls, use_container_width=True)
            st.caption(", ".join(image_urls))
        else:
            st.warning("No image available.")

with col3:
    st.subheader("✨ Enhanced Data")
    enhanced_data_df = pd.DataFrame([enhanced_selected.dropna().to_dict()]).T
    st.dataframe(enhanced_data_df, height=400)

# Display Unified Diff
st.markdown("---")
st.subheader("🔍 Differences (Unified Diff)")

original_json = original_selected.dropna().to_json(indent=2).splitlines()
enhanced_json = enhanced_selected.dropna().to_json(indent=2).splitlines()

diff = list(difflib.unified_diff(
    original_json,
    enhanced_json,
    fromfile='Original',
    tofile='Enhanced',
    lineterm=''
))
if diff:
    diff_text = '\n'.join(diff)
    st.code(diff_text, language='diff')
else:
    st.info("No differences detected between the original and enhanced data.")

# Optional: Expanders for complete datasets
st.markdown("---")
with st.expander("📊 Entire Enhanced Dataset"):
    st.dataframe(enhanced_df)

with st.expander("📁 Entire Original Dataset"):
    st.dataframe(original_df)