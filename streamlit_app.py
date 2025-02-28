import streamlit as st
import pandas as pd
import difflib

st.set_page_config(
    page_title="Enhanced vs Original Data Dashboard",
    layout="wide"
)

# Load datasets
@st.cache_data
def load_data():
    enhanced_df = pd.read_csv("data/Enhanced_Results.csv")
    original_df = pd.read_excel("data/eComMasterDataResultsUpwork.xlsx")

    # Explicitly ensure SKU columns are strings
    enhanced_df['SKU'] = enhanced_df['SKU'].astype(str)
    original_df['SKU'] = original_df['SKU'].astype(str)

    return enhanced_df, original_df

enhanced_df, original_df = load_data()

st.title("🚀 Enhanced Data vs Original Data Explorer")

# Ensure only SKUs present in both datasets are selectable
common_skus = enhanced_df['SKU'][enhanced_df['SKU'].isin(original_df['SKU'])].tolist()
selected_sku = st.selectbox("Select a Product (SKU)", common_skus)

# Safely select rows
enhanced_selected = enhanced_df.loc[enhanced_df['SKU'] == selected_sku].iloc[0]
original_selected = original_df.loc[original_df['SKU'] == selected_sku].iloc[0]

# Display Side-by-Side Data Comparison
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.subheader("📝 Original Data")
    st.json(original_selected.dropna().to_dict())

with col2:
    st.subheader("🖼️ Product Image")
    image_url = enhanced_selected.get('Uploaded URL', '')
    if pd.notna(image_url) and image_url:
        st.image(image_url, use_container_width=True)
        st.caption(image_url)
    else:
        st.warning("No image available.")

with col3:
    st.subheader("✨ Enhanced Data")
    st.json(enhanced_selected.dropna().to_dict())

# Display Unified Diff
st.markdown("---")
st.subheader("🔍 Differences (Unified Diff)")

original_json = original_selected.dropna().to_json(indent=2).splitlines()
enhanced_json = enhanced_selected.dropna().to_json(indent=2).splitlines()

diff = difflib.unified_diff(
    original_json,
    enhanced_json,
    fromfile='Original',
    tofile='Enhanced',
    lineterm=''
)
diff_text = '\n'.join(diff)

st.code(diff_text, language='diff')

# Optional: Expanders for complete datasets
st.markdown("---")
with st.expander("📊 Entire Enhanced Dataset"):
    st.dataframe(enhanced_df)

with st.expander("📁 Entire Original Dataset"):
    st.dataframe(original_df)