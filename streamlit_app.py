import streamlit as st
import pandas as pd
import difflib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Enhanced vs Original Data Dashboard",
    layout="wide"
)


# Load datasets: Enhanced and Original data remain unchanged.
@st.cache_data
def load_data():
    # Load enhanced data from CSV
    enhanced_df = pd.read_csv("data/Enhanced_Results.csv")
    # Load original data from Excel (using sheet name and engine if needed)
    original_df = pd.read_excel("data/eComMasterDataResultsUpwork.xlsx", sheet_name='eComMasterDataResults',
                                engine='openpyxl')

    # Ensure SKU columns are strings
    enhanced_df['SKU'] = enhanced_df['SKU'].astype(str)
    original_df['SKU'] = original_df['SKU'].astype(str)

    # Create display versions (converted to string to avoid pyarrow issues)
    enhanced_df_disp = enhanced_df.astype(str)
    original_df_disp = original_df.astype(str)

    return enhanced_df_disp, original_df_disp, enhanced_df, original_df


enhanced_df_disp, original_df_disp, enhanced_df_raw, original_df_raw = load_data()


# Load the new CSV file with SKU and image links
@st.cache_data
def load_images():
    images_df = pd.read_csv("data/image_links.csv")
    images_df['SKU'] = images_df['SKU'].astype(str)
    return images_df


images_df = load_images()

# Create a tabbed layout for the app
tabs = st.tabs(["Overview", "Missing Values & Statistics", "Visualizations", "Data Chat"])

###########################################
# Overview Tab
###########################################
with tabs[0]:
    st.title("🚀 Enhanced Data vs Original Data Explorer")

    # Ensure only SKUs present in both datasets are selectable
    common_skus = enhanced_df_disp['SKU'][enhanced_df_disp['SKU'].isin(original_df_disp['SKU'])].tolist()
    selected_sku = st.selectbox("Select a Product (SKU)", common_skus)

    # Safely select rows using try/except in case of errors
    try:
        enhanced_selected = enhanced_df_disp.loc[enhanced_df_disp['SKU'] == selected_sku].iloc[0]
        original_selected = original_df_disp.loc[original_df_disp['SKU'] == selected_sku].iloc[0]
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

    st.markdown("---")
    with st.expander("📊 Entire Enhanced Dataset"):
        st.dataframe(enhanced_df_disp)
    with st.expander("📁 Entire Original Dataset"):
        st.dataframe(original_df_disp)

    # Download buttons for the displayed data
    csv_enhanced = enhanced_df_disp.to_csv(index=False).encode('utf-8')
    csv_original = original_df_disp.to_csv(index=False).encode('utf-8')
    st.download_button("Download Enhanced Data", csv_enhanced, "Enhanced_Data.csv", "text/csv")
    st.download_button("Download Original Data", csv_original, "Original_Data.csv", "text/csv")

###########################################
# Missing Values & Statistics Tab
###########################################
with tabs[1]:
    st.title("Missing Values & Statistics")
    st.subheader("Missing Values Analysis")


    def missing_stats(df):
        missing_count = df.isnull().sum()
        total = len(df)
        missing_percent = (missing_count / total * 100).round(2)
        stats_df = pd.DataFrame({
            'Missing Count': missing_count,
            'Missing %': missing_percent
        })
        return stats_df


    st.markdown("**Original Dataset Missing Values**")
    missing_original = missing_stats(original_df_raw)
    st.dataframe(missing_original)

    st.markdown("**Enhanced Dataset Missing Values**")
    missing_enhanced = missing_stats(enhanced_df_raw)
    st.dataframe(missing_enhanced)

    # Combined view for side-by-side comparison
    combined = pd.concat([missing_original, missing_enhanced], axis=1, keys=["Original", "Enhanced"])
    st.markdown("**Combined Missing Values Comparison**")
    st.dataframe(combined)

    # Bar charts visualization
    st.markdown("**Bar Chart: Missing Values (Original)**")
    st.bar_chart(missing_original['Missing Count'])
    st.markdown("**Bar Chart: Missing Values (Enhanced)**")
    st.bar_chart(missing_enhanced['Missing Count'])

###########################################
# Visualizations Tab
###########################################
with tabs[2]:
    st.title("Visualizations")
    st.subheader("Select Dataset and Column for Visualization")

    dataset_choice = st.selectbox("Choose Dataset", ["Original", "Enhanced"])
    if dataset_choice == "Original":
        df_vis = original_df_raw
    else:
        df_vis = enhanced_df_raw

    # Let user choose a column from the dataset
    column_choice = st.selectbox("Select Column", df_vis.columns.tolist())
    chart_type = st.selectbox("Select Chart Type", ["Histogram", "Box Plot"])

    # Try converting the selected column to numeric values
    try:
        data_series = pd.to_numeric(df_vis[column_choice], errors='coerce').dropna()
        if data_series.empty:
            st.warning("No numeric data available for the selected column.")
        else:
            if chart_type == "Histogram":
                fig, ax = plt.subplots()
                ax.hist(data_series, bins=20, color='skyblue', edgecolor='black')
                ax.set_title(f'Histogram of {column_choice} ({dataset_choice})')
                ax.set_xlabel(column_choice)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
            elif chart_type == "Box Plot":
                fig, ax = plt.subplots()
                ax.boxplot(data_series, vert=False)
                ax.set_title(f'Box Plot of {column_choice} ({dataset_choice})')
                st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating plot: {e}")

###########################################
# Data Chat Tab
###########################################
with tabs[3]:
    st.title("Data Chat (Beta)")
    st.markdown("Ask questions about the data. (This feature is under development.)")
    user_query = st.text_input("Enter your query here:")
    if st.button("Submit Query"):
        # For now, we provide a placeholder response.
        st.info("This feature will allow you to chat with the data using OpenAI. Stay tuned!")