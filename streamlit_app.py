import streamlit as st
import pandas as pd
import difflib
import matplotlib.pyplot as plt
import json
import base64
import os
import requests

# ========= NEW IMPORTS FOR OPENAI AND STREAMING ========== #
import openai
from openai import OpenAI

def setup_openai():
    """
    Loads your OpenAI API key and returns an OpenAI client instance.
    In production, it's best to store your key in st.secrets or env variables.
    """
    api_key = st.secrets["OPENAI_API_KEY"]
    if not api_key:
        raise ValueError("API key not found.")
    openai.api_key = api_key
    return OpenAI(api_key=api_key)

client = setup_openai()

# Your known vector store ID (replace with your actual vector store ID if needed)
VECTOR_STORE_ID = st.secrets.get("VECTOR_ID", "YOUR_VECTOR_STORE_ID")

st.set_page_config(
    page_title="Enhanced vs Original Data Dashboard",
    layout="wide"
)

# -------------------------------------------------------------------------
#                 ORIGINAL DATA-LOADING FUNCTIONS
# -------------------------------------------------------------------------
@st.cache_data
def load_data():
    enhanced_df = pd.read_excel('data/Updated_and_Fixed_03_18_25.xlsx', engine='openpyxl')
    original_df = pd.read_excel(
        "data/eComMasterDataResultsUpwork.xlsx",
        sheet_name='eComMasterDataResults',
        engine='openpyxl'
    )
    # Ensure SKU columns are strings
    enhanced_df['SKU'] = enhanced_df['SKU'].astype(str)
    original_df['SKU'] = original_df['SKU'].astype(str)

    # Create display versions for convenience
    enhanced_df_disp = enhanced_df.astype(str)
    original_df_disp = original_df.astype(str)

    return enhanced_df_disp, original_df_disp, enhanced_df, original_df

enhanced_df_disp, original_df_disp, enhanced_df_raw, original_df_raw = load_data()

@st.cache_data
def load_images():
    images_df = pd.read_csv("data/image_links.csv")
    images_df['SKU'] = images_df['SKU'].astype(str)
    return images_df

images_df = load_images()

# -------------------------------------------------------------------------
#        ORIGINAL JSON-SCHEMA STREAMING RESPONSE FUNCTION
# -------------------------------------------------------------------------
def retrieve_and_respond_streaming(query: str) -> dict:
    """
    Use the new OpenAI Responses API in streaming mode, enforcing a JSON Schema
    to ensure well-formed product data. Return the parsed JSON as a dictionary.
    (Original example method that returns a strictly valid JSON object.)
    """
    streamed_text = ""

    # This is an example usage of a model + JSON schema
    response = client.responses.create(
        model="gpt-4o-2024-08-06",   # or whichever model you prefer
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are an e-commerce assistant for an art retailer called Classy Art. "
                            "Generate strictly valid JSON that adheres to the provided JSON schema. "
                            "You have access to a file search tool for searching products by SKU or subject matter. "
                            "If the user is searching for a product image, provide a 'Main Image File' or additional "
                            "'Image N File' fields in the JSON if relevant."
                        )
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": query}
                ]
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "product_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "Internal ID": { "type": "number" },
                        "Inactive": { "type": "string" },
                        "SKU": { "type": "string" },
                        "Size": { "type": "string" },
                        "Product Name": { "type": "string" },
                        "Style": { "type": "string" },
                        "Category": { "type": "string" },
                        "Subject Matter Primary": { "type": "string" },
                        "Subject Matter 2": { "type": "string" },
                        "Subject Matter 3": { "type": "string" },
                        "Occasion": { "type": "string" },
                        "Predominant Color": { "type": "string" },
                        "Orientation": { "type": "string" },
                        "Collection Name": { "type": "string" },
                        "Country of Origin": { "type": "string" },
                        "UPC Code": { "type": "number" },
                        "eCom Price": { "type": "number" },
                        "MAP Price (Sale Price)": { "type": "number" },
                        "MAP Price": { "type": "number" },
                        "Full Retail Price (MSRP)": { "type": "number" },
                        "Item Width (in)": { "type": "number" },
                        "Item Depth (in)": { "type": "number" },
                        "Item Height (in)": { "type": "number" },
                        "Weight": { "type": "number" },
                        "eCom Carton Width (In)": { "type": "number" },
                        "eCom Carton Depth (In)": { "type": "number" },
                        "eCom Carton Height (In)": { "type": "number" },
                        "eCom Carton Weight (lbs)": { "type": "number" },
                        "Page Romance": { "type": "string" },
                        "Frame Color": { "type": "string" },
                        "Frame & Color": { "type": "string" },
                        "Artist": { "type": "string" },
                        "Artist Bio": { "type": "string" },
                        "Country of Manufacture": { "type": "string" },
                        "Marketing Copy": { "type": "string" },
                        "Item Feature 1": { "type": "string" },
                        "Item Feature 2": { "type": "string" },
                        "Item Feature 3": { "type": "string" },
                        "Item Feature 4": { "type": "string" },
                        "Item Feature 5": { "type": "string" },
                        "Keywords/MetaData": { "type": "string" },
                        "Care & Cleaning": { "type": "string" },
                        "Warranty": { "type": "string" },
                        "California Proposition 65 Warning Required": { "type": "string" },
                        "Main Image File": { "type": "string" },
                        "Image 1 File": { "type": "string" },
                        "Image 2 File": { "type": "string" },
                        "Image 3 File": { "type": "string" },
                        "Image 4 File": { "type": "string" },
                        "Image 5 File": { "type": "string" }
                    },
                    "additionalProperties": False,
                    "required": [
                        "Internal ID", "Inactive", "SKU", "Size", "Product Name", "Style",
                        "Category", "Subject Matter Primary", "Subject Matter 2", "Subject Matter 3",
                        "Occasion", "Predominant Color", "Orientation", "Collection Name", "Country of Origin",
                        "UPC Code", "eCom Price", "MAP Price (Sale Price)", "MAP Price",
                        "Full Retail Price (MSRP)", "Item Width (in)", "Item Depth (in)", "Item Height (in)",
                        "Weight", "eCom Carton Width (In)", "eCom Carton Depth (In)", "eCom Carton Height (In)",
                        "eCom Carton Weight (lbs)", "Page Romance", "Frame Color", "Frame & Color", "Artist",
                        "Artist Bio", "Country of Manufacture", "Marketing Copy", "Item Feature 1",
                        "Item Feature 2", "Item Feature 3", "Item Feature 4", "Item Feature 5",
                        "Keywords/MetaData", "Care & Cleaning", "Warranty",
                        "California Proposition 65 Warning Required", "Main Image File",
                        "Image 1 File", "Image 2 File", "Image 3 File", "Image 4 File", "Image 5 File"
                    ]
                }
            }
        },
        tools=[{
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID],
        }],
        max_output_tokens=2048,
        temperature=1,
        top_p=1,
        stream=True
    )

    for event in response:
        if event.type == "response.output_text.delta":
            streamed_text += event.delta
        elif event.type == "response.completed":
            break
        elif event.type == "response.refusal.delta":
            streamed_text += event.delta

    product_data = {}
    try:
        product_data = json.loads(streamed_text)
    except json.JSONDecodeError as e:
        product_data = {"error": f"Failed to parse JSON: {str(e)}"}

    return product_data


# -------------------------------------------------------------------------
#              NEW MULTI-TURN CHAT (STREAMING) EXAMPLE
# -------------------------------------------------------------------------
# We'll maintain a conversation state in st.session_state.
# We'll provide a function for streaming the model's textual response.

def generate_multiturn_response(conversation_history):
    """
    Given a conversation history list of messages (role + content),
    call the model in streaming mode and return the final text.
    For brevity, we won't handle function calls here,
    but you could add them if desired.
    """
    partial_text = ""
    response_stream = client.responses.create(
        model="gpt-4o-2024-08-06",  # or any model that supports streaming
        input=conversation_history,
        # If you want to add function calling or file_search:
        # tools=[{...}, {...}],
        # tool_choice="auto",
        stream=True,
        max_output_tokens=1024,
        temperature=0.7,
    )

    final_text = ""
    for event in response_stream:
        if event.type == "response.output_text.delta":
            partial_text += event.delta
            # If you want real-time UI updates:
            # we might do something like st.session_state.partial_text = partial_text
            # and st.experimental_rerun or a placeholder
        elif event.type == "response.completed":
            final_text = partial_text
            break
        elif event.type == "response.refusal.delta":
            partial_text += event.delta
        elif event.type == "response.error":
            st.error(f"Error: {event.error}")

    return final_text


# -------------------------------------------------------------------------
#                       STREAMLIT TABS
# -------------------------------------------------------------------------
tabs = st.tabs(["Overview", "Missing Values & Statistics", "Visualizations", "Data Chat"])

# ==============================
# TAB 1: OVERVIEW
# ==============================
with tabs[0]:
    st.title("🚀 Enhanced vs Original Data Dashboard")

    # Ensure only SKUs present in both datasets are selectable
    common_skus = enhanced_df_disp['SKU'][enhanced_df_disp['SKU'].isin(original_df_disp['SKU'])].tolist()
    selected_sku = st.selectbox("Select a Product (SKU)", common_skus)

    # Safely select rows using try/except
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
        st.dataframe(
            original_data_df,
            column_config={
                "json": st.column_config.JsonColumn(
                    "JSON Data",
                    help="JSON strings or objects",
                    width="large",
                    height=600,
                ),
            },
            hide_index=True,
        )

    with col2:
        st.subheader("🖼️ Product Image")
        images_row = images_df[images_df['SKU'] == selected_sku]
        if images_row.empty:
            st.warning("No image available.")
        else:
            image_urls = []
            for col_url in ['Cloudinary_1', 'Cloudinary_2', 'Cloudinary_3', 'Cloudinary_4']:
                if col_url in images_row.columns:
                    url = images_row.iloc[0][col_url]
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

    import json
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

    # Download buttons
    csv_enhanced = enhanced_df_disp.to_csv(index=False).encode('utf-8')
    csv_original = original_df_disp.to_csv(index=False).encode('utf-8')
    st.download_button("Download Enhanced Data", csv_enhanced, "Enhanced_Data.csv", "text/csv")
    st.download_button("Download Original Data", csv_original, "Original_Data.csv", "text/csv")


# ==============================
# TAB 2: MISSING VALUES
# ==============================
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

    combined = pd.concat([missing_original, missing_enhanced], axis=1, keys=["Original", "Enhanced"])
    st.markdown("**Combined Missing Values Comparison**")
    st.dataframe(combined)

    st.markdown("**Bar Chart: Missing Values (Original)**")
    st.bar_chart(missing_original['Missing Count'])
    st.markdown("**Bar Chart: Missing Values (Enhanced)**")
    st.bar_chart(missing_enhanced['Missing Count'])


# ==============================
# TAB 3: VISUALIZATIONS
# ==============================
with tabs[2]:
    st.title("Visualizations")
    st.subheader("Select Dataset and Column for Visualization")

    dataset_choice = st.selectbox("Choose Dataset", ["Original", "Enhanced"])
    df_vis = original_df_raw if dataset_choice == "Original" else enhanced_df_raw

    column_choice = st.selectbox("Select Column", df_vis.columns.tolist())
    chart_type = st.selectbox("Select Chart Type", ["Histogram", "Box Plot"])

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


# ==============================
# TAB 4: DATA CHAT
# ==============================
with tabs[3]:
    st.header("Data Chat (Two Methods)")

    # We provide two ways to chat:
    #  1. The original "Strict JSON Output" method (retrieve_and_respond_streaming)
    #  2. A new multi-turn, streaming text method

    chat_mode = st.radio("Choose a Chat Method:", ["Strict JSON (Original)", "Multi-turn Streaming Chat"])

    if chat_mode == "Strict JSON (Original)":
        st.subheader("Method #1: Strict JSON Retrieval w/ Vector Store")
        user_query = st.text_input("Enter your query (for product data retrieval):")
        if st.button("Submit Query"):
            with st.spinner("Querying vector store & streaming response..."):
                final_dict = retrieve_and_respond_streaming(user_query)
            st.success("Done!")
            st.subheader("Parsed JSON Output:")
            st.json(final_dict)

            # Display matching images if "SKU" is in final_dict
            sku = final_dict.get("SKU")
            if sku:
                st.subheader("Related Product Images:")
                images_row = images_df[images_df["SKU"] == sku]
                if images_row.empty:
                    st.warning("No Cloudinary images found for SKU: " + sku)
                else:
                    cloud_urls = []
                    for col_url in ['Cloudinary_1','Cloudinary_2','Cloudinary_3','Cloudinary_4']:
                        if col_url in images_row.columns:
                            link = images_row.iloc[0][col_url]
                            if pd.notna(link) and str(link).strip():
                                cloud_urls.append(link)
                    if cloud_urls:
                        for link in cloud_urls:
                            st.image(link, caption=f"Image for SKU {sku}", width=300)
                    else:
                        st.warning("No image links available.")
        else:
            st.info("Enter a query to get strictly valid JSON output.")

    else:
        st.subheader("Method #2: Multi-turn Chat with Streaming")
        # We'll store the conversation in session_state:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("Enter your message:")
        if st.button("Send"):
            if user_input.strip():
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                # Generate response
                with st.spinner("Assistant is typing..."):
                    assistant_reply = generate_multiturn_response(st.session_state.chat_history)
                # Add assistant message
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

        # Display the conversation
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**User:** {msg['content']}")
            else:
                st.markdown(f"**Assistant:** {msg['content']}")

        st.write("---")
        st.write("Try chatting about your data, e.g. 'How many SKUs do we have?' or general requests.")