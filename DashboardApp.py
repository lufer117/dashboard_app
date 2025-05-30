import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import plotly.express as px
import plotly.graph_objects as go
import ast
import markdown
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import os 
import google.generativeai as genai 

# --- CONFIGURACIÓN DE CONTRASEÑA ---
# CORRECT_PASSWORD = st.secrets.get("app_password", "streamlit123")

# --- GATE DE ACCESO USANDO SESSION STATE ---
# def password_gate():
#     if st.session_state.get("authenticated", False):
#         return True

#     st.title("🔒 App Access")
#     user_input = st.text_input("Enter password:", type="password")

#     if user_input == CORRECT_PASSWORD:
#         st.session_state["authenticated"] = True
#         st.success("Access granted")
#         st.rerun()  # Recarga para limpiar input y continuar
#     elif user_input:
#         st.error("Incorrect password")
#     else:
#         st.info("Please enter the password to continue")

#     return False

# # --- BLOQUEA EL RESTO DE LA APP SI NO ACCEDIÓ ---
# if not password_gate():
#     st.stop()


# Control de funciones experimentales
ENABLE_GEMINI = False  # ← Cambia a True cuando tengas billing/autorización

def gemini_analysis_button(
    location_name,
    selected_product,
    df_metric_by_cooler,
    metric_type="pull",
    model_enabled=True,
    button_label=None
):
    if "gemini_used" not in st.session_state:
        st.session_state.gemini_used = {}

    key_suffix = f"{metric_type}_{selected_product}"
    button_key = f"gemini_{key_suffix}"

    if not button_label:
        label_map = {
            "pull": "🔍 Analyze with Gemini",
            "fill": "🔍 Analyze with Gemini",
            "oos": "🔍 Analyze with Gemini",
            "restock": "🔍 Analyze with Gemini",
            "velocity": "🔍 Analyze with Gemini",
            "index": "🔍 Analyze with Gemini"
        }
        button_label = label_map.get(metric_type, "🔍 Analyze with Gemini")

    if selected_product not in st.session_state.gemini_used:
        if st.button(button_label, key=button_key):
            if model_enabled:
                try:
                    from google.generativeai import configure, GenerativeModel
                    import os

                    # api_key = st.secrets["api_keys"]["gemini"] # to deploy in streamlit
                    api_key = os.getenv("GEMINI_API_KEY") 
                    configure(api_key=api_key)
                    model = GenerativeModel("gemini-2.0-flash")

                    prompt = f"""
                    Analyze the {metric_type} performance of this SKU.

                    Location: {location_name}
                    Product: {selected_product}

                    Total {metric_type.title()}s by Cooler:
                    {df_metric_by_cooler.to_string(index=False)}

                    Short performance summarize, detect anomalies, patterns, and suggest operational actions.
                    """

                    response = model.generate_content(prompt)
                    result = response.text

                    st.markdown(f"**Gemini Insight ({metric_type.title()}):**")
                    st.markdown(f"""
                    <div style="text-align: justify;">
                    {markdown.markdown(result)}
                    </div>
                    """, unsafe_allow_html=True)

                    st.session_state.gemini_used[selected_product] = True

                except Exception as e:
                    st.error("Gemini model is not responding or billing is not active.")
                    st.exception(e)
            else:
                st.info("This feature will be available soon. Gemini integration is not yet active.")
    else:
        st.info(f"You already ran Gemini analysis for this SKU ({metric_type}) in this session.")

def gemini_analysis_by_cooler(
    location_name,
    selected_cooler,
    df_metric_by_product,
    metric_type="pull",
    model_enabled=True,
    button_label=None
):
    import streamlit as st
    import markdown

    if "gemini_used" not in st.session_state:
        st.session_state.gemini_used = {}

    key_suffix = f"{metric_type}_{selected_cooler}"
    button_key = f"gemini_cooler_{key_suffix}"

    if not button_label:
        label_map = {
            "pull": "🔍 Analyze with Gemini",
            "fill": "🔍 Analyze with Gemini",
            "oos": "🔍 Analyze with Gemini",
            "restock": "🔍 Analyze with Gemini",
            "velocity": "🔍 Analyze with Gemini",
            "index": "🔍 Analyze with Gemini"
        }
        button_label = label_map.get(metric_type, "🔍 Analyze with Gemini")

    if selected_cooler not in st.session_state.gemini_used:
        if st.button(button_label, key=button_key):
            if model_enabled:
                try:
                    from google.generativeai import configure, GenerativeModel

                    api_key = st.secrets["api_keys"]["gemini"]
                    configure(api_key=api_key)
                    model = GenerativeModel("gemini-2.0-flash")

                    prompt = f"""
                    Analyze the {metric_type} performance by SKU for this cooler.

                    Location: {location_name}
                    Cooler: {selected_cooler}

                    Total {metric_type.title()}s by Product:
                    {df_metric_by_product.to_string(index=False)}

                    Short performance summary, detect anomalies or patterns, and suggest operational actions.
                    """

                    response = model.generate_content(prompt)
                    result = response.text

                    st.markdown(f"**Gemini Insight ({metric_type.title()}):**")
                    st.markdown(f"""
                    <div style="text-align: justify;">
                    {markdown.markdown(result)}
                    </div>
                    """, unsafe_allow_html=True)

                    st.session_state.gemini_used[selected_cooler] = True

                except Exception as e:
                    st.error("Gemini model is not responding or billing is not active.")
                    st.exception(e)
            else:
                st.info("This feature will be available soon. Gemini integration is not yet active.")
    else:
        st.info(f"You already ran Gemini analysis for this Cooler ({metric_type}) in this session.")


#@st.cache_data
def load_all_data():
    pulls = pd.read_csv("data/pulls_summary_full.csv")
    overview_table = pd.read_csv("data/overview_final_stats_table.csv")
    overview_stats = pd.read_csv("data/overview_final_stats.csv")
    loc_restock_sum= pd.read_csv("data/loc_restocking_summary.csv")
    restock_sum = pd.read_csv("data/restocking_summary.csv")
    loc_oos_sum= pd.read_csv("data/loc_oos_summary.csv")
    oos_restock_sum= pd.read_csv("data/oos_restocking_summary.csv")
    loc_indexes= pd.read_csv("data/index_by_location.csv")
    index_sum= pd.read_csv("data/index_sum.csv")
    
    
    # velocity_hour = pd.read_csv("data/velocity_by_hour.csv")
    # fills = pd.read_csv("data/fills.csv")
    # oos = pd.read_csv("data/oos.csv")
    # temporal = pd.read_csv("data/temporal_patterns.csv")
    # indexes = pd.read_csv("data/indexes.csv")

    # Load the overview summary
    # with open("data/overview_table_summary.json", encoding="utf-8") as f:
    #     overview = json.load(f)

    location_ids = pulls[["Location", "Location Id"]].drop_duplicates().set_index("Location")["Location Id"].to_dict()

    return {
        "overview_table": overview_table,
        "pulls": pulls,
        "overview_stats": overview_stats,
        "loc_restock_sum": loc_restock_sum,
        "restock_sum": restock_sum,
        "loc_oos_sum": loc_oos_sum,
        "oos_restock_sum": oos_restock_sum,
        "loc_indexes": loc_indexes,
        "index_sum": index_sum,
        "location_ids": location_ids
    }

data = load_all_data()


@st.cache_data
def load_general_insights():
    with open("insights/insights_general_lorem.json", "r", encoding="utf-8") as f:
        general_insights = json.load(f)
    return general_insights

general_insights = load_general_insights()

@st.cache_data
def load_insights():
    with open("insights/insights_gem_lorem.json", "r", encoding="utf-8") as f:
        insights = json.load(f)
    return insights

# Cargar los insights
insights = load_insights()

@st.cache_data
def load_key_conclusions():
    with open("insights/insights_key_conclusions_lorem.json", "r", encoding="utf-8") as f:
        general_insights = json.load(f)
    return general_insights

insights_key_conclusions = load_key_conclusions()

# # Obtener todas las locaciones y normalizar para comparación
# ALL_LOCATIONS = list(data["location_ids"].keys())
# normalized_locations = {loc.lower(): loc for loc in ALL_LOCATIONS}

# # Input manual
# user_input = st.sidebar.text_input("Enter location name(s), separated by comma:")

# if user_input:
#     inputs = [i.strip().lower() for i in user_input.split(",")]
#     matched = [normalized_locations[i] for i in inputs if i in normalized_locations]
#     not_matched = [i for i in inputs if i not in normalized_locations]

#     if matched:
#         ACTIVE_LOCATIONS = matched
#         if not_matched:
#             st.sidebar.warning(f"Some locations were not recognized: {', '.join(not_matched)}")
#     else:
#         st.sidebar.error("None of the entered locations were recognized.")
#         st.stop()
# else:
#     st.sidebar.info("Please enter at least one location name.")
#     st.stop()


# Banner

def presentation():
    st.image("assets/presentation_banner.png", use_container_width=True)
    #escribir marckdown
    st.markdown("""<div style="text-align: justify"> 
                This is a generic version based on an app developed for product analysis at points of sale. 
                The data has been modified or generated for illustrative purposes
    </div>
    """, unsafe_allow_html=True)

def dataset():
    st.markdown("<h2 style='font-weight:bold;'>DATA SET</h2>", unsafe_allow_html=True)

    st.markdown("""
    The company has installed 10 coolers across two locations. The idea is to identify what the consumption of coolers product is.
    The following sections cover a general analysis and an analysis of each location.
    """)

    # Highlighted info box
    st.markdown("""
    <div style="border:1px solid #ccc; padding:10px; border-radius:6px; background-color:#f9f9f9; margin-top:15px;">
        <ul style="list-style-type: '◾'; padding-left: 20px;">
            <li><b>Time Period:</b> 01 Oct 2024 - 01 May 2025</li>
            <li><b>Using all week data:</b> Monday to Sunday, 24h</li>
            <li><b>Time zone:</b> American/New York Zone</li>
            <li><b>NOTE:</b> This analysis does not takes into account the records reported under Product ID: -1 </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(" ")

    st.markdown("Data set **'Stock_sense'** contains the following columns:")
    # Dataset name and date range
    
    st.markdown("""
    <div style="
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 16px;
        background-color: #f9f9f9;
        margin-bottom: 20px;
    ">
        <p style="font-family: monospace; color: #333;">
            'Location Local Datetime', 'Location Id', 'Location', 'Deployment Id', 'Deployment',<br>
            'Placement Id', 'Placement', 'Product Id', 'Product', 'Metric', 'Data'
        </p>
        <p style="padding-left: 20px;">
        <b>Metric:</b> Count
        </p>
    </div>
    """, unsafe_allow_html=True)

def methodology():

    tab1, tab2, tab3 = st.tabs(["Data Preparation","Data Exploration", "Data Transformation"])

    with tab1:
        st.markdown("""
        <div style="text-align: justify">        

        <hr>

        <h5> Raw File Consolidation</h5>

        <b>1.1 List and sort CSV files</b><br>
        All individual <code>.csv</code> files containing sensor data were listed from a shared folder and sorted alphabetically to ensure consistent chronological order.<br><br>

        <b>1.2 Chunk-based reading</b><br>
        Each file was read in chunks of 500,000 rows to handle large volumes efficiently and prevent memory overload.<br><br>

        <b>1.3 Merge into a single dataset</b><br>
        All chunks were appended and saved into a unified dataset called <code>merged_stock.csv</code>.

        <hr>

        <h5> Initial Cleaning and Standardization</h5>

        <b>2.1 Handle missing values</b><br>
        Missing values in key columns (<code>Location Id</code>, <code>Deployment Id</code>, <code>Placement Id</code>, <code>Product Id</code>, <code>Data</code>) were filled with <code>-1</code> and cast to <code>Int32</code> format to preserve null handling.<br><br>

        <b>2.2 Save cleaned dataset</b><br>
        The cleaned and type-consistent dataset was saved again under <code>merged_stock.csv</code>.

        <hr>

        <h5> Removal of Consecutive Duplicates (Oscillations)</h5>

        <b>3.1 Sort and group data</b><br>
        The dataset was sorted by <code>Location Local Datetime</code>, <code>Location Id</code>, <code>Deployment Id</code>, <code>Placement Id</code>, and <code>Product Id</code> to ensure correct temporal sequencing within groups.<br><br>

        <b>3.2 Remove consecutive duplicates</b><br>
        For each group, rows where the <code>Data</code> value remained unchanged from the previous record were considered redundant and removed.<br><br>

        <b>3.3 Manage chunk boundaries</b><br>
        The last row of each chunk was preserved and prepended to the next chunk to maintain continuity between chunks.<br><br>

        <b>3.4 Save deduplicated result</b><br>
        The processed dataset, free from consecutive duplicate values, was saved as <code>duplicates_removed_merged_stock.csv</code>.

        <hr>

        <h5> Stock Movement Calculation</h5>
                    
        <b>4.1 Compute <code>Prev_Data</code> and <code>Next_Data</code></b><br>
        Two helper columns were added:<br>
        - <code>Prev_Data</code>: previous value of <code>Data</code><br>
        - <code>Next_Data</code>: next value of <code>Data</code><br><br>
           
        <b>4.1 Compute <code>Data Change</code></b><br>
        A new column <code>Data Change</code> was calculated as the difference between the current and previous <code>Data</code> value within each group.<br><br>
        
        <b>4.3 Classify status of each row</b><br>
        A new column <code>Status</code> was created based on value comparison:<br>
        - <b>Oscillation</b>: if <code>Prev_Data == Next_Data</code><br>
        - <b>Possible Fill</b>: if <code>Prev_Data < Data</code><br>
        - <b>Possible Pull</b>: if <code>Prev_Data > Data</code><br>
        - <b>Stable</b>: otherwise<br><br>

        <b>4.4 Apply logic by group</b><br>
        This classification was applied within each group defined by <code>Product Id</code>, <code>Location Id</code>, <code>Deployment Id</code>, and <code>Placement Id</code>.<br><br>

        <b>4.5 Save classified dataset</b><br>
        The final dataset with movement classification was saved as <code>real_pulls.csv</code>.

        
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        

        <ol style="text-align: justify;">
        <li><b>Dataset Loading:</b> Loaded the <i>real_pulls.csv</i> file containing stock sensor data with 271,509 records and 17 columns.</li>
        
        <li><b>Missing Values Analysis:</b> Examined null values across columns. Notable missing data was identified in the <code>Product</code> column (7674 rows) and in <code>Data Change</code>, <code>Prev_Data</code>, and <code>Next_Data</code> (494 rows each).</li>
        
        <li><b>Null Investigation in "Data Change":</b> Inspected the content of null cells in <code>Data Change</code>. Found non-numeric values like the string "nan" and confirmed these were not malformed characters but actual missing entries.</li>
        
        <li><b>Unique Value Counts:</b> Counted unique values for each column to identify cardinality. Key findings include:
            <ul>
            <li>2 unique locations</li>
            <li>10 unique deployments (coolers)</li>
            <li>84 named products + 7674 unnamed</li>
            <li>4 unique statuses</li>
            </ul>
        </li>
        
        <li><b>Datetime Range Check:</b> Parsed <code>Location Local Datetime</code> and validated the full dataset's date range: from <b>October 1, 2024</b> to <b>May 1, 2025</b>.</li>
        
        <li><b>Object Columns Profiling:</b> Printed all unique values in object-type columns to check for anomalies or encoding issues, including cooler and shelf labels.</li>
        
        <li><b>Invalid Product Records:</b> Identified records with invalid <code>Product Id = -1</code> or <code>Product = NaN</code> as candidates for removal.</li>
        </ol>
        """, unsafe_allow_html=True)




    with tab3:

        # Info box placeholder for later equations

            st.markdown("""
            <div style='text-align: justify'>

        

            1. **Datetime Standardization**
            - Timestamps in the `Location Local Datetime` column were corrected by appending missing microseconds (e.g., `.000000`) to ensure uniform parsing.
            - Timezones were removed while preserving the real clock time, maintaining the original hour context.

            2. **Datetime Parsing & Feature Extraction**
            - Converted datetime strings to `datetime` objects.
            - Extracted useful time-based features including `Date`, `DayOfWeek`, `HourOfDay`, and `Month` to facilitate temporal analysis.

            3. **Cleaning ‘Data Change’ Values**
            - The `Data Change` column, derived from product count variation, was coerced into numeric format and converted to absolute values to represent magnitude.
            - Missing values in `Data Change` for rows labeled as “Stable” in `Status` were filled with 0.

            4. **Removal of Unnamed Products**
            - Entries with `Product Id = -1` and missing `Product` names were excluded.
            - These rows likely represent unidentified or manually moved items without a proper SKU assignment.

            5. **Deployment Name Simplification**
            - The suffix “- Compass Group HQ” was removed from the `Deployment` names to streamline cooler labeling.

            6. **Sorting and Reindexing**
            - The dataset was sorted by `Location`, `Product`, and `Location Local Datetime` to prepare for time series analysis.

            7. **Validation of Final Dataset**
            - The final cleaned dataset was validated for nulls and correct data types, resulting in a clean structure with 263,835 rows.

            </div>
            """, unsafe_allow_html=True)


            st.markdown("""
            <h5>Transformed Data Set Structure:</h5>
            <div style="border:1px solid #ccc; padding:15px; border-radius:8px; background-color:#f9f9f9;">
                <ul>
                    <li><b>Stock_sense:</b> 01/10/2024 – 01/05/2024</li>
                </ul>
                <p style="font-family: monospace;">
                    ['Location Local Datetime', 'Location Id', 'Location', 'Deployment Id', 'Deployment',<br>
                    'Placement Id', 'Placement', 'Product Id', 'Product', 'Metric', 'Data',<br>
                    'Data Change', 'Product Pulls', 'Product Fills', 'Prev_Data', 'Next_Data',<br>
                    'Status', 'Brand', 'Location', 'Month', 'DayOfWeek', 'Hour']
                </p>
                <br>
                <table style="border-collapse: collapse; width: 100%;">
                    <tr>
                        <th style="border: 1px solid #ccc; padding: 6px;">Metric</th>
                        <th style="border: 1px solid #ccc; padding: 6px;">Data Change</th>
                        <th style="border: 1px solid #ccc; padding: 6px;">Status</th>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ccc; padding: 6px;">ProductCount</td>
                        <td style="border: 1px solid #ccc; padding: 6px;">Change count</td>
                        <td style="border: 1px solid #ccc; padding: 6px;">Possible Fill, Possible Pull, Oscillation</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

def metrics_definitions():
    
    st.markdown(r"""
    <div style='text-align: justify'>
                
    <p>
    This section describes the logic used to calculate key performance indicators for cooler performance,
    including product consumption, restocking behavior, and stockout patterns.
    </p>

    ---
    
    <h5> Total Pulls</h5>
    <p> 
    The total number of pull records over the time period was calculated based on the grouping of product, deployment, and location.
    </p>
               
    $\text{Total Pulls} = \sum(\text{Data Change}) \quad \text{where Status = "Possible Pull"}$

    ---
    
    <h5> Product Velocity</h5>
                
    <p>
    Product velocity is defined as the average number of pulls per day, calculated over the entire date range.
    </p>          
    
    - **Velocity (Active Days):** 
                 
        $\text{Velocity}_{\text{active}} = \frac{\text{Total Pulls}}{\text{Days with Data}}$
    
    - **Velocity (Period):**  
                
        $\text{Velocity}_{\text{period}} = \frac{\text{Total Pulls}}{\text{Total Analysis Days}}$

    ---
    
    <h5> Restoking metrics </h5>

    - **Total Restocking Incidents and Total Products Restocked**: The resulting values represent the total restocking actions and the cumulative quantity of products restocked for each location/brand/product
        
        A restocking incident occurs when a product is refilled (indicated by a "possible fill" event in the "Status" column of stock_sense transformed dataset)           

        $\text{Total Restock Incidents} = \text{count(Status = "Possible Fill")}$  
 
                
        $\text{Total Products Restocked} = \sum(\text{Data Change}) \quad \text{where Status = "Possible Fill"}$  
    
    - **Average Time Between Restocks:** This gives the average duration between each restocking incident for each unique location/brand/product.             
    
        $\text{Avg Time Between Restocks} = \text{mean}(\Delta \text{Datetime}_{\text{fills}})$  
    
    - **Restocking Frequency and Average Daily Restocked:** These metrics provide insights into the frequency of restocking incidents and the average number of products restocked per day.
                
        $\text{Restocking Frequency} = \frac{\text{Total Restock Incidents}}{\text{Total Analysis Days}}$  
       
        $\text{Avg Daily Restocked} = \frac{\text{Total Products Restocked}}{\text{Total Analysis Days}}$

    ---
    
    <h5> OOS Incidents</h5>
    
    <p>
    For each location-product pair, events are filtered to include only "Data == 0 and Possible Pull" (representing potential OOS events) and "Possible Fill" (representing restocking events).
    </p
    
    - **Total OOS Incidents and Total OOS Duration:** These metrics provide insights into the frequency and duration of out-of-stock incidents for each unique location/product/deployment.

        $\text{Total OOS Incidents} = \text{count of unique 0 → fill episodes}$  
    
        $\text{Total OOS Duration} = \sum(\text{Fill time} - \text{OOS start time})$  
    
    - **Average Time Between OOS:** This gives the average duration between each OOS incident to the next restocking incident for each unique location/product/deployment. 
                
        $\text{Avg OOS Duration} = \frac{\text{Total OOS Duration}}{\text{Total OOS Incidents}}$  
    
    - **OOS Frequency:** This metric provides insights into the frequency of OOS incidents per day.            
        
        $\text{OOS Frequency} = \frac{\text{Total OOS Incidents}}{\text{Total Analysis Days}}$

    ---
    
    <h5> Indexes</h5>

    - **OOS-to-Restock Ratio:** This metric provides insights into the relationship between OOS incidents and restocking incidents. A higher ratio indicates a greater number of OOS incidents relative to restocking actions.
    
        $\text{OOS-to-Restock Ratio} = \frac{\text{Total OOS Incidents}}{\text{Total Restock Incidents} + 0.01}$  
    
    - **Velocity-to-Restock Ratio:** This metric provides insights into the relationship between product velocity and restocking frequency. A higher ratio indicates a greater number of pulls relative to restocking actions.
        
        $\text{Velocity-to-Restock Ratio} = \frac{\text{Avg Daily Pulls}}{\text{Restocking Frequency (per day)} + 0.01}$  
    
    - **OOS Duration per Fill:** This metric provides insights into the average duration of OOS incidents relative to restocking actions. A higher ratio indicates a longer duration of OOS incidents relative to restocking actions.
        
        $\text{OOS Duration per Fill} = \frac{\text{Total OOS Duration}}{\text{Total Restock Incidents} + 0.01}$  
    
    - **Pulls per Fill:** This metric provides insights into the average number of pulls relative to restocking actions. A higher ratio indicates a greater number of pulls relative to restocking actions.
            
        $\text{Pulls per Fill} = \frac{\text{Total Pulls}}{\text{Total Restock Incidents} + 0.01}$

    </div>
    """, unsafe_allow_html=True)





    # # KEY METRICS: Velocity by location
    # st.markdown("<h2 style='font-weight:bold;'>Metrics</h2>", unsafe_allow_html=True)
    # st.markdown("""
    # <h4>Product velocity:</h4>
    # <ol>
    #     <li>Data is grouped by brand to calculate product velocity per brand.</li>
    #     <li>Velocity is grouped by location using the full date range to count total days.</li>
    #     <li>Calculation: total pulls ÷ total days.</li>
    # </ol>
    # <div style="border: 1px dashed #bbb; padding: 10px; margin-top: 10px;">
    #     <i>Placeholder for velocity formula box</i>
    # </div>
    # """, unsafe_allow_html=True)

    # # Velocity by hour/day
    # st.markdown("""
    # <h4>Product velocity by time of day and day of week:</h4>
    # <ol>
    #     <li>Group data by location, hour, and day of week</li>
    #     <li>Calculate total pulls per hour-day combination</li>
    #     <li>Divide pulls by total days observed</li>
    # </ol>
    # <div style="border: 1px dashed #bbb; padding: 10px;">
    #     <i>Placeholder for hourly velocity formula</i>
    # </div>
    # """, unsafe_allow_html=True)

    # # Restocking
    # st.markdown("""
    # <h4>Restocking incidents:</h4>
    # <ol>
    #     <li>Filtered by 'Possible Fill' events</li>
    #     <li>Grouped by timestamp to find intervals</li>
    #     <li>Calculated average time between restocks</li>
    #     <li>Total incidents + quantity restocked by location/brand/product</li>
    # </ol>
    # <div style="border: 1px dashed #bbb; padding: 10px;">
    #     <i>Placeholder for restocking metric box</i>
    # </div>
    # """, unsafe_allow_html=True)

    # # OOS incidents
    # st.markdown("""
    # <h4>OOS incidents:</h4>
    # <ol>
    #     <li>Filter events with Data == 0 and Status == Possible Pull</li>
    #     <li>Sort by timestamp</li>
    #     <li>Track start and end of OOS windows</li>
    #     <li>Count total OOS incidents per location/product</li>
    # </ol>
    # <div style="border: 1px dashed #bbb; padding: 10px;">
    #     <i>Placeholder for OOS equation box</i>
    # </div>
    # """, unsafe_allow_html=True)

# def general_overview(location_filter, overview_table, overview_stats, general_insights):
#     if not location_filter:
#         st.warning("No locations to display.")
#         return

#     if len(location_filter) == 1:
#         tab = st.tabs(location_filter)
#         with tab[0]:
#             df_overview = overview_table[overview_table["Location"] == location_filter[0]].drop(columns=["Location"]).T
#             df_overview.columns = ["Value"]
#             df_overview.index.name = "Metric"
#             st.dataframe(df_overview, use_container_width=True)
#     else:
#         tabs = st.tabs(location_filter + ["Comparison"])
#         for i, loc in enumerate(location_filter):
#             with tabs[i]:
#                 df = overview_stats[overview_stats["Location"] == loc].drop(columns=["Location"]).T
#                 df.columns = ["Value"]
#                 df.index.name = "Metric"
#                 st.dataframe(df, use_container_width=True)

#         with tabs[-1]:
#             mode = st.radio("Comparison Mode", ["General Comparison", "Compare by Metric", "Compare by Location"])
#             summary_no_total = overview_stats[overview_stats["Location"].isin(location_filter)]

#             if mode == "General Comparison":
#                 melt_df = summary_no_total.melt(id_vars="Location", var_name="Metric", value_name="Value")
#                 melt_df = melt_df[melt_df["Metric"].isin([
#                     'Total Pulls', 'Total Fills', 'OOS Incidents'
#                 ])].copy()

#                 for metric in melt_df["Metric"].unique():
#                     total_metric = melt_df[melt_df["Metric"] == metric]["Value"].sum()
#                     melt_df.loc[melt_df["Metric"] == metric, "%"] = (
#                         melt_df.loc[melt_df["Metric"] == metric, "Value"] / total_metric * 100
#                     )

#                 fig = px.bar(
#                     melt_df,
#                     x="Location",
#                     y="%",
#                     color="Metric",
#                     barmode="group",
#                     title="Comparison of Metrics by Location (%)",
#                     text=melt_df["%"].map("{:.2f}%".format),
#                     color_discrete_map={
#                         'Total Pulls': '#064f14',
#                         'Total Fills': '#FFDEAD',
#                         'OOS Incidents': '#CD5C5C'
#                     },
#                     custom_data=["Value"]
#                 )
#                 fig.update_traces(
#                     textposition='outside',
#                     hovertemplate=(
#                         "Location: %{x}<br>"
#                         "Metric: %{legendgroup}<br>"
#                         "Percentage: %{y:.2f}%<br>"
#                         "Value: %{customdata[0]:,.0f}<extra></extra>"
#                     )
#                 )
#                 fig.update_yaxes(title_text="Percentage (%)")
#                 st.plotly_chart(fig, use_container_width=True)

#                 insight = general_insights.get("General Comparison", "No insight available for this view.")
#                 st.markdown(f"<div style='text-align: justify;'>{markdown.markdown(insight)}</div>", unsafe_allow_html=True)

#             elif mode == "Compare by Metric":
#                 metric = st.selectbox("Select a metric", [
#                     'Total Pulls', 'Total Fills', 'OOS Incidents',
#                     'SKU Count', 'Coolers Count'
#                 ])
#                 color_map = {
#                     'Total Pulls': '#064f14',
#                     'Total Fills': '#FFDEAD',
#                     'OOS Incidents': '#CD5C5C',
#                     'SKU Count': '#B0C4DE',
#                     'Coolers Count': '#778899'
#                 }

#                 if metric in ['Total Pulls', 'Total Fills', 'OOS Incidents','SKU Count', 'Coolers Count']:
#                     total = summary_no_total[metric].sum()
#                     summary_no_total["%"] = summary_no_total[metric] / total * 100

#                     fig = px.bar(
#                         summary_no_total,
#                         x="Location",
#                         y="%",
#                         text=summary_no_total["%"].map("{:.2f}%".format),
#                         title=f"{metric} by Location (%)",
#                         color_discrete_sequence=[color_map[metric]],
#                         custom_data=[summary_no_total[metric]]
#                     )
#                     fig.update_yaxes(title_text="Percentage (%)")
#                     fig.update_traces(
#                         textposition='outside',
#                         hovertemplate=(
#                             "Location: %{x}<br>"
#                             "Percentage: %{y:.2f}%<br>"
#                             f"{metric}: %{{customdata[0]:,.0f}}<extra></extra>"
#                         )
#                     )
#                 else:
#                     fig = px.bar(
#                         summary_no_total,
#                         x="Location",
#                         y=metric,
#                         text=metric,
#                         title=f"{metric} by Location",
#                         color_discrete_sequence=[color_map[metric]]
#                     )
#                     fig.update_yaxes(title_text="Count")
#                     fig.update_traces(textposition='outside')

#                 st.plotly_chart(fig, use_container_width=True)

#                 insight = general_insights.get(f"Compare by Metric - {metric}", "No insight available for this view.")
#                 st.markdown(f"<div style='text-align: justify;'>{markdown.markdown(insight)}</div>", unsafe_allow_html=True)

#             elif mode == "Compare by Location":
#                 location = st.selectbox("Select a location", summary_no_total["Location"].unique())
#                 row = overview_stats[overview_stats["Location"] == location].melt(
#                     id_vars="Location", var_name="Metric", value_name="Value"
#                 )
#                 fig = px.bar(
#                     row,
#                     x="Metric",
#                     y="Value",
#                     color="Metric",
#                     text="Value",
#                     title=f"Metrics for {location}",
#                     color_discrete_map={
#                         'Total Pulls': '#064f14',
#                         'Total Fills': '#FFDEAD',
#                         'OOS Incidents': '#CD5C5C',
#                         'SKU Count': '#B0C4DE',
#                         'Coolers Count': '#778899'
#                     }
#                 )
#                 fig.update_traces(textposition='outside')
#                 st.plotly_chart(fig, use_container_width=True)

#                 insight = general_insights.get(f"Compare by Location - {location}", "No insight available for this view.")
#                 st.markdown(f"<div style='text-align: justify;'>{markdown.markdown(insight)}</div>", unsafe_allow_html=True)

def general_overview(location_filter, overview_table, overview_stats, general_insights):
    if not location_filter:
        st.warning("No locations to display.")
        return

    if len(location_filter) == 1:
        tab = st.tabs(location_filter)
        with tab[0]:
            df_overview = overview_table[overview_table["Location"] == location_filter[0]].drop(columns=["Location"]).T
            df_overview.columns = ["Value"]
            df_overview.index.name = "Metric"
            df_overview["Value"] = df_overview["Value"].astype(str)
            st.dataframe(df_overview, use_container_width=True)
    else:
        tabs = st.tabs(location_filter + ["Comparison"])
        for i, loc in enumerate(location_filter):
            with tabs[i]:
                df = overview_stats[overview_stats["Location"] == loc].drop(columns=["Location"]).T
                df.columns = ["Value"]
                df.index.name = "Metric"
                df["Value"] = df["Value"].astype(str)
                st.dataframe(df, use_container_width=True)
                

        with tabs[-1]:
            

            mode = st.radio("Comparison Mode", ["General Comparison", "Compare by Metric", "Compare by Location"])
            summary_no_total = overview_stats[overview_stats["Location"].isin(location_filter)].copy()
            
            if mode == "General Comparison":
                melt_df = summary_no_total.melt(id_vars="Location", var_name="Metric", value_name="Value")
                melt_df = melt_df[melt_df["Metric"].isin([
                    'Total Pulls', 'Total Fills', 'OOS Incidents'
                ])].copy()

                for metric in melt_df["Metric"].unique():
                    total_metric = melt_df[melt_df["Metric"] == metric]["Value"].sum()
                    melt_df.loc[melt_df["Metric"] == metric, "%"] = (
                        melt_df.loc[melt_df["Metric"] == metric, "Value"] / total_metric * 100
                    )

                fig = px.bar(
                    melt_df,
                    x="Location",
                    y="%",
                    color="Metric",
                    barmode="group",
                    title="Comparison of Metrics by Location (%)",
                    text=melt_df["%"].map("{:.2f}%".format),
                    color_discrete_map={
                        'Total Pulls': '#064f14',
                        'Total Fills': '#FFDEAD',
                        'OOS Incidents': '#CD5C5C'
                    },
                    custom_data=["Value"]
                )
                fig.update_traces(
                    textposition='outside',
                    hovertemplate=(
                        "Location: %{x}<br>"
                        "Metric: %{legendgroup}<br>"
                        "Percentage: %{y:.2f}%<br>"
                        "Value: %{customdata[0]:,.0f}<extra></extra>"
                    )
                )
                fig.update_yaxes(title_text="Percentage (%)")
                st.plotly_chart(fig, use_container_width=True)

                insight = general_insights.get("General Comparison", "No insight available for this view.")
                st.markdown(f"<div style='text-align: justify;'>{markdown.markdown(insight)}</div>", unsafe_allow_html=True)

            elif mode == "Compare by Metric":
                metric = st.selectbox("Select a metric", [
                    'Total Pulls', 'Total Fills', 'OOS Incidents',
                    'SKU Count', 'Coolers Count'
                ])
                color_map = {
                    'Total Pulls': '#064f14',
                    'Total Fills': '#FFDEAD',
                    'OOS Incidents': '#CD5C5C',
                    'SKU Count': '#B0C4DE',
                    'Coolers Count': '#778899'
                }

                percent_metrics = ['Total Pulls', 'Total Fills', 'OOS Incidents']
                absolute_metrics = ['SKU Count', 'Coolers Count']

                if metric in percent_metrics:
                    total = summary_no_total[metric].sum()
                    summary_no_total["%"] = summary_no_total[metric] / total * 100

                    fig = px.bar(
                        summary_no_total,
                        x="Location",
                        y="%",
                        text=summary_no_total["%"].map("{:.2f}%".format),
                        title=f"{metric} by Location (%)",
                        color_discrete_sequence=[color_map[metric]],
                        custom_data=[summary_no_total[metric]]
                    )
                    fig.update_yaxes(title_text="Percentage (%)")
                    fig.update_traces(
                        textposition='outside',
                        hovertemplate=(
                            "Location: %{x}<br>"
                            "Percentage: %{y:.2f}%<br>"
                            f"{metric}: %{{customdata[0]:,.0f}}<extra></extra>"
                        )
                    )

                elif metric in absolute_metrics:
                    fig = px.bar(
                        summary_no_total,
                        x="Location",
                        y=metric,
                        text=metric,
                        title=f"{metric} by Location",
                        color_discrete_sequence=[color_map[metric]]
                    )
                    fig.update_yaxes(title_text="Count")
                    fig.update_traces(textposition='outside')

                st.plotly_chart(fig, use_container_width=True)

                insight = general_insights.get(f"Compare by Metric - {metric}", "No insight available for this view.")
                st.markdown(f"<div style='text-align: justify;'>{markdown.markdown(insight)}</div>", unsafe_allow_html=True)

            elif mode == "Compare by Location":
                location = st.selectbox("Select a location", summary_no_total["Location"].unique())
                row = overview_stats[overview_stats["Location"] == location].melt(
                    id_vars="Location", var_name="Metric", value_name="Value"
                )
                fig = px.bar(
                    row,
                    x="Metric",
                    y="Value",
                    color="Metric",
                    text="Value",
                    title=f"Metrics for {location}",
                    color_discrete_map={
                        'Total Pulls': '#064f14',
                        'Total Fills': '#FFDEAD',
                        'OOS Incidents': '#CD5C5C',
                        'SKU Count': '#B0C4DE',
                        'Coolers Count': '#778899'
                    }
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

                insight = general_insights.get(f"Compare by Location - {location}", "No insight available for this view.")
                st.markdown(f"<div style='text-align: justify;'>{markdown.markdown(insight)}</div>", unsafe_allow_html=True)

def tab1_total_pulls(df_loc,pulls,location_id, location_name, insights):
    st.markdown("### Pulls Analysis")
    # Filtrar los datos para la ubicación seleccionada
    df_loc = pulls[pulls["Location Id"] == location_id].copy()

    if df_loc.empty:
        st.warning(f"No data available for location ID {location_id}.")
    else:
        # Selector de vista principal
        view_option = st.selectbox(
            "Select View",
            ["General Overview Pulls","Pulls by All SKUs","Pulls by Top 10 SKU", "Pulls by Cooler"],
            key="total_pulls_view"
        )

        if view_option == "General Overview Pulls":
            st.markdown("""Total pulls across the entire location, aggregating all products and coolers.
            """)

            col1, col2 = st.columns([0.4, 0.6])
            
            with col1:
                # Vista general de Total Pulls
                st.markdown("##### " \
                "" \
                "" \
                "" \
                "")
                total_pulls = df_loc["Total Pulls"].sum()
                total_pulls_all = pulls["Total Pulls"].sum()
                percentage_pulls = (total_pulls / total_pulls_all) * 100

                st.metric(label="Total Pulls (Count)", value=int(total_pulls))
                st.metric(label="Pulls Contribution (%)", value=f"{percentage_pulls:.2f}%")
            
            with col2:
                pie_data = pd.DataFrame({
                    "Category": [f"{location_name}", "Other Locations"],
                    "Pulls": [total_pulls, total_pulls_all - total_pulls]
                })

                pie_chart = px.pie(
                    pie_data,
                    names="Category",
                    values="Pulls",
                    color="Category",
                    color_discrete_map={f"{location_name}": "#2E8B57", "Other Locations": "#8FBC8F"}
                )
                pie_chart.update_traces(textinfo='percent+label', hoverinfo='label+percent')
                
                filename = f"{view_option}__{location_name}"
                st.plotly_chart(pie_chart, use_container_width=True, key="pie_chart_general",
                                config={"toImageButtonOptions": {"filename": filename}})  

        

            # Mostrar el insight específico para la ubicación y vista
            insight = insights.get(location_name, {}).get(view_option, "No insight available for this view.")
            # Convertir Markdown a HTML
            insight_html = markdown.markdown(insight)

            # Justificar el texto usando HTML
            st.markdown(f"""
            <div style="text-align: justify;">
            {insight_html}
            </div>
            """, unsafe_allow_html=True)       
            
        elif view_option == "Pulls by All SKUs":
            
            all_products = (
                df_loc.groupby("Product")["Total Pulls"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )

            total_pulls_all_products = all_products["Total Pulls"].sum()
            all_products["% Pulls"] = (all_products["Total Pulls"] / total_pulls_all_products) * 100
            all_products = all_products.sort_values(by="% Pulls", ascending=True)

            fig_all = px.bar(
                all_products,
                x="% Pulls",
                y="Product",
                orientation="h",
                title="All Products by % of Total Pulls",
                text=all_products["% Pulls"].map("{:.2f}%".format),
                color_discrete_sequence=["#2E8B57"],
                custom_data=["Total Pulls"]
            )

            fig_all.update_traces(
                textposition="outside",
                hovertemplate=(
                    "Product: %{y}<br>"
                    "Percentage: %{x:.2f}%<br>"
                    "Total Pulls: %{customdata[0]:,.0f}<extra></extra>"
                )
            )

            avg_percent = all_products["% Pulls"].mean()
            fig_all.add_shape(
                type="line",
                x0=avg_percent, x1=avg_percent,
                y0=-0.5, y1=len(all_products)-0.5,
                line=dict(color="black", dash="dash", width=1)
            )
            fig_all.add_annotation(
                x=avg_percent,
                y=len(all_products)-0.5,
                text=f"Avg: {avg_percent:.2f}%",
                showarrow=False,
                font=dict(color="black"),
                xanchor="left",
                yanchor="bottom",
                bgcolor="white"
            )

            fig_all.update_layout(height=20 * len(all_products))

            filename = f"{view_option}__{location_name}"
            st.plotly_chart(fig_all, use_container_width=True, key="pulls_all_skus_chart",
                            config={"toImageButtonOptions": {"filename": filename}})

            insight = insights.get(location_name, {}).get(view_option, "No insight available for this view.")
            insight_html = markdown.markdown(insight)
            st.markdown(f"""
            <div style="text-align: justify;">
            {insight_html}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("**Deep Dive by product**"):
                st.markdown("""Allows selecting a specific product (SKU) 
                            to explore its pull behavior across all coolers in the location.""")
                
                   

                # Selector de búsqueda para SKU
                selected_product = st.selectbox(
                    "Search for a Product (SKU)",
                    df_loc["Product"].dropna().unique(),
                    key="sku_selector"
                )

                # Filtrar los datos para el producto seleccionado
                df_product = df_loc[df_loc["Product"] == selected_product]

                # Agrupar por cooler y calcular los pulls
                pulls_by_cooler = (
                    df_product.groupby("Deployment")["Total Pulls"]
                    .sum()
                    .sort_values(ascending=False)
                    .reset_index()
                )

                # Calcular el porcentaje de pulls por cooler para el SKU seleccionado
                total_pulls_sku = pulls_by_cooler["Total Pulls"].sum()
                pulls_by_cooler["% Pulls"] = (pulls_by_cooler["Total Pulls"] / total_pulls_sku) * 100

                #organizar los datos para el gráfico ascending
                pulls_by_cooler = pulls_by_cooler.sort_values(by="% Pulls", ascending=True)

                # Gráfico de barras horizontales
                fig_cooler = px.bar(
                    pulls_by_cooler,
                    x="% Pulls",
                    y="Deployment",
                    orientation="h",
                    title=f"Total Pulls by Cooler for Product: {selected_product}",
                    text=pulls_by_cooler["% Pulls"].map("{:.2f}%".format),
                    color_discrete_sequence=["#2E8B57"],  # Color verde
                    custom_data=["Total Pulls"]
                )
                fig_cooler.update_traces(
                    textposition="outside",
                    hovertemplate=(
                        "Cooler: %{y}<br>"
                        "Percentage: %{x:.2f}%<br>"
                        "Total Pulls: %{customdata[0]:,.0f}<extra></extra>"
                    )
                )
                # Calcular el promedio de % Pulls para el producto seleccionado
                avg_percent = pulls_by_cooler["% Pulls"].mean()
                # Agregar línea de promedio
                fig_cooler.add_shape(
                    type="line",
                    x0=avg_percent, x1=avg_percent,
                    y0=-0.5, y1=len(pulls_by_cooler)-0.5,
                    line=dict(color="black", dash="dash", width=1),
                )
                fig_cooler.add_annotation(
                    x=avg_percent,
                    y=len(pulls_by_cooler)-0.5,
                    text=f"Avg: {avg_percent:.2f}%",
                    showarrow=False,
                    font=dict(color="black"),
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="white"
                )
                fig_cooler.update_layout(yaxis=dict(title="Cooler"), xaxis=dict(title="% of Total Pulls"))
                st.plotly_chart(fig_cooler, use_container_width=True, key="cooler_for_product_chart")

                gemini_analysis_button(location_name, selected_product, pulls_by_cooler, metric_type="pull", model_enabled=ENABLE_GEMINI)


        elif view_option =="Pulls by Top 10 SKU":
                # Expandible para el análisis del Top 10
            exp_col = st.container()
            with exp_col:
                with st.expander("**📖 Why analysis the Top 10?**"):
                    # Layout de dos columnas
                    col1, col2 = st.columns([0.6, 0.4])  # Define las proporciones de las columnas

                    with col1:
                        # Calcular la contribución de todos los productos al total de la ubicación
                        total_pulls_all_products = df_loc["Total Pulls"].sum()
                        total_pulls_top10 = (
                            df_loc.groupby("Product")["Total Pulls"]
                            .sum()
                            .sort_values(ascending=False)
                            .head(10)
                            .sum()
                        )
                        other_pulls = total_pulls_all_products - total_pulls_top10

                        # Datos para el gráfico de pastel
                        pie_data = pd.DataFrame({
                            "Category": ["Top 10 Products", "Other Products"],
                            "Pulls": [total_pulls_top10, other_pulls]
                        })

                        # Gráfico de pastel
                        pie_chart = px.pie(
                            pie_data,
                            names="Category",
                            values="Pulls",
                            color="Category",
                            color_discrete_map={"Top 10 Products": "#2E8B57", "Other Products": "#8FBC8F"}
                        )
                        pie_chart.update_traces(textinfo='percent', hoverinfo='percent')
                        pie_chart.update_layout(
                            height=300,  # Ajusta la altura del gráfico
                            showlegend=True
                        )

                        filename = f"why top__{location_name}"
                        st.plotly_chart(pie_chart, use_container_width=True, key="sku_pie_chart", 
                                        config={"toImageButtonOptions": {"filename": filename}}
                                        )
                
                    with col2:

                        insight = insights.get(location_name, {}).get("why top", "No insight available for this view.")
                        insight_html = markdown.markdown(insight)
                        st.markdown(f"<div style='text-align: justify;'>{insight_html}</div>", unsafe_allow_html=True)


            # Gráfico de barras horizontales del top 10 general
            total_pulls_all_products = df_loc["Total Pulls"].sum()
            top_products = (
                df_loc.groupby("Product")["Total Pulls"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )

            top_products["% Pulls"] = (top_products["Total Pulls"] / total_pulls_all_products) * 100

            #organizar los datos para el gráfico ascending
            top_products = top_products.sort_values(by="% Pulls", ascending=True)

            #Plot
            fig_general = px.bar(
                top_products,
                x="% Pulls",
                y="Product",
                orientation="h",
                title="Top 10 Products by % of Total Pulls",
                text="% Pulls",
                hover_data={"Total Pulls": True, "% Pulls": ":.2f"},  # Mostrar Total Pulls al pasar el cursor
                color_discrete_sequence=["#2E8B57"],  # Color verde
            )

            # Calcular el promedio de % Pulls del top 10
            avg_percent = top_products["% Pulls"].mean()

            fig_general.update_traces(
                texttemplate="%{x:.2f}%",  # Mostrar el porcentaje en las barras
                textposition="outside"
            )
            fig_general.update_layout(
                yaxis=dict(title="Product"),
                xaxis=dict(title="% of Total Pulls", tickformat=".2f%%")  # Formato de porcentaje en el eje X
            )

            # Agregar línea de promedio
            fig_general.add_shape(
                type="line",
                x0=avg_percent, x1=avg_percent,
                y0=-0.5, y1=len(top_products)-0.5,
                line=dict(color="black", dash="dash", width=1),  # width=1 para línea más delgada
            )
            fig_general.add_annotation(
                x=avg_percent,
                y=len(top_products)-0.5,
                text=f"Avg: {avg_percent:.2f}%",
                showarrow=False,
                font=dict(color="black"),
                xanchor="left",
                yanchor="bottom",
                bgcolor="white"
            )

            filename = f"{view_option}__{location_name}"
            st.plotly_chart(fig_general, use_container_width=True, key="top10_general_chart", 
            config={"toImageButtonOptions": {"filename": filename}})

            insight = insights.get(location_name, {}).get(view_option, "No insight available for this view.")
            # Convertir Markdown a HTML
            insight_html = markdown.markdown(insight)
            # Justificar el texto usando HTML
            st.markdown(f"""
            <div style="text-align: justify;">
            {insight_html}
            </div>
            """, unsafe_allow_html=True)         

        elif view_option == "Pulls by Cooler":
            st.markdown("""Shows overall pulls per cooler, and breaks down the SKUs within each cooler.
                """)
            # Vista de Pulls por Cooler
            pulls_by_cooler = (
                df_loc.groupby("Deployment")["Total Pulls"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            #organizar los datos para el gráfico ascending
            
            # Calcular el porcentaje de pulls por cooler
            total_pulls_location = pulls_by_cooler["Total Pulls"].sum()
            pulls_by_cooler["% Pulls"] = (pulls_by_cooler["Total Pulls"] / total_pulls_location) * 100
            
            pulls_by_cooler = pulls_by_cooler.sort_values(by="% Pulls", ascending=True)
            
            fig_cooler = px.bar(
                pulls_by_cooler,
                x="% Pulls",
                y="Deployment",
                orientation="h",
                title="Total Pulls by Cooler (%)",
                text=pulls_by_cooler["% Pulls"].map("{:.2f}%".format),
                color_discrete_sequence=["#2E8B57"],  # Color verde
                custom_data=["Total Pulls"]
            )
            fig_cooler.update_traces(
                textposition="outside",
                hovertemplate=(
                    "Cooler: %{y}<br>"
                    "Percentage: %{x:.2f}%<br>"
                    "Total Pulls: %{customdata[0]:,.0f}<extra></extra>"
                )
            )

            # Calcular el promedio de % Pulls por cooler
            avg_percent = pulls_by_cooler["% Pulls"].mean()
            # Agregar línea de promedio
            fig_cooler.add_shape(
                type="line",
                x0=avg_percent, x1=avg_percent,
                y0=-0.5, y1=len(pulls_by_cooler)-0.5,
                line=dict(color="black", dash="dash", width=1),
            )
            fig_cooler.add_annotation(
                x=avg_percent,
                y=len(pulls_by_cooler)-0.5,
                text=f"Avg: {avg_percent:.2f}%",
                showarrow=False,
                font=dict(color="black"),
                xanchor="left",
                yanchor="bottom",
                bgcolor="white"
            )

            fig_cooler.update_layout(yaxis=dict(title="Cooler"), xaxis=dict(title="% of Total Pulls"))
            
            filename = f"{view_option}__{location_name}"
            st.plotly_chart(fig_cooler, use_container_width=True, key="cooler_chart",
                            config={"toImageButtonOptions": {"filename": filename}})

            insight = insights.get(location_name, {}).get(view_option, "No insight available for this view.")
            # Convertir Markdown a HTML
            insight_html = markdown.markdown(insight)
            # Justificar el texto usando HTML
            st.markdown(f"""
            <div style="text-align: justify;">
            {insight_html}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("**Deep Dive by cooler**"):
                st.markdown("""
                             This selector allows you to change how products are filtered within the selected cooler:

                - **All Products**: Displays all SKUs available in the selected cooler during the analysis period.
                - **Top 10 General**: Shows only the SKUs that belong to the overall Top 10 most pulled products across the entire location, but filters them within the selected cooler.               
                """)
                            
                
                # Selector de cooler
                selected_cooler = st.selectbox(
                    "Select a Cooler",
                    sorted(df_loc["Deployment"].dropna().unique()),
                    key="cooler_selector"
                )

                # Subselector para elegir entre todos los productos o solo el top 10 general
                product_filter_option = st.radio(
                    "Select Product Filter",
                    ["All Products", "Top 10 General"],
                    key="product_filter_option"
                )

                # Filtrar los datos para el cooler seleccionado
                df_cooler = df_loc[df_loc["Deployment"] == selected_cooler]

                if product_filter_option == "Top 10 General":
                    # Calcular el top 10 general de la ubicación
                    top_10_general = (
                        df_loc.groupby("Product")["Total Pulls"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                    )
                    # Filtrar solo los productos del top 10 general
                    df_cooler = df_cooler[df_cooler["Product"].isin(top_10_general["Product"])]

                # Agrupar por producto y calcular los pulls
                pulls_by_product = (
                    df_cooler.groupby("Product")["Total Pulls"]
                    .sum()
                    .sort_values(ascending=False)
                    .reset_index()
                )

                # Calcular el porcentaje de pulls por producto en el cooler
                total_pulls_cooler = pulls_by_product["Total Pulls"].sum()
                pulls_by_product["% Pulls"] = (pulls_by_product["Total Pulls"] / total_pulls_cooler) * 100

                #organizar los datos para el gráfico ascending
                pulls_by_product = pulls_by_product.sort_values(by="% Pulls", ascending=True)

                # Gráfico de barras horizontales
                fig_product = px.bar(
                    pulls_by_product,
                    x="% Pulls",
                    y="Product",
                    orientation="h",
                    title=f"Total Pulls by Product in Cooler: {selected_cooler} ({product_filter_option})",
                    text=pulls_by_product["% Pulls"].map("{:.2f}%".format),
                    color_discrete_sequence=["#2E8B57"],  # Color verde
                    custom_data=["Total Pulls"]
                )
                fig_product.update_traces(
                    textposition="outside",
                    hovertemplate=(
                        "Product: %{y}<br>"
                        "Percentage: %{x:.2f}%<br>"
                        "Total Pulls: %{customdata[0]:,.0f}<extra></extra>"
                    )
                )
                # Calcular el promedio de % Pulls para los productos en el cooler
                avg_percent = pulls_by_product["% Pulls"].mean()
                # Agregar línea de promedio
                fig_product.add_shape(
                    type="line",
                    x0=avg_percent, x1=avg_percent,
                    y0=-0.5, y1=len(pulls_by_product)-0.5,
                    line=dict(color="black", dash="dash", width=1),
                )
                fig_product.add_annotation(
                    x=avg_percent,
                    y=len(pulls_by_product)-0.5,
                    text=f"Avg: {avg_percent:.2f}%",
                    showarrow=False,
                    font=dict(color="black"),
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="white"
                )
                fig_product.update_layout(yaxis=dict(title="Product"), xaxis=dict(title="% of Total Pulls"))
                st.plotly_chart(fig_product, use_container_width=True, key="product_in_cooler_chart")

                gemini_analysis_by_cooler(location_name, selected_cooler, pulls_by_product, metric_type="pull", model_enabled=ENABLE_GEMINI)


def tab2_product_velocity(df_loc,pulls,location_id, location_name, insights):
    st.markdown("### Product Velocity Analysis")       
    # Filtrar los datos para la ubicación seleccionada
    df_loc = pulls[pulls["Location Id"] == location_id].copy()

    if df_loc.empty:
        st.warning(f"No data available for location ID {location_id}.")
    else:
        # Convertir la columna 'Date' a tipo datetime
        df_loc["Date"] = pd.to_datetime(df_loc["Date"])

        # Calcular total_days fuera de los bloques condicionales
        total_days = (df_loc["Date"].max() - df_loc["Date"].min()).days + 1

        # Selector de vista principal
        view_option = st.selectbox(
            "Select View",
            ["General Product Velocity","PV by All SKUs","PV by Top 10 Sku", "PV by Cooler"],
            key="product_velocity_view"
        )

        if view_option == "General Product Velocity":
            st.markdown("""
                            The velocity at which products are consumed at the location.
                        """)
            # Vista general de Product Velocity
            avg_velocity_active = df_loc["Total Pulls"].sum() / df_loc["Date"].nunique()
            avg_velocity_period = df_loc["Total Pulls"].sum() / total_days

            # Tarjetas de métricas
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Average Velocity (Active Days)", value=f"{avg_velocity_active:.2f} pulls/day")
               
            with col2:
                st.metric(label="Average Velocity (Period)", value=f"{avg_velocity_period:.2f} pulls/day")
                
                
            insight = insights.get(location_name, {}).get(view_option, "No insight available for this view.")
            # Convertir Markdown a HTML
            insight_html = markdown.markdown(insight)
            # Justificar el texto usando HTML
            st.markdown(f"""
            <div style="text-align: justify;">
            {insight_html}
            </div>
            """, unsafe_allow_html=True)
 

        elif view_option == "PV by All SKUs":

            st.markdown(""" The velocity at which all products are consumed across the location.""")

            # Convertir la columna 'Location Local Datetime' a datetime
            df_loc["Location Local Datetime"] = pd.to_datetime(df_loc["Location Local Datetime"], errors='coerce')

            # Calcular la fecha mínima y máxima para obtener el total de días en la data
            total_days = (df_loc["Location Local Datetime"].dt.date.max() - df_loc["Location Local Datetime"].dt.date.min()).days + 1

            # Calcular días únicos con datos por producto
            df_loc["Date"] = df_loc["Location Local Datetime"].dt.date
            days_with_data = df_loc.groupby("Product")["Date"].nunique().reset_index()
            days_with_data.columns = ["Product", "Days with Data"]

            # Agrupar pulls totales por producto (sin cortar al Top 10)
            all_products = (
                df_loc.groupby("Product")["Total Pulls"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )

            # Combinar con días activos
            all_products = all_products.merge(days_with_data, on="Product", how="left")

            # Calcular Product Velocity
            all_products["Velocity (Active Days)"] = all_products["Total Pulls"] / all_products["Days with Data"]
            all_products["Velocity (Period)"] = all_products["Total Pulls"] / total_days

            # Ordenar para gráfico
            all_products = all_products.sort_values(by="Velocity (Active Days)", ascending=True)

            # Crear gráfico
            fig = px.bar(
                all_products,
                x=["Velocity (Active Days)", "Velocity (Period)"],
                y="Product",
                orientation="h",
                title="All SKUs by Velocity",
                barmode="group",
                text_auto=True,
                color_discrete_sequence=["#4682B4", "#ADD8E6"]
            )
            fig.update_layout(
                height=20 * len(all_products),
                yaxis=dict(title="Product"),
                xaxis=dict(title="Velocity (pulls/day)")
            )

            # Promedio de Velocity (Period)
            avg_velocity_period = all_products["Velocity (Period)"].mean()
            fig.add_vline(
                x=avg_velocity_period,
                line_dash="dash",
                line_width=1,
                line_color="black",
                annotation_text=f"Avg Period: {avg_velocity_period:.2f}",
                annotation_position="top",
                annotation_font_color="black"
            )

            
            filename = f"{view_option}__{location_name}"
            st.plotly_chart(fig, use_container_width=True,
                            config={"toImageButtonOptions": {"filename": filename}})

            # Insight
            insight = insights.get(location_name, {}).get(view_option, "No insight available for this view.")
            # Convertir Markdown a HTML
            insight_html = markdown.markdown(insight)
            # Justificar el texto usando HTML
            st.markdown(f"""
            <div style="text-align: justify;">
            {insight_html}
            </div>
            """, unsafe_allow_html=True)


            with st.expander("**Deep dive by product**"):
            # Selector de búsqueda para SKU
                selected_product = st.selectbox(
                    "Search for a Product (SKU)",
                    df_loc["Product"].dropna().unique(),
                    key="sku_selector_velocity"
                )

                # Filtrar los datos para el producto seleccionado
                df_product = df_loc[df_loc["Product"] == selected_product]

               
                # Calcular días únicos con datos para el producto seleccionado
                active_days_product_deployment = (
                    df_product.groupby("Deployment")["Date"]
                    .nunique()
                    .reset_index()
                    .rename(columns={"Date": "Active Days"})
                )

                # Gráfico de Product Velocity por cooler para el SKU seleccionado
                velocity_by_cooler_sku = (
                    df_product.groupby("Deployment")["Total Pulls"]
                    .sum()
                    .reset_index()
                )

                     # Combinar los datos de días activos con los pulls totales
                velocity_by_cooler_sku = velocity_by_cooler_sku.merge(
                     active_days_product_deployment, on="Deployment", how="left"
                )

                velocity_by_cooler_sku["Velocity (Active Days)"] = (
                    velocity_by_cooler_sku["Total Pulls"] / velocity_by_cooler_sku["Active Days"]
                )

                velocity_by_cooler_sku["Velocity (Period)"] = velocity_by_cooler_sku["Total Pulls"] / total_days

                
                #organizar los datos para el gráfico ascending
                velocity_by_cooler_sku = velocity_by_cooler_sku.sort_values(by="Velocity (Active Days)", ascending=True)

                #Figura
                fig_sku = px.bar(
                    velocity_by_cooler_sku,
                    x=["Velocity (Active Days)", "Velocity (Period)"],
                    y="Deployment",
                    orientation="h",
                    title=f"Product Velocity by Cooler for SKU: {selected_product}",
                    barmode="group",
                    text_auto=True,
                    color_discrete_sequence=["#4682B4", "#ADD8E6"]
                )
                fig_sku.update_layout(yaxis=dict(title="Cooler"), xaxis=dict(title="Velocity (pulls/day)"))

                # Calcular el promedio de Velocity (Period)
                avg_velocity_period = velocity_by_cooler_sku["Velocity (Period)"].mean()

                # Agregar línea vertical de promedio solo para "Velocity (Period)"
                fig_sku.add_vline(
                    x=avg_velocity_period,
                    line_dash="dash",
                    line_width=1,
                    line_color="black",
                    annotation_text=f"Avg Period: {avg_velocity_period:.2f}",
                    annotation_position="top",
                    annotation_font_color="black"
                )

                st.plotly_chart(fig_sku, use_container_width=True)

                gemini_analysis_button(location_name, selected_product, velocity_by_cooler_sku, metric_type="velocity", model_enabled=ENABLE_GEMINI)

        elif view_option == "PV by Top 10 Sku":

            st.markdown(""" The velocity at which products are consumed across the top 10 SKUs""")

            # Convertir la columna 'Location Local Datetime' a datetime
            df_loc["Location Local Datetime"] = pd.to_datetime(df_loc["Location Local Datetime"], errors='coerce')

            # Calcular la fecha mínima y máxima para obtener el total de días en la data
            total_days = (df_loc["Location Local Datetime"].dt.date.max() - df_loc["Location Local Datetime"].dt.date.min()).days + 1

            # Calcular días únicos con datos por producto
            df_loc["Date"] = df_loc["Location Local Datetime"].dt.date
            days_with_data = df_loc.groupby("Product")["Date"].nunique().reset_index()
            days_with_data.columns = ["Product", "Days with Data"]

            # Calcular el Top 10 productos con más pulls
            top_10_products = (
                df_loc.groupby("Product")["Total Pulls"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )

            # Combinar el Top 10 con los días con datos
            top_10_products = top_10_products.merge(days_with_data, on="Product", how="left")

            # Calcular Product Velocity
            top_10_products["Velocity (Active Days)"] = top_10_products["Total Pulls"] / top_10_products["Days with Data"]
            top_10_products["Velocity (Period)"] = top_10_products["Total Pulls"] / total_days

            # Ordenar los datos para el gráfico
            top_10_products = top_10_products.sort_values(by="Velocity (Active Days)", ascending=True)

            # Crear el gráfico con Plotly
            fig = px.bar(
                top_10_products,
                x=["Velocity (Active Days)", "Velocity (Period)"],
                y="Product",
                orientation="h",
                title="Top 10 SKUs by Velocity",
                barmode="group",
                text_auto=True,
                color_discrete_sequence=["#4682B4", "#ADD8E6"]
            )
            fig.update_layout(
                yaxis=dict(title="Product"),
                xaxis=dict(title="Velocity (pulls/day)")
            )

            # Calcular el promedio de Velocity (Period)
            avg_velocity_period = top_10_products["Velocity (Period)"].mean()

            # Agregar línea vertical de promedio solo para "Velocity (Period)"
            fig.add_vline(
                x=avg_velocity_period,
                line_dash="dash",
                line_width=1,
                line_color="black",
                annotation_text=f"Avg Period: {avg_velocity_period:.2f}",
                annotation_position="top",
                annotation_font_color="black"
            )

            filename = f"{view_option}__{location_name}"
            st.plotly_chart(fig, use_container_width=True,
                             config={"toImageButtonOptions": {"filename": filename}}
                            )

            # Mostrar el insight específico para la ubicación y vista
            insight = insights.get(location_name, {}).get(view_option, "No insight available for this view.")
            # Convertir Markdown a HTML
            insight_html = markdown.markdown(insight)
            # Justificar el texto usando HTML
            st.markdown(f"""
            <div style="text-align: justify;">
            {insight_html}
            </div>
            """, unsafe_allow_html=True)

            
        elif view_option == "PV by Cooler":
            st.markdown("""
                        The velocity at which products are consumed by the cooler.
                        """)
            
            df_loc["Location Local Datetime"] = pd.to_datetime(df_loc["Location Local Datetime"], errors='coerce')
            
            # Calcular la fecha mínima y máxima para obtener el total de días en la data
            total_days = (df_loc["Location Local Datetime"].dt.date.max() - df_loc["Location Local Datetime"].dt.date.min()).days + 1

            # Calcular días únicos con datos por cooler
            active_days_by_cooler = (
                df_loc.groupby("Deployment")["Date"]
                .nunique()
                .reset_index()
                .rename(columns={"Date": "Active Days"})
            )
            
            # Gráfico general de Product Velocity por cooler
            velocity_by_cooler = (
                df_loc.groupby("Deployment")["Total Pulls"]
                .sum()
                .reset_index()
            )

            # Combinar los datos de días activos con los pulls totales
            velocity_by_cooler = velocity_by_cooler.merge(
                active_days_by_cooler, on="Deployment", how="left"
            )

            # Calcular las métricas de Product Velocity
            velocity_by_cooler["Velocity (Active Days)"] = (
                velocity_by_cooler["Total Pulls"] / velocity_by_cooler["Active Days"]
            )
            
            velocity_by_cooler["Velocity (Period)"] = velocity_by_cooler["Total Pulls"] / total_days

            #organizar los datos para el gráfico ascending
            velocity_by_cooler = velocity_by_cooler.sort_values(by="Velocity (Active Days)", ascending=True)
            
            fig_cooler = px.bar(
                velocity_by_cooler,
                x=["Velocity (Active Days)", "Velocity (Period)"],
                y="Deployment",
                orientation="h",
                title="Product Velocity by Cooler",
                barmode="group",
                text_auto=True,
                color_discrete_sequence=["#4682B4", "#ADD8E6"]
            )
            fig_cooler.update_layout(yaxis=dict(title="Cooler"), xaxis=dict(title="Velocity (pulls/day)"))

            # Calcular el promedio de Velocity (Period)
            avg_velocity_period = velocity_by_cooler["Velocity (Period)"].mean()

            # Agregar línea vertical de promedio solo para "Velocity (Period)"
            fig_cooler.add_vline(
                x=avg_velocity_period,
                line_dash="dash",
                line_width=1,
                line_color="black",
                annotation_text=f"Avg Period: {avg_velocity_period:.2f}",
                annotation_position="top",
                annotation_font_color="black"
            )

            filename = f"{view_option}__{location_name}"
            st.plotly_chart(fig_cooler, use_container_width=True,
                            config={"toImageButtonOptions": {"filename": filename}}
                            )

            # Mostrar el insight específico para la ubicación y vista
            insight = insights.get(location_name, {}).get(view_option, "No insight available for this view.")
            # Convertir Markdown a HTML
            insight_html = markdown.markdown(insight)
            # Justificar el texto usando HTML
            st.markdown(f"""
            <div style="text-align: justify;">
            {insight_html}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("**Deep dive by cooler**"):
                st.markdown("""
                This selector allows you to change how products are filtered within the selected cooler:

                - **All Products**: Displays all SKUs available in the selected cooler during the analysis period.
                - **Top 10 General**: Shows only the SKUs that belong to the overall Top 10 most pulled products across the entire location, but filters them within the selected cooler.               
                """)
                # Selector de cooler
                selected_cooler = st.selectbox(
                    "Select a Cooler",
                    sorted(df_loc["Deployment"].dropna().unique()),
                    key="cooler_selector_velocity"
                )

                # Filtrar los datos para el cooler seleccionado
                df_cooler = df_loc[df_loc["Deployment"] == selected_cooler]

                # Subselector para elegir entre Top 10 General y Top 10 del Cooler
                product_filter_option = st.radio(
                    "Select Product Filter",
                    ["All products", "Top 10 General"],
                    key="product_filter_option_velocity"
                )

                if product_filter_option == "Top 10 General":
                    top_10_general = (
                        df_loc.groupby("Product")["Total Pulls"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                    )
                    df_cooler = df_cooler[df_cooler["Product"].isin(top_10_general["Product"])]

                  # Calcular días únicos con datos por producto en el cooler seleccionado
                active_days_by_cooler = (
                    df_cooler.groupby("Product")["Date"]
                    .nunique()
                    .reset_index()
                    .rename(columns={"Date": "Active Days"})
                )   

                # Gráfico del Top 10 SKUs del cooler seleccionado
                top_velocity_cooler = (
                    df_cooler.groupby("Product")["Total Pulls"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                    .reset_index()
                )

                top_velocity_cooler = top_velocity_cooler.merge(
                    active_days_by_cooler, on="Product", how="left"
                )

                top_velocity_cooler["Velocity (Active Days)"] = (top_velocity_cooler["Total Pulls"] / top_velocity_cooler["Active Days"])   

                top_velocity_cooler["Velocity (Period)"] = top_velocity_cooler["Total Pulls"] / total_days

                #organizar los datos para el gráfico ascending
                top_velocity_cooler = top_velocity_cooler.sort_values(by="Velocity (Active Days)", ascending=True)
                fig_cooler_top = px.bar(
                    top_velocity_cooler,
                    x=["Velocity (Active Days)", "Velocity (Period)"],
                    y="Product",
                    orientation="h",
                    title=f"Top 10 SKUs in Cooler: {selected_cooler}",
                    barmode="group",
                    text_auto=True,
                    color_discrete_sequence=["#4682B4", "#ADD8E6"]
                )
                fig_cooler_top.update_layout(yaxis=dict(title="Product"), xaxis=dict(title="Velocity (pulls/day)"))

                # Calcular el promedio de Velocity (Period)
                avg_velocity_period = top_velocity_cooler["Velocity (Period)"].mean()

                # Agregar línea vertical de promedio solo para "Velocity (Period)"
                fig_cooler_top.add_vline(
                    x=avg_velocity_period,
                    line_dash="dash",
                    line_width=1,
                    line_color="black",
                    annotation_text=f"Avg Period: {avg_velocity_period:.2f}",
                    annotation_position="top",
                    annotation_font_color="black"
                )

                st.plotly_chart(fig_cooler_top, use_container_width=True)

                gemini_analysis_by_cooler(location_name, selected_cooler, top_velocity_cooler, metric_type="velocity", model_enabled=ENABLE_GEMINI)


def tab3_restoking_analysis(df_sku, df_restock, restock_sum, loc_restock_sum, location_id, location_name, insights):
    st.markdown("### Restoking Analysis")          
    
    df_sku = restock_sum[restock_sum["Location Id"] == location_id].copy()
    df_restock = loc_restock_sum[loc_restock_sum["Location Id"] == location_id].copy()
    
    if df_sku.empty or df_restock.empty:
        st.warning(f"No data available for location ID {location_id}.")
    else:
        # Convertir la columna 'Date' a tipo datetime
        # Selector de vista principal
        view_option = st.selectbox(
            "Select View",
            ["General Restocking Data", "Restocking by Top 10 Sku", "Restoking by Cooler"],
            key="restoking_view"
        )

        if view_option == "General Restocking Data":
            row=df_restock.iloc[0]


            # Tarjetas de métricas
            # Métricas agregadas por locación
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Restock Incidents", int(row["Total Restock Incidents"]))
            col2.metric("Avg Time Between Restocks (hours)", f"{row['Avg Time Between Restocks (hours)']:.2f}")
            col3.metric("Restocking Frequency (per day)", f"{row['Restocking Frequency (per day)']:.2f}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Products Restocked", int(row["Total Products Restocked"]))
            col2.metric("Avg Daily Products Restocked", f"{row['Avg Daily Products Restocked']:.2f}")
            col3.metric("Avg Daily Pulls", f"{row['Avg Daily Pulls']:.2f}")

            # Mostrar el insight específico para la ubicación y vista

            insight = insights.get(location_name, {}).get(view_option, "No insight available for this view.")
            # Convertir Markdown a HTML
            insight_html = markdown.markdown(insight)
            # Justificar el texto usando HTML
            st.markdown(f"""
            <div style="text-align: justify;">
            {insight_html}
            </div>
            """, unsafe_allow_html=True)  

        elif view_option == "Restocking by Top 10 Sku":
            st.markdown("""The restocking behavior of the top 10 SKUs in the location""")

            # Obtener el Top 10 productos con más pulls en la ubicación
            top_10_products = (
                df_sku.groupby("Product")["Total Pulls"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .index
            )

            # Filtrar los datos para el Top 10 productos
            df_top_10 = df_sku[df_sku["Product"].isin(top_10_products)].copy()

            # Agrupar por producto y calcular los valores promedio
            df_top_10 = df_top_10.groupby(["Product Id", "Product"]).agg({
                "Restocking Frequency (per day)": "mean",
                "Avg Daily Products Restocked": "mean",
                "Avg Daily Pulls": "mean",
                "Avg Time Between Restocks (hours)": "mean"
            }).reset_index()

            # Crear una nueva columna para etiquetas combinadas de Product Id y Product Name
            df_top_10['Product Label'] = df_top_10['Product'].astype(str)

            # Calcular el promedio de Restocking Frequency (per day)
            average_daily_restock_incidents = df_top_10['Restocking Frequency (per day)'].mean()


            #Ordenar los datos para el gráfico
            df_top_10 = df_top_10.sort_values(by="Restocking Frequency (per day)", ascending=False)
            
            # Crear el gráfico con Plotly
            fig = go.Figure()

            # Agregar barras para Restocking Frequency (per day)
            fig.add_trace(go.Bar(
                x=df_top_10['Product Label'],
                y=df_top_10['Restocking Frequency (per day)'],
                name='Restocking Frequency (per day)',
                marker_color='navajowhite',
                text=[f'{h:.1f}' for h in df_top_10['Restocking Frequency (per day)']],
                textposition='outside',
                yaxis="y1"
            ))

            # Agregar línea para Avg Daily Products Restocked
            fig.add_trace(go.Scatter(
                x=df_top_10['Product Label'],
                y=df_top_10['Avg Daily Products Restocked'],
                mode='lines+markers',
                name='Avg Daily Products Restocked',
                marker=dict(color='green', symbol='x'),
                line=dict(dash='dash', color='green'),
                yaxis="y2"
            ))

            # Agregar línea para Avg Daily Pulls
            fig.add_trace(go.Scatter(
                x=df_top_10['Product Label'],
                y=df_top_10['Avg Daily Pulls'],
                mode='lines+markers',
                name='Avg Daily Pulls',
                marker=dict(color='red', symbol='square'),
                line=dict(dash='dot', color='red'),
                yaxis="y2"
            ))

            # Agregar línea para Avg Time Between Restocks (hours)
            fig.add_trace(go.Scatter(
                x=df_top_10['Product Label'],
                y=df_top_10['Avg Time Between Restocks (hours)'],
                mode='lines+markers',
                name='Avg Time Between Restocks (hours)',
                marker=dict(color='purple', symbol='circle'),
                line=dict(color='purple'),
                yaxis="y3"
            ))

            # Agregar línea horizontal para el promedio de Restocking Frequency
            fig.add_trace(go.Scatter(
                x=df_top_10['Product Label'],  # Usar las mismas etiquetas del eje X
                y=[average_daily_restock_incidents] * len(df_top_10),  # Repetir el valor promedio
                mode='lines',
                name=f"Avg Restocking Frequency ({average_daily_restock_incidents:.2f})",
                line=dict(dash='dash', color='gray'),
                hoverinfo='skip'  # Evitar que aparezca información al pasar el cursor
            ))

            # Configurar diseño del gráfico
            fig.update_layout(
                xaxis=dict(
                    title="Product", 
                    tickangle=90,
                    domain=[0, 0.85],
                    ),
            
                yaxis=dict(
                    title=dict(text="Restocking Frequency (per day)", font=dict(color="goldenrod")),
                    tickfont=dict(color="goldenrod"),
                    title_standoff=10
                ),

                yaxis2=dict(
                    title=dict(text="Average Daily Metrics", font=dict(color="green")),
                    tickfont=dict(color="green"),
                    overlaying="y",
                    side="right",
                    title_standoff=10
                ),

                yaxis3=dict(
                    title=dict(text="Avg Time Between Restocks (hours)", font=dict(color="purple")),
                    tickfont=dict(color="purple"),
                    overlaying="y",
                    side="right",
                    anchor="free",
                    position=0.99, # Adjusted position for better alignment
                    title_standoff=10
                ),

                legend=dict(
                    orientation="h", 
                    yanchor="bottom", 
                    y=1.02, 
                    xanchor="center", 
                    x=0.5,
                ),

                height=800,
                margin=dict(t=50, b=200, r=100)
            )
            
            fig.update_yaxes(tickangle=0)

            # Mostrar el gráfico en Streamlit
            filename = f"{view_option}__{location_name}"
            st.plotly_chart(fig, use_container_width=True,
            config={"toImageButtonOptions": {"filename": filename}})

            # Mostrar el insight específico para la ubicación y vista
            insight = insights.get(location_name, {}).get(view_option, "No insight available for this view.")
            # Convertir Markdown a HTML
            insight_html = markdown.markdown(insight)
            # Justificar el texto usando HTML
            st.markdown(f"""
            <div style="text-align: justify;">
            {insight_html}
            </div>
            """, unsafe_allow_html=True)   
        
            with st.expander("**Deep dive**"):
                st.markdown("""Allows selecting a specific product (SKU) 
                to explore its restocking behavior across all coolers in the location.""")
            # Selector de búsqueda para SKU
                selected_product = st.selectbox(
                    "Search for a Product (SKU)",
                    sorted(df_sku["Product"].dropna().unique()),
                    key="sku_selector_restoking"
                )

              

                df_product = df_sku[df_sku["Product"] == selected_product]

                # Agrupar por cooler y calcular los valores promedio
                restocking_by_cooler = (
                    df_product.groupby("Deployment").agg({
                        "Restocking Frequency (per day)": "mean",
                        "Avg Daily Products Restocked": "mean",
                        "Avg Daily Pulls": "mean",
                        "Avg Time Between Restocks (hours)": "mean"
                    }).reset_index()
                )

                #Ordenar los datos para el gráfico
                restocking_by_cooler = restocking_by_cooler.sort_values(by="Restocking Frequency (per day)", ascending=False)
                # Crear el gráfico para el SKU seleccionado
                fig_cooler = go.Figure()

                # Agregar barras para Restocking Frequency (per day)
                fig_cooler.add_trace(go.Bar(
                    x=restocking_by_cooler["Deployment"],
                    y=restocking_by_cooler["Restocking Frequency (per day)"],
                    name="Restocking Frequency (per day)",
                    marker_color="navajowhite",
                    text=[f'{h:.1f}' for h in restocking_by_cooler["Restocking Frequency (per day)"]],
                    textposition="outside",
                    yaxis="y1"
                ))

                # Agregar línea para Avg Daily Products Restocked
                fig_cooler.add_trace(go.Scatter(
                    x=restocking_by_cooler["Deployment"],
                    y=restocking_by_cooler["Avg Daily Products Restocked"],
                    mode="lines+markers",
                    name="Avg Daily Products Restocked",
                    marker=dict(color="green", symbol="x"),
                    line=dict(dash="dash", color="green"),
                    yaxis="y2"
                ))

                # Agregar línea para Avg Daily Pulls
                fig_cooler.add_trace(go.Scatter(
                    x=restocking_by_cooler["Deployment"],
                    y=restocking_by_cooler["Avg Daily Pulls"],
                    mode="lines+markers",
                    name="Avg Daily Pulls",
                    marker=dict(color="red", symbol="square"),
                    line=dict(dash="dot", color="red"),
                    yaxis="y2"
                ))

                # Agregar línea para Avg Time Between Restocks (hours)
                fig_cooler.add_trace(go.Scatter(
                    x=restocking_by_cooler["Deployment"],
                    y=restocking_by_cooler["Avg Time Between Restocks (hours)"],
                    mode="lines+markers",
                    name="Avg Time Between Restocks (hours)",
                    marker=dict(color="purple", symbol="circle"),
                    line=dict(color="purple"),
                    yaxis="y3"
                ))

                # Configurar diseño del gráfico
                fig_cooler.update_layout(
                    xaxis=dict(
                        title="Cooler",
                        tickangle=90,
                        domain=[0, 0.85]
                    ),
                    yaxis=dict(
                        title=dict(text="Restocking Frequency (per day)", font=dict(color="goldenrod")),
                        tickfont=dict(color="goldenrod"),
                        title_standoff=10
                    ),
                    yaxis2=dict(
                        title=dict(text="Average Daily Metrics", font=dict(color="green")),
                        tickfont=dict(color="green"),
                        overlaying="y",
                        side="right",
                        title_standoff=10
                    ),
                    yaxis3=dict(
                        title=dict(
                            text="Avg Time Between Restocks (hours)", 
                            font=dict(color="purple")),
                        tickfont=dict(color="purple"),
                        overlaying="y",
                        side="right",
                        anchor="free",
                        position=0.99,
                        title_standoff=10
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=12,
                        xanchor="center",
                        x=0.5
                    ),
                    height=600,
                    margin=dict(t=50, b=200, r=150)
                )

                # Mostrar el gráfico en Streamlit
                st.plotly_chart(fig_cooler, use_container_width=True)

                gemini_analysis_button(location_name, selected_product, restocking_by_cooler, metric_type="restock", model_enabled=ENABLE_GEMINI)

        
        elif view_option == "Restoking by Cooler":
            st.markdown("""The restocking behavior across all coolers in the location.""")

            # Agrupar por cooler y calcular las métricas
            restocking_by_cooler = (
                df_sku.groupby("Deployment").agg({
                    "Restocking Frequency (per day)": "mean",
                    "Avg Daily Products Restocked": "mean",
                    "Avg Daily Pulls": "mean",
                    "Avg Time Between Restocks (hours)": "mean"
                }).reset_index()
            )

            # Ordenar los datos para el gráfico
            restocking_by_cooler = restocking_by_cooler.sort_values(by="Restocking Frequency (per day)", ascending=False)

            # Crear el gráfico con Plotly
            fig_cooler = go.Figure()

            # Agregar barras para Restocking Frequency (per day)
            fig_cooler.add_trace(go.Bar(
                x=restocking_by_cooler["Deployment"],
                y=restocking_by_cooler["Restocking Frequency (per day)"],
                name="Restocking Frequency (per day)",
                marker_color="navajowhite",
                text=[f'{h:.1f}' for h in restocking_by_cooler["Restocking Frequency (per day)"]],
                textposition="outside",
                yaxis="y1"
            ))

            # Agregar línea para Avg Daily Products Restocked
            fig_cooler.add_trace(go.Scatter(
                x=restocking_by_cooler["Deployment"],
                y=restocking_by_cooler["Avg Daily Products Restocked"],
                mode="lines+markers",
                name="Avg Daily Products Restocked",
                marker=dict(color="green", symbol="x"),
                line=dict(dash="dash", color="green"),
                yaxis="y2"
            ))

            # Agregar línea para Avg Daily Pulls
            fig_cooler.add_trace(go.Scatter(
                x=restocking_by_cooler["Deployment"],
                y=restocking_by_cooler["Avg Daily Pulls"],
                mode="lines+markers",
                name="Avg Daily Pulls",
                marker=dict(color="red", symbol="square"),
                line=dict(dash="dot", color="red"),
                yaxis="y2"
            ))

            # Agregar línea para Avg Time Between Restocks (hours)
            fig_cooler.add_trace(go.Scatter(
                x=restocking_by_cooler["Deployment"],
                y=restocking_by_cooler["Avg Time Between Restocks (hours)"],
                mode="lines+markers",
                name="Avg Time Between Restocks (hours)",
                marker=dict(color="purple", symbol="circle"),
                line=dict(color="purple"),
                yaxis="y3"
            ))

            # Configurar diseño del gráfico
            fig_cooler.update_layout(
                xaxis=dict(
                    title="Cooler",
                    tickangle=90,
                    domain=[0, 0.85]
                ),
                yaxis=dict(
                    title=dict(text="Restocking Frequency (per day)", font=dict(color="goldenrod")),
                    tickfont=dict(color="goldenrod"),
                    title_standoff=10
                ),
                yaxis2=dict(
                    title=dict(text="Average Daily Metrics", font=dict(color="green")),
                    tickfont=dict(color="green"),
                    overlaying="y",
                    side="right",
                    title_standoff=10
                ),
                yaxis3=dict(
                    title=dict(text="Avg Time Between Restocks (hours)", font=dict(color="purple")),
                    tickfont=dict(color="purple"),
                    overlaying="y",
                    side="right",
                    anchor="free",
                    position=0.99,
                    title_standoff=10
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.4,
                    xanchor="center",
                    x=0.5
                ),
                height=600,
                margin=dict(t=50, b=200, r=150)
            )

            # Mostrar el gráfico en Streamlit
            filename = f"{view_option}__{location_name}"
            st.plotly_chart(fig_cooler, use_container_width=True,
                            config={"toImageButtonOptions": {"filename": filename}})

            # Mostrar el insight específico para la ubicación y vista
            insight = insights.get(location_name, {}).get(view_option, "No insight available for this view.")
            # Convertir Markdown a HTML
            insight_html = markdown.markdown(insight)
            # Justificar el texto usando HTML
            st.markdown(f"""
            <div style="text-align: justify;">
            {insight_html}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("**Deep dive**"):
                st.markdown("""Allows selecting a specific cooler to explore its restocking behavior across SKUs in the location.
                            Use this information for deeper exploration, but note that it does not include specific insights""")
                

                # Selector de cooler
                selected_cooler = st.selectbox(
                    "Select a Cooler",
                    sorted(df_sku["Deployment"].dropna().unique()),
                    key="cooler_selector_restoking"
                )

                # Subselector para elegir entre Top 10 General y Top 10 del Cooler
                product_filter_option = st.radio(
                    "Select Product Filter",
                    ["Top 10 General", "Top 10 Cooler"],
                    key="product_filter_option_restoking"
                )

                # Filtrar los datos para el cooler seleccionado
                df_cooler = df_sku[df_sku["Deployment"] == selected_cooler]

                if product_filter_option == "Top 10 General":
                    # Calcular el Top 10 General de la ubicación
                    top_10_general = (
                        df_sku.groupby("Product")["Total Pulls"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                    )
                    # Filtrar solo los productos del Top 10 General
                    df_cooler = df_cooler[df_cooler["Product"].isin(top_10_general["Product"])]

                # Agrupar por producto y calcular las métricas de restocking
                restocking_metrics = (
                    df_cooler.groupby("Product").agg({
                        "Restocking Frequency (per day)": "mean",
                        "Avg Daily Products Restocked": "mean",
                        "Avg Daily Pulls": "mean",
                        "Avg Time Between Restocks (hours)": "mean"
                    }).reset_index()
                )

                # Ordenar los datos para el gráfico
                restocking_metrics = restocking_metrics.sort_values(by="Restocking Frequency (per day)", ascending=False)

                # Crear el gráfico con Plotly
                fig_cooler = go.Figure()

                # Agregar barras para Restocking Frequency (per day)
                fig_cooler.add_trace(go.Bar(
                    x=restocking_metrics["Product"],
                    y=restocking_metrics["Restocking Frequency (per day)"],
                    name="Restocking Frequency (per day)",
                    marker_color="navajowhite",
                    text=[f'{h:.1f}' for h in restocking_metrics["Restocking Frequency (per day)"]],
                    textposition="outside",
                    yaxis="y1"
                ))

                # Agregar línea para Avg Daily Products Restocked
                fig_cooler.add_trace(go.Scatter(
                    x=restocking_metrics["Product"],
                    y=restocking_metrics["Avg Daily Products Restocked"],
                    mode="lines+markers",
                    name="Avg Daily Products Restocked",
                    marker=dict(color="green", symbol="x"),
                    line=dict(dash="dash", color="green"),
                    yaxis="y2"
                ))

                # Agregar línea para Avg Daily Pulls
                fig_cooler.add_trace(go.Scatter(
                    x=restocking_metrics["Product"],
                    y=restocking_metrics["Avg Daily Pulls"],
                    mode="lines+markers",
                    name="Avg Daily Pulls",
                    marker=dict(color="red", symbol="square"),
                    line=dict(dash="dot", color="red"),
                    yaxis="y2"
                ))

                # Agregar línea para Avg Time Between Restocks (hours)
                fig_cooler.add_trace(go.Scatter(
                    x=restocking_metrics["Product"],
                    y=restocking_metrics["Avg Time Between Restocks (hours)"],
                    mode="lines+markers",
                    name="Avg Time Between Restocks (hours)",
                    marker=dict(color="purple", symbol="circle"),
                    line=dict(color="purple"),
                    yaxis="y3"
                ))

                # Configurar diseño del gráfico
                fig_cooler.update_layout(
                    xaxis=dict(
                        title="Product",
                        tickangle=90,
                        domain=[0, 0.85]
                    ),
                    yaxis=dict(
                        title=dict(text="Restocking Frequency (per day)", font=dict(color="goldenrod")),
                        tickfont=dict(color="goldenrod"),
                        title_standoff=10
                    ),
                    yaxis2=dict(
                        title=dict(text="Average Daily Metrics", font=dict(color="green")),
                        tickfont=dict(color="green"),
                        overlaying="y",
                        side="right",
                        title_standoff=10
                    ),
                    yaxis3=dict(
                        title=dict(text="Avg Time Between Restocks (hours)", font=dict(color="purple")),
                        tickfont=dict(color="purple"),
                        overlaying="y",
                        side="right",
                        anchor="free",
                        position=0.99,
                        title_standoff=10
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=12,
                        xanchor="center",
                        x=0.5
                    ),
                    height=600,
                    margin=dict(t=50, b=200, r=150)
                )

                # Mostrar el gráfico en Streamlit
                st.plotly_chart(fig_cooler, use_container_width=True)

                gemini_analysis_by_cooler(location_name, selected_cooler, restocking_metrics, metric_type="restock", model_enabled=ENABLE_GEMINI)


def tab4_oos_incidents(df_loc_oos,df_oos_restock,loc_oos_sum, oos_restock_sum, location_id, location_name, insights):
    st.markdown("### OOS Incidents")          
    
    df_loc_oos = loc_oos_sum[loc_oos_sum["Location Id"] == location_id].copy()
    df_oos_restock = oos_restock_sum[oos_restock_sum["Location Id"] == location_id].copy()
    
    if df_loc_oos.empty:
        st.warning(f"No data available for location ID {location_id}.")
    else:
        # Selector de vista principal
        view_option = st.selectbox(
            "Select View",
            ["General OOS Data", "OOS by Top 10 Sku", "OOS by Cooler"],
            key="oos_view"
        )

        if view_option == "General OOS Data":
            row=df_loc_oos.iloc[0]

            # Tarjetas de métricas
            # Métricas agregadas por locación
            col1, col2, col3 = st.columns(3)
            col1.metric("Total OOS Incidents", int(row["Total OOS Incidents"]))
            col2.metric("Avg OOS Duration (hours)", f"{row['Avg OOS Duration (hours)']:.2f}")
            col3.metric("OOS Frequency (per day)", f"{row['OOS Frequency (per day)']:.2f}")

            
            # Mostrar el insight específico para la ubicación y vista
            insight = insights.get(location_name, {}).get(view_option, "No insight available for this view.")
            # Convertir Markdown a HTML
            insight_html = markdown.markdown(insight)
            # Justificar el texto usando HTML
            st.markdown(f"""
            <div style="text-align: justify;">
            {insight_html}
            </div>
            """, unsafe_allow_html=True)  

        elif view_option == "OOS by Top 10 Sku":
            st.markdown("""The out-of-stock behavior of the top 10 SKUs in the location""")

             # Obtener el Top 10 productos con más pulls en la ubicación
            top_10_products = (
                df_oos_restock.groupby("Product")["Total Pulls"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .index
            )

            # Filtrar los datos para el Top 10 productos
            df_top_10 = df_oos_restock[df_oos_restock["Product"].isin(top_10_products)].copy()  

            # Agrupar por producto y calcular métricas promedio
            df_top_10 = df_top_10.groupby(["Product Id", "Product"]).agg({
                "OOS Frequency (per day)": "mean",
                "Avg OOS Duration (hours)": "mean",
                "Restocking Frequency (per day)": "mean"
            }).reset_index()    

            # Crear columna de etiqueta
            df_top_10['Product Label'] = df_top_10['Product'].astype(str)

            # Calcular promedios para líneas horizontales
            avg_oos_freq = df_top_10["OOS Frequency (per day)"].mean()
            avg_oos_duration = df_top_10["Avg OOS Duration (hours)"].mean()

            # Ordenar por OOS Frequency
            df_top_10 = df_top_10.sort_values(by="OOS Frequency (per day)", ascending=False)

            # Crear figura
            fig = go.Figure()

            # Barra: OOS Frequency (per day)
            fig.add_trace(go.Bar(
                x=df_top_10['Product Label'],
                y=df_top_10['OOS Frequency (per day)'],
                name='OOS Frequency (per day)',
                marker_color='lightcoral',
                text=[f'{v:.1f}' for v in df_top_10['OOS Frequency (per day)']],
                textposition='outside',
                yaxis="y1"
            ))

            # Barra: Restocking Frequency (per day)
            fig.add_trace(go.Bar(
                x=df_top_10['Product Label'],
                y=df_top_10['Restocking Frequency (per day)'],
                name='Restocking Frequency (per day)',
                marker_color='navajowhite',
                text=[f'{v:.1f}' for v in df_top_10['Restocking Frequency (per day)']],
                textposition='outside',
                yaxis="y1"
            ))

            # Línea: Avg OOS Duration (hours)
            fig.add_trace(go.Scatter(
                x=df_top_10['Product Label'],
                y=df_top_10['Avg OOS Duration (hours)'],
                mode='lines+markers',
                name='Avg OOS Duration (hours)',
                marker=dict(color='purple', symbol='circle'),
                line=dict(color='purple'),
                yaxis="y2"
            ))

            # Línea horizontal: promedio OOS Frequency
            fig.add_trace(go.Scatter(
                x=df_top_10['Product Label'],
                y=[avg_oos_freq] * len(df_top_10),
                mode='lines',
                name=f'Avg OOS Frequency (per day) ({avg_oos_freq:.2f})',
                line=dict(dash='dash', color='lightcoral'),
                hoverinfo='skip',
                yaxis='y1'
            ))

            # Línea horizontal: promedio OOS Duration
            fig.add_trace(go.Scatter(
                x=df_top_10['Product Label'],
                y=[avg_oos_duration] * len(df_top_10),
                mode='lines',
                name=f'Avg OOS Duration (hours) ({avg_oos_duration:.2f})',
                line=dict(dash='dot', color='purple'),
                hoverinfo='skip',
                yaxis='y2'
            ))

            # Layout
            fig.update_layout(
                xaxis=dict(title="Product", tickangle=90),
                yaxis=dict(
                    title="Frequency OOS & Restock Incidents",
                    tickfont=dict(color="black")
                ),
                yaxis2=dict(
                    title="OOS Duration (hours)",
                    overlaying="y",
                    side="right",
                    tickfont=dict(color="purple")
                ),

                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                height=700,
                margin=dict(t=50, b=200, r=100),
            )

            # Mostrar gráfico
            
            filename = f"{view_option}__{location_name}"
            st.plotly_chart(fig, use_container_width=True,
                             config={"toImageButtonOptions": {"filename": filename}})

            # Mostrar el insight específico para la ubicación y vista
            insight = insights.get(location_name, {}).get(view_option, "No insight available for this view.")
            # Convertir Markdown a HTML
            insight_html = markdown.markdown(insight)
            # Justificar el texto usando HTML
            st.markdown(f"""
            <div style="text-align: justify;">
            {insight_html}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("**Deep dive**"):
                st.markdown("""Allows selecting a specific product (SKU) 
                to explore its OOS behavior across all coolers in the location.
                Please Note that this does **not include specific insights**""")

                # Selector de búsqueda para SKU
                selected_product = st.selectbox(
                    "Search for a Product (SKU)",
                    sorted(df_oos_restock["Product"].dropna().unique()),
                    key="sku_selector_oos"
                )

                # Filtrar el dataset para el producto seleccionado
                df_product = df_oos_restock[df_oos_restock["Product"] == selected_product]

                # Agrupar por cooler y calcular métricas promedio de OOS y Restocking
                oos_by_cooler = (
                    df_product.groupby("Deployment").agg({
                        "OOS Frequency (per day)": "mean",
                        "Avg OOS Duration (hours)": "mean",
                        "Restocking Frequency (per day)": "mean"
                    }).reset_index()
                )

                # Calcular los promedios generales (líneas horizontales)
                avg_oos_freq = oos_by_cooler["OOS Frequency (per day)"].mean()
                avg_oos_duration = oos_by_cooler["Avg OOS Duration (hours)"].mean()

                # Ordenar por OOS Frequency
                oos_by_cooler = oos_by_cooler.sort_values(by="OOS Frequency (per day)", ascending=False)

                # Crear el gráfico
                fig_cooler = go.Figure()

                # Barra: OOS Frequency (per day)
                fig_cooler.add_trace(go.Bar(
                    x=oos_by_cooler["Deployment"],
                    y=oos_by_cooler["OOS Frequency (per day)"],
                    name="OOS Frequency (per day)",
                    marker_color="lightcoral",
                    text=[f'{v:.2f}' for v in oos_by_cooler["OOS Frequency (per day)"]],
                    textposition="outside",
                    yaxis="y1"
                ))

                # Barra: Restocking Frequency (per day)
                fig_cooler.add_trace(go.Bar(
                    x=oos_by_cooler["Deployment"],
                    y=oos_by_cooler["Restocking Frequency (per day)"],
                    name="Restocking Frequency (per day)",
                    marker_color="navajowhite",
                    text=[f'{v:.2f}' for v in oos_by_cooler["Restocking Frequency (per day)"]],
                    textposition="outside",
                    yaxis="y1"
                ))

                # Línea: Avg OOS Duration (hours)
                fig_cooler.add_trace(go.Scatter(
                    x=oos_by_cooler["Deployment"],
                    y=oos_by_cooler["Avg OOS Duration (hours)"],
                    mode="lines+markers",
                    name="Avg OOS Duration (hours)",
                    marker=dict(color="purple", symbol="circle"),
                    line=dict(color="purple"),
                    yaxis="y2"
                ))

                # Línea horizontal: promedio OOS Frequency
                fig_cooler.add_trace(go.Scatter(
                    x=oos_by_cooler["Deployment"],
                    y=[avg_oos_freq] * len(oos_by_cooler),
                    mode="lines",
                    name=f"Avg OOS Freq ({avg_oos_freq:.2f})",
                    line=dict(dash="dash", color="lightcoral"),
                    hoverinfo="skip",
                    yaxis="y1"
                ))

                # Línea horizontal: promedio OOS Duration
                fig_cooler.add_trace(go.Scatter(
                    x=oos_by_cooler["Deployment"],
                    y=[avg_oos_duration] * len(oos_by_cooler),
                    mode="lines",
                    name=f"Avg OOS Duration ({avg_oos_duration:.2f} h)",
                    line=dict(dash="dot", color="purple"),
                    hoverinfo="skip",
                    yaxis="y2"
                ))

                # Diseño del gráfico
                fig_cooler.update_layout(
                    xaxis=dict(
                        title="Cooler",
                        tickangle=90,
                        domain=[0, 0.85]
                    ),
                    yaxis=dict(
                        title=dict(text="Frequency (OOS & Restocking)", font=dict(color="black")),
                        tickfont=dict(color="black"),
                        title_standoff=10
                    ),
                    yaxis2=dict(
                        title=dict(text="Avg OOS Duration (hours)", font=dict(color="purple")),
                        tickfont=dict(color="purple"),
                        overlaying="y",
                        side="right",
                        title_standoff=10
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    height=600,
                    margin=dict(t=50, b=200, r=150)
                )

                # Mostrar gráfico
                st.plotly_chart(fig_cooler, use_container_width=True)

                gemini_analysis_button(location_name, selected_product, oos_by_cooler, metric_type="oos", model_enabled=ENABLE_GEMINI)

        
        elif view_option == "OOS by Cooler":
            st.markdown("""The OOS behavior across all coolers in the location.""")

            # Agrupar por Deployment (cooler) y calcular métricas promedio de OOS y Restocking
            oos_by_cooler = (
                df_oos_restock.groupby("Deployment").agg({
                    "OOS Frequency (per day)": "mean",
                    "Avg OOS Duration (hours)": "mean",
                    "Restocking Frequency (per day)": "mean"
                }).reset_index()
            )

            # Calcular promedios generales
            avg_oos_freq = oos_by_cooler["OOS Frequency (per day)"].mean()
            avg_oos_duration = oos_by_cooler["Avg OOS Duration (hours)"].mean()

            # Ordenar por OOS Frequency
            oos_by_cooler = oos_by_cooler.sort_values(by="OOS Frequency (per day)", ascending=False)

            # Crear figura
            fig_cooler = go.Figure()

            # Barra: OOS Frequency (per day)
            fig_cooler.add_trace(go.Bar(
                x=oos_by_cooler["Deployment"],
                y=oos_by_cooler["OOS Frequency (per day)"],
                name="OOS Frequency (per day)",
                marker_color="lightcoral",
                text=[f'{v:.2f}' for v in oos_by_cooler["OOS Frequency (per day)"]],
                textposition="outside",
                yaxis="y1"
            ))

            # Barra: Restocking Frequency (per day)
            fig_cooler.add_trace(go.Bar(
                x=oos_by_cooler["Deployment"],
                y=oos_by_cooler["Restocking Frequency (per day)"],
                name="Restocking Frequency (per day)",
                marker_color="navajowhite",
                text=[f'{v:.2f}' for v in oos_by_cooler["Restocking Frequency (per day)"]],
                textposition="outside",
                yaxis="y1"
            ))

            # Línea: Avg OOS Duration (hours)
            fig_cooler.add_trace(go.Scatter(
                x=oos_by_cooler["Deployment"],
                y=oos_by_cooler["Avg OOS Duration (hours)"],
                mode="lines+markers",
                name="Avg OOS Duration (hours)",
                marker=dict(color="purple", symbol="circle"),
                line=dict(color="purple"),
                yaxis="y2"
            ))

            # Línea horizontal: promedio OOS Frequency
            fig_cooler.add_trace(go.Scatter(
                x=oos_by_cooler["Deployment"],
                y=[avg_oos_freq] * len(oos_by_cooler),
                mode="lines",
                name=f"Avg OOS Frequency ({avg_oos_freq:.2f})",
                line=dict(dash="dash", color="lightcoral"),
                hoverinfo="skip",
                yaxis="y1"
            ))

            # Línea horizontal: promedio OOS Duration
            fig_cooler.add_trace(go.Scatter(
                x=oos_by_cooler["Deployment"],
                y=[avg_oos_duration] * len(oos_by_cooler),
                mode="lines",
                name=f"Avg OOS Duration ({avg_oos_duration:.2f} h)",
                line=dict(dash="dot", color="purple"),
                hoverinfo="skip",
                yaxis="y2"
            ))

            # Configuración del layout
            fig_cooler.update_layout(
                xaxis=dict(
                    title="Cooler",
                    tickangle=90,
                    
                ),
                yaxis=dict(
                    title=dict(text="Frequency (OOS & Restocking)", font=dict(color="black")),
                    tickfont=dict(color="black"),
                    title_standoff=10
                ),
                yaxis2=dict(
                    title=dict(text="Avg OOS Duration (hours)", font=dict(color="purple")),
                    tickfont=dict(color="purple"),
                    overlaying="y",
                    side="right",
                    title_standoff=10
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.2,
                    xanchor="center",
                    x=0.5
                ),
                height=600,
                margin=dict(t=50, b=200, r=150)
            )

            # Mostrar gráfico en Streamlit
            filename = f"{view_option}__{location_name}"
            st.plotly_chart(fig_cooler, use_container_width=True,
                             config={"toImageButtonOptions": {"filename": filename}})

            # Mostrar insight si existe
            insight = insights.get(location_name, {}).get(view_option, "No insight available for this view.")
            insight_html = markdown.markdown(insight)
            st.markdown(f"""
            <div style="text-align: justify;">
            {insight_html}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("**Deep dive**"):
                st.markdown("""Allows selecting a specific cooler to explore its OOS behavior across SKUs in the location. 
                            Use this information for deeper exploration, but note that it does **not include specific insights**.""")

                # Selector de cooler
                selected_cooler = st.selectbox(
                    "Select a Cooler",
                    sorted(df_oos_restock["Deployment"].dropna().unique()),
                    key="cooler_selector_oos"
                )

                # Subselector de filtro
                product_filter_option = st.radio(
                    "Select Product Filter",
                    ["Top 10 General", "Top 10 Cooler"],
                    key="product_filter_option_oos"
                )

                # Filtrar datos por cooler
                df_cooler = df_oos_restock[df_oos_restock["Deployment"] == selected_cooler]

                if product_filter_option == "Top 10 General":
                    top_10_general = (
                        df_oos_restock.groupby("Product")["Total Pulls"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                    )
                    df_cooler = df_cooler[df_cooler["Product"].isin(top_10_general["Product"])]

                # Agrupar por producto para métricas de OOS y Restocking
                oos_metrics = (
                    df_cooler.groupby("Product").agg({
                        "OOS Frequency (per day)": "mean",
                        "Avg OOS Duration (hours)": "mean",
                        "Restocking Frequency (per day)": "mean"
                    }).reset_index()
                )

                # Calcular promedios
                avg_oos_freq = oos_metrics["OOS Frequency (per day)"].mean()
                avg_oos_duration = oos_metrics["Avg OOS Duration (hours)"].mean()

                # Ordenar
                oos_metrics = oos_metrics.sort_values(by="OOS Frequency (per day)", ascending=False)

                # Crear gráfico
                fig_cooler = go.Figure()

                # Barra: OOS Frequency (per day)
                fig_cooler.add_trace(go.Bar(
                    x=oos_metrics["Product"],
                    y=oos_metrics["OOS Frequency (per day)"],
                    name="OOS Frequency (per day)",
                    marker_color="lightcoral",
                    text=[f'{v:.2f}' for v in oos_metrics["OOS Frequency (per day)"]],
                    textposition="outside",
                    yaxis="y1"
                ))

                # Barra: Restocking Frequency (per day)
                fig_cooler.add_trace(go.Bar(
                    x=oos_metrics["Product"],
                    y=oos_metrics["Restocking Frequency (per day)"],
                    name="Restocking Frequency (per day)",
                    marker_color="navajowhite",
                    text=[f'{v:.2f}' for v in oos_metrics["Restocking Frequency (per day)"]],
                    textposition="outside",
                    yaxis="y1"
                ))

                # Línea: Avg OOS Duration
                fig_cooler.add_trace(go.Scatter(
                    x=oos_metrics["Product"],
                    y=oos_metrics["Avg OOS Duration (hours)"],
                    mode="lines+markers",
                    name="Avg OOS Duration (hours)",
                    marker=dict(color="purple", symbol="circle"),
                    line=dict(color="purple"),
                    yaxis="y2"
                ))

                # Línea horizontal: promedio OOS Frequency
                fig_cooler.add_trace(go.Scatter(
                    x=oos_metrics["Product"],
                    y=[avg_oos_freq] * len(oos_metrics),
                    mode="lines",
                    name=f"Avg OOS Frequency ({avg_oos_freq:.2f})",
                    line=dict(dash="dash", color="lightcoral"),
                    hoverinfo="skip",
                    yaxis="y1"
                ))

                # Línea horizontal: promedio OOS Duration
                fig_cooler.add_trace(go.Scatter(
                    x=oos_metrics["Product"],
                    y=[avg_oos_duration] * len(oos_metrics),
                    mode="lines",
                    name=f"Avg OOS Duration ({avg_oos_duration:.2f} h)",
                    line=dict(dash="dot", color="purple"),
                    hoverinfo="skip",
                    yaxis="y2"
                ))

                # Layout
                fig_cooler.update_layout(
                    xaxis=dict(
                        title="Product",
                        tickangle=90,
                        domain=[0, 0.85]
                    ),
                    yaxis=dict(
                        title=dict(text="Frequency (OOS & Restocking)", font=dict(color="black")),
                        tickfont=dict(color="black"),
                        title_standoff=10
                    ),
                    yaxis2=dict(
                        title=dict(text="Avg OOS Duration (hours)", font=dict(color="purple")),
                        tickfont=dict(color="purple"),
                        overlaying="y",
                        side="right",
                        title_standoff=10
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    height=600,
                    margin=dict(t=50, b=200, r=150)
                )

                # Mostrar gráfico
                st.plotly_chart(fig_cooler, use_container_width=True)

                gemini_analysis_by_cooler(location_name, selected_cooler, oos_metrics, metric_type="oos", model_enabled=ENABLE_GEMINI)


def tab5_indexes(df_loc_indexes, loc_indexes, df_index_sku,index_sum, location_id, location_name, insights):
    st.markdown("### Indexes Analysis")     
    df_loc_indexes = loc_indexes[loc_indexes["Location Id"] == location_id].copy()
    with st.expander("ℹ️ What do these indexes mean?"):
        st.markdown("""
    **Indexes Glossary**

    These performance indexes help you understand how well each location is operating in terms of product availability, restocking, and consumption.

    ---
     **Velocity-to-Restock Ratio**

    Measures how many products are pulled for every restock.
    - **< 1** → More restocks than product movement → inefficient.
    - **≈ 1** → One pull per restock → possibly overstocked.
    - **> 1** → Good efficiency. For example, **3.8** means nearly 4 pulls happen per restock.
    
    ---
    **OOS-to-Restock Ratio**

    Measures how often products go out of stock compared to how often they are restocked.
    - **0** → No out-of-stock events (could be good or could mean low demand).
    - **< 1** → Fewer OOS than restocks → preventive or excessive restocking.
    - **≈ 1** → One OOS for each restock → balanced.
    - **> 1** → Too many OOS for the number of restocks → restocking is not keeping up. 

    ---

    **OOS Duration per Fill (hours)**

    Average number of hours the product was out of stock before being refilled.
    - **0** → Perfect. No out-of-stock delays.
    - **< 1–2 hours** → Fast and efficient restocking.
    - **> 4 hours** → Delayed restocking. Can lead to lost sales.
    - **> 24 hours** → Major problem. Indicates products are unavailable for full days.

    ---

    **Pulls per Fill**

    How many product pulls happen after each restock.
    - **0** → Products are restocked but not consumed → unnecessary refill.
    - **≈ 1** → Every restock leads to one pull → not efficient.
    - **> 1** → Efficient. For example, **3.9** means 4 products are pulled per restock.

    ---

    Use these indexes to spot overstocking, missed sales, or restocking inefficiencies.
    """)


    if df_loc_indexes.empty:
        st.warning(f"No data available for location ID {location_id}.")
    else:
        # Selector de vista principal
        view_option = st.selectbox(
            "Select View",
            ["General Indexes Analysis", "Index by Top 10 Sku", "Index by Cooler"],
            key="index_view"
        )

        if view_option == "General Indexes Analysis":
            row=df_loc_indexes.iloc[0]

            # Tarjetas de métricas
            # Métricas agregadas por locación
            col1, col2, col3,col4 = st.columns(4)
            col1.metric("OOS-to-Restock Ratio", int(row["OOS-to-Restock Ratio"]))
            col2.metric("Velocity-to-Restock Ratio", f"{row['Velocity-to-Restock Ratio']:.2f}")
            col3.metric("OOS Duration per Fill (h)", f"{row['OOS Duration per Fill']:.2f}")
            col4.metric("Pulls per Fill", f"{row['Pulls per Fill']:.2f}")

            tab1, tab2 = st.tabs(["Index Table","Radar Chart"])
            
            with tab1:
                
                selected_columns = [
                    "Avg Daily Pulls",
                    "Restocking Frequency (per day)",
                    "OOS Frequency (per day)",
                    "Avg OOS Duration (hours)",
                    "OOS-to-Restock Ratio",
                    "Velocity-to-Restock Ratio",
                    "OOS Duration per Fill",
                    "Pulls per Fill"
                ]

                # Seleccionar la fila y transponerla para mostrar vertical
                row_data = df_loc_indexes[selected_columns].iloc[0].round(2)
                vertical_table = row_data.to_frame(name="Value")
                vertical_table.index.name = "Index"

                st.dataframe(vertical_table, use_container_width=True)


            with tab2:
                
                # Normalización para radar
                df_norm = loc_indexes.copy()
                scaler = MinMaxScaler()
                df_norm[selected_columns] = scaler.fit_transform(df_norm[selected_columns])

                row_scaled = df_norm[df_norm["Location Id"] == location_id].copy()

                if not row_scaled.empty:
                    
                    fig = go.Figure()

                    fig.add_trace(go.Scatterpolar(
                        r=row_scaled[selected_columns].values.flatten().tolist(),
                        theta=selected_columns,
                        fill='toself',
                        name=location_name
                    ))

                    fig.update_layout(
                        width=200,
                        height=300,
                        margin=dict(t=30, b=10),
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=False,
                    )

                    fig.add_annotation(
                        text="Normalized performance indexes (0–1 scale)",
                        xref="paper", yref="paper",
                        x=0.5, y=-0.2,  # Posición debajo del gráfico
                        showarrow=False,
                        font=dict(size=12, color="gray")
                    )

                    filename = f"{view_option}__{location_name}"
                    st.plotly_chart(fig, use_container_width=True,
                    config={"toImageButtonOptions": {"filename": filename}})
            
        
            # Mostrar el insight específico para la ubicación y vista
            insight = insights.get(location_name, {}).get(view_option, "No insight available for this view.")
            # Convertir Markdown a HTML
            insight_html = markdown.markdown(insight)
            # Justificar el texto usando HTML
            st.markdown(f"""
            <div style="text-align: justify;">
            {insight_html}
            </div>
            """, unsafe_allow_html=True)
                
                
        elif view_option == "Index by Top 10 Sku":
            df_index_sku = index_sum[index_sum["Location Id"] == location_id].copy()

            if df_index_sku.empty:
                st.warning("No SKU data available for this location.")
            else:
                # --- Heatmap del Top 10 por Total Pulls (sin repetidos) ---
                df_top10 = (
                    df_index_sku.groupby("Product", as_index=False)
                    .agg({"Total Pulls": "sum"})
                    .sort_values("Total Pulls", ascending=False)
                    .head(10)
                )

                heatmap_metrics = [
                    "Velocity-to-Restock Ratio", "OOS-to-Restock Ratio",
                    "OOS Duration per Fill", "Pulls per Fill"
                ]

                # Filtrar el dataframe original solo para los productos del top 10
                df_heat = (
                    df_index_sku[df_index_sku["Product"].isin(df_top10["Product"])]
                    .drop_duplicates(subset=["Product"])
                    .set_index("Product")
                    .round(2)
                    #.reset_index()                      
                )

                # Crear heatmap con Plotly
                fig_heat = px.imshow(
                    df_heat[heatmap_metrics],
                    text_auto=True,
                    color_continuous_scale="Blues",
                    aspect="auto",
                    labels=dict(x="Metric", y="Product", color="Value")
                )

                fig_heat.update_layout(
                    title="Index Heatmap (Top 10 Products)",
                    xaxis_title="Metric",
                    yaxis_title="Product",
                    margin=dict(l=100, r=20, t=50, b=50),
                    xaxis=dict(tickangle=90)
                )

                filename = f"{view_option}__{location_name}"
                st.plotly_chart(
                    fig_heat,
                    use_container_width=True,
                    config={"toImageButtonOptions": {"filename": filename}}
                )

                insight = insights.get(location_name, {}).get(view_option, "No insight available for this view.")
                # Convertir Markdown a HTML
                insight_html = markdown.markdown(insight)
                # Justificar el texto usando HTML
                st.markdown(f"""
                <div style="text-align: justify;">
                {insight_html}
                </div>
                """, unsafe_allow_html=True)

                              

                # --- Deep Dive para cualquier SKU ---
                with st.expander("Deep Dive by SKU"):
                    sku_options = df_index_sku["Product"].dropna().unique()
                    selected_sku = st.selectbox("Select any SKU", sorted(sku_options))

                    df_selected_sku = df_index_sku[df_index_sku["Product"] == selected_sku]

                    if df_selected_sku.empty:
                        st.info("No data available for this SKU.")
                    else:
                        st.markdown("##### Metrics Overview")

                        row_sku = df_selected_sku.iloc[0][[
                            "Avg Daily Pulls", "Restocking Frequency (per day)", "OOS Frequency (per day)",
                            "Avg OOS Duration (hours)", "OOS-to-Restock Ratio", "Velocity-to-Restock Ratio",
                            "OOS Duration per Fill", "Pulls per Fill"
                        ]].round(2).to_frame(name="Value")
                        row_sku.index.name = None
                        st.dataframe(row_sku, use_container_width=True)

                        st.markdown("##### Heatmap: Cooler-Level Indexes for Selected SKU")

                        # Agrupar por Deployment para el SKU seleccionado
                        df_sku_by_cooler = df_selected_sku.groupby("Deployment")[heatmap_metrics].mean().round(2)

                        if df_sku_by_cooler.empty:
                            st.info("No cooler-level data available for this SKU.")
                        else:
                            fig_cooler, ax_cooler = plt.subplots(figsize=(10, max(4, 0.4 * len(df_sku_by_cooler))))
                            sns.heatmap(df_sku_by_cooler, annot=True, cmap="Blues", linewidths=0.5, ax=ax_cooler)
                            st.pyplot(fig_cooler)

        elif view_option == "Index by Cooler":
            df_index_cooler = index_sum[index_sum["Location Id"] == location_id].copy()

            if df_index_cooler.empty:
                st.warning("No cooler data available for this location.")
            else:
                

                heatmap_metrics = [
                    "Velocity-to-Restock Ratio", "OOS-to-Restock Ratio",
                    "OOS Duration per Fill", "Pulls per Fill"
                ]

                
                df_heat_deplo = df_index_cooler.groupby("Deployment")[heatmap_metrics].mean().round(2)

                # Crear heatmap con Plotly
                fig_heat = px.imshow(
                    df_heat_deplo,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    aspect="auto",
                    labels=dict(x="Metric", y="Cooler", color="Value")
                )

                fig_heat.update_layout(
                    title="Index Heatmap by Cooler",
                    xaxis_title="Metric",
                    yaxis_title="Cooler",
                    margin=dict(l=100, r=20, t=50, b=50),
                    xaxis=dict(tickangle=90)
                )

                # Mostrar el gráfico en Streamlit
                filename = f"{view_option}__{location_name}"
                st.plotly_chart(
                    fig_heat,
                    use_container_width=True,
                    config={"toImageButtonOptions": {"filename": filename}}
                )

                insight = insights.get(location_name, {}).get(view_option, "No insight available for this view.")
                # Convertir Markdown a HTML
                insight_html = markdown.markdown(insight)
                # Justificar el texto usando HTML
                st.markdown(f"""
                <div style="text-align: justify;">
                {insight_html}
                </div>
                """, unsafe_allow_html=True)


                # --- Deep Dive by cooler ---

                with st.expander("Deep Dive"):
                    st.markdown("""
                    Allows selecting a specific cooler to explore performance indexes across SKUs.  
                    Use the filter to switch between the store's Top 10 products overall or the Top 10 within this cooler.
                    """)

                    # Selector de cooler
                    selected_cooler = st.selectbox(
                        "Select a Cooler",
                        sorted(index_sum[index_sum["Location Id"] == location_id]["Deployment"].dropna().unique()),
                        key="cooler_selector_index"
                    )

                    # Subselector: tipo de top 10
                    product_filter_option = st.radio(
                        "Select Product Filter",
                        ["Top 10 General", "Top 10 Cooler"],
                        key="product_filter_option_index"
                    )

                    # Filtrar por cooler
                    df_cooler = index_sum[
                        (index_sum["Location Id"] == location_id) &
                        (index_sum["Deployment"] == selected_cooler)
                    ].copy()

                    if product_filter_option == "Top 10 General":
                        top_10_general = (
                            index_sum[index_sum["Location Id"] == location_id]
                            .groupby("Product")["Total Pulls"]
                            .sum()
                            .sort_values(ascending=False)
                            .head(10)
                            .index
                        )
                        df_cooler = df_cooler[df_cooler["Product"].isin(top_10_general)]

                    # Agrupar por SKU dentro del cooler
                    df_sku_metrics = df_cooler.groupby("Product")[[
                        "Velocity-to-Restock Ratio", "OOS-to-Restock Ratio",
                        "OOS Duration per Fill", "Pulls per Fill"
                    ]].mean().round(2)

                    if df_sku_metrics.empty:
                        st.info("No SKU data available for this cooler.")
                    else:
                        st.markdown("##### SKU Indexes in Selected Cooler")
                        fig_sku, ax_sku = plt.subplots(figsize=(10, max(4, 0.4 * len(df_sku_metrics))))
                        sns.heatmap(
                            df_sku_metrics,
                            annot=True,
                            cmap="YlGnBu",  # o "Blues" si prefieres neutral
                            linewidths=0.5,
                            ax=ax_sku,
                            fmt=".2f"
                        )
                        plt.xticks(rotation=0)
                        st.pyplot(fig_sku)


def key_conclusions(insights_key_conclusions):
    tab1, tab2 = st.tabs(["Store A", "Store B"])

    with tab1:
   
        insight = insights_key_conclusions.get("Store A", "No insight available for this view.")
        # Convertir Markdown a HTML
        insight_html = markdown.markdown(insight)
        # Justificar el texto usando HTML
        st.markdown(f"""
        <div style="text-align: justify;">
        {insight_html}
        </div>
        """, unsafe_allow_html=True)


    with tab2:    
        
        insight = insights_key_conclusions.get("Store B", "No insight available for this view.")
        insight_html = markdown.markdown(insight)
        st.markdown(f"""
        <div style="text-align: justify;">
            {insight_html}
        </div>
        """, unsafe_allow_html=True)
        
        
            
def location_analysis(location_id: int, data: dict):
    pulls = data["pulls"]
    restock_sum = data["restock_sum"]
    oos_restock_sum = data["oos_restock_sum"]
    loc_restock_sum = data["loc_restock_sum"]
    loc_oos_sum = data["loc_oos_sum"]
    loc_indexes = data["loc_indexes"]
    index_sum = data["index_sum"]
    location_name = pulls[pulls["Location Id"] == location_id]["Location"].iloc[0]

    st.title(f"{location_name} Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Total Pulls",
        "Product Velocity",
        "Restoking Analysis",
        "OOS Incidents",
        "Indexes",
        
    ])

    # --- TAB 1: Total Pulls ---
    with tab1:
        tab1_total_pulls(
            df_loc=pulls[pulls["Location Id"] == location_id],
            pulls=pulls,
            location_id=location_id,
            location_name=location_name,
            insights=insights
        )

        
        # --- TAB 2: Product Velocity ---
    with tab2:
        tab2_product_velocity(
            df_loc=pulls[pulls["Location Id"] == location_id],
            pulls=pulls,
            location_id=location_id,
            location_name=location_name,
            insights=insights
        )      
        
    with tab3:
        tab3_restoking_analysis(
            df_sku=restock_sum[restock_sum["Location Id"] == location_id],
            df_restock=loc_restock_sum[loc_restock_sum["Location Id"] == location_id],
            restock_sum=restock_sum,
            loc_restock_sum=loc_restock_sum,
            location_id=location_id,
            location_name=location_name,
            insights=insights
        )   

    with tab4:
        tab4_oos_incidents(
            df_oos_restock=oos_restock_sum[oos_restock_sum["Location Id"] == location_id],
            df_loc_oos=loc_oos_sum[loc_oos_sum["Location Id"] == location_id],
            oos_restock_sum=oos_restock_sum,
            loc_oos_sum=loc_oos_sum,
            location_id=location_id,
            location_name=location_name,
            insights=insights
        )  

    with tab5:  
        tab5_indexes(
            df_loc_indexes=loc_indexes[loc_indexes["Location Id"] == location_id],
            loc_indexes=loc_indexes,
            df_index_sku=index_sum[index_sum["Location Id"] == location_id],
            index_sum=index_sum,
            location_id=location_id,
            location_name=location_name,
            insights=insights
        )              

# st.sidebar.image("assets/logo1.png", width=200)

# Sidebar Navigation
# --- CONFIGURACIÓN DE LOCACIONES ACTIVAS ---
ACTIVE_LOCATIONS = ["Store A", "Store B"]  # ← Cambia esto según lo que quieras mostrar

location_slides = [f"{loc} Analysis" for loc in ACTIVE_LOCATIONS]

slide_options = [
    "1. Presentation",
    "2. Data Set",
    "3. Methodology",
    "4. Metrics Definitions",
    "5. General Overview",
    *location_slides,
    "8. Key Conclusions"
]



slide = st.sidebar.radio("Navigation:", slide_options)


if slide == "1. Presentation":
    presentation()
elif slide == "2. Data Set":
    dataset()
elif slide == "3. Methodology":
    methodology()
elif slide == "4. Metrics Definitions":
    metrics_definitions()


elif slide == "5. General Overview":
    general_overview(ACTIVE_LOCATIONS, data["overview_table"], data["overview_stats"], general_insights)


elif slide == "8. Key Conclusions":
    # Crear una pestaña por cada locación activa
    tabs = st.tabs(ACTIVE_LOCATIONS)

    for i, loc in enumerate(ACTIVE_LOCATIONS):
        with tabs[i]:
            
            insight = insights_key_conclusions.get(loc, "No insight available for this location.")
            insight_html = markdown.markdown(insight)
            st.markdown(f"""
            <div style="text-align: justify;">
            {insight_html}
            </div>
            """, unsafe_allow_html=True)


# Despliegue dinámico por locación
for location in ACTIVE_LOCATIONS:
    if slide == f"{location} Analysis":
        location_analysis(data["location_ids"][location], data)
