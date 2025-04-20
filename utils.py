# utils.py
import streamlit as st
import pandas as pd
from datetime import timedelta
import base64

def load_clusters():
    url = "data/clustered_patients.csv"
    df = pd.read_csv(url)
    
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
    df['Year'] = df['Date of Admission'].dt.year

    df.Cluster += 1     # clusters will be 1-6

    df['Cluster'] = df['Cluster'].apply(lambda x: 'Cluster ' + str(x) )
    
    return df

def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def toggle_help_state():
    st.session_state.show_help = False

@st.cache_data
def load_data():
    url = "data/Healthcare Analysis Dataset.csv"
    df = pd.read_csv(url)

    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
    df['Year'] = df['Date of Admission'].dt.year
    df['Month'] = df['Date of Admission'].dt.month
    df['Quarter'] = df['Date of Admission'].dt.quarter

    return df

def create_page_navigation():
    # Get current page path
    current_page = st.session_state.get('current_page', 'Executive Summary')
    
    # Create navigation with a single continuous line
    st.markdown(f"""
    <div class="stTabs">
        <div data-baseweb="tab-list">
            <a target="_self" href="/" class="nav-link" data-active="{'true' if current_page == 'Executive Summary' else 'false'}">Executive Summary</a>
            <a target="_self" href="/Patient_Demographics" class="nav-link" data-active="{'true' if current_page == 'Patient Demographics' else 'false'}">Patient Demographics</a>
            <a target="_self" href="/Hospital_Performance" class="nav-link" data-active="{'true' if current_page == 'Hospital Performance' else 'false'}">Hospital Performance</a>
            <a target="_self" href="/Insurance_&_Billing" class="nav-link" data-active="{'true' if current_page == 'Insurance & Billing' else 'false'}">Insurance & Billing</a>
            <a target="_self" href="/Trends_&_Forecasting" class="nav-link" data-active="{'true' if current_page == 'Trends & Forecasting' else 'false'}">Trends & Forecasting</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Add the help box after navigation
    if 'show_help' in st.session_state and st.session_state.show_help:

        with st.container(key="help-container"):

            help_header = st.container(key='help_head')
            helpcols = help_header.columns([10,1])
            helpcols[0].markdown("""<h3 class="help-title">Dashboard Help & Information</h3>""", unsafe_allow_html=True)
            if helpcols[1].button("",
                                key="close_help",
                                icon=":material/close:",
                                help="Close the help section",
                                use_container_width=True,
                                on_click=toggle_help_state):
                pass   


            st.markdown("""
            <div class="help-container">
                <div class="help-section" style="margin-top: 20px;">
                    <h3 style="margin-bottom: 0px;">Getting Started</h3>
                    <p>This dashboard provides comprehensive insights into healthcare data. To get the most out of it:</p>
                    <ul>
                        <li>Use the sidebar filters to customize your view</li>
                        <li>Navigate between pages using the tabs at the top</li>
                        <li>Scroll down each page to discover deeper insights and visualizations</li>
                        <li>Hover over charts for additional information</li>
                    </ul>
                </div>
                <div class="help-section">
                    <h3 style="margin-bottom: 0px;">Time Period Options</h3>
                    <ul>
                        <li><strong>Last Month:</strong> Shows data for the current month compared to the previous month</li>
                        <li><strong>Last Quarter:</strong> Shows data for the current quarter compared to the previous quarter</li>
                        <li><strong>Last Year:</strong> Shows data for the current year compared to the previous year</li>
                        <li><strong>Custom:</strong> Allows you to select a custom date range</li>
                    </ul>
                </div>
                <div class="help-section">
                    <h3>Hospital Filter</h3>
                    <p>Select one or multiple hospitals to filter the data. Leave empty to show all hospitals.</p>
                </div>
                <div class="help-section">
                    <h3 style="margin-bottom: 0px;">Page-Specific Information</h3>
                    <div class="page-specific">
                        <p><b>Executive Summary</b>: Provides high-level KPIs and metrics summarizing overall hospital performance, patient volume, and financial status.</p>
                        <p><b>Patient Demographics</b>: Detailed breakdown of patient population by gender, age, blood type, and medical conditions with interactive visualizations.</p>
                        <p><b>Hospital Performance</b>: Analysis of hospital-level metrics including average length of stay, patient satisfaction, and readmission rates with benchmark comparisons.</p>
                        <p><b>Insurance & Billing</b>: Financial analysis of insurance types, billing amounts, and payment patterns with patient segmentation models.</p>
                        <p><b>Trends & Forecasting</b>: Historical patient volume trends with seasonality patterns and predictive forecasting for future periods.</p>
                    </div>
                </div>
                <div class="attribution">
                    <p>This dashboard was created for the <strong>Data DNA</strong> April 2025 challenge<br>
                    Designed and developed by <strong>Sajjad Ahmadi</strong></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Hidden button that actually controls the state
            #if st.button("", key="close_help", icon=":material/close:", help="Close the help section",use_container_width=True):
            #    st.session_state.show_help = False

def process_date_ranges(scenario, max_date):   

    prev_custom_start = None

    if scenario == "Last Month":
        # Get current month dates
        current_month = max_date.month
        current_year = max_date.year
        day_of_month = max_date.day
        
        # Current period: From 1st of current month to current day
        current_start = pd.Timestamp(year=current_year, month=current_month, day=1)
        current_end = max_date
        
        # Previous month
        if current_month == 1:  # January
            prev_month = 12
            prev_year = current_year - 1
        else:
            prev_month = current_month - 1
            prev_year = current_year
        
        # Previous period: From 1st of previous month to same day of previous month
        prev_start = pd.Timestamp(year=prev_year, month=prev_month, day=1)
        
        try:
            prev_end = pd.Timestamp(year=prev_year, month=prev_month, day=day_of_month)
        except ValueError:
            # If day doesn't exist in previous month (e.g., March 31 -> Feb 28/29)
            if prev_month == 2:  # February
                prev_end = pd.Timestamp(year=prev_year, month=prev_month, day=29) if (prev_year % 4 == 0 and (prev_year % 100 != 0 or prev_year % 400 == 0)) else pd.Timestamp(year=prev_year, month=prev_month, day=28)
            else:
                last_day = pd.Timestamp(year=prev_year, month=prev_month+1, day=1) - timedelta(days=1)
                prev_end = last_day
        
        
        comparison_label = "vs PM"
        
    elif scenario == "Last Quarter":
        # Get current quarter dates
        current_quarter = (max_date.month - 1) // 3 + 1
        current_year = max_date.year
        
        # Calculate the start of the current quarter
        current_quarter_start = pd.Timestamp(year=current_year, month=((current_quarter-1)*3)+1, day=1)
        
        # Current period is from start of current quarter to max_date
        current_start = current_quarter_start
        current_end = max_date
        
        # Calculate days into quarter
        days_into_quarter = (max_date - current_quarter_start).days
        
        # Calculate previous quarter
        if current_quarter == 1:  # Q1
            prev_quarter = 4
            prev_year = current_year - 1
        else:
            prev_quarter = current_quarter - 1
            prev_year = current_year
        
        # Previous period start
        prev_start = pd.Timestamp(year=prev_year, month=((prev_quarter-1)*3)+1, day=1)
        
        # Previous period end is same number of days into previous quarter
        prev_end = prev_start + timedelta(days=days_into_quarter) 
        
        comparison_label = "vs PQ"

    elif scenario == "Last Year":
        # Get current year dates up to max_date
        current_year = max_date.year
        
        # Calculate days into year
        year_start = pd.Timestamp(year=current_year, month=1, day=1)
        days_into_year = (max_date - year_start).days - 1
        
        # Current period: From Jan 1 to max_date of current year
        current_start = year_start
        current_end = max_date
        
        # Previous period: Same number of days into previous year
        prev_year = current_year - 1
        prev_start = pd.Timestamp(year=prev_year, month=1, day=1)
        prev_end = prev_start + timedelta(days=days_into_year)
        
        comparison_label = "vs PY"
        
    else:  # Custom
        # Date range picker for custom date selection
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(max_date - timedelta(days=180), max_date),
            min_value=pd.Timestamp('2020-01-01').date(),  # Adjust based on your data
            max_value=max_date.date()
        )
        if len(date_range) == 2:
            current_start = pd.Timestamp(date_range[0])
            current_end = pd.Timestamp(date_range[1])
            prev_custom_start = pd.Timestamp(date_range[0])
            prev_start = None
            prev_end = None
            comparison_label = ""
        else:
            st.error("Please select both start and end dates")
            current_start = max_date - timedelta(days=30)
            current_end = max_date
            prev_start = None
            prev_end = None
            comparison_label = ""

    # Display selected date range
    st.sidebar.markdown(f"""
        <div class="simpleTextFirst">
            <p>Selected range:</p>                
            <p class="textBlak">{current_start.strftime('%b %d, %Y')} - {current_end.strftime('%b %d, %Y')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display comparison date range if available
    if prev_start is not None and prev_end is not None:
        st.sidebar.markdown(f"""
            <div class="simpleText">
                <p>Comparison:</p>
                <p class="textBlak">{prev_start.strftime('%b %d, %Y')} - {prev_end.strftime('%b %d, %Y')}</p>
            </div>    
            """, unsafe_allow_html=True)
        
    st.sidebar.markdown("<hr style='margin-top:45px; margin-bottom: 50px;'>", unsafe_allow_html=True)    

    return current_start, current_end, prev_start, prev_end, comparison_label, prev_custom_start

def initialize_page():
    # Set page configuration
    st.set_page_config(
        page_title="Healthcare Analytics Dashboard",
        layout="wide",
        page_icon= 'assets/images/logo.png',
        initial_sidebar_state="expanded"
    )

    # Load CSS
    with open('assets/styles.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def create_sidebar(df):
    
    logo_path = "assets/images/logo.png"
        
    # Sidebar header with logo
    st.sidebar.markdown(f"""
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <img src="data:image/png;base64,{img_to_base64(logo_path)}" width="35">
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div class="sidebar-header">
        Analytics Dashboard
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-section-title'>Filters</div>", unsafe_allow_html=True)

    # Time period selector
    scenario = st.sidebar.selectbox(
        "Time Period:",
        options=["Last Month", "Last Quarter", "Last Year", "Custom"],
        index=0,
        key="time_period_selector"
    )

    # Hospital multiselect
    # Get unique hospitals and add "All" option
    hospitals = sorted(df['Hospital'].unique().tolist())
    selected_hospitals = st.sidebar.multiselect(
        "Hospital:",
        options=hospitals,
        default= None,
        placeholder= 'All'
    )

    # Process the hospital filter
    if "All" in selected_hospitals or not selected_hospitals:
        filtered_df = df  # No filtering if "All" is selected or nothing is selected
    else:
        filtered_df = df[df['Hospital'].isin(selected_hospitals)]


    # Get current page from session state (default to empty string if not set)
    current_page = st.session_state.get('current_page', 'Executive Summary')

    if current_page == 'Trends & Forecasting':
        
        max_date = df['Date of Admission'].max()
        current_month = max_date.month
        current_year = max_date.year
        day_of_month = max_date.day
        
        # If we're in the early days of the month (< 7), consider the previous month as the last complete one
        if day_of_month < 25:
            # Calculate the last day of the previous month
            if current_month == 1:  # January
                prev_month = 12
                prev_year = current_year - 1
            else:
                prev_month = current_month - 1
                prev_year = current_year
                
            # Last day of previous month
            last_day = pd.Timestamp(year=prev_year, month=prev_month+1, day=1) - timedelta(days=1)
            max_date = last_day  # Use the last day of previous month as max_date
    else:
        max_date = df['Date of Admission'].max()

    # Process date ranges based on scenario and current page
    current_start, current_end, prev_start, prev_end, comparison_label, prev_custom_start = process_date_ranges(
        scenario, max_date
    )


    help_container = st.sidebar.container(key='helpBlock')
    help_cols = help_container.columns([3,1])
    help_cols[0].markdown("<p style='margin-top: 7px; margin-bottom:0'>More info:</p>", unsafe_allow_html=True)
    if help_cols[1].button("", key="help_button", icon=":material/help:", use_container_width=False):
        if 'show_help' not in st.session_state:
            st.session_state.show_help = True
        else:
            st.session_state.show_help = not st.session_state.show_help



    return current_start, current_end, prev_start, prev_end, comparison_label, prev_custom_start, filtered_df
    