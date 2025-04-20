import streamlit as st
from utils import initialize_page, load_data, create_sidebar, create_page_navigation

# Initialize page
initialize_page()

try:
    # Load data
    df = load_data()
    
    # Create sidebar
    current_start, current_end, prev_start, prev_end, comparison_label, prev_custom_start, filtered_df = create_sidebar(df)
    
    # Create navigation
    create_page_navigation()
    
    # Filter data for current period
    df_current = filtered_df[(filtered_df['Date of Admission'] >= current_start) & 
                            (filtered_df['Date of Admission'] <= current_end)]
    
    # Filter data for previous period
    if prev_start is not None and prev_end is not None:
        df_prev = filtered_df[(filtered_df['Date of Admission'] >= prev_start) & 
                             (filtered_df['Date of Admission'] <= prev_end)]
    else:
        df_prev = None
    
    # Store in session state for other pages
    st.session_state['current_page'] = 'Executive Summary'
    st.session_state['df_current'] = df_current
    st.session_state['df_prev'] = df_prev
    st.session_state['comparison_label'] = comparison_label
    
    # Executive Summary content
    st.markdown("""<h3 class="sub">KPIs:</h3>""", unsafe_allow_html=True)
    

    # Calculate metrics for current period
    avg_length_of_stay = df_current['Length of Stay'].mean()
    avg_treatment_cost = df_current['Billing Amount'].mean()
    elective_admission_pct = (
        len(df_current[df_current['Admission Type'] == 'Elective']) / 
        len(df_current) * 100 if len(df_current) > 0 else 0
    )
    inconclusive_pct = (
        len(df_current[df_current['Test Results'] == 'Inconclusive']) / 
        len(df_current) * 100 if len(df_current) > 0 else 0
    )
    
    # Calculate metrics for previous period (if available)
    if df_prev is not None and len(df_prev) > 0:
        prev_avg_los = df_prev['Length of Stay'].mean()
        prev_avg_cost = df_prev['Billing Amount'].mean()
        prev_elective_pct = (
            len(df_prev[df_prev['Admission Type'] == 'Elective']) / 
            len(df_prev) * 100
        )
        prev_inconclusive_pct = (
            len(df_prev[df_prev['Test Results'] == 'Inconclusive']) / 
            len(df_prev) * 100
        )
        
        # Calculate percent changes
        los_change = ((avg_length_of_stay / prev_avg_los) - 1) * 100 if prev_avg_los > 0 else 0
        cost_change = ((avg_treatment_cost / prev_avg_cost) - 1) * 100 if prev_avg_cost > 0 else 0
        elective_change = elective_admission_pct - prev_elective_pct
        inconclusive_change = inconclusive_pct - prev_inconclusive_pct
    else:
        prev_avg_los = None
        prev_avg_cost = None
        prev_elective_pct = None
        prev_inconclusive_pct = None
        los_change = 0
        cost_change = 0
        elective_change = 0
        inconclusive_change = 0
    
    # Display metrics using custom HTML
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Avg Stay (days)</div>
            <div class="kpi-value">{avg_length_of_stay:.1f}</div>
            {f'<div class="kpi-{"up" if los_change > 0 else "down"}">{"▲" if los_change > 0 else "▼"} {abs(los_change):.1f}%</div>' if prev_avg_los is not None else ''}
            {f'<div class="kpi-compare">{comparison_label}: {prev_avg_los:.1f}</div>' if prev_avg_los is not None else ''}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Elective Admission %</div>
            <div class="kpi-value">{elective_admission_pct:.1f}%</div>
            {f'<div class="kpi-{"up" if elective_change > 0 else "down"}">{"▲" if elective_change > 0 else "▼"} {abs(elective_change):.1f}%</div>' if prev_elective_pct is not None else ''}
            {f'<div class="kpi-compare">{comparison_label}: {prev_elective_pct:.1f}%</div>' if prev_elective_pct is not None else ''}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Avg Treatment Cost</div>
            <div class="kpi-value">${avg_treatment_cost:,.0f}</div>
            {f'<div class="kpi-{"down" if cost_change < 0 else "up"}">{"▼" if cost_change < 0 else "▲"} {abs(cost_change):.1f}%</div>' if prev_avg_cost is not None else ''}
            {f'<div class="kpi-compare">{comparison_label}: ${prev_avg_cost:,.0f}</div>' if prev_avg_cost is not None else ''}
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Inconclusive Results</div>
            <div class="kpi-value">{inconclusive_pct:.1f}%</div>
            {f'<div class="kpi-{"up" if inconclusive_change < 0 else "down"}">{"▼" if inconclusive_change < 0 else "▲"} {abs(inconclusive_change):.1f}%</div>' if prev_inconclusive_pct is not None else ''}
            {f'<div class="kpi-compare">{comparison_label}: {prev_inconclusive_pct:.1f}%</div>' if prev_inconclusive_pct is not None else ''}
        </div>
        """, unsafe_allow_html=True)
    
    # Insurance Provider section with title
    st.markdown(f"""
                <div class="numberOfPatients">
                    <h3 class="sub">Total Patients: {len(df_current):,.0f}</h3>
                    <p>breakdown by Insurance Providers:</p>
                </div>
                """, unsafe_allow_html=True)

    # Calculate Insurance Provider metrics
    insurance_counts = df_current['Insurance Provider'].value_counts()
    
    if df_prev is not None:
        prev_insurance_counts = df_prev['Insurance Provider'].value_counts()
    else:
        prev_insurance_counts = None
    
    # Create columns for insurance metrics
    ins_cols = st.columns(len(insurance_counts))
    
    # Display insurance provider metrics with comparison
    for idx, (provider, count) in enumerate(insurance_counts.items()):
        with ins_cols[idx]:
            if df_prev is not None and provider in prev_insurance_counts:
                prev_count = prev_insurance_counts[provider]
                if prev_count > 0:
                    change_pct = ((count / prev_count) - 1) * 100
                    st.markdown(f"""
                    <div class="kpi-container">
                        <div class="kpi-title">{provider}</div>
                        <div class="kpi-value">{count:,.0f}</div>
                        <div class="kpi-{"up" if change_pct > 0 else "down"}">{"▲" if change_pct > 0 else "▼"} {abs(change_pct):.1f}%</div>
                        <div class="kpi-compare">{comparison_label}: {prev_count:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-title">{provider}</div>
                    <div class="kpi-value">{count:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)



except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()