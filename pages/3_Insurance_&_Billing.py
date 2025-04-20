import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import initialize_page, load_data, load_clusters, create_sidebar, create_page_navigation, img_to_base64



# Initialize page
initialize_page()

try:
    # Load data
    df = load_data()

    current_start, current_end, prev_start, prev_end, comparison_label, prev_custom_start, filtered_df = create_sidebar(df)
    
    # Set current page for navigation
    st.session_state['current_page'] = 'Insurance & Billing'
    
    st.markdown("""<style>
                [data-testid="stHorizontalBlock"] {
                background-color: white;
                border-radius: 5px;
                padding: 25px;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
                }
                .kpi-container {
                background-color: none;
                padding: 10px;
                padding-top: 4px;
                box-shadow: none;
                }
                .kpi-title {
                color: #3b3b3b;
                font-size: 15px;
                font-weight: 500;
                margin-bottom: 2px;
                }
                .kpi-value {
                font-size: 32px;
                font-weight: bold;
                margin-bottom: 2px;
                }
                .break {
                margin:0px;
                padding:0px;
                padding-right: 30px;
                }
    </style>""", unsafe_allow_html=True)
    
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

    # Get period label for column title (PM, PQ, PY)
    if comparison_label == "":
        period_label = "(no comparison)"
    else:
        period_label = comparison_label.replace("vs ", "")



    # ------ Section 1: Overview ------

    st.markdown("""<h3 class="sub">Billing Overview</h3>""", unsafe_allow_html=True)

    Total_Billing_Amount_current = df_current['Billing Amount'].sum()
    Billing_Amount_Male_current = df_current[df_current['Gender']=='Male']['Billing Amount'].sum()
    Billing_Amount_Female_current = df_current[df_current['Gender']=='Female']['Billing Amount'].sum()
    
    if df_prev is not None:
        Total_Billing_Amount_prev= df_prev['Billing Amount'].sum()   
        Total_Billing_Amount_Difference = ( ( Total_Billing_Amount_current / Total_Billing_Amount_prev ) - 1 ) * 100  
        Billing_Amount_Male_prev = df_prev[df_prev['Gender']=='Male']['Billing Amount'].sum()
        Billing_Amount_Male_Difference = ( ( Billing_Amount_Male_current / Billing_Amount_Male_prev ) - 1 ) * 100
        Billing_Amount_Female_prev = df_prev[df_prev['Gender']=='Female']['Billing Amount'].sum()
        Billing_Amount_Female_Difference = ( ( Billing_Amount_Female_current / Billing_Amount_Female_prev ) - 1 ) * 100
    else:
        Total_Billing_Amount_prev = None
        Total_Billing_Amount_Difference = None
        Billing_Amount_Male_prev = None
        Billing_Amount_Male_Difference = None
        Billing_Amount_Female_prev = None
        Billing_Amount_Female_Difference = None

    
    column1, column2 = st.columns([1, 2.5], gap="large")

    with column1:

        with st.container(key='insuranceKPIs'):

            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Total Billing Amount</div>
                <div class="kpi-value">${Total_Billing_Amount_current:,.0f}</div>
                {f'<div class="kpi-{"up" if Total_Billing_Amount_Difference > 0 else "down"}">{"▲" if Total_Billing_Amount_Difference > 0 else "▼"} {abs(Total_Billing_Amount_Difference):.1f}%</div>' if Total_Billing_Amount_prev is not None else ''}
                {f'<div class="kpi-compare">{comparison_label}: ${Total_Billing_Amount_prev:,.0f}</div>' if Total_Billing_Amount_prev is not None else ''}
            </div>
            """, unsafe_allow_html=True)      

            st.markdown("<hr class='break'>", unsafe_allow_html=True) 

            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Female Billing Amount</div>
                <div class="kpi-value">${Billing_Amount_Female_current:,.0f}</div>
                {f'<div class="kpi-{"up" if Billing_Amount_Female_Difference > 0 else "down"}">{"▲" if Billing_Amount_Female_Difference > 0 else "▼"} {abs(Billing_Amount_Female_Difference):.1f}%</div>' if Billing_Amount_Female_prev is not None else ''}
                {f'<div class="kpi-compare">{comparison_label}: ${Billing_Amount_Female_prev:,.0f}</div>' if Billing_Amount_Female_prev is not None else ''}
            </div>
            """, unsafe_allow_html=True) 


            st.markdown("<hr class='break'>", unsafe_allow_html=True) 

            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Male Billing Amount</div>
                <div class="kpi-value">${Billing_Amount_Male_current:,.0f}</div>
                {f'<div class="kpi-{"up" if Billing_Amount_Male_Difference > 0 else "down"}">{"▲" if Billing_Amount_Male_Difference > 0 else "▼"} {abs(Billing_Amount_Male_Difference):.1f}%</div>' if Billing_Amount_Male_prev is not None else ''}
                {f'<div class="kpi-compare">{comparison_label}: ${Billing_Amount_Male_prev:,.0f}</div>' if Billing_Amount_Male_prev is not None else ''}
            </div>
            """, unsafe_allow_html=True) 

    with column2:
        st.markdown("<h6 style='text-align: left;'>Billing Amount Distribution</h6>", unsafe_allow_html=True)


        # Create histogram data with fixed number of bins
        hist_data = np.histogram(df_current['Billing Amount'], bins=12)
        bin_counts = hist_data[0]
        bin_edges = hist_data[1] / 1000
        
        # Find the bin with the highest count
        max_count_idx = np.argmax(bin_counts)
        
        # Create colors array - highlight the max bin
        colors = ['#c0c0c0'] * len(bin_counts)
        colors[max_count_idx] = '#2f88ff'  # Highlight color for the max bin
        
        # Create bin labels for x-axis
        bin_labels = [f"${bin_edges[i]:,.1f}K - ${bin_edges[i+1]:,.1f}K" for i in range(len(bin_edges)-1)]
        
        # Create the histogram with custom colors
        fig_hist = go.Figure()
        
        # Add bars one by one to control colors
        for i in range(len(bin_counts)):
            fig_hist.add_trace(go.Bar(
                x=[bin_labels[i]],
                y=[bin_counts[i]],
                marker_color=colors[i],
                text=[bin_counts[i]],
                textposition='inside',
                width=0.85,  # Width of bars (adjust as needed)
                name='Most common' if i == max_count_idx else 'Regular'
            ))
        
        # Add annotation for the highest bin
        max_bin_label = bin_labels[max_count_idx]
        fig_hist.add_annotation(
            x=bin_labels[max_count_idx],
            y=bin_counts[max_count_idx],
            text=f"Most common: {max_bin_label}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(color='black', size=12),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(47,136,255,0.3)',
            borderwidth=2,
            borderpad=3
        )
        
        # Update layout
        fig_hist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            bargap=0.15,
            xaxis_title="Billing Amount Range ($)",
            yaxis_title="Number of Patients",
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
            xaxis=dict(
                tickangle=-45,  # Angle the tick labels for better readability
                tickmode='array',
                tickvals=bin_labels
            )
        )
        
        fig_hist.update_yaxes(
            gridcolor='rgba(211,211,211,0.3)'
        )
        
        # Update hover template
        fig_hist.update_traces(
            hovertemplate="Range: %{x}<br>Patients: %{y}<extra></extra>"
        )

        st.plotly_chart(fig_hist, use_container_width=True)


    # ------ Section 2: Insurance Provider Analysis ------

    st.markdown("""
                <div class="numberOfPatients">
                    <h3 class="sub">Insurance Provider Analysis</h3>
                    <p>breakdown by number of patients and average billing amount</p>
                </div>
                """, unsafe_allow_html=True
    )

    # Calculate metrics per hospital for current and previous periods
    insurance_metrics_current = df_current.groupby('Insurance Provider').agg({
        'Patient ID': 'count',
        'Billing Amount': 'mean'
    }).reset_index()
    insurance_metrics_current.columns = ['Insurance Provider', 'Patient_Count', 'Avg_Billing']

    if df_prev is not None:
        insurance_metrics_prev = df_prev.groupby('Insurance Provider').agg({
            'Patient ID': 'count',
            'Billing Amount': 'mean'
        }).reset_index()
        insurance_metrics_prev.columns = ['Insurance Provider', 'Patient_Count', 'Avg_Billing']
        
        # Merge to get both periods in one dataframe
        insurance_metrics = pd.merge(
            insurance_metrics_current, 
            insurance_metrics_prev,
            on='Insurance Provider', 
            suffixes=('_current', '_prev')
        )
        
        # Calculate percent changes
        insurance_metrics['Patient_Change_Pct'] = ((insurance_metrics['Patient_Count_current'] / insurance_metrics['Patient_Count_prev']) - 1) * 100
        insurance_metrics['Billing_Change_Pct'] = ((insurance_metrics['Avg_Billing_current'] / insurance_metrics['Avg_Billing_prev']) - 1) * 100
        
        # Sort by current patient count
        insurance_metrics = insurance_metrics.sort_values('Patient_Count_current', ascending= True)
    else:
        insurance_metrics = insurance_metrics_current
        insurance_metrics = insurance_metrics.sort_values('Patient_Count', ascending=True)

    # Get the order of hospitals
    insurance_providers_order = insurance_metrics['Insurance Provider'].tolist()

    # Create two columns for the visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h6 style='text-align: left;'>Patient Volume by Insurance Provider</h6>", unsafe_allow_html=True)
        
        if df_prev is not None:
            # Create horizontal bar chart with target lines
            fig = go.Figure()
            
            # Add bars for current period
            fig.add_trace(go.Bar(
                y=insurance_metrics['Insurance Provider'],
                x=insurance_metrics['Patient_Count_current'],
                orientation='h',
                name='Current Period',
                marker_color='#2f88ff',
                text=insurance_metrics['Patient_Change_Pct'].apply(lambda pct: f"↑{pct:.1f}%" if pct > 0 else f"↓{abs(pct):.1f}%"),
                textposition='outside',
                insidetextanchor='start',
                textfont=dict(color=insurance_metrics['Patient_Change_Pct'].apply(lambda pct: 'green' if pct > 0 else 'red')),
                hovertemplate=(
                    "Current: %{x:,}<br>"
                    "Previous: %{customdata[0]:,}<br>"
                    "Change: %{text}<extra></extra>"
                ),
                customdata=insurance_metrics[['Patient_Count_prev']]
            ))
            
            # Add target lines for previous period
            for i, insurance_provider in enumerate(insurance_providers_order):
                row = insurance_metrics[insurance_metrics['Insurance Provider'] == insurance_provider].iloc[0]
                fig.add_shape(
                    type='line',
                    y0=i - 0.47,
                    y1=i + 0.47,
                    x0=row['Patient_Count_prev'],
                    x1=row['Patient_Count_prev'],
                    line=dict(color='#333333', width=2, dash='dot'),
                    name=f'Target: {row["Patient_Count_prev"]:,}'
                )
            
            # Add a custom legend for the target line
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(color='#333333', width=2, dash='dot'),
                name=f'Target ({period_label})'
            ))
                
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=100, t=20, b=0),
                xaxis=dict(
                    title="Number of Patients",
                    range=[0, insurance_metrics['Patient_Count_current'].max() * 1.2]  # Increase max value by 20%
                ),
                yaxis_title=None,
                height=370,
                barmode='group',
                bargap=0.25,
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Simple bar chart for current period only
            fig = px.bar(
                insurance_metrics,
                y='Insurance Provider',
                x='Patient_Count',
                orientation='h',
                text='Patient_Count',
                color_discrete_sequence=['#2f88ff']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title="Number of Patients",
                yaxis_title=None,
                height=370,
                bargap=0.25
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<h6 style='text-align: left;'>Average Billing by Insurance Provider</h6>", unsafe_allow_html=True)
        
        if df_prev is not None:
            # Create horizontal bar chart with target lines
            fig = go.Figure()
            
            # Add bars for current period
            fig.add_trace(go.Bar(
                y=insurance_metrics['Insurance Provider'],
                x=insurance_metrics['Avg_Billing_current'],
                orientation='h',
                name='Current Period',
                marker_color='#2f88ff',
                text=insurance_metrics['Billing_Change_Pct'].apply(lambda pct: f"↑{pct:.1f}%" if pct > 0 else f"↓{abs(pct):.1f}%"),
                textposition='outside',
                insidetextanchor='start',
                textfont=dict(color=insurance_metrics['Billing_Change_Pct'].apply(lambda pct: 'green' if pct > 0 else 'red')),
                hovertemplate=(
                    "Current: $%{x:,.0f}<br>"
                    "Previous: $%{customdata[0]:,.0f}<br>"
                    "Change: %{text}<extra></extra>"
                ),
                customdata=insurance_metrics[['Avg_Billing_prev']]
            ))
            
            # Add target lines for previous period
            for i, insurance_provider in enumerate(insurance_providers_order):
                row = insurance_metrics[insurance_metrics['Insurance Provider'] == insurance_provider].iloc[0]
                fig.add_shape(
                    type='line',
                    y0=i - 0.47,
                    y1=i + 0.47,
                    x0=row['Avg_Billing_prev'],
                    x1=row['Avg_Billing_prev'],
                    line=dict(color='#333333', width=2, dash='dot'),
                    name=f'Target: ${row["Avg_Billing_prev"]:,.0f}'
                )
            
            # Add a custom legend for the target line
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(color='#333333', width=2, dash='dot'),
                name=f'Target ({period_label})'
            ))
                
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=100, t=20, b=0),
                xaxis=dict(
                    title="Average Billing Amount ($)",
                    range=[0, insurance_metrics['Avg_Billing_current'].max() * 1.2]  # Increase max value by 20%
                ),
                yaxis_title=None,
                height=370,
                barmode='group',
                bargap=0.25,
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            fig.update_xaxes(
                tickprefix="$",
                tickformat=",",
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Simple bar chart for current period only
            fig = px.bar(
                insurance_metrics,
                y='Insurance Provider',
                x='Avg_Billing',
                orientation='h',
                text=insurance_metrics['Avg_Billing'].apply(lambda x: f"${x:,.0f}"),
                color_discrete_sequence=['#2f88ff']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title="Total Billing Amount ($)",
                yaxis_title=None,
                height=370,
                bargap=0.25
            )
            fig.update_xaxes(
                tickprefix="$",
                tickformat=",",
            )
            st.plotly_chart(fig, use_container_width=True)


    # ------------------------------------------------------------
    # <---       Section 3: Condition-Specific Billing        --->
    # ------------------------------------------------------------

    st.markdown("""<h3 class="sub">Condition-Specific Billing</h3>""", unsafe_allow_html=True)

    conditions_avg_billing_current = df_current.groupby('Medical Condition').agg({
        'Billing Amount': 'mean'
    }).reset_index()
    conditions_avg_billing_current.columns = ['Medical Condition', 'Avg_Billing']

    if df_prev is not None:

        conditions_avg_billing_prev = df_prev.groupby('Medical Condition').agg({
            'Billing Amount': 'mean'
        }).reset_index()

        conditions_avg_billing_prev.columns = ['Medical Condition', 'Avg_Billing']

        # Merge to get both periods in one dataframe
        conditions_avg_billing = pd.merge(
            conditions_avg_billing_current, 
            conditions_avg_billing_prev,
            on='Medical Condition', 
            suffixes=('_current', '_prev')
        )

        conditions_avg_billing['Billing_Change_Pct'] = ((conditions_avg_billing['Avg_Billing_current'] / conditions_avg_billing['Avg_Billing_prev']) - 1) * 100

        # Sort by current patient count
        conditions_avg_billing = conditions_avg_billing.sort_values('Avg_Billing_current', ascending= True)
    else:
        conditions_avg_billing = conditions_avg_billing_current
        conditions_avg_billing = conditions_avg_billing.sort_values('Avg_Billing', ascending=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h6 style='text-align: left;'>Average Billing Amount by Medical Condition</h6>", unsafe_allow_html=True)
        
        if df_prev is not None:
            # Create horizontal bar chart with target lines
            fig = go.Figure()
            
            # Add bars for current period
            fig.add_trace(go.Bar(
                y=conditions_avg_billing['Medical Condition'],
                x=conditions_avg_billing['Avg_Billing_current'],
                orientation='h',
                name='Current Period',
                marker_color='#2f88ff',
                text=conditions_avg_billing['Billing_Change_Pct'].apply(
                    lambda pct: f"↑{pct:.1f}%" if pct > 0 else f"↓{abs(pct):.1f}%"
                ),
                textposition='outside',
                insidetextanchor='start',
                textfont=dict(color=conditions_avg_billing['Billing_Change_Pct'].apply(
                    lambda pct: 'red' if pct < 0 else 'green'  # Increase in inconclusive is bad
                )),
                hovertemplate=(
                    "Current: %{x:,}<br>"
                    "Previous: %{customdata[0]:,}<br>"
                    "Change: %{text}<extra></extra>"
                ),
                customdata=conditions_avg_billing['Avg_Billing_prev']
            ))
            
            # Add target lines for previous period
            for i, medical_condition in enumerate(conditions_avg_billing['Medical Condition']):
                row = conditions_avg_billing[conditions_avg_billing['Medical Condition'] == medical_condition].iloc[0]
                fig.add_shape(
                    type='line',
                    y0=i - 0.4,
                    y1=i + 0.4,
                    x0=row['Avg_Billing_prev'],
                    x1=row['Avg_Billing_prev'],
                    line=dict(color='#333333', width=2, dash='dot'),
                    name='Target'
                )
            
            # Add a custom legend for the target line
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(color='#333333', width=2, dash='dot'),
                name=f'Target ({period_label})'
            ))
                
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=100, t=20, b=0),
                xaxis=dict(
                    title="Average Billing Amount ($)",
                    range=[0, conditions_avg_billing['Avg_Billing_current'].max() * 1.2]  # Increase max by 20%
                ),
                yaxis_title=None,
                height=500,
                barmode='group',
                bargap=0.25,
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            fig.update_xaxes(
                tickprefix="$",
                tickformat=",",
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            # Simple bar chart for current period only
            fig = px.bar(
                conditions_avg_billing,
                y='Medical Condition',
                x='Avg_Billing',
                orientation='h',
                text=conditions_avg_billing['Avg_Billing'].apply(lambda x: f"${x:,.0f}"),
                color_discrete_sequence=['#2f88ff']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title="Average Billing Amount ($)",
                yaxis_title=None,
                bargap=0.25,
                height=500
            )
            fig.update_xaxes(
                tickprefix="$",
                tickformat=",",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<h6 style='text-align: left;'>Distribution of Billing Amount by Medical Condition</h6>", unsafe_allow_html=True)    

        fig = px.box(df_current, x='Medical Condition', y='Billing Amount' )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(
                title="Billing Amount ($)",
                range=[0, df_current['Billing Amount'].max() * 1.2]  # Increase max by 20%
            ),
            xaxis_title='Medical Condition',
            bargap=0.25,
            height=500
        )
        fig.update_yaxes(
            gridcolor='rgba(211,211,211,0.3)'
        )
        
    
        st.plotly_chart(fig, use_container_width=True)


    # ------------------------------------------------------------
    # <---       Section 4: Patient Risk & Billing Clustering Model        --->
    # ------------------------------------------------------------
    image_path = 'assets/images/machine-learning.png'
    st.markdown(f"""
            <div style="display: flex; flex-direction: row; align-items: center; margin-bottom: 10px; margin-top: 30px; gap: 15px;">
            <img src="data:image/png;base64,{img_to_base64(image_path)}" style="width: 56px; height: 56px;">
            <h3 class="sub">Risk & Billing Clustering Model</h3>
            </div>
    """, unsafe_allow_html=True)

    df_clusters = load_clusters()

    cluster_summary = df_clusters.groupby('Cluster').agg({
        'Age': 'mean',
        'Billing Amount': ['mean', 'std'],
        'Length of Stay': ['mean', 'std'],
        'Patient ID': 'count',
        'Medical Condition': lambda x: x.mode().iloc[0],
        'Medication': lambda x: x.mode().iloc[0],
        'Admission Type': lambda x: x.mode().iloc[0]
    }).reset_index()

    cluster_summary.columns = ['Cluster', 'Avg Age', 'Billing Avg', 'Billing Std', 'Length of Stay Avg', 'Length of Stay Std', 'Patient Count', 'Mode Medical Condition', 'Mode Medication', 'Mode Admission Type' ]

    cluster_summary = cluster_summary.sort_values('Billing Avg', ascending=True)


    # Create a more detailed profile for each cluster including risk score
    cluster_profiles = df_clusters.groupby('Cluster').agg({
        'Age': ['mean', 'std'],
        'Length of Stay': ['mean', 'std'],
        'Billing Amount': ['mean', 'std', 'median'],
        'Patient ID': 'count',
        'Insurance Provider': lambda x: x.mode().iloc[0],
        'Medical Condition': lambda x: x.mode().iloc[0],
        'Admission Type': lambda x: x.mode().iloc[0],
        'Gender': lambda x: (x == 'Male').mean() * 100  # Percentage of males
    }).reset_index()

    # Flatten the column multi-index
    cluster_profiles.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] for col in cluster_profiles.columns]

    # Sort by risk (combination of LOS and billing)
    cluster_profiles['Risk_Score'] = (
        (cluster_profiles['Billing Amount_mean'] / cluster_profiles['Billing Amount_mean'].max()) * 0.7 + 
        (cluster_profiles['Length of Stay_mean'] / cluster_profiles['Length of Stay_mean'].max()) * 0.3
    )
    cluster_profiles['Risk_Category'] = pd.qcut(cluster_profiles['Risk_Score'], 3, labels=['Low', 'Medium', 'High'])
    cluster_profiles = cluster_profiles.sort_values('Risk_Score', ascending=False)


    with st.container():

        with st.container():
            column1, column2 = st.columns(2, gap="large")

            with column1:
            
                st.markdown("""
        
                <div style="text-align: left; padding-right: 40px; margin-bottom:20px;">
                <h6 style='text-align: left;'>About the Patient Segmentation Model</h6>
                
                <p>This model identifies distinct patient groups with similar:</p>
                <ul class="sub2">
                    <li>Financial patterns (billing amounts, length of stay, daily costs)</li>
                    <li>Clinical profiles (medical conditions, medications, admission urgency)</li>
                    <li>Demographic traits (age, gender)</li>
                </ul>
                <div style="margin-top: 16px; font-size: 0.9em; color: #666;">
                    <i>Note: Visuals and metrics are based on the entire dataset. Filtering the data by Time period and Hospital slicers will not change the results.</i><br>
                </div>
                </div>
                        
                """, unsafe_allow_html=True)
            with column2:
            
                st.markdown("""
        
                <div style="text-align: left; padding-right: 40px; margin-bottom:20px;">
                            
                <p>This model uses:<br>
                <ul class="sub2">
                    <li>Uses <b>K-Prototypes clustering</b> - a specialized AI technique that analyzes both numerical data and categorical data<br>
                    </li>
                    <li>Silhouette Score: 0.24 (Weak-Moderate Separation)(only appliable to numerical columns)</li>
                    <li>Davies-Bouldin Index: 1.10 (Moderate Cluster Quality)(only appliable to numerical columns)</li>
                </ul>

                <div style="margin-top: 16px; font-size: 0.9em; color: #666;">
                    <i>Note: I believe the model was trained on synthetic data. Real-world performance may vary.</i>
                </div>
                </div>
                        
                """, unsafe_allow_html=True)

        st.markdown('''<div style="margin-top: 15px;"></div>''', unsafe_allow_html=True) # Adds space


        with st.container():
            column1, column2 = st.columns(2, gap="large")    
            with column1:
                # bar chart
                st.markdown("<h6 style='text-align: left;'>Clusters by Average Billing</h6>", unsafe_allow_html=True)

                fig_top = px.bar(cluster_summary, y='Cluster', x='Billing Avg', 
                            labels={'Billing Avg': 'Average Billing ($)'},
                            text=cluster_summary['Billing Avg'].apply(lambda x: f"${x:,.0f}")
                )
                fig_top.update_traces(marker_color='#2f88ff', textposition='inside')
                fig_top.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=350,
                    barmode='group',
                    bargap=0.25
                )
                fig_top.update_xaxes(
                    tickprefix="$",
                    tickformat=",",
                )
                fig_top.update_yaxes(
                    title=None
                )
                st.plotly_chart(fig_top, use_container_width=True)

            
            with column2:
                # Box Plot
                st.markdown("<h6 style='text-align: left;'>Distribution of Billing Amount by Cluster</h6>", unsafe_allow_html=True)    

                fig = px.box(df_clusters, y='Cluster', x='Billing Amount' )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=20, b=0),
                    yaxis=dict(
                        title=None,
                    ),
                    xaxis_title='Billing Amount ($)',
                    bargap=0.25,
                    height=350
                )
                fig.update_yaxes(
                    gridcolor='rgba(211,211,211,0.3)'
                )
                fig.update_xaxes(
                    tickprefix="$",
                    tickformat=",",
                )
            
                st.plotly_chart(fig, use_container_width=True)    

        st.markdown('''<div style="margin-top: 15px;"></div>''', unsafe_allow_html=True) # Adds space

        with st.container():
            column1, column2 = st.columns(2, gap="large")

            # Column 1: Risk Matrix
            with column1:
                st.markdown("<h6 style='text-align: left;'>Patient Cluster Risk Matrix</h6>", unsafe_allow_html=True)
                
                # Create a risk scatter plot
                fig = px.scatter(
                    cluster_profiles, 
                    x='Length of Stay_mean', 
                    y='Billing Amount_mean',
                    size='Patient ID_count',
                    size_max=45,
                    color='Risk_Category',
                    text='Cluster',
                    color_discrete_map={'High': '#ff6b6b', 'Medium': '#ffcc5c', 'Low': '#88d8b0'},
                    hover_data=['Medical Condition_<lambda>', 'Insurance Provider_<lambda>', 'Admission Type_<lambda>']
                )
                
                # Update layout to match style
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=500,
                    xaxis_title="Average Length of Stay (days)",
                    yaxis_title="Average Billing Amount ($)",
                    font=dict(size=10),
                    legend=dict(title="Risk", orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )
                
                # Update hover template
                fig.update_traces(
                    hovertemplate="<b>Cluster %{text}</b><br>" +
                                "Avg Stay: %{x:.1f} days<br>" +
                                "Avg Billing: $%{y:,.0f}<br>" +
                                "Patients: %{marker.size}<br>" +
                                "Main Condition: %{customdata[0]}<br>" +
                                "Insurance: %{customdata[1]}<br>" +
                                "Admission: %{customdata[2]}<extra></extra>"
                )
                
                fig.update_yaxes(tickprefix="$", tickformat=",", gridcolor='rgba(211,211,211,0.3)' )

                st.plotly_chart(fig, use_container_width=True)

            # Column 2: Cluster Insights & Recommendations
            with column2:
                st.markdown("<h6 style='text-align: left;'>Insights & Recommendations</h6>", unsafe_allow_html=True)
                
                # Create actionable insights table
                insights = []
                for _, row in cluster_profiles.iterrows():
                    cluster = row['Cluster']
                    risk = row['Risk_Category']
                    condition = row['Medical Condition_<lambda>']
                    los = row['Length of Stay_mean']
                    billing = row['Billing Amount_mean']
                    patient_count = row['Patient ID_count']
                    age = f"{row['Age_mean']:.1f}"
                    
                    # Generate tailored recommendations based on risk category
                    if risk == 'High':
                        action = f"Urgent: Implement cost-control measures for {condition} patients. Consider standardized care protocols to reduce {los:.1f}-day stays."
                        impact = f"High Impact: ${int(billing * 0.15):,} potential savings per patient"
                    elif risk == 'Medium':
                        action = f"Review: Optimize treatment pathways for {condition} patients to reduce length of stay from {los:.1f} days."
                        impact = f"Medium Impact: ${int(billing * 0.10):,} potential savings per patient"
                    else:
                        action = f"Monitor: Maintain current efficient protocols for {condition} patients with {los:.1f}-day stays."
                        impact = f"Stable: Continue current approach"
                    
                    insights.append({
                        'Cluster': cluster,
                        'Risk': risk,
                        'Average Age': age,
                        'Patients': patient_count,
                        'Primary Condition': condition,
                        'Recommendation': action,
                        'Potential Impact': impact
                    })
                
                insights_df = pd.DataFrame(insights)
                
                # Create table with colors based on risk
                colors = {'High': '#ff6b6b', 'Medium': '#ffcc5c', 'Low': '#88d8b0'}
                
                fig = go.Figure(data=[go.Table(
                    header=dict(
                        values=list(insights_df.columns),
                        fill_color='#fff',
                        align='center',
                        font=dict(color='#333333', size=12),
                        height=40
                    ),
                    cells=dict(
                        values=[insights_df[col] for col in insights_df.columns],
                        fill_color=[['rgba(255,107,107,0.05)' if risk == 'High' else 'rgba(255,204,92,0.08)' if risk == 'Medium' else 'rgba(136,216,176,0.08)' 
                                    for risk in insights_df['Risk']]],
                        align='center',
                        font=dict(color='#333333', size=12),
                        height=35,
                        line=dict(width=1, color='#f0f0f0')
                    )
                    
                )])
                
                fig.update_layout(
                    height=500,
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('''<div style="margin-top: 15px;"></div>''', unsafe_allow_html=True) # Adds space


    # ------------------------------------------------------------
    # <---              --->
    # -----------------------------------------------------------   -
        # After your existing cluster visualization code (around line 691):



    # Cost Reduction Opportunities Section
    

    col1, col2 = st.columns(2, gap="large")

    with col1:

        st.markdown("<h6 style='text-align: left;'>Cost Reduction Opportunities by Cluster</h6>", unsafe_allow_html=True)
        # Calculate potential savings for each cluster
        savings_data = cluster_profiles.copy()
        savings_data['Target_LOS'] = savings_data['Length of Stay_mean'] * 0.85  # 15% reduction target
        savings_data['Daily_Cost'] = savings_data['Billing Amount_mean'] / savings_data['Length of Stay_mean']
        savings_data['Potential_Savings'] = (savings_data['Length of Stay_mean'] - savings_data['Target_LOS']) * savings_data['Daily_Cost'] * savings_data['Patient ID_count']
        savings_data = savings_data.sort_values('Risk_Category', ascending=False)
        
        # Create savings opportunity chart
        fig = px.bar(
            savings_data,
            x='Cluster',
            y='Potential_Savings',
            color='Risk_Category',
            text=savings_data['Potential_Savings'].apply(lambda x: f"${x/1000:,.0f}K"),
            color_discrete_map={'High': '#ff6b6b', 'Medium': '#ffcc5c', 'Low': '#88d8b0'}
            
        )
        fig.update_traces(
            hovertemplate='<b>Cluster %{x}</b><br>' +
                          'Potential Savings: $%{y:,.0f}<br>' +
                          'Risk: %{marker.color}',
            textposition='outside'               
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500,
            yaxis = dict(
                title="Potential Savings ($)",
                gridcolor='rgba(211,211,211,0.3)'
            ),
            xaxis_title="Patient Cluster",
            bargap=0.25,
            legend=dict(title="Risk", orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        fig.update_yaxes(tickprefix="$", tickformat=",")
        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.markdown("<h6 style='text-align: left;'>Length of Stay Reduction Targets</h6>", unsafe_allow_html=True)


        # Length of Stay Reduction Targets
        los_targets = savings_data[['Cluster', 'Length of Stay_mean', 'Target_LOS', 'Risk_Category']].copy()
        los_targets.columns = ['Cluster', 'Current LOS', 'Target LOS', 'Risk']
        los_targets['Reduction'] = los_targets['Current LOS'] - los_targets['Target LOS']
        los_targets = los_targets.sort_values('Risk', ascending=False)

        fig = go.Figure()
        
        # Add bars for current LOS
        fig.add_trace(go.Bar(
            x=los_targets['Cluster'],
            y=los_targets['Current LOS'],
            name='Current Avg Stay',
            marker_color='#2f88ff',
            text=los_targets['Current LOS'].apply(lambda x: f"{x:.1f} days"),
            textposition='outside',
            hovertemplate=(
            "Cluster: %{x}<br>" +
            "Current Stay: %{y:.1f} days<br>" +
            "Target Stay: %{customdata:.1f} days<br>" +
            "<extra></extra>"  # This removes the secondary box in the hover
        ),
        customdata=los_targets['Target LOS']  # Pass target LOS as custom data
        ))
        
        # Add target lines
        fig.add_trace(go.Scatter(
            x=los_targets['Cluster'],
            y=los_targets['Target LOS'],
            mode='markers+lines',
            name='Target Stay',
            marker=dict(size=8, color='#333333'),
            line=dict(dash='dash', width=2, color='#333333'),
            text=los_targets['Target LOS'].apply(lambda x: f"{x:.1f} days"),
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500,
            yaxis = dict(
                title="Length of Stay (days)",
                gridcolor='rgba(211,211,211,0.3)'
            ),
            xaxis_title="Patient Cluster",
            bargap=0.25,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)









except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()