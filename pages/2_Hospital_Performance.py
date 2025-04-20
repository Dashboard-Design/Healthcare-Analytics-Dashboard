import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import initialize_page, load_data, create_sidebar, create_page_navigation

# Initialize page
initialize_page()

try:
    # Load data
    df = load_data()
    
    current_start, current_end, prev_start, prev_end, comparison_label, prev_custom_start, filtered_df = create_sidebar(df)
    
    # Set current page for navigation
    st.session_state['current_page'] = 'Hospital Performance'
    
    st.markdown("""<style>
                [data-testid="stHorizontalBlock"] {
                background-color: white;
                border-radius: 5px;
                padding: 25px;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
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

    # ------ Hospital Performance Metrics for Underperforming Analysis ------
    # Calculate metrics per hospital for current and previous periods
    hospital_metrics_current = df_current.groupby('Hospital').agg({
        'Patient ID': 'count',
        'Billing Amount': 'sum'
    }).reset_index()
    hospital_metrics_current.columns = ['Hospital', 'Patient_Count', 'Total_Billing']
    
    # Determine underperforming hospitals if there's a previous period
    underperforming_hospitals = []
    
    if df_prev is not None:
        hospital_metrics_prev = df_prev.groupby('Hospital').agg({
            'Patient ID': 'count',
            'Billing Amount': 'sum'
        }).reset_index()
        hospital_metrics_prev.columns = ['Hospital', 'Patient_Count', 'Total_Billing']
        
        # Merge to get both periods in one dataframe
        hospital_metrics = pd.merge(
            hospital_metrics_current, 
            hospital_metrics_prev,
            on='Hospital', 
            suffixes=('_current', '_prev')
        )
        
        # Calculate percent changes
        hospital_metrics['Patient_Change_Pct'] = ((hospital_metrics['Patient_Count_current'] / hospital_metrics['Patient_Count_prev']) - 1) * 100
        hospital_metrics['Billing_Change_Pct'] = ((hospital_metrics['Total_Billing_current'] / hospital_metrics['Total_Billing_prev']) - 1) * 100
        
        # Identify underperforming hospitals (negative patient change)
        underperforming_df = hospital_metrics[hospital_metrics['Patient_Change_Pct'] < 0].sort_values('Patient_Change_Pct')
        underperforming_hospitals = underperforming_df['Hospital'].tolist()
        
        # Sort by current patient count
        hospital_metrics = hospital_metrics.sort_values('Patient_Count_current', ascending=True)
    else:
        hospital_metrics = hospital_metrics_current
        hospital_metrics = hospital_metrics.sort_values('Patient_Count', ascending=True)

    # Get the order of hospitals
    hospitals_order = hospital_metrics['Hospital'].tolist()



    # Create a DataFrame with hospital locations and patient counts
    df_map = df_current.groupby(['Hospital', 'Hospital Latitude', 'Hospital Longitude']).agg({
        'Patient ID': 'count',  # Count of patients
        'Billing Amount': 'sum'  # Total billing amount
    }).reset_index()

    df_map.columns = ['Hospital', 'Latitude', 'Longitude', 'Patient_Count', 'Total_Billing']

    # Normalize the patient count to get reasonable bubble sizes
    max_size = df_map['Patient_Count'].max()
    df_map['size'] = df_map['Patient_Count'] / max_size * 2000  # Adjust the multiplier (2000) to change bubble sizes

    # Create hover text
    df_map['hover_text'] = df_map.apply(
        lambda row: f"<b>{row['Hospital']}</b><br>" +
                f"Patients: {row['Patient_Count']:,}<br>" +
                f"Total Billing: ${row['Total_Billing']:,.0f}",
        axis=1
    )


    # ------
    # ------ Underperforming Hospitals Section ------
    # ------

    if df_prev is not None and len(underperforming_hospitals) > 0:
        st.markdown("""<h3 class="sub">Hospital Performance Insights</h3>""", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            # Calculate total percentage of underperforming hospitals
            underperforming_pct = len(underperforming_hospitals) / len(hospital_metrics) * 100
            
            st.markdown(f"""
            <div class="underperforming">
                <h6 style="color: #d32f2f; margin-top: 0;">Underperforming Hospitals</h6>
                <p>{len(underperforming_hospitals)} out of {len(hospital_metrics)} hospitals ({underperforming_pct:.1f}%) 
                are seeing fewer patients compared to {period_label}.</p>
                <div class="underperforming-list">
                    {"".join([f'<div class="underperforming-item">• {hospital} ({underperforming_df[underperforming_df["Hospital"]==hospital]["Patient_Change_Pct"].values[0]:.1f}%)</div>' for hospital in underperforming_hospitals])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Initialize session state for toggle if not exists
            if 'highlight_underperforming' not in st.session_state:
                st.session_state.highlight_underperforming = False
                
            # Add toggle button with container for alignment
            st.markdown("""
            <div class="toggle-container">
            """, unsafe_allow_html=True)
            
            st.toggle(
                "Highlight underperforming hospitals",
                key="highlight_underperforming",
                help="When enabled, hospitals with decreasing patient numbers will be highlighted while others appear gray"
            )
            
            st.markdown("""
            <div style="margin-top: 13px; margin-right: 25px; font-size: 0.9em; color: #666;">
                <i>Note: Underperforming hospitals are defined as hospitals with a negative patient change compared to the previous period. Selecting the custom option in the time period slicer will make the underperforming hospitals section disappear.</i>
            </div>
            </div>
            """, unsafe_allow_html=True)
    
    
    # Function to determine marker color based on hospital performance
    def get_hospital_color(hospital):
        if not st.session_state.get('highlight_underperforming', False) or df_prev is None:
            return '#2f88ff'  # Default blue
        elif hospital in underperforming_hospitals:
            return '#ff6b6b'  # Red 
        else:
            return '#c0c0c0'  # Gray 

    # ------
    # Map Section
    # ------

    # ------ Map ------
    st.markdown("""<h3 class="sub">Map</h3>""", unsafe_allow_html=True)    
    
    fig = go.Figure()

    # Add scatter mapbox trace with conditional colors
    fig.add_trace(go.Scattermapbox(
        lat=df_map['Latitude'],
        lon=df_map['Longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=(df_map['Patient_Count'] / df_map['Patient_Count'].max()) * 100,  # Normalize size
            color=[get_hospital_color(hospital) for hospital in df_map['Hospital']],
            opacity=0.8
        ),
        text=df_map['hover_text'],
        hoverinfo='text'
    ))

    # Update layout
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',  #'carto-positron',
            zoom=3,  # Adjust zoom level as needed
            center=dict(lat=38, lon=-98.5795)  # Center of USA
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
        showlegend=False
    )

    # Display the map
    st.plotly_chart(fig, use_container_width=True)

    # Add caption based on toggle state
    if st.session_state.get('highlight_underperforming', False) and df_prev is not None and len(underperforming_hospitals) > 0:
        caption = "Bubble sizes represent the number of patients. Red bubbles indicate hospitals with decreasing patient numbers compared to the previous period."
    else:
        caption = "Bubble sizes represent the number of patients in the current period at each hospital location."
    
    st.caption(caption)


    # ------
    # ------ Hospital Performance Metrics ------
    # ------

    st.markdown("""
                <div class="numberOfPatients">
                    <h3 class="sub">Hospital Performance</h3>
                    <p>breakdown by number of patients and total billing amount</p>
                </div>
                """, unsafe_allow_html=True
    )

    # Create two columns for the visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h6 style='text-align: left;'>Patient Volume by Hospital</h6>", unsafe_allow_html=True)
        
        if df_prev is not None:
            # Create horizontal bar chart with target lines
            fig = go.Figure()
            
            # Determine colors based on toggle state
            bar_colors = []
            for hospital in hospital_metrics['Hospital']:
                if st.session_state.get('highlight_underperforming', False):
                    if hospital in underperforming_hospitals:
                        bar_colors.append('#ff6b6b')  # Red for underperforming
                    else:
                        bar_colors.append('#c0c0c0')  # Gray for others
                else:
                    bar_colors.append('#2f88ff')  # Default blue for all
            
            # Add bars for current period
            fig.add_trace(go.Bar(
                y=hospital_metrics['Hospital'],
                x=hospital_metrics['Patient_Count_current'],
                orientation='h',
                name='Current Period',
                marker_color=bar_colors,
                text=hospital_metrics['Patient_Change_Pct'].apply(lambda pct: f"↑{pct:.1f}%" if pct > 0 else f"↓{abs(pct):.1f}%"),
                textposition='outside',
                insidetextanchor='start',
                textfont=dict(color=hospital_metrics['Patient_Change_Pct'].apply(lambda pct: 'green' if pct > 0 else 'red')),
                hovertemplate=(
                    "Current: %{x:,}<br>"
                    "Previous: %{customdata[0]:,}<br>"
                    "Change: %{text}<extra></extra>"
                ),
                customdata=hospital_metrics[['Patient_Count_prev']],
                showlegend=False
            ))

            # Add a custom trace just for the legend that's always gray when toggle is on
            legend_color = '#c0c0c0' if st.session_state.get('highlight_underperforming', False) else '#2f88ff'
            fig.add_trace(go.Bar(
                y=[None],
                x=[None],
                orientation='h',
                name='Current Period',
                marker_color=legend_color,
                showlegend=True
            ))
            
            # Add target lines for previous period
            for i, hospital in enumerate(hospitals_order):
                row = hospital_metrics[hospital_metrics['Hospital'] == hospital].iloc[0]
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
                    range=[0, hospital_metrics['Patient_Count_current'].max() * 1.2]  # Increase max value by 20%
                ),
                yaxis_title=None,
                height=500,
                barmode='group',
                bargap=0.25,
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Simple bar chart for current period only
            fig = px.bar(
                hospital_metrics,
                y='Hospital',
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
                height=500,
                bargap=0.25
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<h6 style='text-align: left;'>Total Billing by Hospital</h6>", unsafe_allow_html=True)
        
        if df_prev is not None:
            # Create horizontal bar chart with target lines
            fig = go.Figure()
            
            # Determine colors based on toggle state
            bar_colors = []
            for hospital in hospital_metrics['Hospital']:
                if st.session_state.get('highlight_underperforming', False):
                    if hospital in underperforming_hospitals:
                        bar_colors.append('#ff6b6b')  # Red for underperforming
                    else:
                        bar_colors.append('#c0c0c0')  # Gray for others
                else:
                    bar_colors.append('#2f88ff')  # Default blue for all
            
            # Add bars for current period
            fig.add_trace(go.Bar(
                y=hospital_metrics['Hospital'],
                x=hospital_metrics['Total_Billing_current'],
                orientation='h',
                name='Current Period',
                marker_color=bar_colors,
                text=hospital_metrics['Billing_Change_Pct'].apply(lambda pct: f"↑{pct:.1f}%" if pct > 0 else f"↓{abs(pct):.1f}%"),
                textposition='outside',
                insidetextanchor='start',
                textfont=dict(color=hospital_metrics['Billing_Change_Pct'].apply(lambda pct: 'green' if pct > 0 else 'red')),
                hovertemplate=(
                    "Current: $%{x:,.0f}<br>"
                    "Previous: $%{customdata[0]:,.0f}<br>"
                    "Change: %{text}<extra></extra>"
                ),
                customdata=hospital_metrics[['Total_Billing_prev']],
                showlegend=False
            ))

            # Add a custom trace just for the legend that's always gray when toggle is on
            legend_color = '#c0c0c0' if st.session_state.get('highlight_underperforming', False) else '#2f88ff'
            fig.add_trace(go.Bar(
                y=[None],
                x=[None],
                orientation='h',
                name='Current Period',
                marker_color=legend_color,
                showlegend=True
            ))
            
            # Add target lines for previous period
            for i, hospital in enumerate(hospitals_order):
                row = hospital_metrics[hospital_metrics['Hospital'] == hospital].iloc[0]
                fig.add_shape(
                    type='line',
                    y0=i - 0.47,
                    y1=i + 0.47,
                    x0=row['Total_Billing_prev'],
                    x1=row['Total_Billing_prev'],
                    line=dict(color='#333333', width=2, dash='dot'),
                    name=f'Target: ${row["Total_Billing_prev"]:,.0f}'
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
                    title="Total Billing Amount ($)",
                    range=[0, hospital_metrics['Total_Billing_current'].max() * 1.2]  # Increase max value by 20%
                ),
                yaxis_title=None,
                height=500,
                barmode='group',
                bargap=0.25,
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Simple bar chart for current period only
            fig = px.bar(
                hospital_metrics,
                y='Hospital',
                x='Total_Billing',
                orientation='h',
                text=hospital_metrics['Total_Billing'].apply(lambda x: f"${x:,.0f}"),
                color_discrete_sequence=['#2f88ff']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title="Total Billing Amount ($)",
                yaxis_title=None,
                height=500,
                bargap=0.25
            )
            st.plotly_chart(fig, use_container_width=True)


    # ------
    # ------ Hospital Performance: Stay Duration and Inconclusive Results ------
    # ------

    st.markdown("""
        <div class="numberOfPatients">
            <h3 class="sub">Hospital Performance</h3>
            <p>breakdown by avg stay and inconclusive results</p>
        </div>
        """, unsafe_allow_html=True
    )

    # For Average Length of Stay
    stay_metrics_current = df_current.groupby('Hospital')['Length of Stay'].mean().reset_index()
    stay_metrics_current.columns = ['Hospital', 'Avg_Stay']

    # For Inconclusive Results 
    inconclusive_current = df_current.groupby('Hospital').apply(
        lambda x: x[x['Test Results'] == 'Inconclusive'].shape[0]  ).reset_index()
    inconclusive_current.columns = ['Hospital', 'Inconclusive_Count']

    if df_prev is not None:
        # For Average Length of Stay
        stay_metrics_prev = df_prev.groupby('Hospital')['Length of Stay'].mean().reset_index()
        stay_metrics_prev.columns = ['Hospital', 'Avg_Stay']
        
        # For Inconclusive Results
        inconclusive_prev = df_prev.groupby('Hospital').apply(
            lambda x: x[x['Test Results'] == 'Inconclusive'].shape[0]  
        ).reset_index()
        inconclusive_prev.columns = ['Hospital', 'Inconclusive_Count']
        
        # Merge to get both periods in one dataframe
        stay_metrics = pd.merge(
            stay_metrics_current, 
            stay_metrics_prev,
            on='Hospital', 
            suffixes=('_current', '_prev')
        )
        
        inconclusive_metrics = pd.merge(
            inconclusive_current, 
            inconclusive_prev,
            on='Hospital', 
            suffixes=('_current', '_prev')
        )
        
        # Calculate percent changes
        stay_metrics['Stay_Change_Pct'] = ((stay_metrics['Avg_Stay_current'] / stay_metrics['Avg_Stay_prev']) - 1) * 100
        
        inconclusive_metrics['Inconclusive_Change_Pct'] = inconclusive_metrics.apply(
            lambda row: row['Inconclusive_Count_current'] * 100 if row['Inconclusive_Count_prev'] == 0
            else ((row['Inconclusive_Count_current'] / row['Inconclusive_Count_prev']) - 1) * 100,
            axis=1
        )
        
        # Sort by current averages
        stay_metrics = stay_metrics.sort_values('Avg_Stay_current', ascending=True)
        inconclusive_metrics = inconclusive_metrics.sort_values('Inconclusive_Count_current', ascending=True)
    else:
        stay_metrics = stay_metrics_current.sort_values('Avg_Stay', ascending=True)
        inconclusive_metrics = inconclusive_current.sort_values('Inconclusive_Count', ascending=True)

    # Get the order of hospitals
    hospitals_order_stay = stay_metrics['Hospital'].tolist()
    hospitals_order_inconclusive = inconclusive_metrics['Hospital'].tolist()

    # Create two columns for the visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h6 style='text-align: left;'>Average Length of Stay by Hospital</h6>", unsafe_allow_html=True)
        
        if df_prev is not None:
            # Create horizontal bar chart with target lines
            fig = go.Figure()
            
            # Determine colors based on toggle state
            bar_colors = []
            for hospital in stay_metrics['Hospital']:
                if st.session_state.get('highlight_underperforming', False):
                    if hospital in underperforming_hospitals:
                        bar_colors.append('#ff6b6b')  # Red for underperforming
                    else:
                        bar_colors.append('#c0c0c0')  # Gray for others
                else:
                    bar_colors.append('#2f88ff')  # Default blue for all
            
            # Add bars for current period
            fig.add_trace(go.Bar(
                y=stay_metrics['Hospital'],
                x=stay_metrics['Avg_Stay_current'],
                orientation='h',
                name='Current Period',
                marker_color=bar_colors,
                text=stay_metrics['Stay_Change_Pct'].apply(lambda pct: f"↑{pct:.1f}%" if pct > 0 else f"↓{abs(pct):.1f}%"),
                textposition='outside',
                insidetextanchor='start',
                textfont=dict(color=stay_metrics['Stay_Change_Pct'].apply(lambda pct: 'red' if pct > 0 else 'green')),  # Lower stay is better
                hovertemplate=(
                    "Current: %{x:.1f} days<br>"
                    "Previous: %{customdata[0]:.1f} days<br>"
                    "Change: %{text}<extra></extra>"
                ),
                customdata=stay_metrics[['Avg_Stay_prev']],
                showlegend=False
            ))

            # Add a custom trace just for the legend that's always gray when toggle is on
            legend_color = '#c0c0c0' if st.session_state.get('highlight_underperforming', False) else '#2f88ff'
            fig.add_trace(go.Bar(
                y=[None],
                x=[None],
                orientation='h',
                name='Current Period',
                marker_color=legend_color,
                showlegend=True
            ))
            
            # Add target lines for previous period
            for i, hospital in enumerate(hospitals_order_stay):
                row = stay_metrics[stay_metrics['Hospital'] == hospital].iloc[0]
                fig.add_shape(
                    type='line',
                    y0=i - 0.4,
                    y1=i + 0.4,
                    x0=row['Avg_Stay_prev'],
                    x1=row['Avg_Stay_prev'],
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
                    title="Average Stay (days)",
                    range=[0, stay_metrics['Avg_Stay_current'].max() * 1.2]  # Increase max by 20%
                ),
                yaxis_title=None,
                height=500,
                barmode='group',
                bargap=0.25,
                showlegend=True,
                legend=dict( orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Simple bar chart for current period only
            fig = px.bar(
                stay_metrics,
                y='Hospital',
                x='Avg_Stay',
                orientation='h',
                text=stay_metrics['Avg_Stay'].apply(lambda x: f"{x:,.1f}"), 
                color_discrete_sequence=['#2f88ff']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title="Average Stay (days)",
                yaxis_title=None,
                bargap=0.25,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<h6 style='text-align: left;'>Inconclusive Results by Hospital</h6>", unsafe_allow_html=True)
        
        if df_prev is not None:
            # Create horizontal bar chart with target lines
            fig = go.Figure()
            
            # Determine colors based on toggle state
            bar_colors = []
            for hospital in inconclusive_metrics['Hospital']:
                if st.session_state.get('highlight_underperforming', False):
                    if hospital in underperforming_hospitals:
                        bar_colors.append('#ff6b6b')  # Red for underperforming
                    else:
                        bar_colors.append('#c0c0c0')  # Gray for others
                else:
                    bar_colors.append('#2f88ff')  # Default blue for all
            
            # Add bars for current period
            fig.add_trace(go.Bar(
                y=inconclusive_metrics['Hospital'],
                x=inconclusive_metrics['Inconclusive_Count_current'],
                orientation='h',
                name='Current Period',
                marker_color=bar_colors,
                text=inconclusive_metrics['Inconclusive_Change_Pct'].apply(
                    lambda pct: f"↑{pct:.1f}%" if pct > 0 else f"↓{abs(pct):.1f}%"
                ),
                textposition='outside',
                insidetextanchor='start',
                textfont=dict(color=inconclusive_metrics['Inconclusive_Change_Pct'].apply(
                    lambda pct: 'red' if pct > 0 else 'green'  # Increase in inconclusive is bad
                )),
                hovertemplate=(
                    "Current: %{x:,}<br>"
                    "Previous: %{customdata[0]:,}<br>"
                    "Change: %{text}<extra></extra>"
                ),
                customdata=inconclusive_metrics[['Inconclusive_Count_prev']],
                showlegend=False
            ))

            # Add a custom trace just for the legend that's always gray when toggle is on
            legend_color = '#c0c0c0' if st.session_state.get('highlight_underperforming', False) else '#2f88ff'
            fig.add_trace(go.Bar(
                y=[None],
                x=[None],
                orientation='h',
                name='Current Period',
                marker_color=legend_color,
                showlegend=True
            ))
            
            # Add target lines for previous period
            for i, hospital in enumerate(hospitals_order_inconclusive):
                row = inconclusive_metrics[inconclusive_metrics['Hospital'] == hospital].iloc[0]
                fig.add_shape(
                    type='line',
                    y0=i - 0.4,
                    y1=i + 0.4,
                    x0=row['Inconclusive_Count_prev'],
                    x1=row['Inconclusive_Count_prev'],
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
                    title="Number of Inconclusive Results",
                    range=[0, inconclusive_metrics['Inconclusive_Count_current'].max() * 1.2]  # Increase max by 20%
                ),
                yaxis_title=None,
                height=500,
                barmode='group',
                bargap=0.25,
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Simple bar chart for current period only
            fig = px.bar(
                inconclusive_metrics,
                y='Hospital',
                x='Inconclusive_Count',
                orientation='h',
                text='Inconclusive_Count',
                color_discrete_sequence=['#2f88ff']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title="Number of Inconclusive Results",
                yaxis_title=None,
                bargap=0.25,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()