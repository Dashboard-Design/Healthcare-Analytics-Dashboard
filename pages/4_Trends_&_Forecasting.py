import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta
from utils import initialize_page, load_data, create_sidebar, create_page_navigation

# Initialize page
initialize_page()

try:
    # Load data
    df = load_data()
    
    # Set current page for navigation
    st.session_state['current_page'] = 'Trends & Forecasting'

    current_start, current_end, prev_start, prev_end, comparison_label, prev_custom_start, filtered_df = create_sidebar(df)
    
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

    if prev_custom_start is not None:
        df_completed  = filtered_df[(filtered_df['Date of Admission'] <= current_end) & (filtered_df['Date of Admission'] >= prev_custom_start)]
        df_completed_section2 = filtered_df[filtered_df['Date of Admission'] <= current_end]
    else:
        df_completed  = filtered_df[filtered_df['Date of Admission'] <= current_end]
        df_completed_section2 = filtered_df[filtered_df['Date of Admission'] <= current_end]

    # Filter data for current period
    df_current = filtered_df[(filtered_df['Date of Admission'] >= current_start) & 
                            (filtered_df['Date of Admission'] <= current_end)]
    
    # Filter data for previous period
    if prev_start is not None and prev_end is not None:
        last_day = pd.Timestamp( year= prev_end.year, month = prev_end.month + 1, day=1 ) - timedelta(days=1)
        if prev_end != last_day:
            df_prev = filtered_df[(filtered_df['Date of Admission'] >= prev_start) & 
                                (filtered_df['Date of Admission'] <= last_day)]
        else:
            df_prev = filtered_df[(filtered_df['Date of Admission'] >= prev_start) & 
                                (filtered_df['Date of Admission'] <= prev_end)]
    else:
        df_prev = None

    # Get period label for column title (PM, PQ, PY)
    if comparison_label == "":
        period_label = "(no comparison)"
    else:
        period_label = comparison_label.replace("vs ", "")
    
    
    # 
    # ------ Section 1: Historical Patient Volume Trends ------
    # 

    st.markdown("""
        <div class="numberOfPatients">
            <h3 class="sub">Patient Volume Trends</h3>
            <p>monthly analysis of patient admissions</p>
        </div>
        """, unsafe_allow_html=True
    )
    
    with st.container(key='background'):
        # Create monthly aggregation for the entire dataset using the filtered data
        df_monthly = df_completed.copy()
        df_monthly['month_year'] = df_monthly['Date of Admission'].dt.to_period('M')
        monthly_patients = df_monthly.groupby(['month_year']).agg({
            'Patient ID': 'count'
        }).reset_index()
        
        # Convert period to datetime for plotting
        monthly_patients['month_year'] = monthly_patients['month_year'].dt.to_timestamp()
        monthly_patients.sort_values('month_year', inplace=True)
        monthly_patients.rename(columns={'Patient ID': 'Patient_Count'}, inplace=True)
        
        # Create marker sizes and colors for highlighting
        marker_sizes = [7] * len(monthly_patients)
        marker_colors = ['#2f88ff'] * len(monthly_patients)
        
        # Set up specific data for the selected periods
        current_data = None
        prev_data = None
        
        if df_prev is not None:
            # Aggregate data for current and previous periods
            df_current['month_year'] = df_current['Date of Admission'].dt.to_period('M')
            current_monthly = df_current.groupby(['month_year']).agg({
                'Patient ID': 'count'
            }).reset_index()
            current_monthly['month_year'] = current_monthly['month_year'].dt.to_timestamp()
            current_monthly.rename(columns={'Patient ID': 'Patient_Count'}, inplace=True)
            
            current_data = {
                'x': current_monthly['month_year'].tolist(),
                'y': current_monthly['Patient_Count'].tolist()
            }
            
            df_prev['month_year'] = df_prev['Date of Admission'].dt.to_period('M')
            prev_monthly = df_prev.groupby(['month_year']).agg({
                'Patient ID': 'count'
            }).reset_index()
            prev_monthly['month_year'] = prev_monthly['month_year'].dt.to_timestamp()
            prev_monthly.rename(columns={'Patient ID': 'Patient_Count'}, inplace=True)
            
            prev_data = {
                'x': prev_monthly['month_year'].tolist(),
                'y': prev_monthly['Patient_Count'].tolist()
            }
        
        # Create line chart
        fig = go.Figure()
        
        # Add main trend line
        fig.add_trace(go.Scatter(
            x=monthly_patients['month_year'],
            y=monthly_patients['Patient_Count'],
            mode='lines+markers',
            name='Monthly Patient Count',
            line=dict(color='#2f88ff', width=2),
            marker=dict(size=marker_sizes, color=marker_colors),
            hovertemplate='%{x|%b %Y}<br>Patients: %{y:,}<extra></extra>'
        ))
        
        # Add highlighted points for current period
        if current_data and current_data['x']:
            fig.add_trace(go.Scatter(
                x=current_data['x'],
                y=current_data['y'],
                mode='markers',
                name='Current Period',
                marker=dict(color='#0000FF', size=12, line=dict(width=2, color='rgba(0,0,0,0)')),
                hovertemplate='Current Period<br>%{x|%b %Y}<br>Patients: %{y:,}<extra></extra>'
            ))
        
        # Add highlighted points for previous period
        if prev_data and prev_data['x']:
            fig.add_trace(go.Scatter(
                x=prev_data['x'],
                y=prev_data['y'],
                mode='markers',
                name=f'Previous Period ({period_label})',
                marker=dict(color='#FFA726', size=12, line=dict(width=2, color='rgba(0,0,0,0)')),
                hovertemplate='Previous Period<br>%{x|%b %Y}<br>Patients: %{y:,}<extra></extra>'
            ))
        
        # Calculate moving average for trend line
        window_size = 3
        if len(monthly_patients) > window_size:
            monthly_patients_copy = monthly_patients.copy()
            monthly_patients_copy['MA'] = monthly_patients_copy['Patient_Count'].rolling(window=window_size).mean()
            
            fig.add_trace(go.Scatter(
                x=monthly_patients_copy['month_year'],
                y=monthly_patients_copy['MA'],
                mode='lines',
                name=f'{window_size}-Month Moving Average',
                line=dict(color='rgba(0,0,0,0.35)', width=2, dash='dot'),
                hovertemplate='%{x|%b %Y}<br>Moving Avg: %{y:.1f}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title='',
            xaxis_title=None,
            yaxis_title=None,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=400,
            xaxis=dict(
                showgrid=False,
                showline=True,
                linecolor='rgba(211,211,211,0.7)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(211,211,211,0.2)',
                title_font=dict(size=14)
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14
            )
        )
        
        st.plotly_chart(fig, use_container_width=False)
        
        # Add caption about the chart
        st.caption("Note: Chart shows patient volume by complete months. The current and previous periods (if selected) are highlighted with larger markers.")

    
    # 
    # ------ Section 2: trend insights ------
    #

    df_monthly2 = df_completed_section2.copy()

    df_monthly2['month_year'] = df_monthly2['Date of Admission'].dt.to_period('M')
    monthly_patients2 = df_monthly2.groupby(['month_year']).agg({
        'Patient ID': 'count'
    }).reset_index()
        
    # Convert period to datetime for plotting
    monthly_patients2['month_year'] = monthly_patients2['month_year'].dt.to_timestamp()   
    monthly_patients2.sort_values('month_year', inplace=True)
    monthly_patients2.rename(columns={'Patient ID': 'Patient_Count'}, inplace=True)
    
    st.markdown("""
        <div class="numberOfPatients">
            <h3 class="sub">Trend Insights</h3>
            <p>Full historical analysis (excluding incomplete current-year data) with hospital-level filtering</p>
        </div>
        """, unsafe_allow_html=True
    )
    
    
    col1, col2 = st.columns(2, gap='large')
    
    with col1:
        st.markdown("<h6 style='text-align: left;'>Key Takeaways</h6>", unsafe_allow_html=True)
        
        # Calculate percentage change from first to last month
        first_month = monthly_patients2.iloc[0]['Patient_Count'] 
        last_month = monthly_patients2.iloc[-1]['Patient_Count']
        total_change_pct = ((last_month - first_month) / first_month) * 100
        
        # Get months with highest and lowest patient counts
        max_month = monthly_patients2.loc[monthly_patients2['Patient_Count'].idxmax()]
        min_month = monthly_patients2.loc[monthly_patients2['Patient_Count'].idxmin()]
        
        # Calculate month-over-month growth
        monthly_patients_growth = monthly_patients2.copy()
        monthly_patients_growth['MoM_Growth'] = monthly_patients_growth['Patient_Count'].pct_change() * 100
        avg_growth = monthly_patients_growth['MoM_Growth'].dropna().mean()
        
        # Calculate recent trend (last 3 months or fewer if not enough data)
        recent_months_count = min(3, len(monthly_patients2))
        recent_months = monthly_patients2.tail(recent_months_count)
        if recent_months_count > 1:
            recent_growth = (recent_months.iloc[-1]['Patient_Count'] / recent_months.iloc[0]['Patient_Count'] - 1) * 100
        else:
            recent_growth = 0
        
        # Year-over-year comparison if we have enough data
        yoy_change = None
        if len(monthly_patients2) >= 13:
            last_year_same_month = monthly_patients2.iloc[-13]['Patient_Count']
            yoy_change = ((last_month - last_year_same_month) / last_year_same_month) * 100
        
        st.markdown(f"""
        <ul style="display: flex; flex-direction: column; gap: 12px; list-style-type: disc; padding-left: 20px; line-height: 1.5;">
            <li>Overall trend shows a <span style="color: {'green' if total_change_pct >= 0 else 'red'}; font-weight: bold;">
            {total_change_pct:.1f}%</span> change in patient volume over the entire period.</li>
            <li>Peak patient volume occurred in <span style="font-weight: bold;">{max_month['month_year'].strftime('%B %Y')}</span> 
            with <span style="font-weight: bold;">{max_month['Patient_Count']:,}</span> patients.</li>
            <li>Lowest patient volume was in <span style="font-weight: bold;">{min_month['month_year'].strftime('%B %Y')}</span> 
            with <span style="font-weight: bold;">{min_month['Patient_Count']:,}</span> patients.</li>
            <li>Average month-over-month growth rate is <span style="color: {'green' if avg_growth >= 0 else 'red'}; font-weight: bold;">
            {avg_growth:.1f}%</span>.</li>
        </ul>
        """, unsafe_allow_html=True)


        st.markdown("<h6 style='text-align: left; margin-top: 15px;'>Seasonality Patterns</h6>", unsafe_allow_html=True)
    
        # Create month-of-year analysis for seasonality using filtered data
        df_seasonal = df_completed_section2.copy()
        df_seasonal['month'] = df_seasonal['Date of Admission'].dt.month
        df_seasonal['month_name'] = df_seasonal['Date of Admission'].dt.strftime('%b')  # 3-letter month abbreviation
        df_seasonal['quarter'] = df_seasonal['Date of Admission'].dt.quarter
        df_seasonal['year'] = df_seasonal['Date of Admission'].dt.year
        df_seasonal = df_seasonal[df_seasonal['year'] < df_seasonal['year'].max()] # last year is not included
        
        monthly_pattern = df_seasonal.groupby('month').agg({
            'Patient ID': 'count'
        }).reset_index()
        
        # Add month names (3-letter abbreviations)
        month_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_pattern['month_name'] = monthly_pattern['month'].apply(lambda x: month_abbr[x-1])
        monthly_pattern.rename(columns={'Patient ID': 'Patient_Count'}, inplace=True)
        
        # Create quarterly data
        quarterly_pattern = df_seasonal.groupby('quarter').agg({
            'Patient ID': 'count'
        }).reset_index()
        quarterly_pattern.rename(columns={'Patient ID': 'Patient_Count', 'quarter': 'Quarter'}, inplace=True)
        quarterly_pattern['Quarter_Name'] = quarterly_pattern['Quarter'].apply(lambda x: f'Q{x}')
        

        # Create bar chart for monthly seasonality
        monthly_pattern_sorted = monthly_pattern.sort_values('month')
        
        fig_season = go.Figure()
        
        # Add bars
        fig_season.add_trace(go.Bar(
            x=monthly_pattern_sorted['month_name'],
            y=monthly_pattern_sorted['Patient_Count'],
            marker=dict(
                color=monthly_pattern_sorted['Patient_Count'],
                colorscale=[[0, '#c2dcff'], [1, '#2f88ff']],
                showscale=False
            ),
            text=monthly_pattern_sorted['Patient_Count'],
            textposition='inside',
            hovertemplate='%{x}<br>Patients: %{y:,}<extra></extra>'
        ))
        
        fig_season.update_layout(
            title='',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            height=200,  # Reduced height to make room for quarterly chart
            xaxis=dict(
                title=None,
                tickmode='array',
                tickvals=month_abbr,  # Use 3-letter abbreviations
                showgrid=False
            ),
            yaxis=dict(
                title=None,
                showgrid=True,
                gridcolor='rgba(211,211,211,0.2)'
            )
        )
        
        st.plotly_chart(fig_season, use_container_width=True)

        # Add note about data aggregation
        st.caption("Note: Data is aggregated across all years to identify consistent seasonal patterns.")


    with col2:

        peak_month = monthly_pattern_sorted.loc[monthly_pattern_sorted['Patient_Count'].idxmax()]['month_name']
        low_month = monthly_pattern_sorted.loc[monthly_pattern_sorted['Patient_Count'].idxmin()]['month_name']
        
        # Identify peak and low quarters
        peak_quarter = quarterly_pattern.loc[quarterly_pattern['Patient_Count'].idxmax()]['Quarter_Name']
        low_quarter = quarterly_pattern.loc[quarterly_pattern['Patient_Count'].idxmin()]['Quarter_Name']

        st.markdown(f"""
            <ul style="display: flex; flex-direction: column; gap: 12px; list-style-type: disc; padding-left:15px; line-height: 1.5; margin-top:35px; margin-bottom: 25px;">
                <li>Recent trend (last {recent_months_count} months) shows <span style="color: {"green" if recent_growth >= 0 else "red"}; font-weight: bold;">{recent_growth:.1f}%</span> growth compared to overall average.</li>
                {f'<li>Year-over-year change is <span style="color: {"green" if yoy_change >= 0 else "red"}; font-weight: bold;">{yoy_change:.1f}%</span> compared to the same month last year.</li>' if yoy_change is not None else ''}
                <li>Peak patient volume typically occurs in <span style="font-weight: bold;">{peak_month}</span>, while <span style="font-weight: bold;">{peak_quarter}</span> shows the highest volume.</li>
                <li>Lowest patient volume tends to be in <span style="font-weight: bold;">{low_month}</span>, with <span style="font-weight: bold;">{low_quarter}</span> showing the lowest quarterly volume.</li>  
            </ul>
                """, unsafe_allow_html=True)
    
        st.markdown("<h6 style='text-align: left; margin-top: 15px;'>Seasonality Patterns</h6>", unsafe_allow_html=True)



        # Create quarterly chart
        fig_quarter = go.Figure()
        
        # Add bars for quarterly data
        fig_quarter.add_trace(go.Bar(
            x=quarterly_pattern['Quarter_Name'],
            y=quarterly_pattern['Patient_Count'],
            marker=dict(
                color=['#8bb9ff', '#5c9fff', '#2f88ff', '#0070f3'],
                showscale=False
            ),
            text=quarterly_pattern['Patient_Count'],
            textposition='inside',
            hovertemplate='%{x}<br>Patients: %{y:,}<extra></extra>'
        ))
        
        fig_quarter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=10),
            height=200,
            xaxis=dict(
                title=None,
                showgrid=False
            ),
            yaxis=dict(
                title=None,
                showgrid=True,
                gridcolor='rgba(211,211,211,0.2)'
            )
        )
        
        st.plotly_chart(fig_quarter, use_container_width=True)

        st.caption("Note: In both charts the last not completed year is not included.")



    
    # 
    # ------ Section 3: Forecasting ------
    # 

    st.markdown("""
        <div class="numberOfPatients">
            <h3 class="sub">Patient Volume Forecast</h3>
            <p>predictions for upcoming months based on historical data</p>
        </div>
        """, unsafe_allow_html=True
    )
    
    with st.container(key='background-2'):
        # Create forecasting model
        forecast_periods = 8  # Number of months to forecast
        
        # Only proceed with forecasting if we have enough data (at least 12 complete months)
        if len(monthly_patients) >= 12:
            # Prepare the time series data using a copy to avoid SettingWithCopyWarning
            time_series = pd.Series(
                monthly_patients['Patient_Count'].values.copy(),
                index=monthly_patients['month_year'].values
            )
            
            try:
                # Create the Exponential Smoothing model with numpy array to avoid copy issues
                model = ExponentialSmoothing(
                    np.array(time_series.values),
                    trend='add', 
                    seasonal='add', 
                    seasonal_periods=12,
                    damped=True
                ).fit()
                
                # Generate forecast
                forecast_values = model.forecast(forecast_periods)
                
                # Create proper DataFrame for forecast
                last_date = time_series.index[-1]

                prediction_index = pd.date_range(
                    start=pd.Timestamp(last_date) + pd.DateOffset(months=1),
                    periods=forecast_periods,
                    freq='MS'
                )
                
                # Calculate prediction intervals
                resid_std = np.std(model.resid)
                lower_bound = np.maximum(0, forecast_values - 1.28 * resid_std)
                upper_bound = forecast_values + 1.28 * resid_std
                
                # Create forecast visualization
                fig_forecast = go.Figure()
                
                # Add historical data
                fig_forecast.add_trace(go.Scatter(
                    x=time_series.index,
                    y=time_series.values,
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='#2f88ff', width=2),
                    marker=dict(size=6)
                ))
                
                # Add forecast
                fig_forecast.add_trace(go.Scatter(
                    x=prediction_index,
                    y=forecast_values,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#ff6b6b', width=2, dash='dash'),
                    marker=dict(size=8, symbol='diamond'),
                    hovertemplate='%{x|%b %Y}<br>Forecast: %{y:.0f} patients<extra></extra>'
                ))
                
                # Add prediction intervals
                fig_forecast.add_trace(go.Scatter(
                    x=np.concatenate([prediction_index, prediction_index[::-1]]),
                    y=np.concatenate([upper_bound, lower_bound[::-1]]),
                    fill='toself',
                    fillcolor='rgba(255, 107, 107, 0.1)',
                    line=dict(color='rgba(255, 107, 107, 0)'),
                    name='80% Confidence Interval',
                    hoverinfo='skip'
                ))
                
                # Update layout
                fig_forecast.update_layout(
                    title="",
                    xaxis_title=None,
                    yaxis_title=None,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    height=400,
                    xaxis=dict(
                        showgrid=False,
                        showline=True,
                        linecolor='rgba(211,211,211,0.7)'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(211,211,211,0.2)',
                        title_font=dict(size=14)
                    ),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12
                    )
                )
                
                st.plotly_chart(fig_forecast, use_container_width=False)
                st.caption("Note: Forecast is based on completed months only. The shaded area represents the 80% confidence interval.")
                

                st.markdown("<h6 style='text-align: left; margin-top:35px; margin-bottom:0px'>Forecast Insights</h6>", unsafe_allow_html=True)

                # Forecast insights
                col1, col2 = st.columns(2, gap='large')

                with col1:
                    # Calculate key metrics from forecast
                    first_forecast = forecast_values[0]
                    last_forecast = forecast_values[-1]
                    current_volume = time_series.values[-1]
                    forecast_change_pct = ((last_forecast - first_forecast) / first_forecast) * 100
                    forecast_vs_current_pct = ((last_forecast - current_volume) / current_volume) * 100
                    total_forecast_volume = np.sum(forecast_values)
                    avg_monthly_forecsat_volume = total_forecast_volume /  forecast_periods
                    
                    # Calculate seasonal adjustments in the forecast
                    expected_peak_month = prediction_index[np.argmax(forecast_values)].strftime('%B %Y')
                    expected_low_month = prediction_index[np.argmin(forecast_values)].strftime('%B %Y')
                    forecast_variation = (np.max(forecast_values) / np.min(forecast_values) - 1) * 100
                    
                    st.markdown(f"""
                    <div style="padding: 5px; border-radius: 5px;">
                     <ul style="display: flex; flex-direction: column; gap: 12px; list-style-type: disc; padding-left:15px; line-height: 1.5;">
                        <li>Patient volume is projected to change by 
                        <span style="color: {'green' if forecast_change_pct >= 0 else 'red'}; font-weight: bold;">
                        {forecast_change_pct:+.1f}%</span> over the next {forecast_periods} months.</li>
                        <li>By {prediction_index[-1].strftime('%B %Y')}, we expect 
                        <span style="font-weight: bold;">{last_forecast:.0f}</span> patients, which is 
                        <span style="color: {'green' if forecast_vs_current_pct >= 0 else 'red'}; font-weight: bold;">
                        {forecast_vs_current_pct:+.1f}%</span> versus current volume.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="padding: 5px; border-radius: 5px;">
                    <ul style="display: flex; flex-direction: column; gap: 12px; list-style-type: disc; padding-left:15px; line-height: 1.5;">
                        <li>Total projected patients for the next {forecast_periods} months: 
                        <span style="font-weight: bold;">{total_forecast_volume:,.0f}</span>, on average <b>{avg_monthly_forecsat_volume:,.0f}</b> for each month.</li>
                        <li>Expected peak is in <span style="font-weight: bold;">{expected_peak_month}</span>, with 
                        expected low in <span style="font-weight: bold;">{expected_low_month}</span> 
                        (a {forecast_variation:.1f}% variation).</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    
            except Exception as e:
                st.warning(f"Forecasting error: {e}. Please ensure you have sufficient complete monthly data for accurate forecasting.")
                st.markdown("""
                <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px;">
                    <p>Suggestions to fix forecasting issues:</p>
                    <ul>
                        <li>Select a wider date range that includes more historical data</li>
                        <li>Ensure the selected hospital(s) have consistent data across all months</li>
                        <li>Try using the "Last Year" time period option for more reliable forecasting</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"Not enough historical data for reliable forecasting. We recommend at least 12 complete months of data for seasonal forecasting. Current data has {len(monthly_patients)} complete months.")
            st.markdown("""
            <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px;">
                <p>To generate a forecast, please:</p>
                <ul>
                    <li>Select "Last Year" from the time period dropdown</li>
                    <li>Ensure you haven't filtered to a specific hospital with limited data</li>
                    <li>Consider using the full dataset without additional filters</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()