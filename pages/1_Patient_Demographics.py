import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import initialize_page, load_data, create_sidebar, create_page_navigation, img_to_base64

# Initialize page
initialize_page()

try:
    # Load data
    df = load_data()
    
    # Create sidebar
    current_start, current_end, prev_start, prev_end, comparison_label, prev_custom_start, filtered_df = create_sidebar(df)
    
    # Set current page for navigation
    st.session_state['current_page'] = 'Patient Demographics'
    
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

    # ------ Key Insights ------
    
    # st.markdown("""<h3 class="sub">Key Insights</h3>""", unsafe_allow_html=True)
    image_path = 'assets/images/Insights.png'
    st.markdown(f"""
            <div style="display: flex; flex-direction: row; margin-top: 32px; align-items: center;  gap: 15px;">
                <div style="margin-bottom: 8px;">
                <img src="data:image/png;base64,{img_to_base64(image_path)}" style="width: 32px; height: 32px;">
                </div>
                <h3 style="color: #3a3a3a;">Key Insights</h3>
            </div>
    """, unsafe_allow_html=True)
    

    # Create age bins
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 150]
    age_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    
    df_current['Age (bins)'] = pd.cut(df_current['Age'], bins=age_bins, labels=age_labels, right=False)
    if df_prev is not None:
        df_prev['Age (bins)'] = pd.cut(df_prev['Age'], bins=age_bins, labels=age_labels, right=False)
    
    # 1. Age Distribution
    age_group_counts = df_current['Age (bins)'].value_counts()
    top_age_group = age_group_counts.idxmax()
    top_age_percentage = (age_group_counts.max() / len(df_current)) * 100


    # 2. Gender Disparity in Admissions
    gender_counts = df_current['Gender'].value_counts()
    total_patients = len(df_current)
    female_count = gender_counts.get('Female', 0)
    male_count = gender_counts.get('Male', 0)
    female_percentage = (female_count / total_patients) * 100
    male_percentage = (male_count / total_patients) * 100
    female_avg_los = df_current[df_current['Gender'] == 'Female']['Length of Stay'].mean()
    male_avg_los = df_current[df_current['Gender'] == 'Male']['Length of Stay'].mean()

    # 3. Prevalent Blood Type
    blood_type_counts = df_current['Blood Type'].value_counts()
    most_common_blood_type = blood_type_counts.idxmax()
    blood_type_percentage = (blood_type_counts.max() / total_patients) * 100
    blood_type_avg_los = df_current[df_current['Blood Type'] == most_common_blood_type]['Length of Stay'].mean()

    # 4. Age and Hospital Stay
    age_group_los = df_current.groupby('Age (bins)')['Length of Stay'].mean()
    longest_los_age_group = age_group_los.idxmax()
    longest_los = age_group_los.max()

    # ------------------------------------------------------------
    # <---       Section 0: Key Insights        --->
    # ------------------------------------------------------------

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown(f"""
        <ul class="sub">
        
        <li>The largest age group is <b>{top_age_group}</b>, representing <b>{top_age_percentage:.1f}%</b> of patients.</li>

        <li>Females constitute <b>{female_percentage:.1f}%</b> of admissions with an average hospital stay of <b>{female_avg_los:.1f} days</b>, compared to males at <b>{male_percentage:.1f}%</b> with <b>{male_avg_los:.1f} days</b>.</li>

        </ul>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <ul class="sub">
        
        <li>The most common blood type is <b>{most_common_blood_type}</b>, accounting for <b>{blood_type_percentage:.1f}%</b> of patients, with an average length of stay of <b>{blood_type_avg_los:.1f} days</b>.</li>

        <li>The <b>{longest_los_age_group}</b> age group has the longest average hospital stay at <b>{longest_los:.1f} days</b>.</li>

        </ul>""", unsafe_allow_html=True)    

    # ------------------------------------------------------------
    # <---       Section 1: Gender        --->
    # ------------------------------------------------------------
    st.markdown("""<h3 class="sub">Gender</h3>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        # Gender distribution with Plotly
        gender_counts = df_current['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Patients']
        gender_counts = gender_counts.sort_values('Patients', ascending=True)

        fig_gender = px.bar(
            gender_counts, 
            y='Gender', 
            x='Patients',
            text='Patients',
            color_discrete_sequence=['#2f88ff'],
            orientation='h'
        )
        fig_gender.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Number of Patients",
            yaxis_title=None,
            height=300
        )
        # Display chart without buttons
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        # Create gender metrics table with Plotly
        gender_metrics = []
        
        for gender in df_current['Gender'].unique():
            gender_data = df_current[df_current['Gender'] == gender]
            avg_los = gender_data['Length of Stay'].mean()
            current_count = len(gender_data)
            
            if df_prev is not None:
                prev_gender_data = df_prev[df_prev['Gender'] == gender]
                prev_count = len(prev_gender_data)
                change_pct = ((current_count / prev_count) - 1) * 100 if prev_count > 0 else 0
            else:
                prev_count = change_pct = 0
            
            # Format the change percentage with arrow (with color)
            if change_pct > 0:
                change_text = f"▲{abs(change_pct):.1f}%"
                change_color = "green"
            elif change_pct < 0:
                change_text = f"▼{abs(change_pct):.1f}%"
                change_color = "red"
            else:
                change_text = "0.0%"
                change_color = "gray"
            
            gender_metrics.append({
                'Gender': gender,
                'Avg Stay (days)': f"{avg_los:.2f}",
                f'Patients {period_label}': f"{prev_count:,}",
                '% Change': change_text,
                'color': change_color
            })
        
        # Create plotly table
        gender_df = pd.DataFrame(gender_metrics)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(gender_df.columns[:-1]),  # Exclude the color column
                fill_color='#f8f9fa',
                align='center',
                height=40,  # Increase header height
                font=dict(color='#333333', size=12),
                line=dict(width=1, color='#f0f0f0') 
            ),
            cells=dict(
                values=[gender_df[col] for col in gender_df.columns[:-1]],
                fill_color='white',
                align='center',
                font=dict(color=['#333333', '#333333', '#333333', 
                                [gender_df['color'][i] for i in range(len(gender_df))]],
                         size=12),
                height=35,  # Increase row height         
                line=dict(width=1, color='#f0f0f0') 
            )
        )])
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300
        )

        # Add scrolling configuration
        st.plotly_chart(fig, use_container_width=True )
    
    # ------------------------------------------------------------
    # <---       Section 2: Age        --->
    # ------------------------------------------------------------

    st.markdown("""<h3 class="sub">Age</h3>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    
    with col1:
        # Age distribution with Plotly
        age_counts = df_current['Age (bins)'].value_counts().reset_index()
        age_counts.columns = ['Age (bins)', 'Patients']
        
        # Sort by age group
        age_counts['Age (bins)'] = pd.Categorical(age_counts['Age (bins)'], categories=age_labels, ordered=True)
        age_counts = age_counts.sort_values('Age (bins)')
        
        fig_age = px.bar(
            age_counts, 
            y='Age (bins)', 
            x='Patients',
            text='Patients',
            color_discrete_sequence=['#2f88ff'],
            orientation='h'
        )
        fig_age.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Number of Patients",
            yaxis_title=None,
            height=300
        )
        fig_age.update_traces(
            textangle=0,     # 0 = horizontal, 90 = vertical
        )
        
        st.plotly_chart(fig_age, use_container_width=True )
    
    with col2:
        # Create age metrics table with Plotly
        age_metrics = []
        
        for age_bin in age_labels:
            age_data = df_current[df_current['Age (bins)'] == age_bin]
            if len(age_data) > 0:
                avg_los = age_data['Length of Stay'].mean()
                current_count = len(age_data)
                
                if df_prev is not None:
                    prev_age_data = df_prev[df_prev['Age (bins)'] == age_bin]
                    prev_count = len(prev_age_data)
                    change_pct = ((current_count / prev_count) - 1) * 100 if prev_count > 0 else 0
                else:
                    prev_count = change_pct = 0
                
                # Format the change percentage with arrow (with color)
                if change_pct > 0:
                    change_text = f"▲{abs(change_pct):.1f}%"
                    change_color = "green"
                elif change_pct < 0:
                    change_text = f"▼{abs(change_pct):.1f}%"
                    change_color = "red"
                else:
                    change_text = "0.0%"
                    change_color = "gray"
                
                age_metrics.append({
                    'Age': age_bin,
                    'Avg Stay (days)': f"{avg_los:.2f}",
                    f'Patients {period_label}': f"{prev_count:,}",
                    '% Change': change_text,
                    'color': change_color
                })
        
        # Create plotly table
        age_df = pd.DataFrame(age_metrics)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(age_df.columns[:-1]),  # Exclude the color column
                fill_color='#f8f9fa',
                align='center',
                height=40,
                font=dict(color='#333333', size=12),
                line=dict(width=1, color='#f0f0f0') 
            ),
            cells=dict(
                values=[age_df[col] for col in age_df.columns[:-1]],
                fill_color='white',
                align='center',
                height=35,
                font=dict(color=['#333333', '#333333', '#333333', 
                                [age_df['color'][i] for i in range(len(age_df))]],
                         size=12),
                line=dict(width=1, color='#f0f0f0')          
            )
        )])
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300
        )

        # Add scrolling configuration
        st.plotly_chart(fig, use_container_width=True)
    
    # ------------------------------------------------------------
    # <---       Section 3: Blood Type        --->
    # ------------------------------------------------------------

    st.markdown("""<h3 class="sub">Blood Type</h3>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        # Blood Type distribution with Plotly
        blood_counts = df_current['Blood Type'].value_counts().reset_index()
        blood_counts.columns = ['Blood Type', 'Patients']
        blood_counts = blood_counts.sort_values('Patients', ascending=True)
        fig_blood = px.bar(
            blood_counts, 
            y='Blood Type', 
            x='Patients',
            text='Patients',
            color_discrete_sequence=['#2f88ff'],
            orientation='h'
        )
        fig_blood.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Number of Patients",
            yaxis_title=None,
            height=300
        )
        
        st.plotly_chart(fig_blood, use_container_width=True, config={"displayModeBar": False})
    
    with col2:
        # Create blood type metrics table with Plotly
        blood_metrics = []
        
        for blood_type in df_current['Blood Type'].unique():
            blood_data = df_current[df_current['Blood Type'] == blood_type]
            avg_los = blood_data['Length of Stay'].mean()
            current_count = len(blood_data)
            
            if df_prev is not None:
                prev_blood_data = df_prev[df_prev['Blood Type'] == blood_type]
                prev_count = len(prev_blood_data)
                change_pct = ((current_count / prev_count) - 1) * 100 if prev_count > 0 else 0
            else:
                prev_count = change_pct = 0
            
            # Format the change percentage with arrow (with color)
            if change_pct > 0:
                change_text = f"▲{abs(change_pct):.1f}%"
                change_color = "green"
            elif change_pct < 0:
                change_text = f"▼{abs(change_pct):.1f}%"
                change_color = "red"
            else:
                change_text = "0.0%"
                change_color = "gray"
            
            blood_metrics.append({
                'Blood Type': blood_type,
                'Avg Stay (days)': f"{avg_los:.2f}",
                f'Patients {period_label}': f"{prev_count:,}",
                '% Change': change_text,
                'color': change_color
            })
        
        # Create plotly table
        blood_df = pd.DataFrame(blood_metrics)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(blood_df.columns[:-1]),  # Exclude the color column
                fill_color='#f8f9fa',
                align='center',
                height=40,
                font=dict(color='#333333', size=12),
                line=dict(width=1, color='#f0f0f0') 
            ),
            cells=dict(
                values=[blood_df[col] for col in blood_df.columns[:-1]],
                fill_color='white',
                align='center',
                height=35,
                font=dict(color=['#333333', '#333333', '#333333', 
                                [blood_df['color'][i] for i in range(len(blood_df))]],
                         size=12),
                line=dict(width=1, color='#f0f0f0')          
            )
        )])
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300
        )

        # Add scrolling configuration
        st.plotly_chart(fig, use_container_width=True)
    
    # ------------------------------------------------------------
    # <---       Section 4: Medical Condition        --->
    # ------------------------------------------------------------
    st.markdown("""<h3 class="sub">Medical Condition</h3>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        # Medical Condition distribution with Plotly (top 10)
        condition_counts = df_current['Medical Condition'].value_counts().head(10).reset_index()
        condition_counts.columns = ['Medical Condition', 'Patients']
        condition_counts = condition_counts.sort_values('Patients', ascending=True)

        fig_condition = px.bar(
            condition_counts, 
            y='Medical Condition', 
            x='Patients',
            text='Patients',
            color_discrete_sequence=['#2f88ff'],
            orientation='h'
        )
        fig_condition.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Number of Patients",
            yaxis_title=None,
            height=300
        )
        
        st.plotly_chart(fig_condition, use_container_width=True)
    
    with col2:
        # Create medical condition metrics table with Plotly (top 10)
        top_conditions = df_current['Medical Condition'].value_counts().head(10).index.tolist()
        condition_metrics = []
        
        for condition in top_conditions:
            condition_data = df_current[df_current['Medical Condition'] == condition]
            avg_los = condition_data['Length of Stay'].mean()
            current_count = len(condition_data)
            
            if df_prev is not None:
                prev_condition_data = df_prev[df_prev['Medical Condition'] == condition]
                prev_count = len(prev_condition_data)
                change_pct = ((current_count / prev_count) - 1) * 100 if prev_count > 0 else 0
            else:
                prev_count = change_pct = 0
            
            # Format the change percentage with arrow (with color)
            if change_pct > 0:
                change_text = f"▲{abs(change_pct):.1f}%"
                change_color = "green"
            elif change_pct < 0:
                change_text = f"▼{abs(change_pct):.1f}%"
                change_color = "red"
            else:
                change_text = "0.0%"
                change_color = "gray"
            
            condition_metrics.append({
                'Medical Condition': condition,
                'Avg Stay (days)': f"{avg_los:.2f}",
                f"Patients {period_label}": f"{prev_count:,}",
                '% Change': change_text,
                'color': change_color
            })
        
        # Create plotly table
        condition_df = pd.DataFrame(condition_metrics)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(condition_df.columns[:-1]),  # Exclude the color column
                fill_color='#f8f9fa',
                align='center',
                height=40,
                font=dict(color='#333333', size=12),
                line=dict(width=1, color='#f0f0f0') 
            ),
            cells=dict(
                values=[condition_df[col] for col in condition_df.columns[:-1]],
                fill_color='white',
                align='center',
                height=35,
                font=dict(color=['#333333', '#333333', '#333333', 
                                [condition_df['color'][i] for i in range(len(condition_df))]],
                            size=12),
                line=dict(width=1, color='#f0f0f0')          
            )
        )])
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)


    # ------------------------------------------------------------
    # <---       Section 5: Sunburst Chart        --->
    # ------------------------------------------------------------

    image_path_sunburst = 'assets/images/interactive.png'
    st.markdown(f"""
            <div style="display: flex; flex-direction: row; margin-top: 32px; align-items: center;  gap: 15px;">
                <div style="margin-bottom: 8px;">
                <img src="data:image/png;base64,{img_to_base64(image_path_sunburst)}" style="width: 32px; height: 32px;">
                </div>
                <h3 style="color: #3a3a3a;">Explore the multiple dimensions of our data</h3>
            </div>
    """, unsafe_allow_html=True)

    # Define the hierarchy options
    hierarchy_options = [
        'Medical Condition → Gender → Age (bins)',
        'Medical Condition → Hospital → Admission Type',
        'Hospital → Admission Type → Test Results',
        'Medication → Medical Condition → Gender',
        'Age (bins) → Insurance Provider → Admission Type'
    ]

    hierarchy_options_2 = [
        ['Medical Condition', 'Gender' , 'Age (bins)'],
        ['Medical Condition', 'Hospital' , 'Admission Type'],
        ['Hospital' , 'Admission Type', 'Test Results'],
        ['Medication' , 'Medical Condition', 'Gender'],
        ['Age (bins)', 'Insurance Provider', 'Admission Type']
    ]

    
    # --- Column Layout ---
    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:
        # Use session state to remember the selection, default to 'Medical Condition'
        if 'sunburst_inner_ring' not in st.session_state:
            st.session_state.sunburst_inner_ring = 'Medical Condition → Gender → Age (bins)'
        
        if 'sunburst_agg' not in st.session_state:
            st.session_state.sunburst_agg = 'Number of Patients'
        
        # Get the selected inner ring from session state
        selected_inner_ring = st.session_state.sunburst_inner_ring
        selected_agg = st.session_state.sunburst_agg

        if  selected_agg == 'Number of Patients':
            selected_agg = 'Patients'
        else:    
            selected_agg = 'Billing'

        index = 0
        for i in range(len(hierarchy_options)):
            if hierarchy_options[i] == selected_inner_ring:
                index = i
        
        
        st.markdown(f"""<h6 class='sub'>Patient Distribution Hierarchy (by {selected_inner_ring})</h6>""", unsafe_allow_html=True)
        
        # --- Dynamic Path Creation ---
        # Define the default path structure, replacing the first element
        path_structure = hierarchy_options_2[index]
        
        # --- Dynamic Data Grouping ---
        # Ensure the selected column exists in the DataFrame
        try:
            # Group by the dynamic path structure
            sunburst_data = df_current.groupby(path_structure).agg(
                Patients = ('Patient ID', 'count'),
                Billing = ('Billing Amount','sum')
            ).reset_index()
            
            
            # --- Create the Sunburst Chart ---
            fig_sunburst = px.sunburst(
                sunburst_data,
                path=path_structure, # Use the dynamic path
                values= selected_agg,
                color= selected_agg,
                color_continuous_scale='Blues',
                maxdepth=3,
                branchvalues='total'
            )

            # Update layout
            fig_sunburst.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=650,
                margin=dict(t=10, l=0, r=0, b=0) # Adjusted top margin
            )

            # Update hover template
            fig_sunburst.update_traces(
                hovertemplate="""
                <b>%{label}</b><br>
                Value: %{value:,.0f}<br>
                Percentage: %{percentParent:.1%}<br>
                <extra></extra>
                """
            )

            st.plotly_chart(fig_sunburst, use_container_width=True)
        
        except Exception as e:
            st.error(f"Could not generate chart with {selected_inner_ring}. Error: {e}")
        

    with col2:
        st.markdown("""
        <h6 class='sub'>How to Read This Chart</h6>
        <p style='margin-bottom: 10px;'>This interactive sunburst chart visualizes the hierarchical relationship, Compare relative sizes of patient segments and explore demographic patterns.</p>
        <div style="margin-left: 30px; margin-bottom: 45px;">    
            <p style='margin-bottom: 10px;'><b>Interactive Features:</b></p>
            <ul style='margin-bottom: 15px; line-height:1.1;'>
                <li>Click on any segment to zoom in</li>
                <li>Click in the center to zoom out</li>
                <li>Hover over segments to see detailed information</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # --- Selectbox for Inner Ring ---
        # Use a key to link the selectbox to session state
        # The on_change will trigger a rerun, updating the chart

        st.radio(
            "Select a path:",
            hierarchy_options,
            key='sunburst_inner_ring', # Link to session state key
            index=hierarchy_options.index(st.session_state.sunburst_inner_ring), 
            horizontal=False 
        )

        st.markdown("""<div style="margin-top:25px"> </div>""", unsafe_allow_html=True) 

        st.radio(
            "Select a metric:",
            ['Number of Patients', 'Total Billing'],
            key='sunburst_agg', # Link to session state key
            index=0, 
            horizontal=False 
        )
  
        
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()