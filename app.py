import streamlit as st
import pandas as pd
import numpy as np
import streamlit_authenticator as stauth
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="COLREGS Performance Dashboard")

# --- AUTHENTICATION ---
credentials = {
    "usernames": {
        "Admin": {
            "email": "marlon.bulan@maap.edu.ph",
            "name": "Sir Marlon",
            "password": "$2b$12$I4J0sbdaLJCl2TzHi.j3JO55W2AFz3M/RxZUpHu9xsa05cn6Ad.bm"
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "colregs_dashboard_cookie",
    "colregs_dashboard_secret",
    1
)

name, auth_status, username = authenticator.login("Login", "main")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    """Load all sheets from the Excel file"""
    student_attempts = pd.read_excel("colregs_analysis_final.xlsx", sheet_name="student_attempts")
    topic_performance = pd.read_excel("colregs_analysis_final.xlsx", sheet_name="topic_performance")
    question_map = pd.read_excel("colregs_analysis_final.xlsx", sheet_name="question_map")
    question_topics = pd.read_excel("colregs_analysis_final.xlsx", sheet_name="question_topics")
    
    # Create full name column
    student_attempts['full_name'] = student_attempts['first_name'] + ' ' + student_attempts['last_name']
    
    # Convert completed_dt to datetime
    student_attempts['completed_dt'] = pd.to_datetime(student_attempts['completed_dt'])
    student_attempts['year'] = student_attempts['completed_dt'].dt.year
    
    return student_attempts, topic_performance, question_map, question_topics

if auth_status:
    try:
        st.sidebar.image("maap_logo.png", width=120)
    except:
        pass  # Logo file not found, skip it
    st.sidebar.title(f"Welcome, {name}")
    authenticator.logout("Logout", "sidebar")

    # Load data
    student_attempts, topic_performance, question_map, question_topics = load_data()

    st.title("⚓ COLREGS Performance Dashboard")
    st.markdown("*Collision Regulations Exam Analysis & Student Progress Tracking*")

    tab1, tab2, tab3 = st.tabs(["👤 Individual Performance", "👥 Cohort Analysis", "📊 Topic Insights"])

    # -----------------------
    # TAB 1: INDIVIDUAL PERFORMANCE
    # -----------------------
    with tab1:
        st.header("Individual Student Performance")

        # Student selector
        student_list = sorted(student_attempts['full_name'].unique())
        selected_student = st.selectbox("Select a Student", student_list)

        # Get student email
        student_email = student_attempts[student_attempts['full_name'] == selected_student]['email'].iloc[0]

        # Filter data for selected student
        student_df = student_attempts[student_attempts['email'] == student_email].sort_values('completed_dt')
        student_topics = topic_performance[topic_performance['email'] == student_email]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Attempts", len(student_df))
        with col2:
            avg_score = student_df['percent'].mean()
            st.metric("Average Score", f"{avg_score:.1f}%")
        with col3:
            best_score = student_df['percent'].max()
            st.metric("Best Score", f"{best_score:.1f}%")
        with col4:
            total_questions = student_df['total_presented'].sum()
            st.metric("Total Questions Answered", f"{total_questions:.0f}")

        # Attempt history
        st.subheader("📋 Exam Attempt History")
        display_df = student_df[['exam_id', 'completed_dt', 'total_presented', 'total_correct', 'percent']].copy()
        display_df.columns = ['Exam ID', 'Date', 'Questions', 'Correct', 'Score (%)']
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(display_df, use_container_width=True)

        # Score progression over time
        if len(student_df) > 1:
            st.subheader("📈 Score Progression Over Time")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(student_df['completed_dt'], student_df['percent'], marker='o', linewidth=2, markersize=8)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Score (%)')
            ax1.set_title(f"{selected_student} - Score Progression")
            ax1.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig1)

        # Topic performance breakdown
        if not student_topics.empty:
            st.subheader("🎯 Topic Performance Breakdown")
            
            # Aggregate by topic
            topic_summary = student_topics.groupby('pred_topic').agg({
                'n_presented': 'sum',
                'n_correct': 'sum'
            }).reset_index()
            topic_summary['accuracy'] = (topic_summary['n_correct'] / topic_summary['n_presented'] * 100).round(1)
            topic_summary = topic_summary.sort_values('accuracy', ascending=False)

            # Bar chart
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            bars = ax2.barh(topic_summary['pred_topic'], topic_summary['accuracy'])
            
            # Color bars by performance
            for i, (bar, acc) in enumerate(zip(bars, topic_summary['accuracy'])):
                if acc >= 80:
                    bar.set_color('#28a745')  # Green
                elif acc >= 60:
                    bar.set_color('#ffc107')  # Yellow
                else:
                    bar.set_color('#dc3545')  # Red
            
            ax2.set_xlabel('Accuracy (%)')
            ax2.set_title(f"{selected_student} - Performance by COLREGS Topic")
            ax2.axvline(x=80, color='green', linestyle='--', alpha=0.3, label='Excellent (80%+)')
            ax2.axvline(x=60, color='orange', linestyle='--', alpha=0.3, label='Good (60%+)')
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)

            # Topic details table
            st.subheader("📊 Detailed Topic Statistics")
            topic_display = topic_summary[['pred_topic', 'n_presented', 'n_correct', 'accuracy']].copy()
            topic_display.columns = ['Topic', 'Questions Presented', 'Correct', 'Accuracy (%)']
            st.dataframe(topic_display, use_container_width=True)

    # -----------------------
    # TAB 2: COHORT ANALYSIS
    # -----------------------
    with tab2:
        st.header("Cohort Performance Analysis")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            years = sorted(student_attempts['year'].unique())
            selected_years = st.multiselect("Select Year(s)", years, default=years)
        with col2:
            exam_ids = sorted(student_attempts['exam_id'].unique())
            selected_exams = st.multiselect("Select Exam(s)", exam_ids, default=exam_ids)

        # Filter data
        cohort_df = student_attempts[
            (student_attempts['year'].isin(selected_years)) &
            (student_attempts['exam_id'].isin(selected_exams))
        ]
        cohort_topics = topic_performance[
            topic_performance['email'].isin(cohort_df['email'])
        ]

        if not cohort_df.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Students", cohort_df['email'].nunique())
            with col2:
                st.metric("Total Attempts", len(cohort_df))
            with col3:
                st.metric("Avg Score", f"{cohort_df['percent'].mean():.1f}%")
            with col4:
                st.metric("Avg Questions/Exam", f"{cohort_df['total_presented'].mean():.0f}")

            # Score distribution
            st.subheader("📊 Score Distribution")
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.hist(cohort_df['percent'], bins=20, edgecolor='black', alpha=0.7)
            ax3.axvline(cohort_df['percent'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {cohort_df['percent'].mean():.1f}%")
            ax3.axvline(cohort_df['percent'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: {cohort_df['percent'].median():.1f}%")
            ax3.set_xlabel('Score (%)')
            ax3.set_ylabel('Number of Attempts')
            ax3.set_title('Score Distribution Across Cohort')
            ax3.legend()
            plt.tight_layout()
            st.pyplot(fig3)

            # Performance by exam
            st.subheader("📈 Average Performance by Exam")
            exam_summary = cohort_df.groupby('exam_id')['percent'].agg(['mean', 'count', 'std']).reset_index()
            exam_summary.columns = ['Exam ID', 'Average Score (%)', 'Attempts', 'Std Dev']
            exam_summary['Average Score (%)'] = exam_summary['Average Score (%)'].round(1)
            exam_summary['Std Dev'] = exam_summary['Std Dev'].round(1)
            st.dataframe(exam_summary, use_container_width=True)

            # Performance over time
            if len(selected_years) > 1:
                st.subheader("📅 Performance Trends by Year")
                yearly_summary = cohort_df.groupby('year')['percent'].mean().reset_index()
                
                fig4, ax4 = plt.subplots(figsize=(10, 5))
                ax4.plot(yearly_summary['year'], yearly_summary['percent'], marker='o', linewidth=2, markersize=10)
                ax4.set_xlabel('Year')
                ax4.set_ylabel('Average Score (%)')
                ax4.set_title('Average Performance by Year')
                ax4.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig4)

    # -----------------------
    # TAB 3: TOPIC INSIGHTS
    # -----------------------
    with tab3:
        st.header("COLREGS Topic Performance Insights")

        # Filters
        years_topic = sorted(student_attempts['year'].unique())
        selected_years_topic = st.multiselect("Select Year(s)", years_topic, default=years_topic, key="topic_years")

        # Filter
        filtered_emails = student_attempts[student_attempts['year'].isin(selected_years_topic)]['email'].unique()
        filtered_topics = topic_performance[topic_performance['email'].isin(filtered_emails)]

        if not filtered_topics.empty:
            # Aggregate topic performance
            topic_agg = filtered_topics.groupby('pred_topic').agg({
                'n_presented': 'sum',
                'n_correct': 'sum'
            }).reset_index()
            topic_agg['accuracy'] = (topic_agg['n_correct'] / topic_agg['n_presented'] * 100).round(1)
            topic_agg = topic_agg.sort_values('accuracy', ascending=False)

            # Overall topic performance
            st.subheader("🎯 Overall Topic Performance")
            fig5, ax5 = plt.subplots(figsize=(12, 8))
            bars = ax5.barh(topic_agg['pred_topic'], topic_agg['accuracy'])
            
            # Color by performance
            for bar, acc in zip(bars, topic_agg['accuracy']):
                if acc >= 80:
                    bar.set_color('#28a745')
                elif acc >= 60:
                    bar.set_color('#ffc107')
                else:
                    bar.set_color('#dc3545')
            
            ax5.set_xlabel('Accuracy (%)')
            ax5.set_title('COLREGS Topic Performance Across All Students')
            ax5.axvline(x=80, color='green', linestyle='--', alpha=0.3)
            ax5.axvline(x=60, color='orange', linestyle='--', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig5)

            # Topic statistics table
            st.subheader("📊 Topic Statistics")
            topic_display = topic_agg[['pred_topic', 'n_presented', 'n_correct', 'accuracy']].copy()
            topic_display.columns = ['Topic', 'Total Questions', 'Total Correct', 'Accuracy (%)']
            
            # Add performance category
            def categorize(acc):
                if acc >= 80:
                    return "✅ Excellent"
                elif acc >= 60:
                    return "⚠️ Good"
                else:
                    return "❌ Needs Improvement"
            
            topic_display['Performance'] = topic_display['Accuracy (%)'].apply(categorize)
            st.dataframe(topic_display, use_container_width=True)

            # Heatmap: Topic performance by exam
            st.subheader("🔥 Topic Performance Heatmap by Exam")
            
            # Create pivot table
            pivot_data = filtered_topics.groupby(['exam_id', 'pred_topic']).agg({
                'n_correct': 'sum',
                'n_presented': 'sum'
            }).reset_index()
            pivot_data['accuracy'] = (pivot_data['n_correct'] / pivot_data['n_presented'] * 100).round(1)
            pivot_table = pivot_data.pivot(index='pred_topic', columns='exam_id', values='accuracy')
            
            if not pivot_table.empty:
                fig6, ax6 = plt.subplots(figsize=(14, 10))
                sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=70, 
                           vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'}, ax=ax6)
                ax6.set_title('Topic Performance Heatmap by Exam')
                ax6.set_xlabel('Exam ID')
                ax6.set_ylabel('COLREGS Topic')
                plt.tight_layout()
                st.pyplot(fig6)

            # Most challenging topics
            st.subheader("⚠️ Most Challenging Topics")
            challenging = topic_agg.nsmallest(5, 'accuracy')[['pred_topic', 'accuracy', 'n_presented']]
            challenging.columns = ['Topic', 'Accuracy (%)', 'Questions Asked']
            st.dataframe(challenging, use_container_width=True)

            # Best performing topics
            st.subheader("✅ Best Performing Topics")
            best = topic_agg.nlargest(5, 'accuracy')[['pred_topic', 'accuracy', 'n_presented']]
            best.columns = ['Topic', 'Accuracy (%)', 'Questions Asked']
            st.dataframe(best, use_container_width=True)

else:
    st.warning("Please enter your credentials to access the COLREGS Dashboard.")
    st.info("Contact your administrator for login credentials.")
