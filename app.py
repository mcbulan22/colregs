import streamlit as st
import pandas as pd
import numpy as np
import streamlit_authenticator as stauth
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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
    """Load and prepare the COLREGS data"""
    df = pd.read_excel("colregs.xlsx")
    
    # Create full name
    df['full_name'] = df['First name'] + ' ' + df['Last name']
    
    # Parse datetime
    df['completed_dt'] = pd.to_datetime(df['Completed'], format='%d/%m/%Y %H:%M', errors='coerce')
    df['started_dt'] = pd.to_datetime(df['Started'], format='%d/%m/%Y %H:%M', errors='coerce')
    
    # Extract year
    df['year'] = df['completed_dt'].dt.year
    
    # Clean class names and filter out "Not Found"
    df['Class'] = df['Class'].fillna('Unknown')
    df = df[df['Class'] != 'Not Found']
    
    # Convert score to numeric
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    
    # Clean pred_topic - remove NaN values
    df['pred_topic'] = df['pred_topic'].fillna('Unknown Topic')
    
    return df

@st.cache_data
def get_student_summary(df):
    """Generate per-student summary statistics"""
    summary = df.groupby(['email', 'full_name', 'Class', 'exam_id']).agg({
        'score': 'sum',
        'question_number': 'count',
        'completed_dt': 'first',
        'Duration': 'first',
        'grade_raw': 'first'
    }).reset_index()
    
    summary.columns = ['email', 'full_name', 'Class', 'exam_id', 'total_correct', 
                       'total_questions', 'completed_dt', 'duration', 'grade_raw']
    summary['percentage'] = (summary['total_correct'] / summary['total_questions'] * 100).round(1)
    
    return summary

@st.cache_data
def get_topic_summary(df):
    """Generate topic-level performance"""
    topic_perf = df.groupby(['email', 'full_name', 'Class', 'exam_id', 'pred_topic']).agg({
        'score': ['sum', 'count']
    }).reset_index()
    
    topic_perf.columns = ['email', 'full_name', 'Class', 'exam_id', 'topic', 'correct', 'total']
    topic_perf['accuracy'] = (topic_perf['correct'] / topic_perf['total'] * 100).round(1)
    
    return topic_perf

@st.cache_data
def get_question_difficulty(df):
    """Calculate question difficulty across all students"""
    q_diff = df.groupby(['global_qid', 'question_text', 'pred_topic']).agg({
        'score': ['sum', 'count', 'mean']
    }).reset_index()
    
    q_diff.columns = ['global_qid', 'question_text', 'topic', 'correct', 'attempts', 'difficulty']
    q_diff['difficulty'] = (q_diff['difficulty'] * 100).round(1)
    
    return q_diff

if auth_status:
    try:
        st.sidebar.image("maap_logo.png", width=120)
    except:
        pass
    st.sidebar.title(f"Welcome, {name}")
    authenticator.logout("Logout", "sidebar")

    # Load data
    df = load_data()
    student_summary = get_student_summary(df)
    topic_summary = get_topic_summary(df)
    question_difficulty = get_question_difficulty(df)

    st.title("⚓ COLREGS Performance Dashboard")
    st.markdown("*Collision Regulations Exam Analysis & Student Progress Tracking*")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "👤 Student Profile", 
        "🎯 Topic Analysis", 
        "❓ Question Insights",
        "📈 Class Comparison",
        "⏱️ Time & Efficiency"
    ])

    # -----------------------
    # TAB 1: OVERVIEW DASHBOARD
    # -----------------------
    with tab1:
        st.header("Overview Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            unique_students = df['email'].nunique()
            st.metric("Unique Students", unique_students)
        with col2:
            total_exams = df['exam_id'].nunique()
            st.metric("Unique Exams", total_exams)
        with col3:
            avg_score = student_summary['percentage'].mean()
            st.metric("Average Score", f"{avg_score:.1f}%")
        with col4:
            total_attempts = len(student_summary)
            st.metric("Total Attempts", total_attempts)
        
        # Additional context metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            unique_questions = df['global_qid'].nunique()
            st.metric("Unique Questions", unique_questions)
        with col2:
            total_responses = len(df)
            st.metric("Total Responses", total_responses)
        with col3:
            avg_attempts = (len(student_summary) / df['email'].nunique())
            st.metric("Avg Attempts/Student", f"{avg_attempts:.1f}")
        with col4:
            overall_accuracy = (df['score'].sum() / len(df) * 100)
            st.metric("Overall Accuracy", f"{overall_accuracy:.1f}%")

        # Class performance comparison
        st.subheader("📊 Performance by Class")
        
        # Calculate class performance considering unique students and their average scores
        class_perf = student_summary.groupby('Class').agg({
            'percentage': 'mean',
            'email': 'nunique'  # Count unique students
        }).reset_index()
        class_perf.columns = ['Class', 'Avg Score', 'Unique Students']
        class_perf['Avg Score'] = class_perf['Avg Score'].round(1)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            bars = ax1.bar(class_perf['Class'], class_perf['Avg Score'], color='steelblue', alpha=0.8)
            ax1.axhline(y=avg_score, color='red', linestyle='--', label=f'Overall Avg: {avg_score:.1f}%')
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Average Score (%)')
            ax1.set_title('Average Score by Class (Based on Student Averages)')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            st.dataframe(class_perf, use_container_width=True, hide_index=True)

        # Score distribution
        st.subheader("📈 Overall Score Distribution")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.hist(student_summary['percentage'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(student_summary['percentage'].mean(), color='red', linestyle='--', 
                    linewidth=2, label=f"Mean: {student_summary['percentage'].mean():.1f}%")
        ax2.axvline(student_summary['percentage'].median(), color='green', linestyle='--', 
                    linewidth=2, label=f"Median: {student_summary['percentage'].median():.1f}%")
        ax2.set_xlabel('Score (%)')
        ax2.set_ylabel('Number of Attempts')
        ax2.set_title('Score Distribution')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

        # Top performers
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏆 Top 10 Performers")
            top_students = student_summary.nlargest(10, 'percentage')[
                ['full_name', 'Class', 'percentage', 'exam_id']
            ]
            top_students.columns = ['Student', 'Class', 'Score (%)', 'Exam']
            st.dataframe(top_students, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("📉 Students Needing Support")
            bottom_students = student_summary.nsmallest(10, 'percentage')[
                ['full_name', 'Class', 'percentage', 'exam_id']
            ]
            bottom_students.columns = ['Student', 'Class', 'Score (%)', 'Exam']
            st.dataframe(bottom_students, use_container_width=True, hide_index=True)

    # -----------------------
    # TAB 2: STUDENT PROFILE
    # -----------------------
    with tab2:
        st.header("Individual Student Profile")
        
        # Student selector
        student_list = sorted(df['full_name'].unique())
        selected_student = st.selectbox("Select a Student", student_list)
        
        # Get student data
        student_email = df[df['full_name'] == selected_student]['email'].iloc[0]
        student_class = df[df['full_name'] == selected_student]['Class'].iloc[0]
        student_data = student_summary[student_summary['email'] == student_email].copy()
        student_questions = df[df['email'] == student_email]
        student_topics = topic_summary[topic_summary['email'] == student_email]
        
        # Sort attempts by completion date (ascending)
        student_data = student_data.sort_values(by='completed_dt', ascending=True)

        # Header info
        st.markdown(f"**Class:** {student_class} | **Email:** {student_email}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Attempts", len(student_data))
        with col2:
            st.metric("Average Score", f"{student_data['percentage'].mean():.1f}%")
        with col3:
            st.metric("Best Score", f"{student_data['percentage'].max():.1f}%")
        with col4:
            st.metric("Questions Answered", len(student_questions))

        # Attempt history
        st.subheader("📋 Exam Attempt History")
        display_attempts = student_data[['exam_id', 'completed_dt', 'total_questions', 
                                         'total_correct', 'percentage', 'duration']].copy()
        display_attempts['completed_dt'] = display_attempts['completed_dt'].dt.strftime('%Y-%m-%d %H:%M')
        display_attempts.columns = ['Exam', 'Date', 'Questions', 'Correct', 'Score (%)', 'Duration']
        st.dataframe(display_attempts, use_container_width=True, hide_index=True)

        # Score progression
        if len(student_data) > 1:
            st.subheader("📈 Score Progression")
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(student_data['completed_dt'], student_data['percentage'].values, 
                     marker='o', linewidth=2, markersize=8, color='steelblue')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Score (%)')
            ax3.set_title(f"{selected_student} - Score Progression Over Time")
            ax3.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig3)

        # Topic performance
        if not student_topics.empty:
            st.subheader("🎯 Performance by Topic")
            
            topic_agg = student_topics.groupby('topic').agg({
                'correct': 'sum',
                'total': 'sum'
            }).reset_index()
            topic_agg['accuracy'] = (topic_agg['correct'] / topic_agg['total'] * 100).round(1)
            topic_agg = topic_agg.sort_values('accuracy', ascending=True)
            
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            bars = ax4.barh(topic_agg['topic'], topic_agg['accuracy'])
            
            # Color code by performance
            for bar, acc in zip(bars, topic_agg['accuracy']):
                if acc >= 80:
                    bar.set_color('#28a745')
                elif acc >= 60:
                    bar.set_color('#ffc107')
                else:
                    bar.set_color('#dc3545')
            
            ax4.set_xlabel('Accuracy (%)')
            ax4.set_title(f"{selected_student} - Topic Performance")
            ax4.axvline(x=80, color='green', linestyle='--', alpha=0.3, label='Excellent (80%+)')
            ax4.axvline(x=60, color='orange', linestyle='--', alpha=0.3, label='Good (60%+)')
            ax4.legend()
            plt.tight_layout()
            st.pyplot(fig4)
            
            # Topic details
            st.subheader("📊 Topic Statistics")
            topic_display = topic_agg[['topic', 'total', 'correct', 'accuracy']].copy()
            topic_display.columns = ['Topic', 'Questions', 'Correct', 'Accuracy (%)']
            st.dataframe(topic_display, use_container_width=True, hide_index=True)

        # Individual question breakdown
        st.subheader("❓ Question-by-Question Breakdown")
        question_breakdown = student_questions[['question_number', 'question_text', 'pred_topic', 
                                                'score', 'exam_id']].copy()
        question_breakdown['result'] = question_breakdown['score'].apply(lambda x: '✅' if x == 1 else '❌')
        question_breakdown = question_breakdown[['exam_id', 'question_number', 'pred_topic', 'result', 'question_text']]
        question_breakdown.columns = ['Exam', 'Q#', 'Topic', 'Result', 'Question Text']
        st.dataframe(question_breakdown, use_container_width=True, hide_index=True)

    # -----------------------
    # TAB 3: TOPIC ANALYSIS
    # -----------------------
    with tab3:
        st.header("COLREGS Topic Analysis")
        
        # Overall topic performance
        st.subheader("🎯 Overall Topic Performance")
        
        overall_topics = df.groupby('pred_topic').agg({
            'score': ['sum', 'count', 'mean']
        }).reset_index()
        overall_topics.columns = ['topic', 'correct', 'total', 'accuracy']
        overall_topics['accuracy'] = (overall_topics['accuracy'] * 100).round(1)
        overall_topics = overall_topics.sort_values('accuracy', ascending=True)
        
        fig5, ax5 = plt.subplots(figsize=(12, 8))
        bars = ax5.barh(overall_topics['topic'], overall_topics['accuracy'])
        
        for bar, acc in zip(bars, overall_topics['accuracy']):
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
        
        # Topic statistics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ Strongest Topics")
            strong_topics = overall_topics.nlargest(5, 'accuracy')[['topic', 'accuracy', 'total']]
            strong_topics.columns = ['Topic', 'Accuracy (%)', 'Questions']
            st.dataframe(strong_topics, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("⚠️ Topics Needing Focus")
            weak_topics = overall_topics.nsmallest(5, 'accuracy')[['topic', 'accuracy', 'total']]
            weak_topics.columns = ['Topic', 'Accuracy (%)', 'Questions']
            st.dataframe(weak_topics, use_container_width=True, hide_index=True)

        # Topic performance by class
        st.subheader("📊 Topic Performance by Class")
        
        class_topics = df.groupby(['Class', 'pred_topic']).agg({
            'score': ['sum', 'count']
        }).reset_index()
        class_topics.columns = ['Class', 'Topic', 'correct', 'total']
        class_topics['accuracy'] = (class_topics['correct'] / class_topics['total'] * 100).round(1)
        
        pivot_class_topics = class_topics.pivot(index='Topic', columns='Class', values='accuracy')
        
        if not pivot_class_topics.empty:
            fig6, ax6 = plt.subplots(figsize=(12, 10))
            sns.heatmap(pivot_class_topics, annot=True, fmt='.1f', cmap='RdYlGn', 
                       center=70, vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'}, ax=ax6)
            ax6.set_title('Topic Performance Heatmap by Class')
            ax6.set_xlabel('Class')
            ax6.set_ylabel('COLREGS Topic')
            plt.tight_layout()
            st.pyplot(fig6)

    # -----------------------
    # TAB 4: QUESTION INSIGHTS
    # -----------------------
    with tab4:
        st.header("Question-Level Insights")
        
        # Question difficulty distribution
        st.subheader("📊 Question Difficulty Distribution")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            easy = len(question_difficulty[question_difficulty['difficulty'] >= 80])
            st.metric("Easy Questions (80%+)", easy)
        with col2:
            medium = len(question_difficulty[(question_difficulty['difficulty'] >= 50) & 
                                            (question_difficulty['difficulty'] < 80)])
            st.metric("Medium Questions (50-79%)", medium)
        with col3:
            hard = len(question_difficulty[question_difficulty['difficulty'] < 50])
            st.metric("Hard Questions (<50%)", hard)

        fig7, ax7 = plt.subplots(figsize=(10, 5))
        ax7.hist(question_difficulty['difficulty'], bins=20, color='steelblue', 
                alpha=0.7, edgecolor='black')
        ax7.set_xlabel('Difficulty (%)')
        ax7.set_ylabel('Number of Questions')
        ax7.set_title('Question Difficulty Distribution')
        ax7.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Hard threshold')
        ax7.axvline(x=80, color='green', linestyle='--', alpha=0.5, label='Easy threshold')
        ax7.legend()
        plt.tight_layout()
        st.pyplot(fig7)

        # Most challenging questions
        st.subheader("❌ Most Challenging Questions")
        challenging = question_difficulty.nsmallest(10, 'difficulty')[
            ['question_text', 'topic', 'difficulty', 'attempts']
        ]
        challenging.columns = ['Question', 'Topic', 'Success Rate (%)', 'Attempts']
        st.dataframe(challenging, use_container_width=True, hide_index=True)

        # Easiest questions
        st.subheader("✅ Easiest Questions")
        easiest = question_difficulty.nlargest(10, 'difficulty')[
            ['question_text', 'topic', 'difficulty', 'attempts']
        ]
        easiest.columns = ['Question', 'Topic', 'Success Rate (%)', 'Attempts']
        st.dataframe(easiest, use_container_width=True, hide_index=True)

        # Question difficulty by topic
        st.subheader("📈 Average Question Difficulty by Topic")
        topic_q_diff = question_difficulty.groupby('topic')['difficulty'].mean().sort_values()
        
        fig8, ax8 = plt.subplots(figsize=(10, 6))
        ax8.barh(topic_q_diff.index, topic_q_diff.values, color='coral', alpha=0.8)
        ax8.set_xlabel('Average Success Rate (%)')
        ax8.set_title('Question Difficulty by Topic')
        ax8.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig8)

    # -----------------------
    # TAB 5: CLASS COMPARISON
    # -----------------------
    with tab5:
        st.header("Class Performance Comparison")
        
        # Class selector
        available_classes = sorted(df['Class'].unique())
        selected_classes = st.multiselect(
            "Select Classes to Compare", 
            available_classes, 
            default=available_classes
        )
        
        if len(selected_classes) > 0:
            # Filter and ensure ordered by class
            class_data = student_summary[student_summary['Class'].isin(selected_classes)].copy()
            class_data['Class'] = pd.Categorical(class_data['Class'], categories=available_classes, ordered=True)
            class_data = class_data.sort_values(by='Class')

            # Summary statistics
            st.subheader("📊 Class Statistics")
            class_stats = (
                class_data
                .groupby('Class', observed=True)
                .agg({
                    'percentage': ['mean', 'median', 'std', 'min', 'max'],
                    'email': 'nunique'
                })
                .round(1)
                .reindex(available_classes)  # keep consistent class order
            )
            class_stats.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Students']
            st.dataframe(class_stats, use_container_width=True)

            # Box plot comparison (ordered by class)
            st.subheader("📦 Score Distribution by Class")
            fig9, ax9 = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=class_data, x='Class', y='percentage', order=available_classes, ax=ax9)
            ax9.set_xlabel('Class')
            ax9.set_ylabel('Score (%)')
            ax9.set_title('Score Distribution Comparison')
            ax9.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig9)

            # Violin plot (ordered by class)
            fig10, ax10 = plt.subplots(figsize=(12, 6))
            sns.violinplot(data=class_data, x='Class', y='percentage', order=available_classes, ax=ax10)
            ax10.set_xlabel('Class')
            ax10.set_ylabel('Score (%)')
            ax10.set_title('Score Distribution (Violin Plot)')
            ax10.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig10)

            # Topic comparison across classes
            st.subheader("🎯 Topic Performance Comparison")
            
            # Get available topics and filter out NaN
            available_topics = [t for t in df['pred_topic'].unique() if pd.notna(t)]
            selected_topic = st.selectbox(
                "Select a Topic", 
                sorted(available_topics)
            )
            
            topic_class_data = (
                df[df['pred_topic'] == selected_topic]
                .groupby('Class', observed=True)
                .agg({'score': ['mean', 'count']})
                .reset_index()
            )
            topic_class_data.columns = ['Class', 'Accuracy', 'Questions']
            topic_class_data['Accuracy'] = (topic_class_data['Accuracy'] * 100).round(1)

            # Keep class order consistent
            topic_class_data['Class'] = pd.Categorical(topic_class_data['Class'], categories=available_classes, ordered=True)
            topic_class_data = topic_class_data.sort_values('Class')

            col1, col2 = st.columns([2, 1])
            with col1:
                fig11, ax11 = plt.subplots(figsize=(10, 5))
                ax11.bar(topic_class_data['Class'], topic_class_data['Accuracy'], color='steelblue', alpha=0.8)
                ax11.set_xlabel('Class')
                ax11.set_ylabel('Accuracy (%)')
                ax11.set_title(f'Performance on: {selected_topic}')
                ax11.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig11)
            
            with col2:
                st.dataframe(topic_class_data, use_container_width=True, hide_index=True)

    # -----------------------
    # TAB 6: TIME & EFFICIENCY (with Dropdown Selector)
    # -----------------------
    
    # Add these new imports at the top of your main script file
    import numpy as np
    import streamlit as st
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture # For Option 1
    import statsmodels.api as sm # For Option 2
    
    with tab6:
        st.header("Time Management & Efficiency Analysis")
    
        # --- Your existing duration_to_minutes function and data loading ---
        def duration_to_minutes(duration_str):
            try:
                if 'min' in str(duration_str):
                    parts = str(duration_str).split()
                    minutes = 0
                    for i, part in enumerate(parts):
                        if 'min' in part and i > 0:
                            minutes += int(parts[i-1])
                        elif 'sec' in part and i > 0:
                            minutes += int(parts[i-1]) / 60
                    return minutes
            except:
                return None
            return None
    
        student_summary['duration_minutes'] = student_summary['duration'].apply(duration_to_minutes)
        valid_duration = student_summary.dropna(subset=['duration_minutes', 'percentage'])
    
        if not valid_duration.empty and len(valid_duration) > 2:
            st.subheader("⏱️ Time Spent vs Score Analysis")
    
            # --- Dropdown Selector ---
            plot_type = st.selectbox(
                "Choose an analysis method for the scatter plot:",
                (
                    "1. Identify Clusters (GMM)",
                    "2. Flexible Trendline (LOWESS)",
                    "3. Curved Trendline (Polynomial Fit)"
                )
            )
    
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Duration", f"{valid_duration['duration_minutes'].mean():.1f} mins")
            with col2:
                st.metric("Median Duration", f"{valid_duration['duration_minutes'].median():.1f} mins")
            with col3:
                correlation = valid_duration['duration_minutes'].corr(valid_duration['percentage'])
                st.metric("Time-Score Correlation", f"{correlation:.3f} (Overall)")
    
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
    
            # --- OPTION 1: GMM Clustering ---
            if plot_type == "1. Identify Clusters (GMM)":
                st.info("This method automatically finds distinct groups of behavior.")
                X = valid_duration[['duration_minutes', 'percentage']]
                gmm = GaussianMixture(n_components=2, random_state=42).fit(X)
                labels = gmm.predict(X)
                
                scatter = ax.scatter(valid_duration['duration_minutes'], valid_duration['percentage'],
                                     c=labels, cmap='viridis', alpha=0.6, s=50)
                ax.set_title('Relationship Between Time Spent and Score (Identified Clusters)')
                ax.legend(handles=scatter.legend_elements()[0], labels=['Cluster 0', 'Cluster 1'])
    
                # Display a summary table for the clusters
                st.write("#### 🧠 Cluster Characteristics")
                valid_duration['cluster'] = labels
                cluster_stats = valid_duration.groupby('cluster').agg(
                    Count=('cluster', 'size'),
                    Avg_Duration=('duration_minutes', 'mean'),
                    Avg_Score=('percentage', 'mean'),
                    Score_Std_Dev=('percentage', 'std')
                ).round(1)
                st.dataframe(cluster_stats, use_container_width=True)
    
    
            # --- OPTION 2: LOWESS Flexible Trendline ---
            elif plot_type == "2. Flexible Trendline (LOWESS)":
                st.info("This method creates a smooth line that follows the local trend of the data.")
                ax.scatter(valid_duration['duration_minutes'], valid_duration['percentage'],
                           alpha=0.6, s=50, color='steelblue')
                
                plot_data = valid_duration.sort_values('duration_minutes')
                lowess_result = sm.nonparametric.lowess(plot_data['percentage'], plot_data['duration_minutes'], frac=0.4)
                
                ax.plot(lowess_result[:, 0], lowess_result[:, 1], 'r--', alpha=0.8, linewidth=2, label='Flexible Trend (LOWESS)')
                ax.set_title('Relationship Between Time Spent and Score (LOWESS Fit)')
                ax.legend()
    
    
            # --- OPTION 3: Polynomial Fit ---
            elif plot_type == "3. Curved Trendline (Polynomial Fit)":
                st.info("This method fits a single mathematical curve (degree 2) to all the data points.")
                ax.scatter(valid_duration['duration_minutes'], valid_duration['percentage'],
                           alpha=0.6, s=50, color='steelblue')
                
                degree = 2
                z = np.polyfit(valid_duration['duration_minutes'], valid_duration['percentage'], degree)
                p = np.poly1d(z)
                xp = np.linspace(valid_duration['duration_minutes'].min(), valid_duration['duration_minutes'].max(), 100)
                
                ax.plot(xp, p(xp), "r--", alpha=0.8, linewidth=2, label=f'Trend (Degree {degree})')
                ax.set_title('Relationship Between Time Spent and Score (Polynomial Fit)')
                ax.legend()
    
            # Common plot settings and display
            ax.set_xlabel('Duration (minutes)')
            ax.set_ylabel('Score (%)')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            # Duration distribution
            st.subheader("📊 Duration Distribution")
            fig13, ax13 = plt.subplots(figsize=(10, 5))
            ax13.hist(valid_duration['duration_minutes'], bins=20, color='coral', 
                     alpha=0.7, edgecolor='black')
            ax13.axvline(valid_duration['duration_minutes'].mean(), color='red', 
                        linestyle='--', linewidth=2, label='Mean')
            ax13.axvline(valid_duration['duration_minutes'].median(), color='green', 
                        linestyle='--', linewidth=2, label='Median')
            ax13.set_xlabel('Duration (minutes)')
            ax13.set_ylabel('Number of Attempts')
            ax13.set_title('Exam Duration Distribution')
            ax13.legend()
            plt.tight_layout()
            st.pyplot(fig13)

            # Fast vs Slow performers
            st.subheader("🏃 Speed vs Performance Analysis")
            
            median_time = valid_duration['duration_minutes'].median()
            valid_duration['speed_category'] = valid_duration['duration_minutes'].apply(
                lambda x: 'Fast (<median)' if x < median_time else 'Slow (>=median)'
            )
            
            speed_analysis = valid_duration.groupby('speed_category')['percentage'].agg(['mean', 'count']).reset_index()
            speed_analysis.columns = ['Category', 'Avg Score (%)', 'Count']
            speed_analysis['Avg Score (%)'] = speed_analysis['Avg Score (%)'].round(1)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig14, ax14 = plt.subplots(figsize=(8, 5))
                ax14.bar(speed_analysis['Category'], speed_analysis['Avg Score (%)'], 
                        color=['lightcoral', 'lightblue'], alpha=0.8)
                ax14.set_ylabel('Average Score (%)')
                ax14.set_title('Performance: Fast vs Slow Test Takers')
                ax14.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig14)
            
            with col2:
                st.dataframe(speed_analysis, use_container_width=True, hide_index=True)

            # Time by class
            st.subheader("⏰ Average Duration by Class")
            class_duration = valid_duration.groupby('Class')['duration_minutes'].agg(['mean', 'std']).reset_index()
            class_duration.columns = ['Class', 'Avg Duration (mins)', 'Std Dev']
            class_duration = class_duration.round(1)
            
            fig15, ax15 = plt.subplots(figsize=(10, 5))
            ax15.bar(class_duration['Class'], class_duration['Avg Duration (mins)'], 
                    color='mediumseagreen', alpha=0.8)
            ax15.set_xlabel('Class')
            ax15.set_ylabel('Average Duration (minutes)')
            ax15.set_title('Average Exam Duration by Class')
            ax15.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig15)
            
            st.dataframe(class_duration, use_container_width=True, hide_index=True)
        
        else:
            st.warning("Duration data not available for analysis.")

else:
    st.warning("Please enter your credentials to access the COLREGS Dashboard.")
    st.info("Contact your administrator for login credentials.")
