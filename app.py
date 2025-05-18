import streamlit as st
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
from datetime import datetime
import logging
import sys
from pathlib import Path

# Import the CorporateGovernanceAgent class and CGSConfig
from scoring_topics_agentic_langchain import CorporateGovernanceAgent, CGSConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Corporate Governance Scoring System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #1E3A8A;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 10px;
        color: #2563EB;
    }
    .status-message {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .success-message {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
    }
    .info-message {
        background-color: #DBEAFE;
        border-left: 5px solid #3B82F6;
    }
    .warning-message {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
    }
    .error-message {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
    .option-card {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .results-container {
        background-color: #F9FAFB;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'setup_done' not in st.session_state:
    st.session_state.setup_done = False
if 'last_action' not in st.session_state:
    st.session_state.last_action = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'source_map' not in st.session_state:
    st.session_state.source_map = None
if 'last_operation_time' not in st.session_state:
    st.session_state.last_operation_time = None
if 'operation_status' not in st.session_state:
    st.session_state.operation_status = "None"

def get_company_symbols():
    """Get list of company symbols from parent directory"""
    base_path = st.session_state.get('base_path', '')
    if not base_path:
        return []
        
    parent_path = Path(base_path).parent
    
    try:
        return [f.name for f in os.scandir(parent_path) 
                if f.is_dir() and f.name not in ['95_all_results', '.git', '__pycache__']]
    except Exception as e:
        logger.error(f"Error getting company symbols: {e}")
        return []

def get_questions_list():
    """Get list of questions from prompts.csv"""
    base_path = st.session_state.get('base_path', '')
    if not base_path:
        return []
        
    parent_path = Path(base_path).parent
    prompts_path = parent_path / 'prompts.csv'
    
    if not prompts_path.exists():
        return []
        
    try:
        df = pd.read_csv(prompts_path)
        # Return a list of tuples (sr_no, que_no, cat, message)
        return [(row['sr_no'], row['que_no'], row['cat'], 
                 row.get('disp_message', row['message'])) 
                for _, row in df.iterrows()]
    except Exception as e:
        logger.error(f"Error loading questions: {e}")
        return []

def get_scoring_criteria():
    """Get list of scoring criteria topics"""
    base_path = st.session_state.get('base_path', '')
    if not base_path:
        return []
        
    parent_path = Path(base_path).parent
    criteria_path = parent_path / 'scoring_creteria.csv'
    
    if not criteria_path.exists():
        return []
        
    try:
        df = pd.read_csv(criteria_path)
        # Return a list of tuples (topic_no, topic_name, category)
        return [(row['topic_no'], row.get('topic_name', f"Topic {row['topic_no']}"), 
                 row.get('category', '')) 
                for _, row in df.iterrows()]
    except Exception as e:
        logger.error(f"Error loading scoring criteria: {e}")
        return []

def init_agent(company_sym, retrieval_method="hybrid"):
    """Initialize agent with specified company and retrieval method"""
    try:
        # Display status
        status_placeholder = st.empty()
        status_placeholder.info("Initializing agent... This may take a moment.")
        
        # Get base path
        base_path = st.session_state.get('base_path', '')
        if not base_path:
            base_path = None
        
        # Create config and set retrieval method
        config = CGSConfig(company_sym, base_path)
        config.retrieval_method = retrieval_method
        
        # Update session state
        st.session_state.config = config
        st.session_state.base_path = config.base_path
        
        # Initialize agent
        agent = CorporateGovernanceAgent(company_sym, base_path=base_path, config=config)
        st.session_state.agent = agent
        
        # Update status
        status_placeholder.success(f"Agent initialized for company {company_sym} with retrieval method: {retrieval_method}")
        return True
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        logger.error(f"Error initializing agent: {e}")
        return False

def setup_agent():
    """Set up the agent by downloading documents and creating source maps"""
    if not st.session_state.agent:
        st.error("Agent not initialized. Please initialize the agent first.")
        return False
        
    try:
        # Display status
        status_placeholder = st.empty()
        status_placeholder.info("Setting up agent... This may take a moment.")
        
        # Setup agent
        start_time = time.time()
        agent = st.session_state.agent.setup()
        end_time = time.time()
        
        # Update session state
        st.session_state.agent = agent
        st.session_state.setup_done = True
        st.session_state.source_map = agent.source_map
        st.session_state.last_operation_time = end_time - start_time
        st.session_state.operation_status = "Success"
        
        # Update status
        status_placeholder.success(f"Agent setup completed in {end_time - start_time:.2f} seconds. {len(agent.source_map)} documents processed.")
        return True
    except Exception as e:
        st.error(f"Error setting up agent: {e}")
        logger.error(f"Error setting up agent: {e}")
        st.session_state.operation_status = "Error"
        return False

def process_questions(sr_no_list=None, load_all_fresh=False):
    """Process questions from prompts.csv"""
    if not st.session_state.agent or not st.session_state.setup_done:
        st.error("Agent not initialized or setup not completed. Please initialize and setup the agent first.")
        return False
        
    try:
        # Display status
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Get number of questions to process for progress tracking
        if sr_no_list:
            total_questions = len(sr_no_list)
            status_placeholder.info(f"Processing {total_questions} selected questions...")
        else:
            # Estimate number of questions
            questions = get_questions_list()
            total_questions = len(questions)
            status_placeholder.info(f"Processing all {total_questions} questions...")
        
        # Start timer
        start_time = time.time()
        
        # Create a function to update progress
        current_question = 0
        
        def progress_callback():
            nonlocal current_question
            current_question += 1
            progress = min(current_question / total_questions, 1.0)
            progress_bar.progress(progress)
            status_placeholder.info(f"Processing question {current_question}/{total_questions}...")
        
        # Original function doesn't have callback, so we'll simulate it
        # by setting up a background execution and periodic checks
        
        # Start processing in session state to track it
        st.session_state.processing_started = True
        st.session_state.processing_complete = False
        st.session_state.processing_error = None
        
        # Process questions
        results_df = st.session_state.agent.process_questions(
            load_all_fresh=load_all_fresh, 
            sr_no_list=sr_no_list
        )
        
        # Mark as complete
        st.session_state.processing_complete = True
        st.session_state.last_result = results_df
        
        # Update stats
        end_time = time.time()
        st.session_state.last_operation_time = end_time - start_time
        st.session_state.operation_status = "Success"
        
        # Update status
        progress_bar.progress(1.0)
        status_placeholder.success(f"Processing completed in {end_time - start_time:.2f} seconds. {len(results_df) if results_df is not None else 0} results generated.")
        return True
    except Exception as e:
        st.error(f"Error processing questions: {e}")
        logger.error(f"Error processing questions: {e}")
        st.session_state.operation_status = "Error"
        st.session_state.processing_error = str(e)
        return False

def score_topic(topic_no):
    """Score a specific topic"""
    if not st.session_state.agent or not st.session_state.setup_done:
        st.error("Agent not initialized or setup not completed. Please initialize and setup the agent first.")
        return False
        
    try:
        # Display status
        status_placeholder = st.empty()
        status_placeholder.info(f"Scoring topic {topic_no}... This may take a moment.")
        
        # Start timer
        start_time = time.time()
        
        # Score topic
        score_result = st.session_state.agent.score_topic(topic_no)
        
        # Update stats
        end_time = time.time()
        st.session_state.last_operation_time = end_time - start_time
        st.session_state.operation_status = "Success"
        st.session_state.last_result = score_result
        
        # Update status
        status_placeholder.success(f"Topic {topic_no} scoring completed in {end_time - start_time:.2f} seconds.")
        return True
    except Exception as e:
        st.error(f"Error scoring topic: {e}")
        logger.error(f"Error scoring topic: {e}")
        st.session_state.operation_status = "Error"
        return False

def score_category(category_num):
    """Score all topics in a specific category"""
    if not st.session_state.agent or not st.session_state.setup_done:
        st.error("Agent not initialized or setup not completed. Please initialize and setup the agent first.")
        return False
        
    try:
        # Display status
        status_placeholder = st.empty()
        status_placeholder.info(f"Scoring category {category_num}... This may take a moment.")
        
        # Start timer
        start_time = time.time()
        
        # Score category
        st.session_state.agent.score_category(category_num)
        
        # Update stats
        end_time = time.time()
        st.session_state.last_operation_time = end_time - start_time
        st.session_state.operation_status = "Success"
        
        # Update status
        status_placeholder.success(f"Category {category_num} scoring completed in {end_time - start_time:.2f} seconds.")
        return True
    except Exception as e:
        st.error(f"Error scoring category: {e}")
        logger.error(f"Error scoring category: {e}")
        st.session_state.operation_status = "Error"
        return False

def score_all_categories():
    """Score all categories"""
    if not st.session_state.agent or not st.session_state.setup_done:
        st.error("Agent not initialized or setup not completed. Please initialize and setup the agent first.")
        return False
        
    try:
        # Display status
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Start timer
        start_time = time.time()
        
        # Score all categories (4 categories)
        for i, category in enumerate(range(1, 5)):
            status_placeholder.info(f"Scoring category {category}/4...")
            progress_bar.progress((i) / 4)
            st.session_state.agent.score_category(category)
            progress_bar.progress((i+1) / 4)
        
        # Update stats
        end_time = time.time()
        st.session_state.last_operation_time = end_time - start_time
        st.session_state.operation_status = "Success"
        
        # Update status
        progress_bar.progress(1.0)
        status_placeholder.success(f"All categories scored in {end_time - start_time:.2f} seconds.")
        return True
    except Exception as e:
        st.error(f"Error scoring all categories: {e}")
        logger.error(f"Error scoring all categories: {e}")
        st.session_state.operation_status = "Error"
        return False

def aggregate_results():
    """Aggregate results from multiple companies"""
    if not st.session_state.agent:
        st.error("Agent not initialized. Please initialize the agent first.")
        return False
        
    try:
        # Display status
        status_placeholder = st.empty()
        status_placeholder.info("Aggregating results... This may take a moment.")
        
        # Start timer
        start_time = time.time()
        
        # Aggregate results
        all_prompt_results, all_que_results = st.session_state.agent.aggregate_results()
        
        # Update stats
        end_time = time.time()
        st.session_state.last_operation_time = end_time - start_time
        st.session_state.operation_status = "Success"
        st.session_state.last_result = {
            "all_prompt_results": all_prompt_results,
            "all_que_results": all_que_results
        }
        
        # Update status
        status_placeholder.success(f"Results aggregated in {end_time - start_time:.2f} seconds.")
        return True
    except Exception as e:
        st.error(f"Error aggregating results: {e}")
        logger.error(f"Error aggregating results: {e}")
        st.session_state.operation_status = "Error"
        return False

def load_and_display_scores():
    """Load and display scores from que_wise_scores_final.csv"""
    if not st.session_state.agent:
        st.error("Agent not initialized. Please initialize the agent first.")
        return False
        
    try:
        # Get file path
        file_path = os.path.join(st.session_state.agent.config.results_path, 'que_wise_scores_final.csv')
        
        if not os.path.exists(file_path):
            st.warning("No scores file found. Please score topics first.")
            return False
            
        # Load scores
        scores_df = pd.read_csv(file_path)
        
        # Return dataframe
        return scores_df
    except Exception as e:
        st.error(f"Error loading scores: {e}")
        logger.error(f"Error loading scores: {e}")
        return False


def create_score_visualization(scores_df):
    """Create visualization of scores with adjusted scale (0-2)"""
    if scores_df is None or len(scores_df) == 0:
        st.warning("No scores data available for visualization.")
        return
        
    try:
        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by category and calculate mean score
        category_scores = scores_df.groupby('category')['score'].mean().reset_index()
        
        # Create bar chart with adjusted scale
        bars = sns.barplot(x='category', y='score', data=category_scores, ax=ax, palette='viridis')
        
        # Add value labels on top of bars
        for bar in bars.patches:
            bars.annotate(f'{bar.get_height():.1f}',
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='bottom',
                        fontsize=10)
        
        # Add labels and title
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.set_title('Average Scores by Category', fontsize=14)
        
        # Set y-axis to appropriate scale (0-2)
        ax.set_ylim(0, 2.2)  # Slightly higher than 2 to accommodate labels
        
        # Add gridlines at 0, 0.5, 1, 1.5, and 2
        ax.set_yticks([0, 0.5, 1, 1.5, 2])
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Return figure
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        logger.error(f"Error creating visualization: {e}")
        return None

def display_topic_score_chart(scores_df):
    """Display topic-wise scores with appropriate scale (0-2)"""
    if scores_df is None or len(scores_df) == 0:
        st.warning("No scores data available for visualization.")
        return
        
    try:
        # Create horizontal bar chart for topic scores
        fig, ax = plt.subplots(figsize=(12, max(8, len(scores_df) * 0.4)))
        
        # Sort by question number
        sorted_df = scores_df.sort_values('que_no')
        
        # Create horizontal bar chart with adjusted scale
        bars = sns.barplot(x='score', y='que_no', data=sorted_df, ax=ax, palette='viridis', orient='h')
        
        # Add value labels beside bars
        for i, bar in enumerate(bars.patches):
            ax.text(
                bar.get_width() + 0.05,
                bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.0f}",
                ha='left', va='center'
            )
        
        # Add labels and title
        ax.set_xlabel('Score', fontsize=12)
        ax.set_ylabel('Question Number', fontsize=12)
        ax.set_title('Scores by Question', fontsize=14)
        ax.set_xlim(0, 2.2)  # Slightly higher than 2 for labels
        
        # Set x-axis ticks to 0, 1, and 2
        ax.set_xticks([0, 1, 2])
        
        # Add grid
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        return fig
    except Exception as e:
        st.error(f"Error creating topic score chart: {e}")
        logger.error(f"Error creating topic score chart: {e}")
        return None

def display_company_comparison_chart(company_scores):
    """Display company comparison with appropriate scale (0-2)"""
    if company_scores is None or len(company_scores) == 0:
        st.warning("No company scores data available for visualization.")
        return
        
    try:
        # Create bar chart for company comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by score descending
        company_scores = company_scores.sort_values('score', ascending=False)
        
        # Create bar chart with adjusted scale
        bars = sns.barplot(x='company', y='score', data=company_scores, ax=ax, palette='viridis')
        
        # Add value labels on top of bars
        for bar in bars.patches:
            ax.annotate(f'{bar.get_height():.1f}',
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='bottom',
                        fontsize=10)
        
        # Add labels and title
        ax.set_xlabel('Company', fontsize=12)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.set_title('Average Scores by Company', fontsize=14)
        ax.set_ylim(0, 2.2)  # Adjusted for 0-2 scale with room for labels
        
        # Set y-axis to show 0, 1, and 2
        ax.set_yticks([0, 0.5, 1, 1.5, 2])
        
        # Rotate x-labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        return fig
    except Exception as e:
        st.error(f"Error creating company comparison chart: {e}")
        logger.error(f"Error creating company comparison chart: {e}")
        return None

def display_category_heatmap(category_data, selected_category):
    """Display category heatmap with appropriate scale (0-2)"""
    if category_data is None or len(category_data) == 0:
        st.warning("No category data available for visualization.")
        return
        
    try:
        # Create pivot table: companies vs questions
        if 'que_no' in category_data.columns and 'company' in category_data.columns:
            # Group by company and question, get average score
            pivot_data = category_data.pivot_table(
                index='company', 
                columns='que_no', 
                values='score',
                aggfunc='mean'
            )
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, len(pivot_data) * 0.5 + 2))
            
            # Create heatmap with adjusted scale
            sns.heatmap(
                pivot_data, 
                annot=True, 
                cmap='YlGnBu', 
                cbar_kws={'label': 'Score'}, 
                ax=ax,
                vmin=0,
                vmax=2,
                fmt='.0f'  # Format as integers
            )
            
            # Add labels and title
            ax.set_xlabel('Question Number', fontsize=12)
            ax.set_ylabel('Company', fontsize=12)
            ax.set_title(f'Company Comparison for Category: {selected_category}', fontsize=14)
            
            return fig
        return None
    except Exception as e:
        st.error(f"Error creating category heatmap: {e}")
        logger.error(f"Error creating category heatmap: {e}")
        return None




# Main app layout
def main():
    # Header
    st.markdown('<p class="main-header">Corporate Governance Scoring System</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown('## Configuration')
    
    # Base path input
    base_path = st.sidebar.text_input(
        "Base Path",
        value=st.session_state.get('base_path', ''),
        help="Base path for company data (leave empty for default)",
        key="base_path_input"
    )
    
    if base_path:
        st.session_state.base_path = base_path

    # Company selection
    company_symbols = get_company_symbols()
    company_options = [""] + company_symbols if company_symbols else [""]
    selected_company = st.sidebar.selectbox(
        "Select Company", 
        options=company_options,
        index=0,
        help="Select a company to analyze"
    )
    
    # Retrieval method
    retrieval_method = st.sidebar.selectbox(
        "Retrieval Method",
        options=["hybrid", "bm25", "vector", "direct"],
        index=0,
        help="Method used to retrieve information from documents"
    )
    
    # Initialize agent button
    if st.sidebar.button("Initialize Agent"):
        if not selected_company:
            st.sidebar.error("Please select a company")
        else:
            init_agent(selected_company, retrieval_method)
    
    # Setup agent button
    if st.sidebar.button("Setup Agent"):
        if not st.session_state.agent:
            st.sidebar.error("Please initialize the agent first")
        else:
            setup_agent()
    
    # Display agent status
    if st.session_state.agent:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Agent Status")
        st.sidebar.markdown(f"**Company:** {st.session_state.agent.config.company_sym}")
        st.sidebar.markdown(f"**Retrieval Method:** {st.session_state.agent.config.retrieval_method}")
        st.sidebar.markdown(f"**Setup Completed:** {'Yes' if st.session_state.setup_done else 'No'}")
        
        if st.session_state.source_map:
            st.sidebar.markdown(f"**Documents:** {len(st.session_state.source_map)}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Process Questions", 
        "Score Topics", 
        "View Results", 
        "Visualize Scores",
        "Aggregate Results"
    ])
    
    # Tab 1: Process Questions
    with tab1:
        st.markdown('<p class="section-header">Process Questions</p>', unsafe_allow_html=True)
        
        if not st.session_state.agent or not st.session_state.setup_done:
            st.warning("Please initialize and setup the agent first.")
        else:
            # Option selection
            process_option = st.radio(
                "Select processing option:",
                ["Process All Questions", "Process Selected Questions", "Process Fresh (All Questions)"]
            )
            
            if process_option == "Process Selected Questions":
                # Get questions list
                questions = get_questions_list()
                
                if not questions:
                    st.warning("No questions found in prompts.csv")
                else:
                    # Create a multiselect for questions
                    question_options = [f"{sr_no}: Q{que_no} ({cat}) - {msg[:50]}..." for sr_no, que_no, cat, msg in questions]
                    selected_questions = st.multiselect(
                        "Select questions to process:",
                        options=question_options
                    )
                    
                    # Extract sr_no from selected questions
                    selected_sr_nos = [int(q.split(':')[0]) for q in selected_questions]
            
            # Process button
            if st.button("Start Processing"):
                if process_option == "Process All Questions":
                    process_questions()
                elif process_option == "Process Selected Questions":
                    if not selected_sr_nos:
                        st.error("Please select at least one question")
                    else:
                        process_questions(sr_no_list=selected_sr_nos)
                elif process_option == "Process Fresh (All Questions)":
                    process_questions(load_all_fresh=True)
    
    # Tab 2: Score Topics
    with tab2:
        st.markdown('<p class="section-header">Score Topics</p>', unsafe_allow_html=True)
        
        if not st.session_state.agent or not st.session_state.setup_done:
            st.warning("Please initialize and setup the agent first.")
        else:
            # Option selection
            score_option = st.radio(
                "Select scoring option:",
                ["Score Specific Topic", "Score Category", "Score All Categories"]
            )
            
            if score_option == "Score Specific Topic":
                # Get scoring topics
                topics = get_scoring_criteria()
                
                if not topics:
                    st.warning("No scoring criteria found")
                else:
                    # Create a selectbox for topics
                    topic_options = [f"Topic {topic_no}: {topic_name} ({category})" 
                                    for topic_no, topic_name, category in topics]
                    selected_topic = st.selectbox(
                        "Select topic to score:",
                        options=topic_options
                    )
                    
                    # Extract topic_no from selected topic
                    selected_topic_no = int(selected_topic.split(':')[0].replace('Topic ', ''))
            
            elif score_option == "Score Category":
                # Select category
                selected_category = st.selectbox(
                    "Select category to score:",
                    options=[
                        "Category 1: Rights and equitable treatment of shareholders",
                        "Category 2: Role of stakeholders",
                        "Category 3: Transparency and disclosure",
                        "Category 4: Responsibility of the board"
                    ]
                )
                
                # Extract category number
                selected_category_no = int(selected_category.split(':')[0].replace('Category ', ''))
            
            # Score button
            if st.button("Start Scoring"):
                if score_option == "Score Specific Topic":
                    if not 'selected_topic_no' in locals():
                        st.error("Please select a topic")
                    else:
                        score_topic(selected_topic_no)
                elif score_option == "Score Category":
                    if not 'selected_category_no' in locals():
                        st.error("Please select a category")
                    else:
                        score_category(selected_category_no)
                elif score_option == "Score All Categories":
                    score_all_categories()
    
    # Tab 3: View Results
    with tab3:
        st.markdown('<p class="section-header">View Results</p>', unsafe_allow_html=True)
        
        if not st.session_state.agent:
            st.warning("Please initialize the agent first.")
        else:
            # Option selection
            result_option = st.radio(
                "Select result type:",
                ["Question Results", "Topic Scores"]
            )
            
            if result_option == "Question Results":
                # Load question results
                result_path = os.path.join(st.session_state.agent.config.results_path, 'prompts_result.csv')
                
                if not os.path.exists(result_path):
                    st.warning("No question results found")
                else:
                    results_df = pd.read_csv(result_path)
                    
                    # Filter options
                    que_no_filter = st.text_input("Filter by Question Number (leave empty for all)")
                    source_filter = st.text_input("Filter by Source (leave empty for all)")
                    
                    # Apply filters
                    filtered_df = results_df
                    if que_no_filter:
                        filtered_df = filtered_df[filtered_df['que_no'] == int(que_no_filter)]
                    if source_filter:
                        filtered_df = filtered_df[filtered_df['source'].str.contains(source_filter, case=False)]
                    
                    # Display results
                    st.dataframe(filtered_df)
                    
                    # Download button
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name=f"question_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            elif result_option == "Topic Scores":
                # Load scores
                scores_df = load_and_display_scores()
                
                if isinstance(scores_df, pd.DataFrame):
                    # Filter options
                    category_filter = st.text_input("Filter by Category (leave empty for all)")
                    
                    # Apply filters
                    filtered_df = scores_df
                    if category_filter:
                        filtered_df = filtered_df[filtered_df['category'].str.contains(category_filter, case=False)]
                    
                    # Display results
                    st.dataframe(filtered_df)
                    
                    # Download button
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download Scores as CSV",
                        data=csv,
                        file_name=f"topic_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    
    # Tab 4: Visualize Scores
    with tab4:
        st.markdown('<p class="section-header">Visualize Scores</p>', unsafe_allow_html=True)
        
        if not st.session_state.agent:
            st.warning("Please initialize the agent first.")
        else:
            # Load scores
            scores_df = load_and_display_scores()
            
            if isinstance(scores_df, pd.DataFrame):
                # Create category average visualization
                st.markdown("### Category Average Scores")
                fig1 = create_score_visualization(scores_df)
                if fig1:
                    st.pyplot(fig1)
                
                # Topic-wise scores
                st.markdown("### Topic-wise Scores")
                fig2 = display_topic_score_chart(scores_df)
                if fig2:
                    st.pyplot(fig2)
                
                # Category-wise analysis
                st.markdown("### Category-wise Analysis")
                
                # Group by category
                if 'category' in scores_df.columns:
                    category_scores = scores_df.groupby('category').agg({
                        'score': ['mean', 'min', 'max', 'count']
                    }).reset_index()
                    
                    category_scores.columns = ['Category', 'Average Score', 'Min Score', 'Max Score', 'Number of Topics']
                    
                    # Display table
                    st.dataframe(category_scores)
    
    # Tab 5: Aggregate Results
    with tab5:
        st.markdown('<p class="section-header">Aggregate Results</p>', unsafe_allow_html=True)
        
        if not st.session_state.agent:
            st.warning("Please initialize the agent first.")
        else:
            # Aggregate button
            if st.button("Aggregate Results from All Companies"):
                aggregate_results()
            
            # Check if aggregated results exist
            agg_path = os.path.join(
                Path(st.session_state.agent.config.parent_path), 
                '95_all_results', 
                'all_que_results.csv'
            )
            
            if os.path.exists(agg_path):
                st.success("Aggregated results are available")
                
                # Load aggregated scores
                try:
                    agg_scores = pd.read_csv(agg_path)
                    
                    # Display aggregated scores
                    st.markdown("### Aggregated Scores from All Companies")
                    
                    # Add filters for better analysis
                    company_filter = st.text_input("Filter by company:", "")
                    category_filter = st.text_input("Filter by category:", "")
                    
                    # Apply filters
                    filtered_agg = agg_scores
                    if company_filter:
                        filtered_agg = filtered_agg[filtered_agg['company'].str.contains(company_filter, case=False)]
                    if category_filter:
                        filtered_agg = filtered_agg[filtered_agg['category'].str.contains(category_filter, case=False)]
                    
                    # Display filtered results
                    st.dataframe(filtered_agg)
                    
                    # Download button for filtered results
                    csv = filtered_agg.to_csv(index=False)
                    st.download_button(
                        label="Download Filtered Results",
                        data=csv,
                        file_name=f"filtered_aggregated_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Create comparative visualization
                    st.markdown("### Company Comparison")
                    
                    if len(agg_scores) > 0:
                        # Group by company and calculate mean score
                        company_scores = agg_scores.groupby('company')['score'].mean().reset_index()
                        
                        # Create bar chart for company comparison
                        fig3, ax3 = plt.subplots(figsize=(12, 8))
                        
                        # Sort by score descending
                        company_scores = company_scores.sort_values('score', ascending=False)
                        
                        # Create bar chart with adjusted scale
                        bars = sns.barplot(x='company', y='score', data=company_scores, ax=ax3, palette='viridis')
                        
                        # Add value labels on top of bars
                        for bar in bars.patches:
                            ax3.annotate(f'{bar.get_height():.1f}',
                                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                        ha='center', va='bottom',
                                        fontsize=10)
                        
                        # Add labels and title
                        ax3.set_xlabel('Company', fontsize=12)
                        ax3.set_ylabel('Average Score', fontsize=12)
                        ax3.set_title('Average Scores by Company', fontsize=14)
                        ax3.set_ylim(0, 2.2)  # Adjusted for 0-2 scale with room for labels
                        
                        # Set y-axis to show 0, 0.5, 1, 1.5, and 2
                        ax3.set_yticks([0, 0.5, 1, 1.5, 2])
                        
                        # Rotate x-labels for better readability
                        plt.xticks(rotation=45, ha='right')
                        
                        # Add grid
                        ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
                        
                        # Display plot
                        st.pyplot(fig3)
                        
                        # Category-wise comparison
                        st.markdown("### Category-wise Company Comparison")
                        
                        # Add category selector
                        if 'category' in agg_scores.columns:
                            categories = sorted(agg_scores['category'].unique())
                            selected_category = st.selectbox(
                                "Select category for comparison:",
                                options=categories
                            )
                            
                            # Filter by selected category
                            category_data = agg_scores[agg_scores['category'] == selected_category]
                            
                            # Create pivot table: companies vs questions
                            if 'que_no' in category_data.columns and len(category_data) > 0:
                                # Group by company and question, get average score
                                pivot_data = category_data.pivot_table(
                                    index='company', 
                                    columns='que_no', 
                                    values='score',
                                    aggfunc='mean'
                                )
                                
                                # Create heatmap
                                fig4, ax4 = plt.subplots(figsize=(12, len(pivot_data) * 0.5 + 2))
                                
                                # Create heatmap with adjusted scale for 0-2
                                sns.heatmap(
                                    pivot_data, 
                                    annot=True, 
                                    cmap='YlGnBu', 
                                    cbar_kws={'label': 'Score'}, 
                                    ax=ax4,
                                    vmin=0,
                                    vmax=2,
                                    fmt='.1f'  # Format with one decimal place
                                )
                                
                                # Add labels and title
                                ax4.set_xlabel('Question Number', fontsize=12)
                                ax4.set_ylabel('Company', fontsize=12)
                                ax4.set_title(f'Company Comparison for Category: {selected_category}', fontsize=14)
                                
                                # Display plot
                                st.pyplot(fig4)
                        
                        # Add question-specific comparison
                        st.markdown("### Question-Specific Company Comparison")
                        
                        # Add question selector
                        if 'que_no' in agg_scores.columns:
                            questions = sorted(agg_scores['que_no'].unique())
                            selected_question = st.selectbox(
                                "Select question for comparison:",
                                options=questions
                            )
                            
                            # Filter by selected question
                            question_data = agg_scores[agg_scores['que_no'] == selected_question]
                            
                            # Create bar chart for specific question
                            if len(question_data) > 0:
                                fig5, ax5 = plt.subplots(figsize=(12, 8))
                                
                                # Sort by score descending
                                question_data = question_data.sort_values('score', ascending=False)
                                
                                # Create bar chart with adjusted scale
                                bars = sns.barplot(x='company', y='score', data=question_data, ax=ax5, palette='viridis')
                                
                                # Add value labels on top of bars
                                for bar in bars.patches:
                                    ax5.annotate(f'{bar.get_height():.0f}',
                                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                            ha='center', va='bottom',
                                            fontsize=10)
                                
                                # Add labels and title
                                ax5.set_xlabel('Company', fontsize=12)
                                ax5.set_ylabel('Score', fontsize=12)
                                ax5.set_title(f'Company Scores for Question {selected_question}', fontsize=14)
                                ax5.set_ylim(0, 2.2)  # Adjusted for 0-2 scale
                                
                                # Set y-axis to show exactly 0, 1, and 2
                                ax5.set_yticks([0, 1, 2])
                                
                                # Rotate x-labels for better readability
                                plt.xticks(rotation=45, ha='right')
                                
                                # Add grid
                                ax5.grid(True, axis='y', linestyle='--', alpha=0.7)
                                
                                # Display plot
                                st.pyplot(fig5)
                                
                                # Add justification view
                                st.markdown("### Score Justifications")
                                
                                if 'justification' in question_data.columns:
                                    for _, row in question_data.iterrows():
                                        with st.expander(f"{row['company']} - Score: {row['score']}"):
                                            st.write(row['justification'])
                        
                        # Add correlation analysis
                        st.markdown("### Score Correlation Analysis")
                        
                        if len(agg_scores) > 0:
                            # Create pivot table for correlation analysis
                            try:
                                # Pivot data to get company x question matrix
                                corr_pivot = agg_scores.pivot_table(
                                    index='company', 
                                    columns='que_no', 
                                    values='score',
                                    aggfunc='mean'
                                )
                                
                                # Calculate correlation between questions
                                corr_matrix = corr_pivot.corr()
                                
                                # Create heatmap for correlation
                                fig6, ax6 = plt.subplots(figsize=(12, 10))
                                
                                # Create heatmap with custom scale for correlation (-1 to 1)
                                sns.heatmap(
                                    corr_matrix, 
                                    annot=True, 
                                    cmap='coolwarm', 
                                    cbar_kws={'label': 'Correlation'}, 
                                    ax=ax6,
                                    vmin=-1,
                                    vmax=1,
                                    fmt='.2f'  # Format with two decimal places
                                )
                                
                                # Add labels and title
                                ax6.set_xlabel('Question Number', fontsize=12)
                                ax6.set_ylabel('Question Number', fontsize=12)
                                ax6.set_title('Correlation Between Question Scores', fontsize=14)
                                
                                # Display plot
                                st.pyplot(fig6)
                                
                                # Add explanation
                                st.info("""
                                **Correlation Analysis Explanation:**
                                - **1.0**: Questions that always receive the same scores (perfect positive correlation)
                                - **0.0**: Questions with no pattern between their scores (no correlation)
                                - **-1.0**: Questions that receive opposite scores (perfect negative correlation)
                                
                                High positive correlations may indicate redundant questions, while negative correlations might show trade-offs between different governance aspects.
                                """)
                            except Exception as e:
                                st.warning(f"Could not perform correlation analysis: {e}")
                
                except Exception as e:
                    st.error(f"Error loading or processing aggregated results: {e}")
                    
            else:
                st.info("No aggregated results found. Click 'Aggregate Results from All Companies' to generate them.")
                
                # Suggest alternatives if no aggregated results
                st.markdown("""
                ### Alternative Analyses
                
                While waiting for aggregated results, you can:
                
                1. Score individual topics for your current company
                2. View single-company results in the 'View Results' tab
                3. Create visualizations for the current company in 'Visualize Scores' tab
                """)
            
    
    # Display operation status
    if st.session_state.operation_status != "None":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Last Operation")
        
        if st.session_state.operation_status == "Success":
            st.sidebar.markdown('<div class="status-message success-message">Success</div>', unsafe_allow_html=True)
        elif st.session_state.operation_status == "Error":
            st.sidebar.markdown('<div class="status-message error-message">Error</div>', unsafe_allow_html=True)
        
        if st.session_state.last_operation_time:
            st.sidebar.markdown(f"**Duration:** {st.session_state.last_operation_time:.2f} seconds")

if __name__ == "__main__":
    main()
                    
#/Users/monilshah/Documents/02_NWU/01_capstone/04_Code_v3/PAYTM/                    
                    
                    