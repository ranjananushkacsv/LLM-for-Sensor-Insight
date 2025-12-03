import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import subprocess
import json
import requests
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Smart Thermostat AI Assistant",
    page_icon="🌡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #DBEAFE;
        border-left: 4px solid #3B82F6;
    }
    .bot-message {
        background-color: #F3F4F6;
        border-left: 4px solid #10B981;
    }
    .stButton button {
        background-color: #3B82F6;
        color: white;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ollama_running' not in st.session_state:
    st.session_state.ollama_running = False

def check_ollama_running():
    """Check if Ollama service is running"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        return response.status_code == 200
    except:
        return False

def start_ollama():
    """Start Ollama service if not running"""
    try:
        # Try to start Ollama (Windows)
        subprocess.Popen(['ollama', 'serve'], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        import time
        time.sleep(5)  # Wait for service to start
        
        # Check again
        if check_ollama_running():
            st.session_state.ollama_running = True
            return True
        return False
    except Exception as e:
        st.error(f"Failed to start Ollama: {e}")
        return False

def load_data():
    """Load thermostat data"""
    try:
        df = pd.read_csv('enhanced_thermostat_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        st.session_state.df = df
        st.session_state.data_loaded = True
        return True
    except Exception as e:
        st.error(f"Data load error: {e}")
        return False

def query_ollama_phi(question):
    """Query Ollama's Phi model with thermostat data context"""
    if not st.session_state.ollama_running:
        if not start_ollama():
            return "❌ Ollama service is not running. Please start Ollama first."
    
    try:
        # Get data context if available
        data_context = ""
        if st.session_state.data_loaded and st.session_state.df is not None:
            df = st.session_state.df
            
            # Extract relevant statistics based on the question
            if 'temperature' in question.lower() or 'temp' in question.lower():
                if 'indoor_temp_c' in df.columns:
                    avg_temp = df['indoor_temp_c'].mean()
                    min_temp = df['indoor_temp_c'].min()
                    max_temp = df['indoor_temp_c'].max()
                    current_temp = df['indoor_temp_c'].iloc[-1] if len(df) > 0 else "N/A"
                    
                    data_context = f"""
                    USER'S THERMOSTAT DATA:
                    - Average indoor temperature: {avg_temp:.1f}°C
                    - Minimum indoor temperature: {min_temp:.1f}°C
                    - Maximum indoor temperature: {max_temp:.1f}°C
                    - Current indoor temperature: {current_temp:.1f}°C
                    - Outdoor temperature range: {df['outdoor_temp_c'].min():.1f}°C to {df['outdoor_temp_c'].max():.1f}°C
                    """
            
            elif 'energy' in question.lower() or 'bill' in question.lower() or 'cost' in question.lower():
                if 'energy_consumption_kwh' in df.columns:
                    total_energy = df['energy_consumption_kwh'].sum()
                    avg_daily = df['energy_consumption_kwh'].mean()
                    total_cost = df['energy_cost_usd'].sum() if 'energy_cost_usd' in df.columns else "N/A"
                    
                    data_context = f"""
                    USER'S ENERGY DATA:
                    - Total energy consumption: {total_energy:.1f} kWh
                    - Average daily consumption: {avg_daily:.2f} kWh
                    - Total energy cost: ${total_cost:.2f} (if available)
                    - Peak consumption hour: {df['hour'].mode()[0] if 'hour' in df.columns else 'N/A'}
                    """
            
            elif 'humidity' in question.lower():
                if 'indoor_humidity' in df.columns:
                    avg_humidity = df['indoor_humidity'].mean()
                    data_context = f"Average indoor humidity: {avg_humidity:.1f}%"
            
            elif 'schedule' in question.lower() or 'time' in question.lower():
                # Add time-based patterns
                df['hour'] = df['timestamp'].dt.hour
                peak_hour = df.groupby('hour')['energy_consumption_kwh'].sum().idxmax() if 'energy_consumption_kwh' in df.columns else "N/A"
                data_context = f"Peak energy usage hour: {peak_hour}:00"
        
        # Prepare system prompt with data context
        system_prompt = f"""You are a smart thermostat AI assistant. You have access to the user's actual thermostat data.
        
        {data_context if data_context else "No specific data available. Provide general advice based on HVAC best practices."}
        
        IMPORTANT: Answer the user's question based on their actual data if available. 
        If data is available, reference the specific numbers.
        If no specific data is available, provide general advice.
        
        Keep answers concise, practical, and helpful.
        """
        
        # Create the request payload
        payload = {
            "model": "phi",
            "prompt": f"User Question: {question}\n\nSystem Context: {system_prompt}\n\nAssistant Answer:",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 300
            }
        }
        
        # Make the API call
        response = requests.post(
            'http://localhost:11434/api/generate',
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', 'No response received')
            
            # Clean up the response
            if "Assistant Answer:" in answer:
                answer = answer.split("Assistant Answer:")[-1].strip()
            
            return answer
        else:
            return f"Error: API returned status code {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to Ollama. Make sure Ollama is running.\nRun this command in terminal: `ollama serve`"
    except Exception as e:
        return f"Error: {str(e)}"

def get_sample_questions():
    """Return sample questions for quick access"""
    return [
        "How can I reduce my energy bill?",
        "What is the ideal temperature setting?",
        "Should I run my HVAC at night?",
        "How to improve HVAC efficiency?",
        "Best temperature for sleeping?",
        "Save energy during peak hours?"
    ]

def initialize_ollama():
    """Initialize Ollama connection"""
    try:
        with st.spinner("🔄 Connecting to Ollama..."):
            if check_ollama_running():
                st.session_state.ollama_running = True
                return True
            else:
                # Try to start Ollama
                if start_ollama():
                    st.session_state.ollama_running = True
                    return True
                return False
    except Exception as e:
        st.error(f"Ollama initialization error: {e}")
        return False

def main():
    # Header
    st.markdown('<h1 class="main-header">🌡️ Smart Thermostat AI Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Setup")
        
        # Ollama Status Check
        if check_ollama_running():
            st.success("✅ Ollama is running")
            st.session_state.ollama_running = True
        else:
            st.warning("⚠️ Ollama not running")
            if st.button("🚀 Start Ollama"):
                if start_ollama():
                    st.success("✅ Ollama started!")
                    st.session_state.ollama_running = True
                    st.rerun()
        
        st.markdown("---")
        
        if st.button("📊 Load Data"):
            if load_data():
                st.success("✅ Data loaded!")
        
        st.markdown("---")
        
        if st.session_state.data_loaded and st.session_state.df is not None:
            df = st.session_state.df
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Energy", f"{df['energy_consumption_kwh'].sum():.1f} kWh")
                st.metric("Avg Temp", f"{df['indoor_temp_c'].mean():.1f}°C")
            with col2:
                st.metric("Total Cost", f"${df['energy_cost_usd'].sum():.2f}")
                if 'cop_efficiency' in df.columns:
                    st.metric("Avg COP", f"{df[df['cop_efficiency'] > 0]['cop_efficiency'].mean():.2f}")
        
        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        st.info("• Keep thermostat at 24°C in summer")
        st.info("• Use programmable schedules")
        st.info("• Clean filters monthly")
        st.info("• Seal windows and doors")
        
        # Ollama Model Info
        st.markdown("---")
        st.markdown("### 🤖 AI Model")
        st.success("Using: Phi (via Ollama)")
        st.caption("Local, Fast, No download needed")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Chat with AI Assistant")
        
        # Check if Ollama is ready
        if not st.session_state.ollama_running:
            st.warning("""
            **Ollama is not running!**
            
            Please follow these steps:
            1. Open a new terminal/command prompt
            2. Run: `ollama serve`
            3. Keep that terminal open
            4. Click "Start Ollama" button in sidebar
            """)
        
        # Display chat history
        if st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong> {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 Assistant:</strong> {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Start by asking a question about your thermostat!")
        
        # Input area
        user_input = st.text_area("Type your question:", height=100)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Send Message", use_container_width=True) and user_input:
                if not st.session_state.ollama_running:
                    st.error("Please start Ollama first!")
                else:
                    # Add user message
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_input
                    })
                    
                    # Get AI response
                    with st.spinner("Thinking..."):
                        response = query_ollama_phi(user_input)
                    
                    # Add bot response
                    st.session_state.chat_history.append({
                        'role': 'bot',
                        'content': response
                    })
                    
                    st.rerun()
        
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    
    with col2:
        st.markdown("###  Quick Stats")
        
        if st.session_state.data_loaded and st.session_state.df is not None:
            df = st.session_state.df
            
            # Simple bar chart of energy by hour
            if 'timestamp' in df.columns:
                df['hour'] = df['timestamp'].dt.hour
                if 'energy_consumption_kwh' in df.columns:
                    hourly_energy = df.groupby('hour')['energy_consumption_kwh'].mean().reset_index()
                    
                    fig = px.bar(
                        hourly_energy, 
                        x='hour', 
                        y='energy_consumption_kwh',
                        title='Average Energy by Hour',
                        color_discrete_sequence=['#3B82F6']
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Temperature stats
            temp_cols = [col for col in ['indoor_temp_c', 'outdoor_temp_c'] if col in df.columns]
            if temp_cols:
                temp_stats = df[temp_cols].describe()
                st.dataframe(temp_stats.head(3), use_container_width=True)
            
        else:
            st.info("Load data to see statistics")
        
        st.markdown("---")
        st.markdown("### Sample Questions")
        
        for question in get_sample_questions():
            if st.button(f" {question}", key=question):
                if not st.session_state.ollama_running:
                    st.error("Start Ollama first!")
                else:
                    # Add to chat
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': question
                    })
                    
                    # Get response
                    with st.spinner("Thinking..."):
                        response = query_ollama_phi(question)
                    
                    st.session_state.chat_history.append({
                        'role': 'bot',
                        'content': response
                    })
                    
                    st.rerun()
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"Ollama: {'Running' if st.session_state.ollama_running else ' Stopped'}")
    with col2:
        st.caption(f"Data: {'Loaded' if st.session_state.data_loaded else ' Not Loaded'}")
    with col3:
        st.caption("Powered by Phi via Ollama")

if __name__ == "__main__":
    main()