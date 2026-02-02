import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- LANGCHAIN IMPORTS (Modern version) ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
st.set_page_config(page_title="InsightForge BI Assistant", layout="wide")

# --- PART 1: DATA PREPARATION & KNOWLEDGE BASE ---
@st.cache_data
def load_and_process_data():
    """Load and process sales data, create knowledge base"""
    try:
        df = pd.read_csv('sales_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Pre-calculate statistics for RAG retrieval
        stats = {
            "total_revenue": float(df['Sales'].sum()),
            "average_transaction": float(df['Sales'].mean()),
            "median_sales": float(df['Sales'].median()),
            "std_dev": float(df['Sales'].std()),
            "best_selling_product": df.groupby('Product')['Sales'].sum().idxmax(),
            "sales_by_product": df.groupby('Product')['Sales'].sum().to_dict(),
            "sales_by_region": df.groupby('Region')['Sales'].sum().to_dict(),
            "monthly_trend": {str(k): int(v) for k, v in df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum().to_dict().items()},
            "avg_customer_age": float(df['Customer_Age'].mean()),
            "avg_satisfaction": float(df['Customer_Satisfaction'].mean())
        }
        return df, stats
    except FileNotFoundError:
        st.error("❌ sales_data.csv not found!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# --- PART 2: LLM CONFIGURATION ---
def get_llm():
    """Initialize LLM with OpenAI configuration"""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if not api_key:
        st.error("⚠️ OpenAI API key not found! Please set it in terminal with: set OPENAI_API_KEY=your_key")
        st.stop()
    
    return ChatOpenAI(
        api_key=api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.3
    )

# --- PART 3: CUSTOM RETRIEVER ---
def custom_retriever(query, stats):
    """Custom retriever that extracts relevant statistics based on query"""
    query_lower = query.lower()
    context_parts = []
    
    if any(word in query_lower for word in ['total', 'revenue', 'overall', 'sum']):
        context_parts.append(f"Total Revenue: ${stats['total_revenue']:,.2f}")
    
    if any(word in query_lower for word in ['average', 'mean', 'avg']):
        context_parts.append(f"Average Transaction: ${stats['average_transaction']:,.2f}")
        context_parts.append(f"Average Customer Age: {stats['avg_customer_age']:.1f} years")
        context_parts.append(f"Average Satisfaction: {stats['avg_satisfaction']:.2f}/5.0")
    
    if any(word in query_lower for word in ['product', 'widget', 'best', 'top']):
        context_parts.append(f"Best Selling Product: {stats['best_selling_product']}")
        context_parts.append(f"Sales by Product: {stats['sales_by_product']}")
    
    if any(word in query_lower for word in ['region', 'location', 'area', 'where']):
        context_parts.append(f"Sales by Region: {stats['sales_by_region']}")
    
    if any(word in query_lower for word in ['trend', 'month', 'time', 'period']):
        recent_months = list(stats['monthly_trend'].items())[-6:]
        context_parts.append(f"Recent 6 Months Trend: {dict(recent_months)}")
    
    if not context_parts:
        context_parts.append(f"Overview: Total Revenue ${stats['total_revenue']:,.2f}, Top Product: {stats['best_selling_product']}")
    
    return "\n".join(context_parts)

# --- PART 4: RAG CHAIN (Modern LCEL Pattern) ---
def create_rag_chain(llm):
    """Create RAG chain using modern LangChain Expression Language"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are InsightForge, an expert Business Intelligence Analyst.
Use the provided statistical context to answer questions accurately and professionally.
If the answer is not in the context, politely say you don't have that information."""),
        ("human", """STATISTICAL CONTEXT:
{context}

QUESTION: {question}

Provide a clear, professional business analysis:""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain

# --- MAIN APPLICATION ---
st.title("📊 InsightForge: AI-Powered Business Intelligence")
st.markdown("*Powered by LangChain + RAG + OpenRouter*")

# Load data
df, kb = load_and_process_data()

# Initialize LLM
try:
    llm = get_llm()
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = create_rag_chain(llm)
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.stop()

# Initialize message history
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = [
        {"role": "assistant", "content": "Hello! I'm InsightForge, your AI Business Intelligence Assistant. Ask me about sales trends, products, regions, or any business metrics!"}
    ]

# Sidebar
page = st.sidebar.radio("📑 Navigate", ["💬 Chat Assistant", "📈 Dashboard", "✅ Model Evaluation"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
st.sidebar.metric("Total Revenue", f"${kb['total_revenue']:,.0f}")
st.sidebar.metric("Avg Transaction", f"${kb['average_transaction']:.2f}")
st.sidebar.metric("Best Product", kb['best_selling_product'])

# --- PAGE 1: CHAT ---
if page == "💬 Chat Assistant":
    st.header("💬 Chat with Your Data")
    st.markdown("Ask questions about sales performance, products, regions, and trends.")
    
    for msg in st.session_state.ui_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if prompt := st.chat_input("Ask about your business data..."):
        st.session_state.ui_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    context_str = custom_retriever(prompt, kb)
                    response = st.session_state.rag_chain.invoke({
                        "context": context_str,
                        "question": prompt
                    })
                    st.write(response)
                    st.session_state.ui_messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.ui_messages.append({"role": "assistant", "content": error_msg})

# --- PAGE 2: DASHBOARD ---
elif page == "📈 Dashboard":
    st.header("📈 Interactive Business Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${kb['total_revenue']:,.0f}")
    col2.metric("Avg Transaction", f"${kb['average_transaction']:.2f}")
    col3.metric("Avg Satisfaction", f"{kb['avg_satisfaction']:.2f}/5.0")
    col4.metric("Transactions", f"{len(df):,}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales by Product")
        fig_prod = px.bar(
            df.groupby('Product')['Sales'].sum().reset_index(),
            x='Product', y='Sales', color='Product',
            title="Product Performance"
        )
        st.plotly_chart(fig_prod, use_container_width=True)
    
    with col2:
        st.subheader("Sales by Region")
        fig_reg = px.pie(
            df.groupby('Region')['Sales'].sum().reset_index(),
            values='Sales', names='Region',
            title="Regional Distribution"
        )
        st.plotly_chart(fig_reg, use_container_width=True)
    
    st.subheader("Monthly Sales Trend")
    monthly_data = df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum().reset_index()
    monthly_data['Date'] = monthly_data['Date'].astype(str)
    fig_line = px.line(monthly_data, x='Date', y='Sales', markers=True)
    st.plotly_chart(fig_line, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Age Distribution")
        fig_age = px.histogram(df, x='Customer_Age', nbins=20)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        st.subheader("Customer Satisfaction")
        fig_sat = px.histogram(df, x='Customer_Satisfaction', nbins=20)
        st.plotly_chart(fig_sat, use_container_width=True)

# --- PAGE 3: EVALUATION ---
elif page == "✅ Model Evaluation":
    st.header("✅ Model Evaluation")
    st.markdown("Testing RAG system accuracy against ground truth.")
    
    if st.button("🚀 Run Evaluation", type="primary"):
        with st.spinner("Evaluating..."):
            try:
                test_cases = [
                    {"query": "What is the total revenue?", "expected": f"${kb['total_revenue']:,.2f}"},
                    {"query": "Which product sells the most?", "expected": kb['best_selling_product']},
                    {"query": "What is the average transaction?", "expected": f"${kb['average_transaction']:.2f}"}
                ]
                
                results = []
                for test in test_cases:
                    context = custom_retriever(test['query'], kb)
                    prediction = st.session_state.rag_chain.invoke({
                        "context": context,
                        "question": test['query']
                    })
                    
                    # Simple accuracy check
                    contains_answer = test['expected'].lower() in prediction.lower()
                    
                    results.append({
                        "Question": test['query'],
                        "Expected": test['expected'],
                        "Model Response": prediction[:150] + "...",
                        "Status": "✅ PASS" if contains_answer else "⚠️ CHECK"
                    })
                
                st.success("✅ Evaluation Complete!")
                st.dataframe(pd.DataFrame(results), use_container_width=True)
                
                passed = sum(1 for r in results if "✅" in r['Status'])
                st.metric("Accuracy", f"{(passed/len(results)*100):.0f}%")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.markdown("*InsightForge - Capstone Project - January 2026*")
