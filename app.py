import os
import streamlit as st
import google.generativeai as genai

# Configure Streamlit Page Layout
st.set_page_config(page_title="Free Token Optimizer Proxy", layout="wide", page_icon="⚡")

st.title("⚡ Free AI Context & Token Optimizer")
st.markdown("Compress massive log files, codebases, or PDFs using **Gemini 2.5 Flash** for free before chatting with Claude.")

# Sidebar Configuration for Key Management
st.sidebar.header("Configuration")
gemini_key = st.sidebar.text_input("Google Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))

if gemini_key:
    # Initialize the Google API client
    genai.configure(api_key=gemini_key)
else:
    st.warning("⚠️ Please input your Google Gemini API Key in the sidebar or save it to your system profile to begin.")
    st.stop()

# Layout UI Division Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Raw Input Payload")
    user_prompt = st.text_area("Paste your heavy code blocks, document text, or massive prompt here:", height=400)
    process_btn = st.button("Compress & Optimize Context", type="primary")

with col2:
    st.subheader("📉 Optimization Metrics")
    metrics_placeholder = st.empty()
    status_placeholder = st.empty()

if process_btn and user_prompt.strip():
    with status_placeholder.container():
        st.info("🔄 Connecting to Google AI Studio Sandbox...")
        
        # Calculate approximate original token count (approx 4 chars per token)
        raw_char_count = len(user_prompt)
        est_raw_tokens = int(raw_char_count / 4)
        
        # Instruction configuration for the compressor model
        system_instruction = (
            "Act as an aggressive Context & Token Optimizer. Your task is to condense the provided text "
            "so it can fit cleanly into another AI assistant without hitting limits. Strip out all conversational filler, "
            "redundant system logs, formatting metadata, and legal jargon. Retain 100% of the raw data values, "
            "critical architectural rules, constraints, and operational logic. Output the final result in an ultra-dense, "
            "token-efficient Markdown summary. Do not include introductory text like 'Here is your summary'."
        )
        
        try:
            # Call the ultra-fast free tier Gemini model
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=system_instruction
            )
            
            response = model.generate_content(user_prompt)
            optimized_text = response.text
            
            # Calculate new token sizing metrics
            est_opt_tokens = int(len(optimized_text) / 4)
            tokens_saved = est_raw_tokens - est_opt_tokens
            pct_saved = round((tokens_saved / max(1, est_raw_tokens)) * 100, 2)
            
            # Render visual comparison dashboard metrics
            with metrics_placeholder.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Est. Original Tokens", f"{est_raw_tokens:,}")
                m2.metric("Est. Compressed Tokens", f"{est_opt_tokens:,}")
                m3.metric("Footprint Reduction", f"{tokens_saved:,}", f"-{pct_saved}%")
                
            st.success("🎉 Compression complete! Copy the text block below into your standard Claude Chat.")
            
            # Display output container with an integrated copy button
            st.markdown("---")
            st.subheader("🤖 Claude-Ready Optimized Context")
            st.code(optimized_text, language="markdown")
            
        except Exception as e:
            st.error(f"An error occurred while contacting the Gemini API: {str(e)}")
