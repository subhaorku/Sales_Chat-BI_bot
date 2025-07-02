import streamlit as st

import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()
from chatbot_model_context3 import (
    SalesDataChatbot,
)  # Ensure this points to the updated file


from chatbot_model_context3 import SalesDataChatbot  
st.set_page_config(page_title="Sales Data Chatbot", layout="wide")

try:
    st.sidebar.image("metayb-logo.png", use_container_width=True)
except Exception:
    st.sidebar.warning("Logo image 'metayb-logo.png' not found.")


@st.cache_resource(show_spinner=True)
def load_chatbot():
    data_file = "sales_rt.parquet"
    try:
        return SalesDataChatbot(data_file)
    except Exception as e:
        st.error(
            f"Failed to load chatbot. Ensure '{data_file}' exists and is valid. Error: {e}"
        )
        return None


st.title("💬 Sales Q&A Assistant ")
st.write(
    "Ask questions about your sales data. You can ask for sales figures, forecasts, or plots (e.g., 'plot sales for INDOMIE PULL in NORTH 1 last month' or 'show quantity trend for DANO')."
)

chatbot = load_chatbot()

if chatbot:
    if "history_mc" not in st.session_state:
        st.session_state.history_mc = []

    for q_hist, a_hist_val in st.session_state.history_mc:
        with st.chat_message("user"):
            st.markdown(q_hist)
        with st.chat_message("assistant"):
            if (
                isinstance(a_hist_val, dict)
                and a_hist_val.get("type") == "plot_figure_data"
            ):
                # If you stored figure data, you might need to re-create the figure or handle it.
                # For simplicity, we'll just show a placeholder for past plots in history.
                st.markdown("A plot was previously displayed for this query.")
            elif isinstance(a_hist_val, str):
                st.markdown(a_hist_val)
            else:
                st.markdown("A plot was previously displayed.")

    if prompt := st.chat_input("Type your question..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            history_for_query = [
                (
                    q,
                    str(a)
                    if not (isinstance(a, dict) and a.get("type") == "plot")
                    else "A plot was generated.",
                )
                for q, a in st.session_state.history_mc
            ]
            answer_data = chatbot.answer_query(prompt, history=history_for_query)

        with st.chat_message("assistant"):
            history_entry_answer = answer_data  # Default to storing the raw response

            if isinstance(answer_data, dict) and answer_data.get("type") == "plot":
                fig = answer_data.get("figure")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    # For history, store a placeholder or a serializable version of the plot if needed
                    # For simplicity, we'll store a string placeholder for the plot in history.
                    history_entry_answer = "Displayed a plot based on your query."
                else:
                    st.markdown("Sorry, I couldn't generate the plot.")
                    history_entry_answer = "Sorry, I couldn't generate the plot."
            elif isinstance(answer_data, str):
                st.markdown(answer_data)
            else:  # Should not happen with current chatbot logic but good to have a fallback
                st.markdown("Received an unexpected response format.")
                history_entry_answer = "Received an unexpected response format."

        st.session_state.history_mc.append((prompt, history_entry_answer))

else:
    st.error(
        "Chatbot could not be initialized. Please check the data file and configurations."
    )
