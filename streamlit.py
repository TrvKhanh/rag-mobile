import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import time
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"

# Custom CSS for enhanced UI styling and layout
st.markdown("""
<style>

/* Restrict scrolling to chat container only */
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.chat-message {
    max-width: 60%;
    padding: 3px 4px;
    border-radius: 12px;
    margin-bottom: 12px;
    display: inline-block;
    word-break: break-word;
    font-size: 1rem;
    position: relative;
    clear: both;
}

.user-message {
    background: #ffb6c1;
    color: #fff;
    margin-left: auto;
    margin-right: 0;
    text-align: right;
    float: right;
    padding: 4px
}

.bot-message {
    background: #fff;
    color: #222;
    margin-right: auto;
    margin-left: 0;
    text-align: left;
    float: left;
    border: 1px solid #e0e0e0;
}

.message-time {
    font-size: 0.8rem;
    color: #888;
    margin-top: 4px;
    text-align: right;
}

.stButton > button {
    border-radius: 10px;
    background: #808080	;
    color: white;
    border: none;
    padding: 0.3rem 2rem;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.sidebar-content {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}

.status-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 8px;
}


/* Prevent main content area from scrolling */
.main-content {
    height: calc(100vh - 150px);
    overflow: hidden !important;
    position: relative;
}

/* Fixed-height chat container with vertical scrolling */
.chat-container {
    height: 650px;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    background-color: #fafafa;
    position: relative;
}

/* Sticky input area positioned at bottom */
.input-container {
    position: sticky;
    bottom: 0;
    background-color: white;
    padding: 10px 0;
    border-top: 1px solid #e0e0e0;
    z-index: 100;
}
</style>
""", unsafe_allow_html=True)

# Auto-scroll JavaScript injection for chat container
# Ensures chat messages are always visible by scrolling to bottom
def add_auto_scroll_script():
    components.html("""
    <script>
    function scrollChatToBottom() {
        // Scroll only the chat container, not the entire page
        const containers = parent.document.querySelectorAll('div[style*="height: 650px"]');
        containers.forEach(container => {
            // Verify this is the chat container by checking for message indicators
            if (container.innerHTML.includes('chat-message') || container.innerHTML.includes('💬')) {
                container.scrollTop = container.scrollHeight;
                console.log('Chat container scrolled to bottom');
            }
        });
        
        // Fallback: locate scrollable containers with overflow-y and fixed height
        const scrollableContainers = parent.document.querySelectorAll('div');
        scrollableContainers.forEach(div => {
            const computedStyle = parent.window.getComputedStyle(div);
            if ((computedStyle.overflowY === 'auto' || computedStyle.overflowY === 'scroll') && 
                computedStyle.height === '650px') {
                div.scrollTop = div.scrollHeight;
            }
        });
    }
    
    // Execute scroll with progressive delays for DOM stability
    setTimeout(scrollChatToBottom, 100);
    setTimeout(scrollChatToBottom, 300);
    setTimeout(scrollChatToBottom, 600);
    
    // Monitor DOM mutations specifically for chat message changes
    const observer = new MutationObserver(function(mutations) {
        let chatChanged = false;
        mutations.forEach(function(mutation) {
            // Trigger scroll only when new messages are added
            if (mutation.addedNodes.length > 0) {
                for (let node of mutation.addedNodes) {
                    if (node.nodeType === 1 && 
                        (node.innerHTML.includes('chat-message') || 
                         node.classList.contains('chat-message'))) {
                        chatChanged = true;
                        break;
                    }
                }
            }
        });
        if (chatChanged) {
            setTimeout(scrollChatToBottom, 100);
        }
    });
    
    setTimeout(() => {
        observer.observe(parent.document.body, { 
            childList: true, 
            subtree: true 
        });
    }, 500);
    </script>
    """, height=0)

# Fixed header component at top of page
st.markdown("""
<div id="custom-header" style="
    position: fixed;
    top: 0;
    left: 2;
    width: 79vw;
    height: 64px;
    background: #fff;
    color: #111;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.2rem;
    font-weight: bold;
    z-index: 9999;
    letter-spacing: 2px;
">
    🤖 AI Chatbot
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "api_status" not in st.session_state:
    st.session_state.api_status = "unknown"
if "scroll_key" not in st.session_state:
    st.session_state.scroll_key = 0




def send_message_user(message: str, thread_id: Optional[str] = None) -> Optional[dict]:
    """
    Send user message to chat API endpoint.
    
    Args:
        message: User input message to send
        thread_id: Optional conversation thread ID for maintaining context
        
    Returns:
        Response dictionary with bot reply and conversation ID, or None on error
    """
    try:
        payload = {"message": message}
        if thread_id:
            payload["thread_id"] = thread_id
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=60)  
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as json_error:
                st.error(f"JSON parsing error from API: {json_error}")
                st.error(f"Response text: {response.text[:200]}")
                return None
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timeout - API took too long to respond. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Unable to connect to API. Please verify the server is running.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Connection error: {str(e)}")
        return None



def main():

    with st.sidebar:
        st.markdown("### ⚙️ Cài đặt")

        top_k = st.slider("Số lượng kết quả tìm kiếm", min_value=1, max_value=10, value=st.session_state.get('top_k', 5), key="top_k")
        print(top_k)
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

       
        if st.button("📥 Xuất lịch sử chat", use_container_width=True):
                chat_data = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "conversation_id": st.session_state.conversation_id,
                    "messages": st.session_state.messages
                }
                st.download_button(
                    label="💾 Tải xuống JSON",
                    data=json.dumps(chat_data, ensure_ascii=False, indent=2),
                    file_name=f"chat_history_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )    

        st.markdown("---")
        st.markdown("### 📊 Thống kê")

        
        user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
        bot_messages = len([msg for msg in st.session_state.messages if msg["role"] == "assistant"])
        st.markdown(f"**Tin nhắn người dùng:** {user_messages}")
        st.markdown(f"**Phản hồi bot:** {bot_messages}")    

        

    # Main chat interface layout
    col1, col2, col3 = st.columns([1, 30, 1])
    with col2:
        
            chat_container = st.container(height=650, key=f"chat_container_{st.session_state.scroll_key}")
            with chat_container:
                if not st.session_state.messages:
                    st.markdown(
                        "<div style='color:#888; text-align:center; margin-top:40px;'>💬 Hello! How can I help you today?</div>",
                        unsafe_allow_html=True
                    )
                else:
                    # Render chat message history
                    for i, message in enumerate(st.session_state.messages):
                        if message["role"] == "user":
                            st.markdown(
                                f'''<div class="chat-message user-message" id="message-{i}">
                                {message['content']}
                                <div class="message-time">{message['timestamp']}</div>
                                </div>''',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'''<div class="chat-message bot-message" id="message-{i}">
                                {message['content']}
                                <div class="message-time">{message['timestamp']}</div>
                                </div>''',
                                unsafe_allow_html=True
                            )
                    st.markdown('<div style="clear:both;"></div>', unsafe_allow_html=True)

            # User input form with fixed positioning
            with st.form("user_input_form", clear_on_submit=True):
                col_input, col_btn = st.columns([16, 1])
                with col_input:
                    user_input = st.text_input("Enter message", placeholder="Type your message...", key="user_input_text", label_visibility="collapsed")
                    print(user_input)
                with col_btn:
                    send_btn = st.form_submit_button("Send")

                if send_btn and user_input:
                    # Append user message to session state
                    st.session_state.messages.append({
                        "role": "user",
                        "content": user_input,
                        "timestamp": time.strftime("%H:%M")
                    })
                    
                    # Send message to API and handle response
                    with st.spinner("Sending message..."):
                        response_data = send_message_user(user_input, st.session_state.conversation_id)
                        bot_message = None
                        
                        if response_data:
                            # Extract bot response and conversation ID
                            bot_message = response_data.get("response") or response_data.get("content")
                            conversation_id = response_data.get("conversation_id") or response_data.get("thread_id")
                            if conversation_id:
                                st.session_state.conversation_id = conversation_id

                        if bot_message:
                            # Append bot response to session state
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": bot_message,
                                "timestamp": time.strftime("%H:%M")
                            })
                           
                            st.session_state.scroll_key += 1
                        else:
                            st.error("Bot did not respond. Please try again.")
                    
                   
                    st.rerun()
                   

    add_auto_scroll_script()

if __name__ == "__main__":

    main()