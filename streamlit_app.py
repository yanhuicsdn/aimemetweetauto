import requests
from bs4 import BeautifulSoup
import streamlit as st
import re
import json
import os
import tweepy
from typing import List, Dict, Optional
from datetime import datetime

# 默认配置
DEFAULT_PROMPT = """
Creative Meme Coin Content Creator
Turn your meme coin into the next viral sensation! Let's create engaging content that captures the community's attention.

First, analyze all the trending topics and select the most suitable one for the meme coin by considering:
1. Relevance to crypto/blockchain/technology
2. Potential for creative connection with the token's theme
3. Current popularity and engagement potential

Selected trend: {selected_trend}

Content Generation Parameters
{context_info}
Token Details

Name: {token_name}
Description: {token_description}

Content Style Guide

Language: English/Chinese (as specified)
Tone: Casual, clever, community-focused
Format: Optimized for Twitter
Elements: Text + Emojis + Hashtags

Content Strategy

Engaging Hook
- Attention-grabbing opener
- Relate to current trends
- Use compelling language

Core Message
- Highlight unique features
- Connect with community
- Include meme references

Viral Elements
- Strategic emoji placement
- Trending hashtag integration
- Call-to-action

Community Focus
- Foster engagement
- Encourage sharing
- Build connections

Output Format
Tweet format should follow:
[Hook] + [Core Message] + [Community Element] + [Call to Action] + [Trending Tags]

Content Requirements
- Keep it fun and shareable
- Blend humor with value
- Stay relevant to trends
- Encourage interaction
- Maintain brand voice

Ready to create your next viral tweet! 🚀

Note: Each piece of content will be uniquely crafted based on the provided parameters while maintaining optimal engagement potential.
Using the above guidelines and context, create a creative and engaging tweet for the meme coin "{token_name}" based on its description: "{token_description}". Ensure that the tweet content is strongly related to the selected trending hashtag to maximize engagement.
"""

DEFAULT_TOKEN_NAME = "LEGENDARY HUMANITY"
DEFAULT_TOKEN_DESCRIPTION = "Merging fashion, art, and #AI into #Web3 assets. Empowering designers and artists with community-driven #meme coins. $VIVI is the governance token."
DEFAULT_IMAGE_DESCRIPTION = "A vibrant and humorous illustration representing the essence of the tweet, with logo 'LEGENDARY HUMANITY' and 'VIVI'."

# Save tokens to local storage
def save_tokens_to_local_storage(tokens: Dict[str, str]) -> None:
    st.experimental_set_query_params(**tokens)

# Load tokens from local storage
def load_tokens_from_local_storage() -> Optional[Dict[str, str]]:
    params = st.experimental_get_query_params()
    if params:
        return {key: params[key][0] for key in params}
    return None

# Authentication function
def authenticate_twitter(credentials: Dict[str, str]) -> Optional[tweepy.Client]:
    try:
        client = tweepy.Client(
            consumer_key=credentials['consumer_key'],
            consumer_secret=credentials['consumer_secret'],
            access_token=credentials['access_token'],
            access_token_secret=credentials['access_token_secret'],
            bearer_token=credentials['bearer_token']
        )
        return client
    except Exception as e:
        st.error(f"Authentication Error: {e}")
        return None

# Load existing tokens
tokens = load_tokens_from_local_storage()

if tokens:
    client = authenticate_twitter(tokens)
    if client:
        st.success("Authenticated successfully using stored tokens!")
    else:
        st.error("Failed to authenticate with stored tokens.")
else:
    # If no stored tokens, prompt user for credentials
    st.sidebar.header("Twitter API Settings")
    consumer_key = st.sidebar.text_input("Consumer Key (API Key)", type="password")
    consumer_secret = st.sidebar.text_input("Consumer Secret (API Secret)", type="password")
    access_token = st.sidebar.text_input("Access Token", type="password")
    access_token_secret = st.sidebar.text_input("Access Token Secret", type="password")
    bearer_token = st.sidebar.text_input("Bearer Token", type="password")

    if st.sidebar.button("Authenticate"):
        credentials = {
            'consumer_key': consumer_key,
            'consumer_secret': consumer_secret,
            'access_token': access_token,
            'access_token_secret': access_token_secret,
            'bearer_token': bearer_token
        }
        client = authenticate_twitter(credentials)
        if client:
            save_tokens_to_local_storage(credentials)
            st.success("Authenticated successfully and tokens saved!")
        else:
            st.error("Failed to authenticate with Twitter.")

def post_tweet(client: tweepy.Client, content: str) -> None:
    try:
        if not content.strip():
            st.error("Please write something before posting!")
            st.stop()

        if len(content) > 280:
            st.error("Tweet exceeds 280 characters limit!")
            st.stop()

        response = client.create_tweet(text=content)
        tweet_id = response.data['id']

        st.markdown("""
        <div style='padding: 1rem; border-radius: 0.5rem; background-color: #f0f2f6; margin: 1rem 0;'>
            <h3 style='color: #0066cc;'>✨ Tweet Posted Successfully!</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Tweet Details")
        st.markdown("---")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Tweet ID:**")
            st.code(tweet_id, language=None)
        with col2:
            st.markdown("**View on X:**")
            st.markdown(f"🔗 [Click to view tweet](https://twitter.com/user/status/{tweet_id})")

        st.markdown("**Content:**")
        st.code(content, language=None)

        st.markdown("**Posted at:**")
        st.code(str(datetime.now()), language=None)

        st.stop()

    except tweepy.TweepError as e:
        st.error(f"Error posting tweet: {str(e)}")
        st.markdown("""
        <div style='padding: 1rem; border-radius: 0.5rem; background-color: #ffebee; margin: 1rem 0;'>
            <h3 style='color: #d32f2f;'>❌ Tweet Posting Failed</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Error Details")
        st.markdown("---")
        st.markdown("**Error Message:**")
        st.code(str(e), language=None)
        st.markdown("**Time:**")
        st.code(str(datetime.now()), language=None)
        st.stop()

def handle_tweet_button():
    st.session_state.show_tweet_button = True

def rag_search(keywords: List[str]) -> str:
    """
    Perform RAG search for given keywords and return relevant content.
    """
    url = "https://google.serper.dev/search"
    query = " ".join(keywords)

    headers = {
        'X-API-KEY': '0cadc0b6ceff6e6f6eebecf2e4c924de082e3616',
        'Content-Type': 'application/json'
    }

    payload = json.dumps({"q": query})

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        results = response.json().get('organic', [])

        content = []
        for result in results[:5]:
            snippet = result.get('snippet', '')
            if snippet:
                content.append(snippet)

        return " ".join(content)
    except Exception as e:
        st.error(f"RAG Search Error: {str(e)}")
        return ""

def get_latest_global_trends():
    url = "https://trends24.in/united-states/"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')

        trend_cards = soup.find_all('ol', class_='trend-card__list')
        trends_list = []
        count = 0

        for card in trend_cards:
            trends = card.find_all('li')
            for trend in trends:
                if count >= 30:
                    break
                trend_text = trend.get_text(strip=True)
                trend_name = re.sub(r'\s*\d+[KM]$', '', trend_text)
                trends_list.append(trend_name)
                count += 1
            if count >= 30:
                break

        return trends_list
    except Exception as e:
        st.error(f"Error fetching trends: {str(e)}")
        return []

def select_best_trend(trends: List[str], token_name: str, token_description: str) -> tuple[str, str]:
    """
    让AI从趋势列表中选择最适合的一个，并解释选择原因
    返回: (selected_trend, explanation)
    """
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-ydcvskcyyzictsylxplbpqmqlpillcpkqznxclfjyohkefwt",
        "Content-Type": "application/json"
    }

    selection_prompt = f"""
    Given these trending topics:
    {', '.join(trends)}

    And this meme token information:
    Token Name: {token_name}
    Token Description: {token_description}

    1. Select the single most suitable trending topic for creating a viral meme tweet.
    2. Explain why this trend is the best choice in 2-3 sentences.

    Format your response as JSON with two fields:
    {{
        "selected_trend": "the selected trend",
        "explanation": "your explanation"
    }}
    """

    payload = {
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "messages": [{"role": "user", "content": selection_prompt}],
        "stream": False,
        "max_tokens": 200,
        "temperature": 0.9,
        "response_format": {"type": "text"}
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        result = json.loads(data['choices'][0]['message']['content'])
        return result["selected_trend"], result["explanation"]
    except Exception as e:
        st.error(f"Trend Selection Error: {str(e)}")
        return trends[0] if trends else "", "Error occurred during selection explanation"

def generate_meme_tweet(token_name, token_description, trends, prompt):
    # 选择最佳趋势并获取解释
    selected_trend, trend_explanation = select_best_trend(trends, token_name, token_description)

    # 存储到 session state 中以供显示
    st.session_state['selected_trend'] = selected_trend
    st.session_state['trend_explanation'] = trend_explanation

    # 获取趋势相关上下文
    trend_context = rag_search([selected_trend])

    # 组合趋势标签
    selected_hashtag = f"#{selected_trend}"
    context_info = f"Current Trend Context:\n{trend_context}"

    formatted_prompt = prompt.format(
        token_name=token_name,
        token_description=token_description,
        selected_trend=selected_trend,
        context_info=context_info
    )

    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-ydcvskcyyzictsylxplbpqmqlpillcpkqznxclfjyohkefwt",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "messages": [{"role": "user", "content": formatted_prompt}],
        "stream": False,
        "max_tokens": 512,
        "stop": ["<string>"],
        "temperature": 0.9,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"}
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        tweet_content = data['choices'][0]['message']['content']
        return tweet_content.strip()
    except Exception as e:
        st.error(f"Tweet Generation Error: {str(e)}")
        return None

def generate_image_from_text(image_prompt):
    url = 'https://api.siliconflow.cn/v1/image/generations'
    headers = {
        "Authorization": "Bearer sk-ydcvskcyyzictsylxplbpqmqlpillcpkqznxclfjyohkefwt",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt": image_prompt,
        "image_size": "1024x1024",
        "seed": 4999999999
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['images'][0]['url']
    except Exception as e:
        st.error(f"Image Generation Error: {str(e)}")
        return None

def save_config(config_name, config_data):
    config_path = f"configs/{config_name}.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)
    st.success(f"Configuration saved as {config_name}.json")

def load_config(config_name):
    config_path = f"configs/{config_name}.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        st.error(f"Configuration file {config_name}.json not found.")
        return None

def get_saved_configs():
    config_dir = "configs"
    if not os.path.exists(config_dir):
        return []
    return [f for f in os.listdir(config_dir) if f.endswith('.json')]

# Streamlit UI
st.title("AI Crypto Meme Tweet Generator")
st.write("This app automatically uses AI to generate meme tweets based on the most suitable current trend")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")

    # Meme Token Configuration
    with st.expander("Meme Token Settings", expanded=True):
        token_name = st.text_input("Token Name", DEFAULT_TOKEN_NAME)
        token_description = st.text_area("Token Description", DEFAULT_TOKEN_DESCRIPTION)
        image_description = st.text_area(
            "Image Description",
            DEFAULT_IMAGE_DESCRIPTION,
            help="Describe the image to be generated that will accompany the tweet."
        )

    # Advanced Configuration
    with st.expander("Advanced Settings"):
        prompt = st.text_area("Prompt Template", DEFAULT_PROMPT)

    # Configuration Management
    with st.expander("Configuration Management"):
        config_name = st.text_input("Configuration Name")
        if st.button("Save Configuration"):
            config_data = {
                "token_name": token_name,
                "token_description": token_description,
                "image_description": image_description,
                "prompt": prompt
            }
            save_config(config_name, config_data)

        saved_configs = get_saved_configs()
        selected_config = st.selectbox("Load Configuration", saved_configs)
        if st.button("Load Configuration"):
            if selected_config:
                config_data = load_config(selected_config.replace('.json', ''))
                if config_data:
                    token_name = config_data.get("token_name", DEFAULT_TOKEN_NAME)
                    token_description = config_data.get("token_description", DEFAULT_TOKEN_DESCRIPTION)
                    image_description = config_data.get("image_description", DEFAULT_IMAGE_DESCRIPTION)
                    prompt = config_data.get("prompt", DEFAULT_PROMPT)

# 主要生成按钮
if st.button("🚀 AI Generate Meme Tweet"):
    if token_name and token_description:
        # 创建进度显示区域
        status_container = st.empty()
        result_container = st.container()

        with result_container:
            # 1. 获取最新趋势
            status_container.info("🔍 AI Fetching latest trends...")
            trends = get_latest_global_trends()
            if not trends:
                st.error("Failed to fetch trends. Please try again.")
                st.stop()

            st.subheader("AI Search Trending Topics")
            st.write(", ".join(trends))

            # 2. AI 选择趋势
            status_container.info("🤔 AI is selecting the most suitable trend...")
            selected_trend, trend_explanation = select_best_trend(trends, token_name, token_description)

            st.subheader("AI Select Trend")
            st.write(f"AI Selected Trend: **{selected_trend}**")
            st.write("Why this trend?")
            st.write(trend_explanation)

            # 3. 生成推文
            status_container.info("✍️ AI Generating tweet...")
            tweet = generate_meme_tweet(
                token_name,
                token_description,
                trends,
                prompt
            )

        if tweet:
            st.subheader("AI Generated Tweet")
            st.write(tweet)

            # 4. 生成配图
            status_container.info("🎨 Creating accompanying image...")
            combined_image_prompt = f"{image_description}\n\nTweet Content: {tweet}"
            image_url = generate_image_from_text(combined_image_prompt)

            if image_url:
                st.subheader("AI Generated Image")
                st.image(image_url, caption="Generated Meme Tweet Image")

                # 使用用户输入的认证信息
                if 'client' in locals() and client:
                    try:
                        # 尝试发送推文
                        response = client.create_tweet(text=tweet)
                        tweet_id = response.data['id']

                        # 显示成功信息
                        st.success(f"Tweet posted successfully! Tweet ID: {tweet_id}")
                        st.markdown(f"[View your tweet](https://twitter.com/user/status/{tweet_id})")

                        # 显示发推按钮
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.button("🐦 Post Another Tweet")

                    except Exception as e:
                        # 显示错误信息
                        st.error(f"Error posting tweet: {str(e)}")

                        # 显示重试按钮
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.button("🔄 Retry Posting")
                else:
                    st.error("Failed to authenticate with Twitter. Please check your credentials.")
            else:
                st.error("Failed to generate image.")

            # 清除状态信息，显示完成消息
            status_container.success("✨ AI Generation completed!")
        else:
            status_container.error("Failed to generate tweet. Please try again.")
    else:
        st.error("Please enter token information in the sidebar.")

# 添加页面底部说明
st.markdown("---")
st.markdown("""
    💡 **How it works:**
    1. AI Fetches real-time trending topics
    2. AI selects the most suitable trend for your token
    3. AI Generates an engaging tweet
    4. AI Creates a matching image
    5. Optionally post directly to X (Twitter)

    Configure your token details in the sidebar to get started!
""")
