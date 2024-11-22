import requests
from bs4 import BeautifulSoup
import streamlit as st
import re
import json
import os
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import tweepy
from datetime import datetime

# Constants
DEFAULT_TOKEN_NAME = "LEGENDARY HUMANITY"
DEFAULT_TOKEN_DESCRIPTION = "Merging fashion, art, and #AI into #Web3 assets. Empowering designers and artists with community-driven #meme coins. $VIVI is the governance token."
DEFAULT_IMAGE_DESCRIPTION = "A vibrant and humorous illustration representing the essence of the tweet, with logo 'LEGENDARY HUMANITY' and 'VIVI'."

# Retry decorator for API calls
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def api_call_with_retry(func, *args, **kwargs):
    return func(*args, **kwargs)

def rag_search(trend: str) -> str:
    """
    获取趋势的上下文信息
    """
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-ydcvskcyyzictsylxplbpqmqlpillcpkqznxclfjyohkefwt",
        "Content-Type": "application/json"
    }

    prompt = f"""
    Analyze this trend "{trend}" and provide:
    1. What is this trend about?
    2. Why is it trending?
    3. Current sentiment around it

    Keep response brief and focused on viral/meme potential.
    """

    payload = {
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 200
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        return content
    except Exception as e:
        st.error(f"RAG Search Error: {str(e)}")
        return ""

def preprocess_trends(trends: List[str]) -> List[Dict]:
    """
    预处理趋势数据，添加元数据
    """
    processed = []
    seen = set()

    for trend in trends:
        # 清理并提取数据
        match = re.match(r'^(.*?)(?:\s+(\d+)([KM])?)?$', trend)
        if match:
            name, count, unit = match.groups()

            # 标准化名称
            clean_name = name.strip()
            if clean_name.lower() in seen:
                continue

            # 计算实际数量
            engagement = 0
            if count:
                count = int(count)
                if unit == 'K':
                    engagement = count * 1000
                elif unit == 'M':
                    engagement = count * 1000000
                else:
                    engagement = count

            processed.append({
                'name': clean_name,
                'original': trend,
                'engagement': engagement,
                'is_hashtag': clean_name.startswith('#'),
            })
            seen.add(clean_name.lower())

    return processed

def score_trend(trend: Dict, token_name: str, token_description: str) -> float:
    """
    对趋势进行评分
    """
    score = 0

    # 参与度分数 (0-0.3)
    if trend['engagement'] > 500000:
        score += 0.3
    elif trend['engagement'] > 100000:
        score += 0.2
    elif trend['engagement'] > 10000:
        score += 0.1

    # 相关性分数 (0-0.3)
    relevant_keywords = ['crypto', 'nft', 'web3', 'ai', 'tech', 'digital', 'art', 'game']
    trend_text = f"{trend['name']} {token_name} {token_description}".lower()
    relevance = sum(1 for keyword in relevant_keywords if keyword in trend_text)
    score += min(0.3, relevance * 0.1)

    # 话题类型分数 (0-0.2)
    if trend['is_hashtag']:
        score += 0.1
    if len(trend['name'].split()) <= 3:  # 简短话题更容易传播
        score += 0.1

    # 多样性分数 (0-0.2)
    if bool(re.search(r'[^a-zA-Z0-9\s]', trend['name'])):  # 包含特殊字符
        score += 0.1
    if any(char.isupper() for char in trend['name']):  # 包含大写字母
        score += 0.1

    return score

def get_latest_global_trends() -> List[Dict]:
    """
    获取最新趋势并预处理
    """
    url = "https://trends24.in/united-states/"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        trends_list = []
        trend_cards = soup.find_all('ol', class_='trend-card__list')

        for card in trend_cards[:3]:  # 只取前3个时段的数据
            trends = card.find_all('li')
            for trend in trends:
                trends_list.append(trend.get_text(strip=True))

        return preprocess_trends(trends_list)
    except Exception as e:
        st.error(f"Error fetching trends: {str(e)}")
        return []

def select_best_trend(trends: List[Dict], token_name: str, token_description: str) -> tuple[str, str, str]:
    """
    选择最佳趋势并提供解释
    返回: (trend_name, explanation, context)
    """
    # 给每个趋势评分
    scored_trends = [(trend, score_trend(trend, token_name, token_description))
                    for trend in trends]
    # 排序并选择前3个
    top_trends = sorted(scored_trends, key=lambda x: x[1], reverse=True)[:3]

    # 让AI从前3个中选择
    selection_prompt = f"""
    Analyze these trending topics for a crypto meme token and return the analysis in JSON format:
    {[t[0]['original'] for t in top_trends]}

    Token: {token_name}
    Description: {token_description}

    Select the best trend considering:
    1. Viral potential and engagement
    2. Creative connection possibilities
    3. Meme creation potential
    4. Community engagement angle

    Format your response EXACTLY like this:
    {{
        "selected_trend": "chosen trend name",
        "explanation": "2-3 sentences explaining why this trend is perfect",
        "usage_angle": "specific idea how to use this trend"
    }}

    Important: Return ONLY the JSON, no additional text.
    """

    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-ydcvskcyyzictsylxplbpqmqlpillcpkqznxclfjyohkefwt",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "messages": [{"role": "user", "content": selection_prompt}],
        "temperature": 0.7,
        "max_tokens": 300
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        ai_response = response.json()['choices'][0]['message']['content'].strip()

        # 清理响应文本，确保它是有效的JSON
        ai_response = ai_response.replace('\n', ' ').strip()
        if ai_response.startswith("```json"):
            ai_response = ai_response[7:]
        if ai_response.endswith("```"):
            ai_response = ai_response[:-3]
        ai_response = ai_response.strip()

        try:
            result = json.loads(ai_response)
        except json.JSONDecodeError:
            # 如果JSON解析失败，使用正则表达式提取信息
            import re
            selected_trend = re.search(r'"selected_trend"\s*:\s*"([^"]+)"', ai_response)
            explanation = re.search(r'"explanation"\s*:\s*"([^"]+)"', ai_response)
            usage_angle = re.search(r'"usage_angle"\s*:\s*"([^"]+)"', ai_response)

            result = {
                "selected_trend": selected_trend.group(1) if selected_trend else top_trends[0][0]['original'],
                "explanation": explanation.group(1) if explanation else "Top trending topic with high engagement potential.",
                "usage_angle": usage_angle.group(1) if usage_angle else "Leverage the trend's popularity for viral content."
            }

        # 获取选中趋势的上下文
        context = rag_search(result['selected_trend'])

        return (
            result['selected_trend'],
            result['explanation'],
            f"{result['usage_angle']}\n\nTrend Context: {context}"
        )
    except Exception as e:
        st.error(f"Trend Selection Error: {str(e)}")
        # 发生错误时返回分数最高的趋势
        return (
            top_trends[0][0]['original'],
            "Using highest scored trend for maximum engagement potential.",
            "Leveraging trending topic for viral content creation."
        )

def generate_meme_tweet(token_name: str, token_description: str,
                       trend: str, context: str) -> str:
    """
    生成meme tweet
    """
    prompt = f"""
    Create a viral crypto meme tweet that perfectly blends the trend with our token!

    Trend: {trend}
    Context: {context}

    Token: {token_name}
    Description: {token_description}

    Requirements:
    1. Under 280 characters
    2. Include 2-3 relevant emojis
    3. Creative connection to the trend
    4. Memorable hook or punchline
    5. Call-to-action
    6. 2-3 hashtags (including trend if relevant)

    Make it catchy, humorous, and shareable!
    Return the tweet text only, no explanations.
    """

    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-ydcvskcyyzictsylxplbpqmqlpillcpkqznxclfjyohkefwt",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 200
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        tweet = response.json()['choices'][0]['message']['content']
        return tweet.strip()
    except Exception as e:
        st.error(f"Tweet Generation Error: {str(e)}")
        return None

def generate_image(tweet_text: str, image_description: str) -> str:
    """
    基于tweet生成配图
    """
    prompt = f"""
    Create a meme image based on this tweet:
    {tweet_text}

    Image requirements:
    {image_description}

    Style: Meme-worthy, eye-catching, humorous
    Include: Token branding elements
    Mood: Viral and shareable
    """

    url = 'https://api.siliconflow.cn/v1/image/generations'
    headers = {
        "Authorization": "Bearer sk-ydcvskcyyzictsylxplbpqmqlpillcpkqznxclfjyohkefwt",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt": prompt,
        "image_size": "1024x1024",
        "seed": int(time.time()) % 1000000  # 动态种子
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['images'][0]['url']
    except Exception as e:
        st.error(f"Image Generation Error: {str(e)}")
        return None

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

# Streamlit UI
def main():
    st.title("🚀 AI Crypto Meme Generator Pro")
    st.write("Creating viral crypto memes using real-time trends!")

    # Sidebar configuration
    with st.sidebar:
        st.header("Token Configuration")
        token_name = st.text_input("Token Name", DEFAULT_TOKEN_NAME)
        token_description = st.text_area("Token Description", DEFAULT_TOKEN_DESCRIPTION)
        image_description = st.text_area("Image Style", DEFAULT_IMAGE_DESCRIPTION)

        st.markdown("---")
        st.markdown("""
        💡 **Tips:**
        - Keep token name memorable
        - Description should be clear
        - Image style guides the AI
        """)

    # Main content
    if st.button("✨ Generate Viral Meme", use_container_width=True):
        if not (token_name and token_description):
            st.error("Please fill in token details first!")
            return

        with st.spinner("🔍 AI Analyzing trends..."):
            # 1. 获取趋势
            trends = get_latest_global_trends()
            if not trends:
                st.error("Couldn't fetch trends. Please try again.")
                return

            # 显示所有趋势
            with st.expander("📊 Current Trends"):
                for trend in trends:
                    st.write(f"- {trend['original']}")

            # 2. 选择最佳趋势
            trend, explanation, context = select_best_trend(
                trends, token_name, token_description
            )

            st.subheader("🎯 AI Selected Trend")
            st.write(f"**{trend}**")
            st.write(explanation)

            # 3. 生成Tweet
            tweet = generate_meme_tweet(
                token_name, token_description,
                trend, context
            )

            if tweet:
                st.subheader("📝 AI Generated Tweet")
                st.code(tweet, language="text")

                # 4. 生成配图
                with st.spinner("🎨 Creating meme image..."):
                    image_url = generate_image(tweet, image_description)
                    if image_url:
                        st.subheader("🖼️ Meme Image")
                        st.image(image_url, caption="AI Generated Meme Image")

                        # 提供下载选项
                        st.markdown(f"[Download Image]({image_url})")
                    else:
                        st.error("Failed to generate image.")

                # 复制按钮
                st.button("📋 Copy Tweet", on_click=lambda: st.write("Tweet copied!"))

                # 发推按钮
                if 'client' in globals() and client:
                    if st.button("🐦 Post Tweet to X", use_container_width=True):
                        post_tweet(client, tweet)
                else:
                    st.error("Please authenticate with Twitter first!")
            else:
                st.error("Failed to generate tweet. Please try again.")

    # Footer
    st.markdown("---")
    st.markdown("""
        Made with ❤️ by AI

        **How it works:**
        1. Real-time trend analysis
        2. Smart trend selection
        3. AI-powered content creation
        4. Meme image generation
    """)

if __name__ == "__main__":
    main()
