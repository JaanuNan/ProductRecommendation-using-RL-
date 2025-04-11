import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import pandas as pd
from collections import deque
import seaborn as sns
import base64
import plotly.express as px

# Enhanced Simulated User Data (for context)
user_profiles = {
    "User1": {
        "preferences": ["Laptop", "Smartphone", "Headphones"],
        "history": [],
        "demographics": {"age": 25, "location": "New York", "occupation": "Student"},
        "interests": ["Technology", "Gadgets", "Music"],
        "purchase_frequency": "Weekly",
    },
    "User2": {
        "preferences": ["Tablet", "Smartwatch"],
        "history": ["Smartphone"],
        "demographics": {"age": 38, "location": "London", "occupation": "Engineer"},
        "interests": ["Fitness", "Wearables", "Travel"],
        "purchase_frequency": "Monthly",
    },
    "User3": {
        "preferences": ["Gaming Console", "TV"],
        "history": ["Headphones", "Laptop"],
        "demographics": {"age": 19, "location": "Los Angeles", "occupation": "Artist"},
        "interests": ["Gaming", "Entertainment", "Movies"],
        "purchase_frequency": "Bi-weekly",
    },
    "User4": {
        "preferences": ["Camera", "Speaker", "Monitor"],
        "history": [],
        "demographics": {"age": 45, "location": "Paris", "occupation": "Photographer"},
        "interests": ["Photography", "Audio", "Productivity"],
        "purchase_frequency": "Quarterly",
    },
    "User5": {
        "preferences": ["Laptop", "Tablet"],
        "history": ["Smartwatch", "Gaming Console"],
        "demographics": {"age": 31, "location": "Tokyo", "occupation": "Software Developer"},
        "interests": ["Programming", "Mobile Apps", "E-readers"],
        "purchase_frequency": "Monthly",
    },
    "User6": {
        "preferences": ["Headphones", "Speaker"],
        "history": [],
        "demographics": {"age": 28, "location": "Berlin", "occupation": "Musician"},
        "interests": ["Music Production", "Audio Equipment", "Streaming"],
        "purchase_frequency": "Occasionally",
    },
    "User7": {
        "preferences": ["Smartphone", "Smartwatch"],
        "history": ["Laptop"],
        "demographics": {"age": 52, "location": "Sydney", "occupation": "Business Analyst"},
        "interests": ["Productivity", "Communication", "Health"],
        "purchase_frequency": "Annually",
    },
    "User8": {
        "preferences": ["Gaming Console", "Headphones"],
        "history": ["TV"],
        "demographics": {"age": 22, "location": "Toronto", "occupation": "University Student"},
        "interests": ["Esports", "Online Gaming", "Streaming"],
        "purchase_frequency": "Irregularly",
    },
    "User9": {
        "preferences": ["Camera", "Laptop"],
        "history": ["Monitor"],
        "demographics": {"age": 35, "location": "Rome", "occupation": "Graphic Designer"},
        "interests": ["Visual Arts", "Design Software", "Creative Tools"],
        "purchase_frequency": "Bi-monthly",
    },
    "User10": {
        "preferences": ["Tablet", "Speaker"],
        "history": ["Smartphone", "Headphones"],
        "demographics": {"age": 41, "location": "Madrid", "occupation": "Teacher"},
        "interests": ["Education Technology", "Reading", "Entertainment"],
        "purchase_frequency": "Semi-annually",
    },
}

# Multi-Armed Bandit (Epsilon-Greedy)
class MultiArmBandit:
    def __init__(self, n_arms=10, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)
        self.arm_counts = np.zeros(n_arms)

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.arange(self.n_arms))
        return np.argmax(self.q_values)

    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.arm_counts[arm]

# Q-Learning
class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.q_table.shape[1]))
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (
            reward + self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action]
        )

# Deep Q-Learning (DQN)
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=2000)
        self.batch_size = 32

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values[0]).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states).max(1)[0]
        targets = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

# Actor-Critic
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCriticNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        action_probs, value = self.actor(state), self.critic(state)
        return action_probs, value

class ActorCriticAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = ActorCriticNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.model(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()
        return action, action_probs.squeeze(0)[action].item()

    def update(self, state, action, reward, next_state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        action_probs, value = self.model(state_tensor)
        _, next_value = self.model(next_state_tensor)

        advantage = reward + self.gamma * next_value.item() - value.item()
        action_log_prob = torch.log(action_probs.squeeze(0)[action])
        actor_loss = -action_log_prob * advantage
        critic_loss = nn.MSELoss()(value.squeeze(0), torch.FloatTensor([reward + self.gamma * next_value.item()]))

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

# Simulated Product Data and Environment
products = ["Laptop", "Smartphone", "Headphones", "Tablet", "Smartwatch", "Gaming Console", "TV", "Camera", "Speaker", "Monitor"]
rewards_base = np.array([0.8, 0.9, 0.7, 0.6, 0.5, 0.75, 0.85, 0.65, 0.7, 0.55])
n_products = len(products)
product_images = {
    "Laptop":"https://assets-global.website-files.com/5fac161927bf86485ba43fd0/643bd8cf55ece2e4b13093ef_Dell%20inspiron.webp",
    "Smartphone":"https://cdn.thewirecutter.com/wp-content/media/2024/05/smartphone-2048px-1013.jpg",
    "Headphones": "https://www.leafstudios.in/cdn/shop/files/Mainupdated_35a234be-57a2-41b6-b8db-79b54b7f5a7f_800x.jpg?v=1690372991",
    "Tablet": "https://cdn.thewirecutter.com/wp-content/media/2024/05/protablets-2048px-232431.jpg?auto=webp&quality=75&width=1024",
    "Smartwatch": "https://encrypted-tbn1.gstatic.com/shopping?q=tbn:ANd9GcT-wDBkm395G8bUDsNfr8BQzdisyY5jg-UtnvIIVn-uo0VTOsb7qnLSkoGaXUQDF-fiEHTDwrjkeQz40rimnJ74dm1Dkvur4b6d3jPwhi7Z8y8vTp-YGLTd&usqp=CAE",
    "Gaming Console": "https://akm-img-a-in.tosshub.com/indiatoday/images/story/202412/backbone-one-055930850-16x9_0.jpg?VersionId=fbOfTO1_Kus0obQSrd0RwpDgNTgxFro_&size=690:388",
    "TV": "https://www.intex.in/cdn/shop/products/1_9b8014ad-124e-4742-a628-9a4c4affe617_1024x1024.jpg?v=1648711109",
    "Camera": "https://m.media-amazon.com/images/I/61erc7ly+ZL._AC_UF1000,1000_QL80_.jpg",
    "Speaker": "https://www.bigw.com.au/medias/sys_master/images/images/hbf/h65/66886894059550.jpg",
    "Monitor": "https://m.media-amazon.com/images/I/71HhysWhCWL._SX679_.jpg",
}
product_details = {
    "Laptop": {
        "Description": "Powerful laptop for professionals.",
        "Price": "$1200",
        "Specifications": {"CPU": "Intel i7", "RAM": "16GB", "Storage": "512GB SSD"},
    },
    "Smartphone": {
        "Description": "High-end smartphone with advanced camera.",
        "Price": "$900",
        "Specifications": {"Display": "6.5 inch", "Camera": "48MP", "Battery": "4000mAh"},
    },
    "Headphones": {
        "Description": "Noise-cancelling headphones for immersive audio.",
        "Price": "$250",
        "Specifications": {"Type": "Over-Ear", "Connectivity": "Bluetooth", "Battery Life": "30 hours"},
    },
    "Tablet": {
        "Description": "Portable tablet for entertainment and productivity.",
        "Price": "$400",
        "Specifications": {"Display": "10 inch", "RAM": "4GB", "Storage": "128GB"},
    },
    "Smartwatch": {
        "Description": "Fitness tracking smartwatch with health monitoring.",
        "Price": "$300",
        "Specifications": {"Sensors": "Heart Rate, GPS", "Water Resistance": "5ATM", "Battery Life": "7 days"},
    },
    "Gaming Console": {
        "Description": "Next-gen gaming console for immersive gameplay.",
        "Price": "$500",
        "Specifications": {"Storage": "1TB SSD", "Graphics": "4K", "Controllers": "Wireless"},
    },
    "TV": {
        "Description": "4K UHD TV for home entertainment.",
        "Price": "$800",
        "Specifications": {"Screen Size": "55 inch", "Resolution": "4K", "Smart Features": "Yes"},
    },
    "Camera": {
        "Description": "Mirrorless camera for professional photography.",
        "Price": "$1500",
        "Specifications": {"Sensor": "Full Frame", "Video": "4K", "Lens Mount": "Interchangeable"},
    },
    "Speaker": {
        "Description": "Wireless speaker for high-quality audio.",
        "Price": "$150",
        "Specifications": {"Connectivity": "Bluetooth, Wi-Fi", "Power": "50W", "Battery Life": "12 hours"},
    },
    "Monitor": {
        "Description": "Ultra-wide monitor for productivity and gaming.",
        "Price": "$600",
        "Specifications": {"Screen Size": "34 inch", "Resolution": "1440p", "Refresh Rate": "144Hz"},
    },
}

def simulate_environment(user_profile, action):
    product = products[action]
    reward = 0
    if product in user_profile["preferences"]:
        reward += rewards_base[action] * 0.8
    if product not in user_profile["history"]:
        reward += 0.2
    reward += np.random.normal(0, 0.1)
    reward = np.clip(reward, 0, 1)
    return reward

# Streamlit UI
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def main():
    image_path = "images/background.jpg"  # Ensure correct file path

    base64_image = get_base64_of_image(image_path)

    st.set_page_config(page_title="RL Product Recommendation", layout="wide")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{base64_image}") no-repeat center center fixed;
            background-size: cover;
        }}
        .st-header {{
            background-color: rgba(74, 144, 226, 0.9) !important;
            color: white !important;
        }}
        .st-subheader {{
            color: white;
        }}
        .sidebar .sidebar-content {{
            background-color: rgba(224, 242, 247, 0.9);
            color: #333;
        }}
        .stButton>button {{
            background-color: #4a90e2;
            color: white;
            border-radius: 5px;
            padding: 0.5em 1em;
        }}
        .stButton>button:hover {{
            background-color: #357ab0;
        }}
        .stSuccess, .stInfo, .stWarning, .stError {{
            background-color: rgba(255, 255, 255, 0.9);
            color: #333;
            padding: 1em;
            border-radius: 5px;
        }}
        table {{
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: #333 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🛒 Product Recommendation in Online Advertising")

    with st.sidebar:
        st.header("⚙️ Configuration")
        algorithm = st.radio("Choose an Algorithm", ["Multi-Armed Bandit", "Q-Learning", "DQN", "Actor-Critic"])
        user_id = st.selectbox("Select User", list(user_profiles.keys()))
        user_profile = user_profiles[user_id]
        st.markdown("---")

        st.subheader("👤 User Profile")
        with st.expander(f"View Profile of {user_id}"):
            for key, value in user_profile.items():
                st.write(f"**{key.capitalize()}:** {value}")
        st.markdown("---")

        st.subheader("📊 Simulation Controls")
        episodes = st.slider("Number of Episodes", 1, 500, 100)
        train_button = st.button("🚀 Train Agent", use_container_width=True)
        st.markdown("---")

        st.subheader("💾 Model Management")
        model_name = st.text_input("Model Filename", f"{algorithm.lower()}_model.pth")
        col1, col2 = st.columns(2)
        load_button = col1.button("Load Model", use_container_width=True)
        save_button = col2.button("Save Model", use_container_width=True)
        st.markdown("---")

        st.subheader("📈 Performance Metrics")
        performance_placeholder = st.empty()

    st.subheader(f"🤖 Using Algorithm: {algorithm}")
    recommendation_placeholder = st.empty()
    feedback_placeholder = st.empty()
    analysis_placeholder = st.empty()

    all_rewards = {}

    if algorithm == "Multi-Armed Bandit":
        bandit = MultiArmBandit(n_arms=n_products, epsilon=0.1)
        if train_button:
            cumulative_rewards = []
            for episode in range(episodes):
                action = bandit.select_arm()
                reward = simulate_environment(user_profile, action)
                bandit.update(action, reward)
                cumulative_rewards.append(reward)
            all_rewards["Multi-Armed Bandit"] = cumulative_rewards
            performance_placeholder.metric("Average Reward (Multi-Armed Bandit)", f"{np.mean(cumulative_rewards):.4f}")

        action = bandit.select_arm()
        recommended_product = products[action]
        recommendation_placeholder.success(f"✨ Recommended Product for **{user_id}**: **{recommended_product}**")
        with feedback_placeholder.form("user_feedback_form", clear_on_submit=True):
            st.write("Rate your experience with this recommendation:")
            user_feedback = st.slider("Rating (1-5)", 1, 5, 3)
            feedback_submitted = st.form_submit_button("Submit Feedback")
            if feedback_submitted:
                reward = (user_feedback / 5.0) * rewards_base[action]
                bandit.update(action, reward)
                st.success("👍 Feedback received. Agent updated.")

        st.write("📊 Multi Arm Bandit Q values")
        data_for_q_values = pd.DataFrame({"Products": products, "Estimated Reward (Q-Value)": bandit.q_values})
        st.line_chart(data_for_q_values.set_index("Products"))

        if save_button:
            st.warning("Multi-Armed Bandit does not have a savable model.")
        if load_button:
            st.warning("Multi-Armed Bandit does not have a loadable model.")

    elif algorithm == "Q-Learning":
        q_agent = QLearning(n_states=n_products, n_actions=n_products, epsilon=0.2)
        if train_button:
            cumulative_rewards = []
            for episode in range(episodes):
                state = random.randint(0, n_products - 1)
                action = q_agent.select_action(state)
                reward = simulate_environment(user_profile, action)
                next_state = action
                q_agent.update(state, action, reward, next_state)
                cumulative_rewards.append(reward)
            all_rewards["Q-Learning"] = cumulative_rewards
            performance_placeholder.metric("Average Reward (Q-Learning)", f"{np.mean(cumulative_rewards):.4f}")

        state = random.randint(0, n_products - 1)
        action = q_agent.select_action(state)
        recommended_product = products[action]
        recommendation_placeholder.success(f"✨ Recommended Product for **{user_id}**: **{recommended_product}**")
        with feedback_placeholder.form("user_feedback_form", clear_on_submit=True):
            st.write("Rate your experience with this recommendation:")
            user_feedback = st.slider("Rating (1-5)", 1, 5, 3)
            feedback_submitted = st.form_submit_button("Submit Feedback")
            if feedback_submitted:
                reward = (user_feedback / 5.0) * rewards_base[action]
                q_agent.update(state, action, reward, action)
                st.success("👍 Feedback received. Agent updated.")

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(q_agent.q_table, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=products, yticklabels=products, ax=ax)
        ax.set_xlabel("Action (Recommended Product)")
        ax.set_ylabel("State (Previous Product/Context)")
        ax.set_title("🔥 Q-Learning Q-Table")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        analysis_placeholder.pyplot(fig)

        if save_button:
            np.save(model_name, q_agent.q_table)
            st.success(f"💾 Q-Table saved to `{model_name}`")
        if load_button:
            try:
                q_agent.q_table = np.load(model_name)
                st.success(f"📂 Q-Table loaded from `{model_name}`")
            except FileNotFoundError:
                st.error(f"❌ File `{model_name}` not found.")

        # Interactive Heatmap of Q-values
        fig_heatmap = px.imshow(q_agent.q_table, labels=dict(x="Action (Recommended Product)", y="State (Previous Product/Context)", color="Q-value"), x=products, y=products, title="Interactive Q-Learning Q-Table")
        st.plotly_chart(fig_heatmap)

    elif algorithm == "DQN":
        state_size = n_products
        action_size = n_products
        agent = DQNAgent(state_size, action_size)

        if load_button:
            try:
                agent.load(model_name)
                st.success(f"📂 DQN Model loaded from `{model_name}`")
            except FileNotFoundError:
                st.error(f"❌ File `{model_name}` not found.")

        if train_button:
            cumulative_rewards = []
            for episode in range(episodes):
                state = np.zeros(state_size)
                total_reward = 0
                done = False
                for _ in range(10):
                    action = agent.act(state)
                    reward = simulate_environment(user_profile, action)
                    next_state = np.zeros(state_size)
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    agent.replay()
                    if done:
                        break
                cumulative_rewards.append(total_reward)
            all_rewards["DQN"] = cumulative_rewards
            performance_placeholder.metric("Average Reward (DQN)", f"{np.mean(cumulative_rewards):.4f}")

        state = np.zeros(state_size)
        action = agent.act(state)
        recommended_product = products[action]
        recommendation_placeholder.success(f"✨ Recommended Product for **{user_id}**: **{recommended_product}**")
        with feedback_placeholder.form("user_feedback_form", clear_on_submit=True):
            st.write("Rate your experience with this recommendation:")
            user_feedback = st.slider("Rating (1-5)", 1, 5, 3)
            feedback_submitted = st.form_submit_button("Submit Feedback")
            if feedback_submitted:
                reward = (user_feedback / 5.0) * rewards_base[action]
                next_state = np.zeros(state_size)
                agent.remember(state, action, reward, next_state, False)
                agent.replay()
                recommendation_placeholder.markdown(
                    """
                    <div style='background-color: #FFA500; color: white; padding: 10px; border-radius: 5px;'>
                        <b>👍 Feedback received. Agent updated.</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = agent.model(state_tensor).detach().numpy()[0]
        df_q_values = pd.DataFrame({'Product': products, 'Q-Value': q_values})
        analysis_placeholder.dataframe(df_q_values.sort_values(by='Q-Value', ascending=False))

        if save_button:
            agent.save(model_name)
            st.success(f"💾 DQN Model saved to `{model_name}`")

        # Interactive Bar Chart of Q-values for the current state
        fig_bar = px.bar(df_q_values, x='Product', y='Q-Value', title='Q-Values for Current State')
        st.plotly_chart(fig_bar)

    elif algorithm == "Actor-Critic":
        state_size = n_products
        action_size = n_products
        agent = ActorCriticAgent(state_size, action_size)

        if load_button:
            try:
                agent.load(model_name)
                st.success(f"📂 Actor-Critic Model loaded from `{model_name}`")
            except FileNotFoundError:
                st.error(f"❌ File `{model_name}` not found.")

        if train_button:
            cumulative_rewards = []
            for episode in range(episodes):
                state = np.zeros(state_size)
                total_reward = 0
                done = False
                for _ in range(10):
                    action, _ = agent.act(state)
                    reward = simulate_environment(user_profile, action)
                    next_state = np.zeros(state_size)
                    agent.update(state, action, reward, next_state)
                    state = next_state
                    total_reward += reward
                    if done:
                        break
                cumulative_rewards.append(total_reward)
            all_rewards["Actor-Critic"] = cumulative_rewards
            performance_placeholder.metric("Average Reward (Actor-Critic)", f"{np.mean(cumulative_rewards):.4f}")

        state = np.zeros(state_size)
        action, action_prob = agent.act(state)
        recommended_product = products[action]
        recommendation_placeholder.success(f"✨ Recommended Product for **{user_id}**: **{recommended_product}** (Probability: {action_prob:.2f})")
        with feedback_placeholder.form("user_feedback_form", clear_on_submit=True):
            st.write("Rate your experience with this recommendation:")
            user_feedback = st.slider("Rating (1-5)", 1, 5, 3)
            feedback_submitted = st.form_submit_button("Submit Feedback")
            if feedback_submitted:
                reward = (user_feedback / 5.0) * rewards_base[action]
                next_state = np.zeros(state_size)
                agent.update(state, action, reward, next_state)
                st.success("👍 Feedback received. Agent updated.")

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = agent.model(state_tensor)
        df_probs = pd.DataFrame({'Product': products, 'Probability': action_probs.detach().numpy()[0]})
        analysis_placeholder.dataframe(df_probs.sort_values(by='Probability', ascending=False))

        if save_button:
            agent.save(model_name)
            st.success(f"💾 Actor-Critic Model saved to `{model_name}`")

        # Interactive Bar Chart of Action Probabilities
        fig_probs = px.bar(df_probs, x='Product', y='Probability', title='Action Probabilities')
        st.plotly_chart(fig_probs)
    st.subheader("👥 User Demographics")
    demographics_data = [profile["demographics"] for profile in user_profiles.values()]
    df_demographics = pd.DataFrame(demographics_data)

    st.write("User Distribution by Age, Occupation, and Location:")

    age_counts = df_demographics['age'].value_counts().sort_index()
    occupation_counts = df_demographics['occupation'].value_counts()
    location_counts = df_demographics['location'].value_counts()

    # Create a DataFrame with all unique indices and fill missing values
    all_indices = sorted(list(set(age_counts.index) | set(occupation_counts.index) | set(location_counts.index)))

    data_for_multi_line_chart = pd.DataFrame({
        "Age Users": age_counts.reindex(all_indices, fill_value=0),
        "Occupation Users": occupation_counts.reindex(all_indices, fill_value=0),
        "Location Users": location_counts.reindex(all_indices, fill_value=0),
    })

    st.markdown("""
        <style>
        div[data-testid="stGraph"] {
            border: 1px solid black;
            padding: 5px;
            width: 80%;
            margin: auto;
        }
        </style>
        """, unsafe_allow_html=True)
    st.line_chart(data_for_multi_line_chart)

    if train_button and all_rewards:
        st.subheader("📈 Algorithm Performance Comparison")
        data_for_comparison = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_rewards.items()]))
        st.line_chart(data_for_comparison)
        st.subheader("📊 Algorithm Performance Summary")
        if all_rewards:
            avg_rewards = {algo: np.mean(rewards) for algo, rewards in all_rewards.items()}
            df_avg_rewards = pd.DataFrame(avg_rewards.items(), columns=['Algorithm', 'Average Reward'])
            st.write("Average Rewards:")
            st.dataframe(df_avg_rewards.sort_values(by='Average Reward', ascending=False))
            final_rewards = {algo: rewards[-1] for algo, rewards in all_rewards.items()}
            df_final_rewards = pd.DataFrame(final_rewards.items(), columns=['Algorithm', 'Final Reward'])
            st.write("Final Rewards (Reward in the last episode):")
            st.dataframe(df_final_rewards.sort_values(by='Final Reward', ascending=False))
            data_for_boxplot = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_rewards.items()]))
            st.write("Reward Distribution across Episodes:")
            st.line_chart(data_for_boxplot)
        else:
            st.info("No training data available to visualize performance.")
    st.title("CTR Table ")
    fake_ctr_data = [
        {"Product": "Laptop", "CTR (%)": "15.00"},
        {"Product": "Smartphone", "CTR (%)": "25.50"},
        {"Product": "Headphones", "CTR (%)": "N/A"},
        {"Product": "Tablet", "CTR (%)": "10.00"},
        {"Product": "Smartwatch", "CTR (%)": "30.25"},
        {"Product": "Gaming Console", "CTR (%)": "20.00"},
        {"Product": "TV", "CTR (%)": "N/A"},
        {"Product": "Camera", "CTR (%)": "18.75"},
        {"Product": "Speaker", "CTR (%)": "5.00"},
        {"Product": "Monitor", "CTR (%)": "22.00"},
    ]
    df_ctr = pd.DataFrame(fake_ctr_data)
    st.subheader("🖱️ Click-Through Rate (CTR) Approximation")
    st.dataframe(df_ctr)
    # User Interaction (Detailed Feedback)
    with feedback_placeholder.form("detailed_feedback_form", clear_on_submit=True):
        st.write("Provide more detailed feedback:")
        like_dislike = st.radio("Did you like the recommendation?", ("Like", "Dislike", "Neutral"))
        reason = st.text_area("Reason for your feedback:")
        feedback_submitted_detailed = st.form_submit_button("Submit Detailed Feedback")
        if feedback_submitted_detailed:
            st.success("Detailed feedback received!")
            print(like_dislike, reason)
    # User Profile Editing (Simplified example)
    if st.checkbox("Edit User Profile"):
        user_id = st.session_state.get("user_id", list(user_profiles.keys())[0])
        edited_preferences = st.multiselect("Edit Preferences", products, default=user_profiles[user_id]["preferences"])
        if st.button("Save Changes"):
            user_profiles[user_id]["preferences"] = edited_preferences
            st.success("User profile updated!")



    # Product Image Display (Enhanced)
    st.subheader("🖼️ Product Images")
    cols = st.columns(len(products))
    for i, product in enumerate(products):
        with cols[i]:
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #e0e0e0;
                    padding: 10px;
                    border-radius: 5px;
                    background-color: white;
                    text-align: center;
                ">
                    <img src="{product_images[product]}" alt="{product}" style="max-width: 100%; height: auto;">
                    <p style="margin-top: 5px;">{product}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    products_list = list(product_images.keys())
    st.subheader("🛍️ Product Details")
    selected_product = st.selectbox("Select a Product", products_list)

    # Product Image Display with White Box
    st.markdown(
        f"""
        <div style="
            border: 1px solid #e0e0e0;
            padding: 10px;
            border-radius: 5px;
            background-color: white;
            text-align: center;
        ">
            <img src="{product_images[selected_product]}" alt="{selected_product}" style="max-width: 100%; height: auto;">
            <p style="margin-top: 5px;">{selected_product}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    product = product_details[selected_product]
    st.write(f"**Description:** {product['Description']}")
    st.write(f"**Price:** {product['Price']}")
    st.write("**Specifications:**")
    for key, value in product["Specifications"].items():
        st.write(f"- {key}: {value}")

if __name__ == "__main__":
    main()

    
