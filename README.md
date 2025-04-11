# ProductRecommendation-using-RL

🛒 RL Product Recommendation in Online Advertising 🤖



Overview:

--------

This application showcases the power of Reinforcement Learning (RL) for suggesting products in online advertising. 💡 It simulates how users interact with recommendations and trains intelligent agents to make the best suggestions based on user profiles and their feedback. 🧠



🚀 Try the live application here: https://3txizagfu9hmhe6p9dpe5h.streamlit.app/ 🌐



Key Features:

-------------

- **Algorithm Selection:** Choose your recommendation strategy! 🤔 Options include Multi-Armed Bandit (Epsilon-Greedy), Q-Learning, Deep Q-Learning (DQN), and Actor-Critic.

- **Simulated User Profiles:** Explore recommendations for diverse virtual users 👤 with unique preferences, purchase histories 🛍️, demographics 📊, and interests.

- **Agent Training:** Watch the magic happen! ✨ Train the selected RL agent by setting the number of simulation rounds 🔄.

- **Product Recommendations:** Get personalized suggestions! 🎁 The trained agent will recommend the most promising products.

- **User Feedback:** Your opinion matters! 👍 Rate the recommendations to help the agent learn and improve.

- **Performance Visualization:** See how the agent learns! 📈 Track its progress with average reward charts.

- **Policy Inspection:** Peek into the agent's mind! 👀 Examine the learned Q-values (Q-Learning) and recommendation probabilities (Actor-Critic).

- **User Demographics Analysis:** Understand your virtual audience! 🧑‍🤝‍🧑 Visualize user distribution by age 🎂, occupation 💼, and location 📍.

- **Product Exploration:** Discover the items! 🔍 View details and images 🖼️ of all available products.

- **Simplified Profile Editing:** Experiment with user preferences! ✍️ Make quick changes to see how recommendations adapt.

- **Simulated CTR:** Get a glimpse of engagement! 🖱️ Observe a sample Click-Through Rate (CTR) table for different products.



Technologies Used:

------------------

- Streamlit: Building the interactive web interface. 💻

- NumPy: Number crunching and array wizardry. <0xF0><0x9F><0xA7><0xAE>

- PyTorch: Powering the advanced RL models (DQN, Actor-Critic). 🔥

- Matplotlib: Creating static charts (for Q-Learning insights). 📊

- Pandas: Managing and manipulating data tables. 🐼

- Seaborn: Enhancing data visualization with style. 🎨

- Plotly Express: Making interactive and dynamic charts. 📊✨



How to Use (via the Streamlit App):

-----------------------------------

1. **Select an Algorithm:** Head to the sidebar and pick your recommendation algorithm. 🤔

2. **Select a User:** Choose a user from the dropdown to see tailored suggestions. 👤

3. **Train Agent (Optional):** Set the training episodes and hit "Train Agent" to start the learning process. 🚀

4. **View Recommendation:** The top recommendation for the current user will appear. 🎁

5. **Provide Feedback:** Rate the suggestion using the slider and click "Submit Feedback." 👍👎

6. **Explore Visualizations:** Check out the performance graphs, Q-tables, and probability charts. 📈📊

7. **Interact with Other Features:** Dive into user profiles, product details, and the CTR table. 🔍



Local Setup (for development or running locally):

------------------------------------------------

1. **Make sure you have Python 3.x installed.** 🐍

2. **Install the necessary libraries:**
