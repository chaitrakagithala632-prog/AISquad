import os
import json
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load API key
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

st.set_page_config(
    page_title="TaskPilot AI",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ TaskPilot AI")
st.subheader("Your AI-Powered Workflow Assistant")

# Check API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("API key not found. Check your .env file.")
    st.stop()

# Connect to Groq
client = Groq(api_key=api_key)

instruction = st.text_area(
    "Describe the work you want to organize",
    placeholder="Example: Organize a college technical fest with technical and fun events.",
    height=150
)

if st.button("🚀 Generate Workflow"):

    if not instruction.strip():
        st.warning("Please enter your instructions.")

    else:
        with st.spinner("AI is planning your workflow..."):

            try:
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": """
                            You are TaskPilot AI, an intelligent workflow assistant.
                            Create a practical workflow based on the user's instructions.
                            Return ONLY a valid JSON array containing 4 to 6 tasks.
                            Each task must contain:
                            - task: clear task description
                            - priority: High, Medium, or Low
                            - deadline: suggested timeframe
                            - status: Pending
                            Arrange tasks in logical order.
                            Do not invent specific calendar dates.
                            Do not include Markdown code fences.
                            """
                        },
                        {
                            "role": "user",
                            "content": instruction
                        }
                    ],
                    temperature=0.3
                )

                result = response.choices[0].message.content
                tasks = json.loads(result)

                if not isinstance(tasks, list):
                    raise ValueError("The AI response was not a task list.")

                st.success("Workflow generated successfully!")
                st.subheader("📋 Manage Your Tasks")

                for number, task in enumerate(tasks, start=1):

                    with st.container(border=True):

                        st.write(f"### Task {number}")

                        task["task"] = st.text_input(
                            "Task Description",
                            value=task["task"],
                            key=f"task_{number}"
                        )

                        priorities = ["High", "Medium", "Low"]
                        current_priority = task.get("priority", "Medium")

                        if current_priority not in priorities:
                            current_priority = "Medium"

                        task["priority"] = st.selectbox(
                            "Priority",
                            priorities,
                            index=priorities.index(current_priority),
                            key=f"priority_{number}"
                        )

                        task["deadline"] = st.text_input(
                            "Suggested Deadline",
                            value=task.get("deadline", "Not specified"),
                            key=f"deadline_{number}"
                        )

                        statuses = ["Pending", "In Progress", "Completed"]
                        current_status = task.get("status", "Pending")

                        if current_status not in statuses:
                            current_status = "Pending"

                        task["status"] = st.selectbox(
                            "Status",
                            statuses,
                            index=statuses.index(current_status),
                            key=f"status_{number}"
                        )

                st.download_button(
                    label="⬇️ Download Updated Workflow",
                    data=json.dumps(tasks, indent=2),
                    file_name="taskpilot_workflow.json",
                    mime="application/json"
                )

            except Exception as error:
                st.error("Unable to generate the workflow.")
                st.write("Error details:", str(error))