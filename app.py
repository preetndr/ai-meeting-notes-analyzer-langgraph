import streamlit as st
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

class MeetingState(TypedDict):
    transcript: str
    topics: str
    summary: str
    action_items: str
    owners: str
    priority: str
    final_report: str

# Step 1: Input Node
def input_node(state: MeetingState):
    return {"transcript": state["transcript"]}

# Step 2: Topic Extraction Agent
def topic_agent(state: MeetingState):
    prompt = f"Identify the main discussion topics as a numbered list from this transcript:\n{state['transcript']}"
    return {"topics": llm.invoke(prompt).text.strip()}

# Step 3: Meeting Summary Agent
def summary_agent(state: MeetingState):
    prompt = f"Write a short meeting summary (3 to 5 sentences) for this transcript:\n{state['transcript']}"
    return {"summary": llm.invoke(prompt).text.strip()}

# Step 4: Action Item Extraction Agent
def action_item_agent(state: MeetingState):
    prompt = (
        "Extract action items from this transcript as a numbered list (Task - Owner). "
        "If the owner is not mentioned, label it as 'Owner: Not specified'. "
        "If there are no action items, reply with exactly: No action items identified in this meeting.\n"
        f"{state['transcript']}"
    )
    actions = llm.invoke(prompt).text.strip()

    if "No action items" in actions:
        return {"action_items": "No action items identified in this meeting.", "owners": "None"}

    owner_prompt = f"List the task owners for each task from these action items:\n{actions}"
    owners = llm.invoke(owner_prompt).text.strip()
    return {"action_items": actions, "owners": owners}

# Step 5: Priority Classification Agent
def priority_agent(state: MeetingState):
    prompt = (
        "Determine the priority level of these tasks (High Priority, Medium Priority, or Low Priority) "
        "based on urgency phrases like 'urgent', 'ASAP', 'by tomorrow', 'this week', or 'before Friday'. "
        "Return only the priority level.\n"
        f"Transcript:\n{state['transcript']}\nTasks:\n{state['action_items']}"
    )
    return {"priority": llm.invoke(prompt).text.strip()}

# Step 6: Conditional Logic
def check_action_items(state: MeetingState):
    if "No action items" in state["action_items"]:
        return "final_output"
    return "priority_agent"

# Step 7: Final Output Node
def final_output(state: MeetingState):
    priority = state.get("priority", "Skipped (No action items identified in this meeting.)")
    report = (
        f"### Meeting Summary\n{state['summary']}\n\n"
        f"### Key Topics\n{state['topics']}\n\n"
        f"### Action Items\n{state['action_items']}\n\n"
        f"### Task Owners\n{state['owners']}\n\n"
        f"### Priority\n{priority}"
    )
    return {"final_report": report}

# Build LangGraph
workflow = StateGraph(MeetingState)
workflow.add_node("input_node", input_node)
workflow.add_node("topic_agent", topic_agent)
workflow.add_node("summary_agent", summary_agent)
workflow.add_node("action_item_agent", action_item_agent)
workflow.add_node("priority_agent", priority_agent)
workflow.add_node("final_output", final_output)

workflow.add_edge(START, "input_node")
workflow.add_edge("input_node", "topic_agent")
workflow.add_edge("topic_agent", "summary_agent")
workflow.add_edge("summary_agent", "action_item_agent")
workflow.add_conditional_edges(
    "action_item_agent",
    check_action_items,
    {"priority_agent": "priority_agent", "final_output": "final_output"}
)
workflow.add_edge("priority_agent", "final_output")
workflow.add_edge("final_output", END)

app = workflow.compile()

# Simple Streamlit UI
st.title("AI Meeting Notes Analyzer")

# Get API Key from the user with a direct link to Google AI Studio
user_api_key = st.text_input(
    "Enter your Google Gemini API Key ([Get your free API key from Google AI Studio](https://aistudio.google.com/app/apikey)):",
    type="password"
)

default_transcript = """John: We need to improve the website performance.
Sarah: Yes, page load time is too slow.
David: I will optimize the database queries this week.
Sarah: I will redesign the homepage layout.
John: Let's try to finish these tasks before Friday."""

transcript = st.text_area("Enter Meeting Transcript:", value=default_transcript, height=180)

if st.button("Analyze"):
    if not user_api_key:
        st.error("Please enter your Gemini API Key above before analyzing.")
    else:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=user_api_key
        )
        result = app.invoke({"transcript": transcript})
        st.markdown(result["final_report"])