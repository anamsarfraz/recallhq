# **RecallHQ (Tech Event Summarizer & Q&A System)**

## **Overview**
In order to help quickly understand both the visual content and spoken discussion during an event (which are hours long or multi-day), we want to build an assistant that leverages LLMs to summarize recordings, interpret visuals, and answer follow-up questions related to the event. This system would significantly reduce the time needed to review long events, and would event content more accessible and actionable.

## **Key Features and Functionality**
### **Event Summarization** 
- Summarizes key points from the event, including discussions, keynote presentations, and Q&A sessions.
- The LLM captures the essence of talks and discussions, extracting key themes, recommendations, and conclusions. It can also create topic-based summaries (e.g., by speaker, session, or panel).

### **Visual Content Interpretation (Language + Vision Model)**
- Interprets slides, charts, and any other visual media shared during the event.
- The LLM can relate visual elements (like infographics or data visualizations) to the topics being discussed, providing context and summary explanations.

### **Interactive Q&A System**
- Allows users to ask follow-up questions about specific sessions, speakers, or visual content after the event.
- Answers specific queries like, “What was the main takeaway from Speaker X’s session?” or “What were the key statistics shown in the slide about market trends?”

### **Session Categorization and Indexing**
- Automatically segments and categorizes different parts of the event based on topics, speakers, or themes. Given the longer duration of events, this feature would be critical to ensure smooth navigation of multi-day or multi-session events.
- The LLM organizes event content into searchable categories, making it easy for users to find specific parts of the event they are interested in (e.g., “AI in Healthcare” or “Keynote by John Doe”).

### **Discord Integration for Events** 
- After an event on Discord, like a webinar, live Q&A, or AMA (Ask Me Anything), RecallHQ could be used to generate the summary of the discussion highlighting key takeaways. The users can review the key points and ask questions without scrolling through long chats.

## **Tech Stack**
### Language
- Python 3.12
- Packages are noted in the requirements.txt

## **How to run the app**
- Make sure you have Python 3.12 installed
- Setup venv and activate it
    * `python3.12 -m venv venv`
    * `source venv/bin/activate`
- Run `pip install -r requirements.txt` to install the dependencies
- Run `chainlit run Home.py` to start the app
