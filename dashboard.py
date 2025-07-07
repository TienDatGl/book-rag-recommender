import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import gradio as gr

from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.schema.messages import HumanMessage

load_dotenv()

CHROMA_PATH = './vector_store/chroma'
BOOK_PATH = './data_cleaned/books_with_emotions.csv'

# --- Load Data ---
books = pd.read_csv(BOOK_PATH)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "./images/cover-not-found.png",
    books["large_thumbnail"],
)

def format_authors(authors_string):
    authors = authors_string.split(";")
    if len(authors) > 1:
        return ", ".join(authors[:-1]) + " and " + authors[-1]
    return authors[0]


# --- Prompt Builder ---
def build_prompt(desc: str, user_query: str, user_category: str, user_emotion: str) -> str:
    prompt = "You are an assistant that explains why a particular book is recommended to a reader.\n"

    if user_query:
        prompt += f'The user described their interest as: "{user_query.strip()}".\n'

    has_cat = user_category and user_category.strip().lower() != "all"
    has_emotion = user_emotion and user_emotion.strip().lower() != "all"

    if has_cat and has_emotion:
        prompt += f'They prefer books in the category: "{user_category}" and tone: "{user_emotion}".\n'
    elif has_cat:
        prompt += f'They prefer books in the category: "{user_category}".\n'
    elif has_emotion:
        prompt += f'They prefer an emotional tone: "{user_emotion}".\n'
    else:
        prompt += "They have not specified particular preferences.\n"

    prompt += f"\nBook description:\n\"{desc.strip()}\"\n\n"
    prompt += (
        "In 2–4 sentences, explain clearly and persuasively why this book might appeal to the user. "
        "Focus on matching themes, emotional elements, or storyline."
    )
    return prompt


# --- Explanation Generator ---
def generate_explanations_batch(desc_list, user_query, user_category, user_emotion, llm, batch_size=8):
    prompts = [
        [HumanMessage(content=build_prompt(desc, user_query, user_category, user_emotion))]
        for desc in desc_list
    ]
    reasons = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating reasons"):
        batch = prompts[i:i + batch_size]
        results = llm.generate(batch)
        for r in results.generations:
            reasons.append(r[0].text.strip())
    return reasons


# --- Main Recommendation Logic (RAG) ---
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 5,
) -> pd.DataFrame:
    # Step 1: Semantic search via vector DB
    recs = vectorstore.similarity_search(query, k=initial_top_k)

    # Step 2: Extract ISBNs efficiently using list comprehension
    books_list = [
        int(doc.page_content.split()[0].strip('"'))
        for doc in recs
        if doc.page_content.split()[0].strip('"').isdigit()
    ]

    # Step 3: Filter metadata DataFrame by ISBN
    book_recs = books[books["isbn13"].isin(books_list)].copy()

    # Step 4: Optional category filter
    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    # Step 5: Re-rank by emotional tone
    tone_col_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }
    if tone in tone_col_map and tone_col_map[tone] in book_recs.columns:
        book_recs = book_recs.sort_values(by=tone_col_map[tone], ascending=False)

    # Step 6: Limit to top N
    return book_recs.head(final_top_k)


# --- Display Function for Gradio ---
def recommend_books(query, category, tone):
    book_recs = retrieve_semantic_recommendations(query, category, tone)

    if book_recs.empty:
        return []

    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.4)

    reasons = generate_explanations_batch(
        desc_list=book_recs["description"].tolist(),
        user_query=query,
        user_category=category,
        user_emotion=tone,
        llm=llm
    )
    book_recs["recommendation_reason"] = reasons

    results = []
    book_data = []
    for _, row in book_recs.iterrows():
        short_reason = " ".join(row["recommendation_reason"].split()[:30]) + "..."
        author_str = format_authors(row["authors"])
        caption = f"{row['title']} by {author_str}: {short_reason}"
        results.append((row["large_thumbnail"], caption))

        # store full book data in state
        book_data.append({
            "title": row["title"],
            "authors": author_str,
            "reason": row["recommendation_reason"],
            "description": row["description"],
        })

    return results, book_data


def show_book_detail(evt: gr.SelectData, book_data: list):
    idx = evt.index
    book = book_data[idx]
    return f"""
### {book['title']}  
**Authors:** {book['authors']}

#### 🧠 Why it was recommended:
{book['reason']}

### Book summary:
{book['description']}
"""


# --- Gradio UI ---
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=5, rows=1)
    detail_output = gr.Markdown(label="Book Details")

    # Store book detail info in state
    book_detail_state = gr.State()

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=[output, book_detail_state])

    # On book click, show full details
    output.select(fn=show_book_detail, inputs=book_detail_state, outputs=detail_output)

if __name__ == "__main__":
    dashboard.launch()